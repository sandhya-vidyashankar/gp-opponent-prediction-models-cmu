#include "mpc_model.h"
using namespace casadi;

MPCSolver::MPCSolver(){
    this->opti = Opti();
    this->all = Slice();
}

MPCSolver::MPCSolver(Config config){
    Slice all;
    this->all = all;
    this->config = config;
}

void MPCSolver::get_waypoints(Waypoints waypoints){
    this->full_ref_states = waypoints;
    return;
}

void MPCSolver::update_car_state(CarState& car_state) {
    this->car_state = car_state;
    return;
}

int MPCSolver::find_nearest_waypoint() {
    float min_dist = 1000000;
    int min_index = 0;
    for(int i=0; i < int(this->full_ref_states.x.size()); i++) {
        float dist = sqrt(pow(this->full_ref_states.x[i]-this->car_state.x, 2) + pow(this->full_ref_states.y[i]-this->car_state.y, 2));
        if(dist < min_dist) {
            min_dist = dist;
            min_index = i;
        }
    }
    //this->progress = min_index; 
    return min_index;
}
void MPCSolver::get_reference_states_within_horizon(int closest_index){ //Closest index is s from mpcc
    // std::cout << "Starting to ccollect reference states";
    Waypoints ref_states;
    int i = closest_index;
    float lookahead_vel = max(this->car_state.v, float(config.v_max));
    while(int(ref_states.x.size()) < this->config.N+1) {
        ref_states.x.push_back(this->full_ref_states.x[i]);
        ref_states.y.push_back(this->full_ref_states.y[i]);
        float c_theta = this->full_ref_states.theta[i];
        if(c_theta - this->car_state.theta > 4.5){
            c_theta -= 2*M_PI;
        }
        else if(this->car_state.theta - c_theta > 4.5){
            c_theta += 2*M_PI;
        }
        ref_states.theta.push_back(c_theta);
        i = min((i + int(lookahead_vel*this->config.dt/this->config.dtk)), int(this->full_ref_states.x.size()) - 1);
    }
    
    this->Xref = DM::vertcat({DM(ref_states.x).T(), DM(ref_states.y).T(), DM(ref_states.theta).T()});
    
    return;
    
}

MX MPCSolver::f(MX X, MX U) {
    MX x_dot = MX::cos(X(2))*X(3);//MX::cos(X(2))*U(1);
    MX y_dot = MX::cos(X(2))*X(3);//MX::sin(X(2))*U(1);
        //0.33 is wheelbase of F1/10
    MX theta_dot = X(3)*MX::tan(U(1))/this->config.wheelbase;
    MX v_dot = U(0);

    MX X_dot = MX::vertcat({x_dot, y_dot, theta_dot, v_dot});
    return X_dot;
}



tuple<double,  double, vector<tuple<double, double>>> MPCSolver::solve(CarState& car_state, float delta0, float a0) {
    // Cost function
    this->update_car_state(car_state);
    DM Q = DM::zeros(3,3);
    Q(0,0) = this->config.Q[0];
    Q(1,1) = this->config.Q[1];
    Q(2,2) = this->config.Q[2];
    DM R = DM::zeros(2,2);
    R(0,0) = this->config.R[0];
    R(1,1) = this->config.R[1];
    DM QN = DM::zeros(3,3);
    QN(0,0) = this->config.QN[0];
    QN(1,1) = this->config.QN[1];
    QN(2,2) = this->config.QN[2];
    this->X0 = DM::vertcat({this->car_state.x, this->car_state.y, this->car_state.theta, this->car_state.v});
    
    DM Qc = DM::zeros(1,1);
    DM Ql = DM::zeros(1,1);
    DM Qs = DM::zeros(1,1);
    Qc = this->config.Q[0];
    Ql = this->config.Q[1];
    Qs = this->config.Q[2];
    
    // Variables
    this->opti = Opti();
    this->X = this->opti.variable(4, this->config.N+1); //still the state
    //MX Xstate = X(casadi::Slice(0, X.size1()), casadi::Slice(0, 3));
    this->U = this->opti.variable(2, this->config.N);
    this->x = this->X(0, this->all);
    this->y = this->X(1, this->all);
    this->theta = this->X(2, this->all);
    this->v = this->X(3, this->all);
    this->u_a = this->U(0, this->all);
    this->u_delta = this->U(1, this->all);
    this->s = this->opti.variable(1, this->config.N+1); // Track progress


    // Updates Xref
    get_reference_states_within_horizon(find_nearest_waypoint());
    MX Xref = this->Xref;

    MX Cost = MX::zeros(1,1);
    MX U_prev;
    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "set up variables");
    // Path cost 
    // for(int i=0; i < this->config.N; i++) {
    //     Cost = Cost + MX::mtimes(MX::mtimes((this->X(all,i)-this->Xref(all,i)).T(), Q), (this->X(all,i)-this->Xref(all,i))) + MX::mtimes(MX::mtimes(this->U(all,i).T(), R), this->U(all,i));
    // }
    // RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "i max %i", this->config.N);
    for(int i=0; i < this->config.N -1; i++) {
        // Linearized contouring and lag error
        MX e_cont = -sin(Xref(2, i))* (Xref(0, i) - x(i)) + cos(Xref(2,i))*(Xref(1, i) - y(i));

        MX e_lag = sin(Xref(2, i))* (Xref(0, i) - x(i)) + cos(Xref(2,i))*(Xref(1, i) - y(i)); //maybe e_lag x ref can be replaced with GP of TV
        
        // Actual cost is contouring + lag + input - progress + delta_input
        Cost += bilin(Qc, e_cont, e_cont) + bilin(Ql, e_lag, e_lag); //contouring + lag
        Cost +=  bilin(R, U(this->all, i)) ;
        
        Cost -= Qs * s(i) * config.dt ; //input - progress
        // RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Got s");
        if(i>0){
            Cost += bilin(R, (U(this->all, i) - U_prev)); //delta_input 
            U_prev = U(this->all, i);
            // RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Running delta input");
        }
        else{
            Cost += bilin(R, U(this->all, i))*0.5; //delta_input
            U_prev = U(this->all, i);
        }
        // RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "i %i", i);
    }
    //TODO: Make an obstacle slack
    // Terminal cost
    //Cost = Cost + MX::mtimes(MX::mtimes((this->X(all,this->config.N)-this->Xref(all,this->config.N)).T(), QN), (this->X(all,this->config.N)-this->Xref(all,this->config.N)));
    //  RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "DONE");

    // Dynamic Constraints - RK4
    for(int i=0; i < this->config.N; i++) {
        //RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "i %i", i);
        MX k1 = this->f(this->X(all,i), this->U(all,i));
        MX k2 = this->f(this->X(all,i) + k1*this->config.dt/2, this->U(all,i));
        MX k3 = this->f(this->X(all,i) + k2*this->config.dt/2, this->U(all,i));
        MX k4 = this->f(this->X(all,i) + k3*this->config.dt, this->U(all,i));
        //  RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Got ks");
        this->opti.subject_to(this->X(all,i+1) == this->X(all,i) + (k1 + 2*k2 + 2*k3 + k4)*(this->config.dt/6));
        //  RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Why am I alive");
         if(i<this->config.N - 1){
            this->opti.subject_to(s(0, i+1) == s(0, i) + this->v(0, i)*config.dt);
         }
        
    }
    // RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "DONE");

    // Control constraints
    this->opti.subject_to(this->X(all,0) == this->X0);
    // RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "X0");
    this->opti.subject_to(this->u_a >= this->config.v_min);
    this->opti.subject_to(this->u_a <= this->config.v_max);
    // RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "v");
    this->opti.subject_to(this->u_delta <= this->config.delta_max);
    this->opti.subject_to(this->u_delta >= this->config.delta_min);
    this->opti.subject_to(s(0) == 0);

    //TODO- add track constraints
    // Setting solver options
    casadi::Dict opts;
    opts["ipopt.print_level"] = 0;
    opts["print_time"] = false;
    opts["ipopt.mu_strategy"] = "adaptive";
    opts["ipopt.tol"] = 1e-5;

    // RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "Initial guessing:");
    // Warm solve with initial guess
    DM Xref_with_v = DM::vertcat({this->Xref, 1*DM::ones(1, Xref.size2())});
    DM initial_X = DM::horzcat({this->X0, Xref_with_v(all, Slice(0, this->config.N))});
    // RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "inirial x");
    // DM initial_U = DM::horzcat({DM::vertcat({a0, delta0}), DM::zeros(2, this->config.N-1)});
     DM initial_U = DM::zeros(2, this->config.N);
    this->opti.set_initial(this->X, initial_X);
    // this->opti.set_initial(this->x, initial_X(0, all));
    // this->opti.set_initial(this->y, initial_X(1, all));
    // this->opti.set_initial(this->theta, initial_X(2, all));
    this->opti.set_initial(this->U, initial_U);
    // OptiAdvanced mycopy = debug();
    // cout << "Values" << mycopy.value(this->X);
    // Solution step
    this->opti.solver("ipopt", opts);
    this->opti.minimize(Cost);
    OptiSol solution = this->opti.solve();
    
    auto sol = solution.value(this->U);
    RCLCPP_INFO(rclcpp::get_logger("rclcpp"), "deltas %f %f %f %f", double(sol(1,0)), double(sol(1,1)), double(sol(1,2)), double(sol(1,3)));
    // Formatting and sending solution to controller
    double acc = double(sol(0,0));
    double delta = double(sol(1,0));
    vector<tuple<double, double>> path;
    for(int i=0; i < this->config.N+1; i++) {
        path.push_back(make_tuple(double(solution.value(this->X(0,i))), double(solution.value(this->X(1,i)))));
    }
    return make_tuple(delta, acc, path);
}


