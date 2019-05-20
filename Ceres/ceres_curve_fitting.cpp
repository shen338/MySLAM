#include "ceres/ceres.h"
#include "glog/logging.h"
#include <chrono>
#include <iostream>
#include <vector> 
#include <opencv2/core/core.hpp>

using namespace ceres;
using namespace std;

// Curve fitting for function "y = e^(a*x*x + b*x + c)"
struct CostFunctor{
    CostFunctor(double x, double y): x_(x), y_(y) {} // This initialization is for input data points 
    template<typename T> bool operator()(const T* const parameters, T* residual) const {
        residual[0] = T(y_) - ceres::exp(parameters[0] * T(x_) * T(x_) + parameters[1] * T(x_) + parameters[2]);
        return true;
    }
    private: 
        double x_, y_;
};

int main(){
    cv::RNG rng;  
    double a=1.0, b=2.0, c=1.0;   // Real parameters     
    int N = 200; 
    double w_sigma=1.0;                   

    vector<double> x_data, y_data;      // generated data

    cout<<"generating data: "<<endl;
    for ( int i=0; i<N; i++ )
    {
        double x = i/100.0;
        x_data.push_back ( x );
        y_data.push_back (
            exp ( a*x*x + b*x + c ) + rng.gaussian ( w_sigma )
        );
        cout<<x_data[i]<<" "<<y_data[i]<<endl;
    }

    double parameters[3] = {0,0,0};  // Parameters need to be solved

    ceres::Problem problem;              // setup a new problem
    for(int i = 0; i<N; i++){
        problem.AddResidualBlock(
            // <error/residual function, output dimension, input dimension>
            new ceres::AutoDiffCostFunction<CostFunctor, 1, 3> (
                new CostFunctor(x_data[i], y_data[i])
            ),
            new CauchyLoss(0.5),   // Loss functions: http://ceres-solver.org/nnls_modeling.html#instances
            parameters
        );
    }

    // Setup solver
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;         // Linear equation solver type
    options.minimizer_progress_to_stdout = true; 

    ceres::Solver::Summary summary;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    ceres::Solve (options, &problem, &summary);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();

    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
    cout<<"solve time cost = "<<time_used.count()<<" seconds. "<<endl;

    // cout << summary.FullReport() << endl;
    cout<<summary.BriefReport() <<endl;
    cout<<"estimated a,b,c = ";
    for ( auto a:parameters ) cout<<a<<" ";
    cout<<endl;

}