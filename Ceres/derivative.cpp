#include "ceres/ceres.h"
#include "glog/logging.h"

#include <chrono>
#include <iostream>
#include <vector> 
#include <opencv2/core/core.hpp>

using namespace ceres;
using namespace std;

struct CostFunctor {
   template <typename T>
   bool operator()(const T* const x, T* residual) const {
     residual[0] = T(10.0) - x[0];
     return true;
   }
};

// Numerical functor for (10-x)^2
struct NumericDiffCostFunctor {
  bool operator()(const double* const x, double* residual) const {
    residual[0] = (10.0 - x[0])*(10.0 - x[0]);
    return true;
  }
};

// Analystic derivative for 10-x
class QuadraticCostFunction : public ceres::SizedCostFunction<1, 1> {
 public:
  virtual ~QuadraticCostFunction() {}
  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    const double x = parameters[0][0];
    residuals[0] = 10 - x;

    // Compute the Jacobian if asked for.
    if (jacobians != NULL && jacobians[0] != NULL) {
      jacobians[0][0] = -1;
    }
    return true;
  }
};

int main(int argc, char** argv){

    // Automatic derivative
    google::InitGoogleLogging(argv[0]);

    // The variable to solve for with its initial value.
    double initial_x = 5.0;
    double x = initial_x;

    // Build the problem.
    Problem problem;

    // Set up the only cost function (also known as residual). This uses
    // auto-differentiation to obtain the derivative (jacobian).
    CostFunction* cost_function =
        new AutoDiffCostFunction<CostFunctor, 1, 1>(new CostFunctor);
    problem.AddResidualBlock(cost_function, NULL, &x);

    // Run the solver!
    Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    Solver::Summary summary;
    Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << "\n";
    std::cout << "x : " << initial_x
                << " -> " << x << "\n";

    // In some cases, its not possible to define a templated cost functor, 
    // for example when the evaluation of the residual involves a call to a library function that you do not have control over.
    // Numerical derivative
    CostFunction* cost_function_numerical =
        new NumericDiffCostFunction<NumericDiffCostFunctor, ceres::CENTRAL, 1, 1>(
            new NumericDiffCostFunctor);
    problem.AddResidualBlock(cost_function_numerical, NULL, &x);

    // In some cases, using automatic differentiation is not possible. 
    // For example, it may be the case that it is more efficient to compute the derivatives 
    // in closed form instead of relying on the chain rule used by the automatic differentiation code.
    // Analystic derivative. 

    CostFunction* cost_function_analystic = new QuadraticCostFunction;
    problem.AddResidualBlock(cost_function_analystic, NULL, &x);

    return 0;
}