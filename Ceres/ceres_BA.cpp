#include "ceres/ceres.h"
#include "glog/logging.h"
#include "flags/command_args.h"
#include "common/BALProblem.h"
#include "common/BundleParams.h"
#include <chrono>
#include <iostream>
#include <vector> 
#include <opencv2/core/core.hpp>

using namespace ceres;
using namespace std;

// camera parameter is 9 dimensional. Three translation, three rotation, three distortion 
struct SnavelyReprojectionError {
    SnavelyReprojectionError(double observed_x, double observed_y)
        : observed_x(observed_x), observed_y(observed_y) {}
        
    template <typename T>
    bool operator()(const T* const camera,
                    const T* const point,
                    T* residuals) const {
        // camera[0,1,2] are the angle-axis rotation.
        T p[3];
        ceres::AngleAxisRotatePoint(camera, point, p);
        // camera[3,4,5] are the translation.
        p[0] += camera[3]; p[1] += camera[4]; p[2] += camera[5];

        // Compute the center of distortion. The sign change comes from
        // the camera model that Noah Snavely's Bundler assumes, whereby
        // the camera coordinate system has a negative z axis.
        T xp = - p[0] / p[2];
        T yp = - p[1] / p[2];

        // Apply second and fourth order radial distortion.
        const T& l1 = camera[7];
        const T& l2 = camera[8];
        T r2 = xp*xp + yp*yp;
        T distortion = T(1.0) + r2  * (l1 + l2  * r2);

        // Compute final projected point position.
        const T& focal = camera[6];
        T predicted_x = focal * distortion * xp;
        T predicted_y = focal * distortion * yp;

        // The error is the difference between the predicted and observed position.
        residuals[0] = predicted_x - T(observed_x);
        residuals[1] = predicted_y - T(observed_y);
        return true;
    }

    // Factory to hide the construction of the CostFunction object from
    // the client code.
    static ceres::CostFunction* Create(const double observed_x,
                                        const double observed_y) {
        return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>(
                    new SnavelyReprojectionError(observed_x, observed_y)));
    }

    double observed_x;
    double observed_y;

};

void buildProblem(BALProblem* bal_problem, Problem* problem, const BundleParams& params){
    const int point_block_size = bal_problem->point_block_size();
    const int camera_block_size = bal_problem->camera_block_size();
    double* points = bal_problem->mutable_points();
    double* cameras = bal_problem->mutable_cameras();

    const double* observations = bal_problem->observations();

    for(int i = 0;i<bal_problem->num_observations;i++){
        CostFunction* cost_function;

        cost_function = SnavelyReprojectionError::Create(observations[2*i], observations[2*i+1]);

        LossFunction* loss_function = new HuberLoss(1.0);

        double* camera = camera + camera_block_size * bal_problem->camera_index()[i];
        double* point = point + point_block_size * bal_problem->point_index()[i];

        problem->AddResidualBlock(cost_function, loss_function, camera, point);

    }
}

void setOptimOrder(BALProblem* bal_problem, Solver::Options* options, const BundleParams& params){
    const int point_block_size = bal_problem->point_block_size();
    const int camera_block_size = bal_problem->camera_block_size();
    double* points = bal_problem->mutable_points();
    double* cameras = bal_problem->mutable_cameras();

    ParameterBlockOrdering* ordering = new ParameterBlockOrdering;
    for(int i = 0;i<bal_problem->num_points();i++){
        ordering->AddElementToGroup(points + point_block_size*i, 0);
    }

    for(int i = 0;i<bal_problem->num_cameras();i++){
        ordering->AddElementToGroup(cameras + camera_block_size*i, 1);
    }

    options->linear_solver_ordering.reset(ordering);
}