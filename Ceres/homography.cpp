#include "ceres/ceres.h"
#include "glog/logging.h"
#include <eigen3/Eigen/Core>

using namespace std;
using namespace ceres;


typedef Eigen::NumTraits<double> EigenDouble;

typedef Eigen::MatrixXd Mat;
typedef Eigen::VectorXd Vec;
typedef Eigen::Matrix<double, 3, 3> Mat3;
typedef Eigen::Matrix<double, 2, 1> Vec2;
typedef Eigen::Matrix<double, Eigen::Dynamic, 8> MatX8;
typedef Eigen::Vector3d Vec3;


struct estimateHomographyOptions{
    estimateHomographyOptions():
        max_num_iterations(50), 
        expected_average_symmetric_distance(1e-10) {}
        
    int max_num_iterations;
    int expected_average_symmetric_distance;

};

template <typename T>
void SymmetricGeometricDistanceTerms(const Eigen::Matrix<T, 3, 3> &H, 
                                    const Eigen::Matrix<T, 2, 1> &x1,
                                    const Eigen::Matrix<T, 2, 1> &x2,
                                    T forward_error[2],
                                    T backward_error[2]){
    
    typedef Eigen::Matrix<T, 3, 1> Vec3;
    Vec3 x(x1(0), x1(1), T(1.0));
    Vec3 y(x2(0), x2(1), T(1.0));

    Vec3 H_x = H * x;
    Vec3 Hinv_y = H.inverse() * y;

    H_x /= H_x(2);
    Hinv_y /= Hinv_y(2);  // homogeneous

    forward_error[0] = H_x(0) - y(0);
    forward_error[1] = H_x(1) - y(1);
    backward_error[0] = Hinv_y(0) - x(0);
    backward_error[1] = Hinv_y(1) - x(1);
                                
}

double SymmetricGeometricDistance(const Mat3 &H,
                                  const Vec2 &x1,
                                  const Vec2 &x2) {
  Vec2 forward_error, backward_error;
  SymmetricGeometricDistanceTerms<double>(H,
                                          x1,
                                          x2,
                                          forward_error.data(),
                                          backward_error.data());
                                          // Template above
  return forward_error.squaredNorm() +
         backward_error.squaredNorm();
}

template<typename T = double>
class Homography2DNormalizedParameterization {
    public:
    typedef Eigen::Matrix<T, 8, 1> Parameters;     // a, b, ... g, h
    typedef Eigen::Matrix<T, 3, 3> Parameterized;  // H

    // Convert 8 to 9
    static void Convert8to9(const Parameters &p, Parameterized &h){
        h << p(0), p(1), p(2),
          p(3), p(4), p(5),
          p(6), p(7), 1.0;
    }

    static void Convert9to8(const Parameterized &h, Parameters &p){
        p << h(0, 0), h(0, 1), h(0, 2),
          h(1, 0), h(1, 1), h(1, 2),
          h(2, 0), h(2, 1);
    }
};

// First, solve Homography problem with Eigen
// 2D Homography transformation estimation in the case that points are in
// euclidean coordinates.
//
//   x = H y
//
// x and y vector must have the same direction, we could write
//
//   crossproduct(|x|, * H * |y| ) = |0|
//
//   | 0 -1  x2|   |a b c|   |y1|    |0|
//   | 1  0 -x1| * |d e f| * |y2| =  |0|
//   |-x2  x1 0|   |g h 1|   |1 |    |0|
//
// That gives:
//
//   (-d+x2*g)*y1    + (-e+x2*h)*y2 + -f+x2          |0|
//   (a-x1*g)*y1     + (b-x1*h)*y2  + c-x1         = |0|
//   (-x2*a+x1*d)*y1 + (-x2*b+x1*e)*y2 + -x2*c+x1*f  |0|
//

bool Homography2DFromCorrespondencesLinearEuc(
    const Mat &x1,
    const Mat &x2,
    Mat3 &H,
    double expected_precision) {
    assert(2 == x1.rows());
    assert(4 <= x1.cols());
    assert(x1.rows() == x2.rows());
    assert(x1.cols() == x2.cols());
    int n = x1.cols();
    MatX8 L = Mat::Zero(n * 3, 8);
    Mat b = Mat::Zero(n * 3, 1);
    for (int i = 0; i < n; ++i) {
        int j = 3 * i;
        L(j, 0) =  x1(0, i);             // a
        L(j, 1) =  x1(1, i);             // b
        L(j, 2) =  1.0;                  // c
        L(j, 6) = -x2(0, i) * x1(0, i);  // g
        L(j, 7) = -x2(0, i) * x1(1, i);  // h
        b(j, 0) =  x2(0, i);             // i
        ++j;
        L(j, 3) =  x1(0, i);             // d
        L(j, 4) =  x1(1, i);             // e
        L(j, 5) =  1.0;                  // f
        L(j, 6) = -x2(1, i) * x1(0, i);  // g
        L(j, 7) = -x2(1, i) * x1(1, i);  // h
        b(j, 0) =  x2(1, i);             // i
        // This ensures better stability
        // TODO(julien) make a lite version without this 3rd set
        ++j;
        L(j, 0) =  x2(1, i) * x1(0, i);  // a
        L(j, 1) =  x2(1, i) * x1(1, i);  // b
        L(j, 2) =  x2(1, i);             // c
        L(j, 3) = -x2(0, i) * x1(0, i);  // d
        L(j, 4) = -x2(0, i) * x1(1, i);  // e
        L(j, 5) = -x2(0, i);             // f
    }
    // Solve Lx=B
    const Vec h = L.fullPivLu().solve(b);
    Homography2DNormalizedParameterization<double>::Convert8to9(h, H);
    return (L * h).isApprox(b, expected_precision);
}

class HomographySymmetricGeometricCostFunctor{
    public:
    HomographySymmetricGeometricCostFunctor(const Vec2 &x, const Vec2 &y):
        x_(x), y_(y) {}
    
    template<typename T> 
    bool operator()(const T* homography_parameters, T* residuals) const{
        typedef Eigen::Matrix<T, 3, 3> Mat3;
        typedef Eigen::Matrix<T, 2, 1> Vec2;
        Mat3 H(homography_parameters);
        Vec2 x(T(x_(0)), T(x_(1)));
        Vec2 y(T(y_(0)), T(y_(1)));

        SymmetricGeometricDistanceTerms<T>(H, x, y, 
                                    &residuals[0], &(residuals[2]));
        return true;
    }
    const Vec2 x_;
    const Vec2 y_;
};

class terminationCheckingCallback: public ceres::IterationCallback{
    private: 
    const estimateHomographyOptions &options_;
    const Mat &x1_;
    const Mat &x2_;
    Mat3 H_;
    public:
    terminationCheckingCallback(const Mat &x1, const Mat &x2, 
                               const estimateHomographyOptions &options, 
                               Mat3 &H):
                options_(options), x1_(x1), x2_(x2), H_(H) {}

    virtual CallbackReturnType operator()(const IterationSummary &summary){
        if(!summary.step_is_successful){
            return SOLVER_CONTINUE;
        }

        double average_distance = 0.0;
        for(int i = 0;i<x1_.cols();i++){
            average_distance += SymmetricGeometricDistance(H_, 
                                                    x1_.col(i), x2_.col(i));
        }
        average_distance /= x1_.cols();

        if(average_distance <= options_.expected_average_symmetric_distance){
            return SOLVER_TERMINATE_SUCCESSFULLY;
        }

        return SOLVER_CONTINUE;
    }
};
//Solver main function 
bool EstimateHomography2DFromCorrespondences(
    const Mat &x1,
    const Mat &x2,
    const estimateHomographyOptions &options,
    Mat3 &H) {
    assert(2 == x1.rows());
    assert(4 <= x1.cols());
    assert(x1.rows() == x2.rows());
    assert(x1.cols() == x2.cols());

    Homography2DFromCorrespondencesLinearEuc(x1, x2, H, 
                            EigenDouble::dummy_precision());

    LOG(INFO) << "Estimated matrix after algebraic estimation:\n" << H;

    ceres::Problem problem;
    for(int i = 0;i<x1.cols();i++){
        HomographySymmetricGeometricCostFunctor *homograghy_cost_functor = 
        new HomographySymmetricGeometricCostFunctor(x1.col(i), x2.col(i));

        problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<
            HomographySymmetricGeometricCostFunctor,
            4,  // num_residuals
            9>(homograghy_cost_functor),
        NULL,
        H.data());
    }

    Solver::Options solver_options;
    solver_options.linear_solver_type = ceres::DENSE_QR;
    solver_options.max_num_iterations = options.max_num_iterations;
    solver_options.update_state_every_iteration = true;

    terminationCheckingCallback callback(x1, x2, options, H);
    solver_options.callbacks.push_back(&callback);

    Solver::Summary summary;
    Solve(solver_options, &problem, &summary);

    LOG(INFO) << "Summary:\n" << summary.FullReport();
    LOG(INFO) << "Final refined matrix:\n" << H;
    return summary.IsSolutionUsable();
}


int main(int argc, char **argv) {
    google::InitGoogleLogging(argv[0]);
    Mat x1(2, 100);
    for (int i = 0; i < x1.cols(); ++i) {
        x1(0, i) = rand() % 1024;
        x1(1, i) = rand() % 1024;
    }

    Mat3 homography_matrix;
    homography_matrix << 1.243715, -0.461057, -111.964454,
                       0.0,       0.617589, -192.379252,
                       0.0,      -0.000983,    1.0;
    Mat x2 = x1;
    for (int i = 0; i < x2.cols(); ++i) {
        Vec3 homogenous_x1 = Vec3(x1(0, i), x1(1, i), 1.0);
        Vec3 homogenous_x2 = homography_matrix * homogenous_x1;
        x2(0, i) = homogenous_x2(0) / homogenous_x2(2);
        x2(1, i) = homogenous_x2(1) / homogenous_x2(2);
        // Apply some noise so algebraic estimation is not good enough.
        x2(0, i) += static_cast<double>(rand() % 1000) / 5000.0;
        x2(1, i) += static_cast<double>(rand() % 1000) / 5000.0;
    }

    Mat3 estimated_matrix;

    estimateHomographyOptions options;
    options.expected_average_symmetric_distance = 0.02;

    EstimateHomography2DFromCorrespondences(x1, x2, options, estimated_matrix);
    estimated_matrix /= estimated_matrix(2,2);
    
    std::cout << "Original matrix:\n" << homography_matrix << "\n";
    std::cout << "Estimated matrix:\n" << estimated_matrix << "\n";
    
    return EXIT_SUCCESS;

}