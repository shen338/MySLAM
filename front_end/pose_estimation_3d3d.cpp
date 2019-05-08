#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <chrono>

using namespace std;
using namespace cv;

Point2d pixel2cam ( const Point2d& p, const Mat& K )
{
    return Point2d
           (
               ( p.x - K.at<double> ( 0,2 ) ) / K.at<double> ( 0,0 ),
               ( p.y - K.at<double> ( 1,2 ) ) / K.at<double> ( 1,1 )
           );
}

class EdgeProjectXYZRGBDPoseOnly: public g2o::BaseUnaryEdge<3, Eigen::Vector3d, g2o::VertexSE3Expmap>
{
    public: 
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    EdgeProjectXYZRGBDPoseOnly(const Eigen::Vector3d &point): _point(point) {}

    virtual void computeError(){
        const g2o::VertexSE3Expmap* pose = static_cast<const g2o::VertexSE3Expmap*> (_vertices[0]);
        _error = _measurement - pose->estimate().map(_point);
    }

    virtual void linearizeOplus(){
        g2o::VertexSE3Expmap* pose = static_cast<g2o::VertexSE3Expmap*> (_vertices[0]);
        g2o::SE3Quat T(pose->estimate());
        Eigen::Vector3d xyz_trans = T.map(_point);
        double x = xyz_trans[0];
        double y = xyz_trans[1];
        double z = xyz_trans[2];

        _jacobianOplusXi(0, 0) = 0;
        _jacobianOplusXi(0, 0) = -z;
        _jacobianOplusXi(0, 0) = y;
        _jacobianOplusXi(0, 0) = -1;
        _jacobianOplusXi(0, 0) = 0;
        _jacobianOplusXi(0, 0) = 0;

        _jacobianOplusXi(0, 0) = z;
        _jacobianOplusXi(0, 0) = 0;
        _jacobianOplusXi(0, 0) = -x;
        _jacobianOplusXi(0, 0) = 0;
        _jacobianOplusXi(0, 0) = -1;
        _jacobianOplusXi(0, 0) = 0;

        _jacobianOplusXi(0, 0) = -y;
        _jacobianOplusXi(0, 0) = x;
        _jacobianOplusXi(0, 0) = 0;
        _jacobianOplusXi(0, 0) = 0;
        _jacobianOplusXi(0, 0) = 0;
        _jacobianOplusXi(0, 0) = -1;
    }

    bool read(istream &in) {}
    bool write(ostream &out) const {}
    protected:
    Eigen::Vector3d _point;
};

void feature_matching(const Mat& img1, const Mat& img2,
    std::vector<KeyPoint>& keypoints1,
    std::vector<KeyPoint>& keypoints2,
    std::vector< DMatch >& matches)
    {
        Mat descriptors_1, descriptors_2;
        Ptr<FeatureDetector> detector = ORB::create();
        Ptr<DescriptorExtractor> descriptor = ORB::create();
        
        // compute FAST corners 
        detector->detect(img1, keypoints1);
        detector->detect(img2, keypoints2);

        // compute BRIEF descriptor
        descriptor->compute(img1, keypoints1, descriptors_1);
        descriptor->compute(img2, keypoints2, descriptors_2);

        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(4);
        /*
        FLANNBASED = 1, 
        BRUTEFORCE = 2, 
        BRUTEFORCE_L1 = 3, 
        BRUTEFORCE_HAMMING = 4, 
        BRUTEFORCE_HAMMINGLUT = 5, 
        BRUTEFORCE_SL2 = 6 */

        vector<DMatch> allmatches;
        matcher->match(descriptors_1, descriptors_2, allmatches);

        // Filter all match points
        // Find maximum and minimum distance
        double min_dist = 1000000, max_dist = 0;
        for ( int i = 0; i < descriptors_1.rows; i++ )
        {
            double dist = allmatches[i].distance;
            if ( dist < min_dist ) min_dist = dist;
            if ( dist > max_dist ) max_dist = dist;
        }

        cout << "max distance: " << max_dist << endl;
        cout << "min distance: " << min_dist << endl;

        for ( int i = 0; i < descriptors_1.rows; i++ )
        {
            if ( allmatches[i].distance <= max ( 2*min_dist, 30.0 ) )
            {
                matches.push_back ( allmatches[i] );
            }
        }
    }

void pose_estimation_3d3d(const vector<Point3f> &pts1, 
    const vector<Point3f> &pts2, Mat &R, Mat &t){
        Point3f p1, p2; // Center of mass to reduce variance
        int N = pts1.size();
        for(int i = 0;i<N;i++){
            p1 += pts1[i];
            p2 += pts2[i];
        }
        p1 /= N;
        p2 /= N;

        vector<Point3f> q1(N), q2(N);
        for(int i = 0;i<N;i++){
            q1[i] = pts1[i] - p1;
            q2[i] = pts2[i] - p2;
        }

        Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
        for(int i = 0;i<N;i++){
            W += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) * Eigen::Vector3d(q2[i].x, q2[i].y, q2[i].z).transpose(); 
        }

        cout << "W " << W << endl;

        // SVD on W
        Eigen::JacobiSVD<Eigen::Matrix3d> svd ( W, Eigen::ComputeFullU|Eigen::ComputeFullV );
        
        Eigen::Matrix3d U = svd.matrixU();
        Eigen::Matrix3d V = svd.matrixV();

        cout << "Matrix U: " << U << endl;
        cout << "Matrix V: " << V << endl;

        Eigen::Matrix3d R_ = U * (V.transpose());
        Eigen::Vector3d t_ = Eigen::Vector3d(p1.x, p1.y, p1.z) - R_ * Eigen::Vector3d(p2.x, p2.y, p2.z);
        R = (Mat_<double>(3,3) << 
           R_(0, 0), R_(0, 1), R_(0, 2),
           R_(1, 0), R_(1, 1), R_(1, 2),
           R_(2, 0), R_(2, 1), R_(2, 2));
        t = (Mat_<double>(3,1) << t_(0, 0), t_(1, 0), t_(2, 0));
    }

void bundleAdjustment(const vector<Point3f> &pts1, 
    const vector<Point3f> &pts2, Mat &R, Mat &t){

        typedef g2o::BlockSolver< g2o::BlockSolverTraits<6, 3>> Block;
        
        // Block::LinearSolverType* linearSolver = new g2o::LinearSolverEigen<Block::PoseMatrixType>(); 
        Block::LinearSolverType* linearSolver = new g2o::LinearSolverEigen<Block::PoseMatrixType>();
        Block *solver_ptr = new Block(linearSolver);

        g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);

        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm( solver );

        // vertex
        g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap(); // camera pose
        pose->setId(0);
        pose->setEstimate( g2o::SE3Quat(
            Eigen::Matrix3d::Identity(),
            Eigen::Vector3d( 0,0,0 )));
        
        optimizer.addVertex( pose );

        // Add unary edges
        for(int i = 0;i<pts1.size();i++){
            EdgeProjectXYZRGBDPoseOnly* edge = new EdgeProjectXYZRGBDPoseOnly(
                    Eigen::Vector3d(pts2[i].x, pts2[i].y, pts2[i].z) );
            edge->setId(i+1);
            edge->setVertex(0, dynamic_cast<g2o::VertexSE3Expmap*> (pose));          // Dynamic cast
            edge->setMeasurement(Eigen::Vector3d(pts1[i].x, pts1[i].y, pts1[i].z));  // Reprojection error
            edge->setInformation(Eigen::Matrix3d::Identity()*1e4);
            optimizer.addEdge(edge);
        }

        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
        optimizer.setVerbose( true );
        optimizer.initializeOptimization();
        optimizer.optimize(10);
        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        // Routine
        chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2-t1);
        cout<<"optimization costs time: "<<time_used.count()<<" seconds."<<endl;

        cout<<endl<<"after optimization:"<<endl;
        cout<<"T="<<endl<<Eigen::Isometry3d( pose->estimate() ).matrix()<<endl;

    }

int main(int argc, char** argv){

     if ( argc != 5 )
    {
        cout<<"usage: pose_estimation_3d2d img1 img2 depth1 depth2"<<endl;
        return 1;
    }

    Mat img_1 = imread ( argv[1], CV_LOAD_IMAGE_COLOR );
    Mat img_2 = imread ( argv[2], CV_LOAD_IMAGE_COLOR );

    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    feature_matching ( img_1, img_2, keypoints_1, keypoints_2, matches );
    cout<<"Total "<<matches.size() <<" Matching Points" <<endl;

    Mat d1 = imread ( argv[3], CV_LOAD_IMAGE_UNCHANGED );    
    Mat d2 = imread ( argv[4], CV_LOAD_IMAGE_UNCHANGED );     

    Mat K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );
    vector<Point3f> pts1_3d, pts2_3d;
    // vector<Point2f> pts_2d;
    for ( DMatch m:matches )
    {
        ushort depth1 = d1.ptr<unsigned short> (int ( keypoints_1[m.queryIdx].pt.y )) [ int ( keypoints_1[m.queryIdx].pt.x ) ];
        ushort depth2 = d2.ptr<unsigned short> (int ( keypoints_2[m.trainIdx].pt.y )) [ int ( keypoints_2[m.trainIdx].pt.x ) ];
        if ( depth1 == 0 || depth2 == 0)   
            continue;
        float dd1 = depth1/5000.0;
        float dd2 = depth2/5000.0;
        Point2d p1 = pixel2cam ( keypoints_1[m.queryIdx].pt, K );
        Point2d p2 = pixel2cam ( keypoints_2[m.trainIdx].pt, K );
        pts1_3d.push_back ( Point3f ( p1.x*dd1, p1.y*dd1, dd1 ) );  // Find feature point's 3D coordinates from Depth image
        pts2_3d.push_back ( Point3f ( p2.x*dd2, p2.y*dd2, dd2 ) );
    }

    cout<<"3d-2d pairs: "<<pts1_3d.size() <<endl;

    Mat R, t;
    pose_estimation_3d3d ( pts1_3d, pts2_3d, R, t );
    cout<<"ICP via SVD results: "<<endl;
    cout<<"R = "<<R<<endl;
    cout<<"t = "<<t<<endl;
    cout<<"R_inv = "<<R.t() <<endl;
    cout<<"t_inv = "<<-R.t() *t<<endl;

    // Bundle Adjustment

    bundleAdjustment( pts1_3d, pts2_3d, R, t );

    cout<<"ICP via BA results: "<<endl;
    cout<<"R = "<<R<<endl;
    cout<<"t = "<<t<<endl;
    cout<<"R_inv = "<<R.t() <<endl;
    cout<<"t_inv = "<<-R.t() *t<<endl;

    // verify p1 = R*p2 + t
    for ( int i=0; i<5; i++ )
    {
        cout<<"p1 = "<<pts1_3d[i]<<endl;
        cout<<"p2 = "<<pts2_3d[i]<<endl;
        cout<<"(R*p2+t) = "<<
            R * (Mat_<double>(3,1)<<pts2_3d[i].x, pts2_3d[i].y, pts2_3d[i].z) + t
            <<endl;
        cout<<endl;
    }

    return 0;
}

