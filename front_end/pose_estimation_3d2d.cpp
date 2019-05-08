#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <chrono>

using namespace std;
using namespace cv;

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

Point2d pixel2cam ( const Point2d& p, const Mat& K )
{
    return Point2d
           (
               ( p.x - K.at<double> ( 0,2 ) ) / K.at<double> ( 0,0 ),
               ( p.y - K.at<double> ( 1,2 ) ) / K.at<double> ( 1,1 )
           );
}

void bundleAdjustment(
    const vector<Point3f> points_3d,
    const vector<Point2f> points_2d,
    const Mat &K,
    Mat &R, Mat &t){
        typedef g2o::BlockSolver< g2o::BlockSolverTraits<6, 3>> Block;
        // typedef g2o::BlockSolver_6_3 Block;

        Block::LinearSolverType* linearSolver = new g2o::LinearSolverCSparse<Block::PoseMatrixType>();
        
        // Routine in pose estimation
        Block *solver_ptr = new Block(linearSolver);
        g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm (solver);
        optimizer.setVerbose(true);

        // Vertex SE(3)
        g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap(); // Camera pose SE(3)
        Eigen::Matrix3d R_mat;

        R_mat <<
            R.at<double> ( 0,0 ), R.at<double> ( 0,1 ), R.at<double> ( 0,2 ),
            R.at<double> ( 1,0 ), R.at<double> ( 1,1 ), R.at<double> ( 1,2 ),
            R.at<double> ( 2,0 ), R.at<double> ( 2,1 ), R.at<double> ( 2,2 );

        pose->setId(0);
        pose->setEstimate(g2o::SE3Quat(
            R_mat,
            Eigen::Vector3d(t.at<double> ( 0,0 ), t.at<double> ( 1,0 ), t.at<double> ( 2,0 ))
        ));     // SE3Quat is 6 dimensional, first three rotation, last three translation. 
                // Actually it's using Quaternion
        optimizer.addVertex(pose);
        
        int index = 1;
        // landmarks, use g2o::VertexSBAPointXYZ
        for (const Point3f p:points_3d){
            g2o::VertexSBAPointXYZ* point = new g2o::VertexSBAPointXYZ();
            point->setId(index++);
            point->setEstimate(Eigen::Vector3d(p.x, p.y, p.z));   // easy to set up, only need Eigen::Vector3d
            point->setMarginalized(true);
            optimizer.addVertex(point);
        }

        // parameter: camera intrinsics. Routine setup 
        g2o::CameraParameters* camera = new g2o::CameraParameters (
            K.at<double> ( 0,0 ), Eigen::Vector2d ( K.at<double> ( 0,2 ), K.at<double> ( 1,2 ) ), 0
        );
        camera->setId ( 0 );
        optimizer.addParameter ( camera );

        index = 1;
        // Error terms. use edge g2o::EdgeProjectXYZ2UV from 3d to 2d. Routine
        for(const Point2f p:points_2d){
            g2o::EdgeProjectXYZ2UV* edge = new g2o::EdgeProjectXYZ2UV();
            edge->setId(index);
            edge->setVertex(0, dynamic_cast<g2o::VertexSBAPointXYZ*> (optimizer.vertex(index++)));  
            edge->setVertex(1, pose);    // One side to pose, another side to each landmark
            edge->setMeasurement(Eigen::Vector2d(p.x, p.y));
            edge->setParameterId(0, 0);
            edge->setInformation(Eigen::Matrix2d::Identity());  //  infomation matrix, covariance matrix
            optimizer.addEdge(edge);
        }

        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
        // Setup and initial optimization. Routine
        optimizer.setVerbose ( true );
        optimizer.initializeOptimization();
        optimizer.optimize ( 100 );
        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
        chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>> ( t2-t1 );
        cout<<"optimization costs time: "<<time_used.count() <<" seconds."<<endl;

        cout<<endl<<"after optimization:"<<endl;
        cout<<"T="<<endl<<Eigen::Isometry3d ( pose->estimate() ).matrix() <<endl;

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
    Mat K = ( Mat_<double> ( 3,3 ) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1 );
    vector<Point3f> pts_3d;
    vector<Point2f> pts_2d;
    for ( DMatch m:matches )
    {
        ushort d = d1.ptr<unsigned short> (int ( keypoints_1[m.queryIdx].pt.y )) [ int ( keypoints_1[m.queryIdx].pt.x ) ];
        if ( d == 0 )   
            continue;
        float dd = d/5000.0;
        Point2d p1 = pixel2cam ( keypoints_1[m.queryIdx].pt, K );
        pts_3d.push_back ( Point3f ( p1.x*dd, p1.y*dd, dd ) );  // Find feature point's 3D coordinates from Depth image
        pts_2d.push_back ( keypoints_2[m.trainIdx].pt );
    }

    cout<<"3d-2d pairs: "<<pts_3d.size() <<endl;

    Mat r, t;
    solvePnP ( pts_3d, pts_2d, K, Mat(), r, t, false ); 
    Mat R;
    cv::Rodrigues ( r, R ); // Rodrigues formula: Rotation vector to rotation matrix

    cout<<"R="<<endl<<R<<endl;
    cout<<"t="<<endl<<t<<endl;

    cout<<"calling bundle adjustment"<<endl;

    bundleAdjustment ( pts_3d, pts_2d, K, R, t );
}
