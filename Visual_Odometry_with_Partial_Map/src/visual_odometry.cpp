#include "myslam/visual_odometry.h"
#include "myslam/config.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <algorithm> 
#include <boost/timer.hpp>

namespace myslam{
    VisualOdometry::VisualOdometry():
    state_ ( INITIALIZING ), ref_ ( nullptr ), curr_ ( nullptr ), map_ ( new Map ), num_lost_ ( 0 ), num_inliers_ ( 0 )
    {
        num_of_features_    = Config::get<int> ( "number_of_features" );
        scale_factor_       = Config::get<double> ( "scale_factor" );
        level_pyramid_      = Config::get<int> ( "level_pyramid" );
        match_ratio_        = Config::get<float> ( "match_ratio" );
        max_num_lost_       = Config::get<float> ( "max_num_lost" );
        min_inliers_        = Config::get<int> ( "min_inliers" );
        key_frame_min_rot_   = Config::get<double> ( "keyframe_rotation" );
        key_frame_min_trans_ = Config::get<double> ( "keyframe_translation" );
        orb_ = cv::ORB::create ( num_of_features_, scale_factor_, level_pyramid_ );
    }

    VisualOdometry::~VisualOdometry(){

    }

    bool VisualOdometry::addFrame(Frame::Ptr frame){
        switch (state_)
        {
        case INITIALIZING:{
            state_ = OK;
            curr_ = frame;
            ref_ = frame;
            map_->insertKeyFrame(frame);
            // Extract feature from first frame
            extractKeyPoints();
            computerDescriptors();
            // calculate 3d points in world coordinates as ref
            setRef3DPoints();
            break;
        }

        case OK:{
            curr_ = frame;
            extractKeyPoints();
            computerDescriptors();
            featureMatching();
            poseEstimationPnP();

            if (checkEstimatedPose() == true){
                curr_->T_w_c_ = T_c_r_estimated_ * ref_->T_w_c_; // update current camera pose
                ref_ = curr_;
                setRef3DPoints();
                num_lost_ = 0;
                if (checkKeyFrame() == true){  // if current frame is a keyframe, add it 
                    addKeyFrame();
                }
            }
            else{
                num_lost_++;
                if(num_lost_ > max_num_lost_){
                    state_ = LOST;
                }
                return false;
            }
            
            break;
        }

        case LOST:{
            // Nothing we can do to invert. 
            cout << "SLAM have lost" << endl;
            break;
        } 
        default:
            break;
        }

        return true;
    }

    void VisualOdometry::extractKeyPoints(){
        orb_->detect(curr_->color_, keypoints_curr_);
    }

    void VisualOdometry::computerDescriptors(){
        orb_->compute(curr_->color_, keypoints_curr_, descriptor_curr_);
    }

    void VisualOdometry::featureMatching(){
        vector<cv::DMatch> matches;
        cv::BFMatcher matcher (cv::NORM_HAMMING);
        matcher.match(descriptor_curr_, descriptor_ref_, matches);

        float min_dis = std::min_element(matches.begin(), matches.end(), 
                          [](const cv::DMatch& m1, const cv::DMatch& m2){
                              return m1.distance < m2.distance;
                          })->distance;
        
        feature_matches_.clear();
        // filter out long distance matches
        for (cv::DMatch& m:matches){
            if(m.distance < 2*min_dis){
                feature_matches_.push_back(m);
            }
        }
        cout << "good matches number: " << feature_matches_.size() << endl;
    }

    void VisualOdometry::setRef3DPoints(){
        pts_3d_ref_.clear();
        descriptor_ref_ = Mat();

        for(int i = 0;i<keypoints_curr_.size();i++){
            double depth = ref_->findDepth(keypoints_curr_[i]);
            if (depth > 0){
                Vector3d p_cam = ref_->camera_->pixel2camera(
                    Vector2d(keypoints_curr_[i].pt.x, keypoints_curr_[i].pt.y), depth
                );
                pts_3d_ref_.push_back(cv::Point3f(p_cam(0,0), p_cam(1,0), p_cam(2, 0)));
                descriptor_ref_.push_back(descriptor_curr_.row(i));
            }
        }
    }

    void VisualOdometry::poseEstimationPnP(){
        vector<cv::Point3f> pts3d;
        vector<cv::Point2f> pts2d;

        for (cv::DMatch m:feature_matches_){
            pts3d.push_back(pts_3d_ref_[m.queryIdx]);
            pts2d.push_back(keypoints_curr_[m.trainIdx].pt);
        }

        Mat K = (cv::Mat_<double>(3, 3) << 
            ref_->camera_->fx_, 0, ref_->camera_->cx_, 
            0, ref_->camera_->fy_, ref_->camera_->cy_, 
            0, 0, 1);
        
        Mat rvec, tvec, inliers;
        cv::solvePnPRansac(pts3d, pts2d, K, Mat(), rvec, tvec, false, 100, 4.0, 0.99, inliers);
        num_inliers_ = inliers.rows;
        cout<<"pnp inliers: "<<num_inliers_<<endl;

        T_c_r_estimated_ = SE3(
            SO3(rvec.at<double>(0, 0), rvec.at<double>(1, 0), rvec.at<double>(2, 0)),
            Vector3d(tvec.at<double>(0, 0), tvec.at<double>(1, 0), tvec.at<double>(2, 0))
        );
    }

    void VisualOdometry::poseEstimationPnPwithBA(){

        // construct the 3d 2d observations
        vector<cv::Point3f> pts3d;
        vector<cv::Point2f> pts2d;
        
        for ( cv::DMatch m:feature_matches_ )
        {
            pts3d.push_back( pts_3d_ref_[m.queryIdx] );
            pts2d.push_back( keypoints_curr_[m.trainIdx].pt );
        }
        
        Mat K = ( cv::Mat_<double>(3,3)<<
            ref_->camera_->fx_, 0, ref_->camera_->cx_,
            0, ref_->camera_->fy_, ref_->camera_->cy_,
            0,0,1
        );
        Mat rvec, tvec, inliers;
        cv::solvePnPRansac( pts3d, pts2d, K, Mat(), rvec, tvec, false, 100, 4.0, 0.99, inliers );
        num_inliers_ = inliers.rows;
        cout<<"pnp inliers: "<<num_inliers_<<endl;
        T_c_r_estimated_ = SE3(
            SO3(rvec.at<double>(0,0), rvec.at<double>(1,0), rvec.at<double>(2,0)), 
            Vector3d( tvec.at<double>(0,0), tvec.at<double>(1,0), tvec.at<double>(2,0))
        );

        typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 2>> Block;
        Block::LinearSolverType* linearSolver = new g2o::LinearSolverDense<Block::PoseMatrixType>();
        Block* solver_ptr = new Block(linearSolver);
        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(solver_ptr);
        g2o::SparseOptimizer optimizer;
        optimizer.setAlgorithm(solver);
        
        // add vertices
        g2o::VertexSE3Expmap* pose = new g2o::VertexSE3Expmap();
        pose->setId(0);
        pose->setEstimate(g2o::SE3Quat(
            T_c_r_estimated_.rotation_matrix(), T_c_r_estimated_.translation()
        ));
        optimizer.addVertex(pose);

        // Add edge
        for(int i = 0;i<inliers.rows; i++){
            int index = inliers.at<int>(i, 0);

            EdgeProjectXYZ2UVPoseOnly* edge = new EdgeProjectXYZ2UVPoseOnly();
            edge->setId(i);
            edge->setVertex(0, pose);
            edge->camera_ = curr_->camera_.get();
            edge->point_ = Vector3d(pts3d[index].x, pts3d[index].y, pts3d[index].z);
            edge->setMeasurement(Vector2d(pts2d[index].x, pts2d[index].y));
            edge->setInformation(Eigen::Matrix2d::Identity());
            optimizer.addEdge(edge);
        }

        optimizer.initializeOptimization();
        optimizer.optimize(10);

        T_c_r_estimated_ = SE3(pose->estimate().rotation(), pose->estimate().translation);
    }

    bool VisualOdometry::checkEstimatedPose(){
        if(num_inliers_ < min_inliers_){
            cout<<"reject because inlier is too small: "<<num_inliers_<<endl;
            return false;
        }
        // if the motion is too large, it is probably wrong
        Sophus::Vector6d d = T_c_r_estimated_.log();
        if ( d.norm() > 5.0 )
        {
            cout<<"reject because motion is too large: "<<d.norm()<<endl;
            return false;
        }
        return true;
    }

    bool VisualOdometry::checkKeyFrame(){
        // simple key frame classification. norm > threshold as Key Frame
        Sophus::Vector6d d = T_c_r_estimated_.log();
        Vector3d trans = d.head<3>();
        Vector2d rot = d.tail<3>();
        if(trans.norm() > key_frame_min_trans_ || rot.norm() > key_frame_min_rot_){
            return true;
        }
        return false;
    }

    void VisualOdometry::addKeyFrame(){
        map_->insertKeyFrame(curr_);
    }
}