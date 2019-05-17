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
            // Treat No.1 Frame as keyframe
            addKeyFrame();
            break;
        }

        case OK:{
            curr_ = frame;
            curr_->T_w_c_ = ref_->T_w_c_;
            extractKeyPoints();
            computerDescriptors();
            featureMatching();
            poseEstimationPnP();

            if (checkEstimatedPose() == true){
                curr_->T_w_c_ = T_c_r_estimated_; // update current camera pose
                
                optimizeMap();
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
        orb_->compute(curr_->color_, keypoints_curr_, descriptors_curr_);
    }

    void VisualOdometry::featureMatching(){
        // Select a portion of Map Points to match current feature
        vector<cv::DMatch> matches;
        boost::timer timer;

        Mat descriptor_map;
        vector<MapPoint::Ptr> candidate;

        for(auto& allpoints:map_->map_points_){
            MapPoint::Ptr& p = allpoints.second;                    // Map as unordered_map
            if(curr_->isInFrame(p->pos_)){
                p->visible_times_++;
                candidate.push_back(p);
                descriptor_map.push_back(p->descriptor_);
            }
        }
        
        matcher_flann_.match(descriptor_map, descriptors_curr_, matches);

        float min_dis = std::min_element(matches.begin(), matches.end(), 
                          [](const cv::DMatch& m1, const cv::DMatch& m2){
                              return m1.distance < m2.distance;
                          })->distance;
        
        match_3dpts_.clear();
        match_2dkp_index_.clear();
        // filter out long distance matches
        for (cv::DMatch& m:matches){
            if(m.distance < 2*min_dis){
                match_3dpts_.push_back(candidate[m.queryIdx]);
                match_2dkp_index_.push_back(m.trainIdx);
            }
        }
        cout << "good matches number: " << match_3dpts_.size() << endl;
    }

    void VisualOdometry::poseEstimationPnP(){
        vector<cv::Point3f> pts3d;
        vector<cv::Point2f> pts2d;

        for ( int index:match_2dkp_index_ )
        {
            pts2d.push_back ( keypoints_curr_[index].pt );
        }
        for ( MapPoint::Ptr pt:match_3dpts_ )
        {
            pts3d.push_back( pt->getPositionCV() );
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
        
        for ( int index:match_2dkp_index_ )
        {
            pts2d.push_back ( keypoints_curr_[index].pt );
        }
        for ( MapPoint::Ptr pt:match_3dpts_ )
        {
            pts3d.push_back( pt->getPositionCV() );
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
        if(map_->keyframes_.empty()){
            for(size_t i = 0;i<keypoints_curr_.size();i++){
                double d = curr_->findDepth ( keypoints_curr_[i] );
                if ( d < 0 ) 
                    continue;
                Vector3d p_world = ref_->camera_->pixel2world (
                    Vector2d ( keypoints_curr_[i].pt.x, keypoints_curr_[i].pt.y ), curr_->T_w_c_, d
                );
                Vector3d n = p_world - ref_->getCamCenter();
                n.normalize();
                MapPoint::Ptr map_point = MapPoint::createMapPoint(
                    p_world, n, descriptors_curr_.row(i).clone(), curr_.get()
                );
                map_->insertMapPoint( map_point );
            }
        }
        map_->insertKeyFrame(curr_);
        ref_ = curr_;
    }

    void VisualOdometry::optimizeMap(){
        for(auto iter = map_->map_points_.begin(); iter!=map_->map_points_.end();){
            if(! curr_->isInFrame(iter->second->pos_)){
                iter = map_->map_points_.erase(iter);
                continue;
            }

            float match_ratio = float(iter->second->matched_times_) / iter->second->visible_times_;
            if(match_ratio < map_point_erase_ratio_){
                iter = map_->map_points_.erase(iter);
                continue;
            }

            double angle = getViewAngle( curr_, iter->second);
            if(angle > M_PI/6.){
                iter = map_->map_points_.erase(iter);
            }

            if(iter->second->good_ = false){
                // Triangle
                
            }
            iter++;

        }

        if(match_2dkp_index_.size() < 100){
            addMapPoints();
        }
        if ( map_->map_points_.size() > 1000 )  
        {
            // TODO map is too large, remove some one 
            map_point_erase_ratio_ += 0.05;
        }
        else 
            map_point_erase_ratio_ = 0.1;
        cout<<"map points: "<<map_->map_points_.size()<<endl;
    }

    double VisualOdometry::getViewAngle(Frame::Ptr frame, MapPoint::Ptr point){
        Vector3d n = point->pos_ - frame->getCamCenter();
        n.normalize();
        return acos(n.transpose() * point->norm_);  // Inner product of two norm vector to calculate the angle between
                                                    // camera axis and mappoint
    }
}