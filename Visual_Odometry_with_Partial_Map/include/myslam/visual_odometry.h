#ifndef VISUALODOMETRY_H
#define VISUALODOMETRY_H


#include "myslam/common_include.h"
#include "myslam/map.h"
#include "myslam/frame.h"
#include "myslam/g2o_types.h"

#include <opencv2/features2d/features2d.hpp>


// No concept of reference frame anymore, use partial map instead
namespace myslam{

    class VisualOdometry{
        public:
        typedef shared_ptr<VisualOdometry> Ptr;

        enum VOState {
            INITIALIZING = -1,
            OK = 0,
            LOST
        };

        VOState state_;
        Map::Ptr map_;

        Frame::Ptr ref_;
        Frame::Ptr curr_;

        // cv::Ptr<cv::ORB> orb_;
        // vector<cv::Point3f> pts_3d_ref_;
        // vector<cv::KeyPoint> keypoints_curr_;
        // Mat descriptor_curr_;
        // Mat descriptor_ref_;
        // vector<cv::DMatch> feature_matches_;

        cv::Ptr<cv::ORB> orb_;  
        vector<cv::KeyPoint>    keypoints_curr_;    
        Mat                     descriptors_curr_;

        cv::FlannBasedMatcher matcher_flann_;
        vector<MapPoint::Ptr> match_3dpts_;
        vector<int>           match_2dkp_index_;

        SE3 T_c_r_estimated_;
        int num_inliers_;
        int num_lost_;

        int num_of_features_;
        double scale_factor_;
        int level_pyramid_;
        float match_ratio_;
        int max_num_lost_;
        int min_inliers_; 

        double key_frame_min_rot_;
        double key_frame_min_trans_;
        double map_point_erase_ratio_;

    public:
        VisualOdometry();
        ~VisualOdometry();

        bool addFrame(Frame::Ptr frame);

    protected: 

        void extractKeyPoints();
        void computerDescriptors();
        void featureMatching();
        void poseEstimationPnP();
        void poseEstimationPnPwithBA();

        void addKeyFrame();
        void addMapPoints();
        bool checkEstimatedPose();
        bool checkKeyFrame();

        void optimizeMap();
        double getViewAngle(Frame::Ptr frame, MapPoint::Ptr point);
        
    };
}

#endif
