#include "myslam/frame.h"

namespace myslam{
    Frame::Frame(){

    }

    Frame::Frame( long id, double time_stamp=0, SE3 T_w_c=SE3(), Camera::Ptr camera=nullptr, Mat color=Mat(), Mat depth=Mat() ):
    id_(id), time_stamp_(time_stamp), T_w_c_(T_w_c), camera_(camera), color_(color), depth_(depth)
    {}

    Frame::~Frame()
    {

    }

    double Frame::findDepth(const cv::KeyPoint& kp){
        int x = cvRound(kp.pt.x);
        int y = cvRound(kp.pt.y);
        ushort d = depth_.ptr<ushort>(y)[x];
        if (d!=0){
            return double(d) / camera_->depth_scale_;
        }
        else{
            // if the depth is 0, filtering using nearby points
            int dx[4] = {-1,0,1,0};
            int dy[4] = {0,-1,0,1};
            for ( int i=0; i<4; i++ )
            {
                d = depth_.ptr<ushort>( y+dy[i] )[x+dx[i]];
                if ( d!=0 )
                {
                    return double(d)/camera_->depth_scale_;
                }
            } 
        }
        // still zero, return -1 as invalid
        return -1.0;
    }

    Frame::Ptr Frame::createFrame()
    {
        static long factory_id = 0;
        // parameters other than id is pre-defined
        return Frame::Ptr( new Frame(factory_id++) );
    }

    Vector3d Frame::getCamCenter() const{
        return T_w_c_.inverse().translation();
    }
    bool Frame::isInFrame(const Vector3d& p_w){
        Vector3d p_cam = camera_->world2camera(p_w, T_w_c_);
        // if not in front of camera, not in the frame
        if(p_cam(2, 0) < 0) return false;
        Vector2d p_p = camera_->world2pixel(p_w, T_w_c_);
        // if not in current image range, not in frame
        return p_p(0, 0) > 0 && p_p(1, 0) > 0 && p_p(0, 0) < color_.cols 
            && p_p(1, 0) < color_.rows;
    }
}