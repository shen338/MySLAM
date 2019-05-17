#ifndef MAPPOINT_H
#define MAPPOINT_H

#include "myslam/common_include.h"

namespace myslam
{
    
class Frame;
class MapPoint
{
public:
    typedef shared_ptr<MapPoint> Ptr;
    unsigned long      id_; // ID
    static unsigned long factory_id_;
    bool good_;
    Vector3d    pos_;       // Position in world
    Vector3d    norm_;      // Normal of viewing direction 
    Mat         descriptor_; // Descriptor for matching 
    int         matched_times_;   
    int         visible_times_;     
    list<Frame*> observed_frames_;
    
    MapPoint();
    MapPoint( unsigned long id, const Vector3d& position, const Vector3d& norm, Frame* frame=nullptr, const Mat& descriptor=Mat());
    
    // factory function
    static MapPoint::Ptr createMapPoint();
    static MapPoint::Ptr createMapPoint(
        const Vector3d& pos_world,
        const Vector3d& norm_, 
        const Mat& descriptor,
        Frame* frame
    );

    inline cv::Point3f getPositionCV() const{
        return cv::Point3f(pos_(0, 0), pos_(1, 0), pos_(2, 0));
    }
};
}

#endif // MAPPOINT_H