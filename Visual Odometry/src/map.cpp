#include "myslam/map.h"

namespace myslam{
    Map::Map(){

    }
    
    void Map::insertKeyFrame(Frame::Ptr frame){
        keyframes_[frame->id_] = frame;
    }

    void Map::insertMapPoint(MapPoint::Ptr map_point){
        map_points_[map_point->id_] = map_point;
    }

}