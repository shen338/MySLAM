
#include <fstream>
#include <boost/timer.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/viz.hpp>

#include "myslam/config.h"
#include "myslam/visual_odometry.h"

int main(int argc, char **argv)
{
    if (argc != 2)
    {
        cout << "Need a parameter file" << endl;
        return 1;
    }

    myslam::Config::setParameterFile(argv[1]);
    myslam::VisualOdometry::Ptr vo(new myslam::VisualOdometry);

    string dataset_dir = myslam::Config::get<string>("dataset_dir");
    cout << "dataset: " << dataset_dir << endl;

    ifstream fin(dataset_dir + "/associate.txt");
    if (!fin)
    {
        cout << "No data association file" << endl;
        return 1;
    }

    vector<string> rgb_files, depth_files;
    vector<double> rgb_times, depth_times;

    while (!fin.eof())
    {
        string rgb_time, rgb_file, depth_time, depth_file;
        fin >> rgb_time >> rgb_file >> depth_time >> depth_file;
        rgb_times.push_back(atof(rgb_time.c_str()));
        depth_times.push_back(atof(depth_time.c_str()));
        rgb_files.push_back(dataset_dir + "/" + rgb_file);
        depth_files.push_back(dataset_dir + "/" + depth_file);

        if (fin.good() == true)
            break;
    }

    myslam::Camera::Ptr camera(new myslam::Camera);

    cv::viz::Viz3d vis("Visual Odometry");
    cv::viz::WCoordinateSystem world_coor(1.0), camera_coor(0.5);
    cv::Point3d cam_pos(0, -1, -1), camera_focal_point(0, 0, 0), cam_y_dir(0, 1, 0);
    cv::Affine3d cam_pose = cv::viz::makeCameraPose(cam_pos, camera_focal_point, cam_y_dir);
    vis.setViewerPose(cam_pose);

    world_coor.setRenderingProperty(cv::viz::LINE_WIDTH, 2.0);
    camera_coor.setRenderingProperty(cv::viz::LINE_WIDTH, 1.0);

    vis.showWidget("World", world_coor);
    vis.showWidget("Camera", camera_coor);

    cout << "read RGB files: " << rgb_files.size() << endl;

    for(int i = 0;i<rgb_files.size();i++){
        Mat color = cv::imread(rgb_files[i]); 
        Mat depth = cv::imread(depth_files[i], -1);

        if(color.data == nullptr || depth.data == nullptr)
            break;
        
        myslam::Frame::Ptr pFrame = myslam::Frame::createFrame();
        pFrame->camera_ = camera;
        pFrame->color_ = color;
        pFrame->depth_ = depth;
        pFrame->time_stamp_ = rgb_times[i];

        boost::timer timer;
        vo->addFrame(pFrame);
        cout << "VO add frame time cost: " << timer.elapsed() << endl;

        if(vo->state_ == myslam::VisualOdometry::LOST)
            break;
        
        SE3 T_c_w = pFrame->T_w_c_.inverse(); //  pFrame->T_c_w_: transform from world to camera
        cv::Affine3d M(
            cv::Affine3d::Mat3( 
                T_c_w.rotation_matrix()(0,0), T_c_w.rotation_matrix()(0,1), T_c_w.rotation_matrix()(0,2),
                T_c_w.rotation_matrix()(1,0), T_c_w.rotation_matrix()(1,1), T_c_w.rotation_matrix()(1,2),
                T_c_w.rotation_matrix()(2,0), T_c_w.rotation_matrix()(2,1), T_c_w.rotation_matrix()(2,2)
            ), 
            cv::Affine3d::Vec3(
                T_c_w.translation()(0,0), T_c_w.translation()(1,0), T_c_w.translation()(2,0)
            
            )
        );

        cv::imshow("image", color);
        cv::waitKey(1);
        vis.setWidgetPose("camera", M);
        vis.spinOnce();
    }
}