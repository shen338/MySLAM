#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/xfeatures2d.hpp"
using namespace std;
using namespace cv;

/*
Extract ORB and SIFT feature of two images and illustrate matches
usage: 
    feature img1 img2 ${feature_type} 

feature_type = SIFT or ORB
*/
int main(int argc, char** argv){

    Mat img1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Mat img2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);

    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptors_1, descriptors_2;

    if(strcmp(argv[3], "ORB") == 0){
        Ptr<FeatureDetector> detector = ORB::create();
        Ptr<DescriptorExtractor> descriptor = ORB::create();
        
        // compute FAST corners 
        detector->detect(img1, keypoints1);
        detector->detect(img2, keypoints2);

        // compute BRIEF descriptor
        descriptor->compute(img1, keypoints1, descriptors_1);
        descriptor->compute(img2, keypoints2, descriptors_2);

    }
    else if (strcmp(argv[3], "SIFT") == 0){
        Ptr<FeatureDetector> detector_descriptor = xfeatures2d::SIFT::create();

        detector_descriptor->detect( img1, keypoints1 );
        detector_descriptor->detect( img2, keypoints2 );
 
        detector_descriptor->compute( img1, keypoints1, descriptors_1 );
        detector_descriptor->compute( img2, keypoints2, descriptors_2 );
    }
    else {
        cout << "feature type is not supported" << endl;
        return 0;
    }

    Mat outimg1;
    drawKeypoints( img1, keypoints1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
    imshow("ORB feature points",outimg1);
    imshow("img1", img1);

    // match process is the same, Bruteforce-Hamming
    
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(4);
    /*
    FLANNBASED = 1, 
    BRUTEFORCE = 2, 
    BRUTEFORCE_L1 = 3, 
    BRUTEFORCE_HAMMING = 4, 
    BRUTEFORCE_HAMMINGLUT = 5, 
    BRUTEFORCE_SL2 = 6 */

    vector<DMatch> matches;
    matcher->match(descriptors_1, descriptors_2, matches);

    // Filter all match points
    // Find maximum and minimum distance
    double min_dist = 1000000, max_dist = 0;
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        double dist = matches[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

    cout << "max distance: " << max_dist << endl;
    cout << "min distance: " << min_dist << endl;

    std::vector< DMatch > good_matches;
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        if ( matches[i].distance <= max ( 2*min_dist, 30.0 ) )
        {
            good_matches.push_back ( matches[i] );
        }
    }
    cout << matches.size() << " " << good_matches.size() << endl;

    Mat img_match;
    Mat img_goodmatch;
    drawMatches ( img1, keypoints1, img2, keypoints2, matches, img_match );
    drawMatches ( img1, keypoints1, img2, keypoints2, good_matches, img_goodmatch );
    imshow ( "All Matches", img_match );
    imshow ( "Matches after filtering", img_goodmatch );
    waitKey(0);

    return 0;
    

}