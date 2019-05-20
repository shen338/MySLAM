#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#define w 400
using namespace cv;
using namespace std;


void MyEllipse( Mat img, double angle )
{
  int thickness = 2;
  int lineType = 8;
  ellipse( img,
       Point( w/2, w/2 ),
       Size( w/4, w/16 ),
       angle,
       0,
       360,
       Scalar( 255, 0, 0 ),
       thickness,
       lineType );
}

void MyFilledCircle( Mat img, Point center )
{
  circle( img,
      center,
      w/32,
      Scalar( 0, 0, 255 ),
      FILLED,
      LINE_8 );
}

void MyLine( Mat img, Point start, Point end )
{
  int thickness = 2;
  int lineType = LINE_8;
  line( img,
    start,
    end,
    Scalar( 0, 0, 0 ),
    thickness,
    lineType );
}

int main(){

    // 2d Point
    Point pt;
    pt.x = 10;
    pt.y = 8;

    cout << "Example Point: " << pt << endl;

    // Scalar
    // Template class for a 1~4 element vector derived from Vec.
    Mat m(2, 2, CV_32FC3, Scalar(1,2,3));

    cout << "Scalar initialize Mat: " << m << endl;

    // Example code on drawing simple patterns
    char atom_window[] = "Drawing 1: Atom";
    Mat atom_image = Mat::zeros( w, w, CV_8UC3 );

    MyEllipse( atom_image, 90 );
    MyEllipse( atom_image, 0 );
    MyEllipse( atom_image, 45 );
    MyEllipse( atom_image, -45 );  

    MyFilledCircle( atom_image, Point( w/2, w/2) );

    // imshow( atom_window, atom_image );
    // moveWindow( atom_window, 0, 200 );

    // waitKey(3);

    // image blurring 
    Mat src = imread("../cat.jpg");
    Mat dst_block;
    blur(src, dst_block, Size(3, 3), Point(-1,-1));

    Mat dst_gaussian;
    GaussianBlur(src, dst_gaussian, Size(3, 3), 0, 0);

    Mat dst_median;
    medianBlur( src, dst_median, 3);

    Mat dst_bilateral;
    bilateralFilter(src, dst_bilateral, 3, 3*3, 3/2);

    // erode and dilation
    Mat img_gray, img_bw, img_final;
    cvtColor(src,img_gray,CV_RGB2GRAY);
    adaptiveThreshold(img_gray, img_bw, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 105, 1); 

    dilate(src, img_final, Mat(), Point(-1, -1), 2, 1, 1); // third argument is kernel type: Rectangular box: MORPH_RECT
                                                                                            // Cross: MORPH_CROSS
                                                                                            // Ellipse: MORPH_ELLIPSE

    erode(src, img_final, Mat(), Point(-1, -1), 2, 1, 1);

    // Convert from BGR to HSV colorspace
    // cvtColor(frame, frame_HSV, COLOR_BGR2HSV);
    // Detect the object based on HSV Range Values
    // inRange(frame_HSV, Scalar(low_H, low_S, low_V), Scalar(high_H, high_S, high_V), frame_threshold);

    // Get image derivative 
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    Sobel(img_gray, grad_x, CV_16S, 1, 0, 3, 1, 1, BORDER_DEFAULT);
    Sobel(img_gray, grad_y, CV_16S, 0, 1, 3, 1, 1, BORDER_DEFAULT);

    // Sobel(src_gray, grad_y, ddepth, 0, 1, ksize, scale, delta, BORDER_DEFAULT);
    // We calculate the "derivatives" in x and y directions. For this, we use the function Sobel() as shown below: The function takes the following arguments:

    // src_gray: In our example, the input image. Here it is CV_8U
    // grad_x / grad_y : The output image.
    // ddepth: The depth of the output image. We set it to CV_16S to avoid overflow.
    // x_order: The order of the derivative in x direction.
    // y_order: The order of the derivative in y direction.
    // scale, delta and BORDER_DEFAULT: We use default values.

    // Also, Laplacian Operator: 
    // Laplacian( src_gray, dst, ddepth, kernel_size, scale, delta, BORDER_DEFAULT );
    Laplacian(img_gray, grad_y, CV_16S, 1, 1, 0);

    Mat canny_edges;
    blur( img_gray, canny_edges, Size(3,3) );
    int lowThreshold = 50;
    int ratio = 3;
    Canny( canny_edges, canny_edges, lowThreshold, lowThreshold*ratio, 3);

    Point2f srcTri[3];
    srcTri[0] = Point2f( 0.f, 0.f );
    srcTri[1] = Point2f( src.cols - 1.f, 0.f );
    srcTri[2] = Point2f( 0.f, src.rows - 1.f );
    Point2f dstTri[3];
    dstTri[0] = Point2f( 0.f, src.rows*0.33f );
    dstTri[1] = Point2f( src.cols*0.85f, src.rows*0.25f );
    dstTri[2] = Point2f( src.cols*0.15f, src.rows*0.7f );
    Mat warp_mat = getAffineTransform( srcTri, dstTri );   // Get affine transform matrix from three paired points
    Mat warp_dst = Mat::zeros( src.rows, src.cols, src.type() );    
    warpAffine( src, warp_dst, warp_mat, warp_dst.size() );    // Affine

    // if using perspective, we need four points 

    return 0;

}