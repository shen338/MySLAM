#include <opencv2/core/core.hpp> 
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>

using namespace cv;
using namespace std;

int main(){

    Mat A;
    A = imread("../cat.jpg", IMREAD_COLOR);

    cout << "pixel value in A: " << A.at<Vec3b>(10, 10) << endl;

    // Access to single pixel value
    Vec3b intensity = A.at<Vec3b>(10, 10);
    uchar blue = intensity.val[0];
    uchar green = intensity.val[1];
    uchar red = intensity.val[2];

    Mat B(A);    // Only copy the header

    A.at<Vec3b>(10, 10) = {0, 0, 0};

    cout << "Check pixel value in A: " << B.at<Vec3b>(10, 10) << endl;

    Mat F = A.clone();   // Really copy data
    F.at<Vec3b>(10, 10) = {1, 1, 1};

    cout << "Check pixel value in A: " << B.at<Vec3b>(10, 10) << endl;

    Mat M(2,2, CV_8UC3, Scalar(0,0,255));
    cout << "M = " << endl << " " << M << endl << endl;
    //! [constructor]

    // Create using MATLAB style eye, ones or zero matrix
    //! [matlab]
    Mat E = Mat::eye(4, 4, CV_64F);
    cout << "E = " << endl << " " << E << endl << endl;
    Mat O = Mat::ones(2, 2, CV_32F);
    cout << "O = " << endl << " " << O << endl << endl;
    Mat Z = Mat::zeros(3,3, CV_8UC1);
    cout << "Z = " << endl << " " << Z << endl << endl;
    //! [matlab]

    // create a 3x3 double-precision identity matrix
    //! [comma]
    Mat C = (Mat_<double>(3,3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);  // Mat_
    cout << "C = " << endl << " " << C << endl << endl;
    //! [comma]

    //! [clone]
    Mat RowClone = C.row(1).clone();
    cout << "RowClone = " << endl << " " << RowClone << endl << endl;
    //! [clone]

    // Fill a matrix with random values
    //! [random]
    Mat R = Mat(3, 2, CV_8UC3);
    randu(R, Scalar::all(0), Scalar::all(255));
    //! [random]

    // Demonstrate the output formatting options
    //! [out-default]
    cout << "R (default) = " << endl <<        R           << endl << endl;
    //! [out-default]
    //! [out-python]
    cout << "R (python)  = " << endl << format(R, Formatter::FMT_PYTHON) << endl << endl;
    //! [out-python]
    //! [out-numpy]
    cout << "R (numpy)   = " << endl << format(R, Formatter::FMT_NUMPY ) << endl << endl;
    //! [out-numpy]
    //! [out-csv]
    cout << "R (csv)     = " << endl << format(R, Formatter::FMT_CSV   ) << endl << endl;
    //! [out-csv]
    //! [out-c]
    cout << "R (c)       = " << endl << format(R, Formatter::FMT_C     ) << endl << endl;
    //! [out-c]

    //! [out-point2]
    Point2f P(5, 1);
    cout << "Point (2D) = " << P << endl << endl;
    //! [out-point2]

    //! [out-point3]
    Point3f P3f(2, 6, 7);
    cout << "Point (3D) = " << P3f << endl << endl;
    //! [out-point3]

    Rect r1(10, 10, 100, 100);   // Rect object. 
    Rect r2(10, 10, 100, 100);
    Rect r = r1 | r2;
    Mat crop;
    crop = A(r);
    rectangle(A, r, Scalar(0, 0, 255));

    // Blend two images
    Mat G;
    addWeighted(A, 0.5, F, 0.5, 0.0, G);

    Mat BGR;
    BGR = imread("../cat.jpg");
    Mat I;
    cvtColor(BGR, I, COLOR_BGR2GRAY);

    // Discrete Fourier Transform

    Mat padded;                            //expand input image to optimal size
    int m = getOptimalDFTSize( I.rows );
    int n = getOptimalDFTSize( I.cols ); // on the border add zero values
    copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));
    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);         // Add to the expanded another plane with zeros
    dft(complexI, complexI);            // this way the result may fit in the source matrix
    // compute the magnitude and switch to logarithmic scale
    // => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
    split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
    Mat magI = planes[0];
    magI += Scalar::all(1);                    // switch to logarithmic scale
    log(magI, magI);
    // crop the spectrum, if it has an odd number of rows or columns
    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));
    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = magI.cols/2;
    int cy = magI.rows/2;
    Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right
    Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);
    normalize(magI, magI, 0, 1, NORM_MINMAX); // Transform the matrix with float values into a
                                            // viewable image form (float between values 0 and 1).
    imshow("Input Image"       , I   );    // Show the result
    imshow("spectrum magnitude", magI);
    waitKey();

    return 0;
}