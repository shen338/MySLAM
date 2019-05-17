#define PI 3.1415

#include <vector>
#include <iostream>
#include <eigen3/Eigen/Geometry>


using namespace std;
using namespace Eigen;

int main(){

    // transform from R and t
    float arrVertices [] = { -1.0 , -1.0 , -1.0 ,
                            1.0 , -1.0 , -1.0 ,
                            1.0 , 1.0 , -1.0 ,
                            -1.0 , 1.0 , -1.0 ,
                            -1.0 , -1.0 , 1.0 ,
                            1.0 , -1.0 , 1.0 ,
                            1.0 , 1.0 , 1.0 ,
                            -1.0 , 1.0 , 1.0};
    MatrixXf mVertices = Map < Matrix <float , 3 , 8 > > ( arrVertices ) ;

    cout << mVertices << endl;
    Transform <float , 3 , Affine > t = Transform <float , 3 , Affine >::Identity();
    t.translate ( Vector3f (1, 1, 0) ) ;
    t.scale ( 0.8f ) ;
    t.rotate ( AngleAxisf (0.5f * M_PI , Vector3f :: UnitZ () ) ) ;  // Three operator should be in order. 
    
    cout << t * mVertices.colwise().homogeneous() << endl ;

    // Rotation as axis and angle
    double angle_in_radian = M_PI/2;
    Vector3f axis;
    axis << 0.0, 0.0, 1.0;

    Eigen::Transform<float , 3 , Affine > tt;
    tt = AngleAxis<float>(angle_in_radian, axis);  // Not eficient 
    Translation<float, 3> translation1(1, 1, 0);

    cout <<  Scaling(Vector3f( 0.8, 0.8, 0.8 )) * tt * translation1* mVertices.colwise().homogeneous() << endl ;


    // Transform from Quarternion
    Quaternion<float> q;
    q = AngleAxis<float>(angle_in_radian, axis);



    // direct 
    Eigen::Matrix3f R;
    R.setIdentity();
    R(2,2) = 2;

    Eigen::Vector3f T;
    T.setOnes();

    Eigen::Matrix4f trans;
    trans.setIdentity();
    trans.block<3,3>(0,0) = R;
    trans.block<3,1>(0,3) = T;

    Affine3f aff1;
    aff1.matrix() = trans;
    cout <<  aff1 * mVertices.colwise().homogeneous() << endl ;

    Transform<float,3,Affine> tttt;
    tttt.matrix() = trans;
    cout <<  tttt * mVertices.colwise().homogeneous() << endl ;


    // Use affine multiplication
    Affine3f rotation_;
    rotation_.setIdentity();

    Translation3f ttt(Vector3f(1,1,1));
    Affine3f translation_(ttt);
    Matrix4f m1 = (translation_ * rotation_).matrix(); // option 1
    Matrix4f m2 = translation_.matrix();
    m2 *= rotation_.matrix();

    // Various transforms
    // 2d rotation
    Rotation2D<float> rot2(angle_in_radian);

    // 3D rotation 
    AngleAxis<float> aa(angle_in_radian, Vector3f(0,0,1));  // Vector has to be normalized

    // 3d rotation as Quaternion
    Quaternion<float> qq;
    qq = aa;

    // N-D scaling 
    Scaling(1, 1);   // just define a diagonal matrix and do matrix multiplication 
    Scaling(1,1,1);

    // translation 
    Translation<float, 3>(1, 1, 1);
    
    return 0;

}