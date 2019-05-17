cmake_minimum_required( VERSION 2.8 )
project( Eigen_practice )

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

add_executable( eigen_sparse eigen_sparse.cpp )
add_executable( space_transform space_transform.cpp )