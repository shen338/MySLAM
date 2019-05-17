cmake_minimum_required( VERSION 2.8 )
project( OpenCV_practice )

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

# OpenCV
find_package( OpenCV 3.1 REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${OpenCV_DIRS} )

add_executable(mymat mymat.cpp)
target_link_libraries( mymat ${OpenCV_LIBS} )
add_executable(myyaml yaml_config.cpp)
target_link_libraries( myyaml ${OpenCV_LIBS} )