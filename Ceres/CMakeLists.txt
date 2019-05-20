cmake_minimum_required( VERSION 2.8 )
project( ceres_curve_fitting )

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )


# 寻找Ceres库并添加它的头文件
find_package( Ceres REQUIRED )
include_directories( ${CERES_INCLUDE_DIRS} )

# OpenCV
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_DIRS} )

add_executable( curve_fitting ceres_curve_fitting.cpp )
#与Ceres和OpenCV链接
target_link_libraries( curve_fitting ${CERES_LIBRARIES} ${OpenCV_LIBS} )

add_executable( derivative derivative.cpp )
#与Ceres和OpenCV链接
target_link_libraries( derivative ${CERES_LIBRARIES} ${OpenCV_LIBS} )

add_executable( homography homography.cpp )
#与Ceres和OpenCV链接
target_link_libraries( homography ${CERES_LIBRARIES} )