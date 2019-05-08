cmake_minimum_required( VERSION 2.8 )
project( ceres_curve_fitting )

set( CMAKE_BUILD_TYPE "Release" )
set( CMAKE_CXX_FLAGS "-std=c++11 -O3" )

# 添加cmake模块以使用ceres库
# list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
LIST( APPEND CMAKE_MODULE_PATH /home/tong/g2o/cmake_modules ) 
SET( G2O_ROOT /usr/local/include/g2o )

# 寻找G2O
find_package( G2O REQUIRED )
find_package( CSparse REQUIRED )
include_directories( 
    ${G2O_INCLUDE_DIRS}
    ${CSPARSE_INCLUDE_DIR}
    "/usr/include/eigen3"
)

# 寻找Ceres库并添加它的头文件
find_package( Ceres REQUIRED )
include_directories( ${CERES_INCLUDE_DIRS} )

# OpenCV
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_DIRS} )

add_executable( feature feature.cpp )
# 与Ceres和OpenCV链接
target_link_libraries( feature ${OpenCV_LIBS} )

add_executable( pose_estimation_2d2d pose_estimation_2d2d.cpp )
# 与Ceres和OpenCV链接
target_link_libraries( pose_estimation_2d2d ${OpenCV_LIBS} )

add_executable( triangular_measurement triangular_measurement.cpp )
# 与Ceres和OpenCV链接
target_link_libraries( triangular_measurement ${OpenCV_LIBS} )

add_executable( pose_estimation_3d2d pose_estimation_3d2d.cpp )
# 与G2O和OpenCV链接
target_link_libraries( pose_estimation_3d2d 
    ${OpenCV_LIBS}
    ${CSPARSE_LIBRARY}
    g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension
)

add_executable( pose_estimation_3d3d pose_estimation_3d3d.cpp )
target_link_libraries( pose_estimation_3d3d 
   ${OpenCV_LIBS}
   g2o_core g2o_stuff g2o_types_sba g2o_csparse_extension 
   ${CSPARSE_LIBRARY}
)

add_executable(opticalflow_tracing opticalflow_tracing.cpp )
target_link_libraries( opticalflow_tracing ${OpenCV_LIBS} )