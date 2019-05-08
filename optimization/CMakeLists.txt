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
include_directories( 
    ${G2O_INCLUDE_DIRS}
    "/usr/include/eigen3"
)

# 寻找Ceres库并添加它的头文件
find_package( Ceres REQUIRED )
include_directories( ${CERES_INCLUDE_DIRS} )

# OpenCV
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_DIRS} )

add_executable( curve_fitting ceres_curve_fitting.cpp )
# 与Ceres和OpenCV链接
target_link_libraries( curve_fitting ${CERES_LIBRARIES} ${OpenCV_LIBS} )

add_executable( simple_solver ceres_simple_solver.cpp )
# 与Ceres和OpenCV链接
target_link_libraries( simple_solver ${CERES_LIBRARIES} ${OpenCV_LIBS} )

add_executable( g2o_curve_fitting g2o_curve_fitting.cpp )
# 与G2O和OpenCV链接
target_link_libraries( g2o_curve_fitting 
    ${OpenCV_LIBS}
    g2o_core g2o_stuff
)