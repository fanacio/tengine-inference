
# CMake 最低版本号要求
cmake_minimum_required (VERSION 3.19)

# 项目信息
project (mobilenet_shoot)

#设置c++的版本并添加多线程的使用
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread")

#设置输出文件的路径
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/release)

#添加opencv、cuda的依赖库路径
set(MYLIBPATH /media/firefly/5d7f8be8-2c5b-4ba8-b6ee-f54cdac97129/pengzhikang/tengine-cross-complie/v11-test)
link_directories(/usr/lib)
link_directories(${MYLIBPATH}/lib)
link_directories(/home/firefly/data/pack/Tengine-tengine-lite/3rdparty/acl/lib)
set(OPENCV_dep_LIBS opencv_imgproc opencv_imgcodecs opencv_core opencv_dnn opencv_highgui opencv_videoio )

set(TENGINE_dep_LIBS tengine-lite)
set(ACL_dep_LIBS arm_compute_core arm_compute arm_compute_graph)
#添加opencv和cuda的头文件路径
include_directories(/usr/include)
include_directories(/home/firefly/data/pack/Tengine-tengine-lite/3rdparty/acl/include)
include_directories(${MYLIBPATH}/include)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/common")

# 查找当前目录的所有源文件
#并将名称保存到DIR_SRCS的变量中 
#aux_source_directory(./ DIR_SRCS)

set(REQUIRE_CPP  shoot.cpp "${CMAKE_CURRENT_SOURCE_DIR}/common/tengine_operations.c")
# add_definitions("-D__GNUC__")
#指定生成目标
add_executable(shoot_v1.0 ${REQUIRE_CPP})
#添加链接库,包括opencv和qt5
target_link_libraries(shoot_v1.0 ${OPENCV_dep_LIBS})
target_link_libraries(shoot_v1.0 ${TENGINE_dep_LIBS})
target_link_libraries(shoot_v1.0 ${ACL_dep_LIBS})
target_link_libraries(shoot_v1.0 ${QT5_dep_LIBS})
target_link_libraries(shoot_v1.0 stdc++fs)
