cmake_minimum_required(VERSION 2.8.3)

project(stereo_camera)

add_compile_options(-std=c++11)

#set(OpenCV_DIR /opt/opencv_331/share/OpenCV/)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui calib3d)
if (OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
#    link_directories(/opt/opencv_331/lib)
    link_libraries(${OpenCV_LIBS})
endif ()

find_package(PCL REQUIRED COMPONENTS common io)
if(PCL_FOUND)
    include_directories(${PCL_INCLUDE_DIRS})
    link_libraries(${PCL_LIBRARIES})
    add_definitions(${PCL_DEFINITIONS})
endif()

add_library(stereo_camera SHARED stereo_camera.cc)
