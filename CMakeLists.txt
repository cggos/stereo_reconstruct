cmake_minimum_required(VERSION 2.8.3)
project(stereo_reconstruct)

IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
ENDIF()

# set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -Wall -O3 ")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -Wreorder")

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  dynamic_reconfigure
  geometry_msgs
  image_geometry
  image_transport
  message_filters
  nodelet
  pcl_conversions
  pcl_ros
  roscpp
  rospy
  sensor_msgs
  std_msgs
  std_srvs
  tf
  tf2_ros
  tf_conversions
)

catkin_package(
#  INCLUDE_DIRS include
  LIBRARIES stereo_node
  CATKIN_DEPENDS cv_bridge dynamic_reconfigure geometry_msgs image_geometry image_transport message_filters nodelet pcl_conversions pcl_ros roscpp rospy sensor_msgs std_msgs std_srvs tf tf2_ros tf_conversions
#  DEPENDS system_lib
)

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

include_directories(include ${catkin_INCLUDE_DIRS})

add_library(stereo_node src/stereo_reconstruct.cpp src/stereo_camera.cc)
target_link_libraries(stereo_node ${catkin_LIBRARIES})
add_dependencies(stereo_node ${stereo_node_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# Mark executables and/or libraries for installation
install(TARGETS stereo_node
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
