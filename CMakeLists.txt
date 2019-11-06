cmake_minimum_required(VERSION 2.8.3)
project(stereo_reconstruct)

IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
ENDIF()

set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS}   -Wall -O3 -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -march=native -Wreorder")

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

add_subdirectory(stereo_camera)

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
  LIBRARIES stereo_reconstruct
  CATKIN_DEPENDS cv_bridge dynamic_reconfigure geometry_msgs image_geometry image_transport message_filters nodelet pcl_conversions pcl_ros roscpp rospy sensor_msgs std_msgs std_srvs tf tf2_ros tf_conversions
#  DEPENDS system_lib
)

include_directories(stereo_camera ${catkin_INCLUDE_DIRS})

add_library(${PROJECT_NAME} src/stereo_reconstruct.cpp)
target_link_libraries(${PROJECT_NAME} ${catkin_LIBRARIES} stereo_camera)
add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# Mark executables and/or libraries for installation
install(TARGETS ${PROJECT_NAME}
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
