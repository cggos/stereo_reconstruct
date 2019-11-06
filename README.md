# Stereo Reconstruct

ROS Wrapper for Stereo Reconstruction, generate Depth Image and Point Cloud by left and right images

* seperated from [cggos/cgocv](https://github.com/cggos/cgocv)
* [双目立体视觉三维重建 (CSDN)](https://blog.csdn.net/u011178262/article/details/81156412)

-----

## Build

```
catkin_make
```

## Run

```
roslaunch stereo_reconstruct stereo_reconstruct.launch \
  camera:=mynteye left:=left_rect right:=right_rect mm:=true\
  rviz:=true colormap:=true
```

## ELAS
* [LIBELAS: Library for Efficient Large-scale Stereo Matching](http://www.cvlibs.net/software/libelas/)
* [elas_ros (ROS Wiki)](http://wiki.ros.org/elas_ros)
