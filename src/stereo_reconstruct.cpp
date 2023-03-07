#include <cv_bridge/cv_bridge.h>
#include <image_geometry/stereo_camera_model.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <nodelet/nodelet.h>
#include <pcl/pcl_base.h>
#include <pcl/point_types.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/image_encodings.h>
// #include <pcl/filters/voxel_grid.h>
// #include <pcl/registration/icp.h>

#include <pcl_conversions/pcl_conversions.h>

#include "stereo_camera.h"

namespace stereo_reconstruct {

class StereoNode : public nodelet::Nodelet {
 public:
  StereoNode()
      : approx_sync_stereo_(0),
        exact_sync_stereo_(0),
        is_mm_(true),
        is_use_colormap_(false),
        frame_id_depth_("stereo_depth_optical_frame"),
        frame_id_cloud_("stereo_cloud_optical_frame"),
        depth_frame_(nullptr) {}

  virtual ~StereoNode() {
    if (approx_sync_stereo_) delete approx_sync_stereo_;
    if (exact_sync_stereo_) delete exact_sync_stereo_;
    if (depth_frame_) delete depth_frame_;
  }

 private:
  virtual void onInit() {
    ros::NodeHandle &nh = getNodeHandle();
    ros::NodeHandle &pnh = getPrivateNodeHandle();

    bool approx_sync = true;

    pnh.param("approx_sync", approx_sync, approx_sync);
    pnh.param("is_mm", is_mm_, is_mm_);
    pnh.param("is_use_colormap", is_use_colormap_, is_use_colormap_);
    pnh.param("is_rectified", is_rectified_, is_rectified_);
    pnh.param("frame_id_cloud", frame_id_cloud_, frame_id_cloud_);
    pnh.param("frame_id_depth", frame_id_depth_, frame_id_depth_);

    NODELET_INFO("Approximate time sync = %s", approx_sync ? "true" : "false");

    if (approx_sync) {
      approx_sync_stereo_ = new message_filters::Synchronizer<MyApproxSyncStereoPolicy>(
          MyApproxSyncStereoPolicy(10), image_left_, image_right_, camera_info_left_, camera_info_right_);
      approx_sync_stereo_->registerCallback(boost::bind(&StereoNode::stereo_callback, this, _1, _2, _3, _4));
    } else {
      exact_sync_stereo_ = new message_filters::Synchronizer<MyExactSyncStereoPolicy>(
          MyExactSyncStereoPolicy(10), image_left_, image_right_, camera_info_left_, camera_info_right_);
      exact_sync_stereo_->registerCallback(boost::bind(&StereoNode::stereo_callback, this, _1, _2, _3, _4));
    }

    ros::NodeHandle left_nh(nh, "left");
    ros::NodeHandle right_nh(nh, "right");
    ros::NodeHandle left_pnh(pnh, "left");
    ros::NodeHandle right_pnh(pnh, "right");
    image_transport::ImageTransport left_it(left_nh);
    image_transport::ImageTransport right_it(right_nh);
    image_transport::TransportHints hintsLeft("raw", ros::TransportHints(), left_pnh);
    image_transport::TransportHints hintsRight("raw", ros::TransportHints(), right_pnh);

    image_left_.subscribe(left_it, left_nh.resolveName("image"), 1, hintsLeft);
    image_right_.subscribe(right_it, right_nh.resolveName("image"), 1, hintsRight);
    camera_info_left_.subscribe(left_nh, "camera_info", 1);
    camera_info_right_.subscribe(right_nh, "camera_info", 1);

    cloud_pub_ = nh.advertise<sensor_msgs::PointCloud2>("cloud", 1);

    image_transport::ImageTransport depth_it(nh);
    depth_pub_ = depth_it.advertiseCamera("depth", 1, false);

    if (!is_rectified_) {
      std::string param_file = "rs_t265.yaml";
      pnh.param("param_file", param_file, param_file);
      std::cout << "param_file: " << param_file << std::endl;
      get_rect_map(param_file);
    }
  }

  void get_rect_map(const std::string &param_file) {
    cv::Mat R;
    cv::Vec3d t;
    cv::Size img_size, new_size;
    int vfov_now = 60;

    cv::FileStorage fs(param_file, cv::FileStorage::READ);
    if (!fs.isOpened()) {
      std::cerr << "Failed to open calibration parameter file." << std::endl;
      exit(1);
    }
    fs["K1"] >> K1_;
    fs["K2"] >> K2_;
    fs["D1"] >> D1_;
    fs["D2"] >> D2_;
    fs["xi1"] >> xi1_;
    fs["xi2"] >> xi2_;
    fs["R"] >> R;
    fs["T"] >> t;
    fs["img_w"] >> img_size.width;
    fs["img_h"] >> img_size.height;
    fs["new_w"] >> new_size.width;
    fs["new_h"] >> new_size.height;
    fs["fov_v"] >> vfov_now;
    fs.release();

#if 0
    cv::Mat Q;
    cv::fisheye::stereoRectify(
        K1_, D1_, K2_, D2_, img_size, R, t, R1_, R2_, P1_, P2_, Q, CV_CALIB_ZERO_DISPARITY, new_size, 0.0, 1.1);
    cv::fisheye::initUndistortRectifyMap(K1_, D1_, R1_, P1_, new_size, CV_16SC2, rect_map_[0][0], rect_map_[0][1]);
    cv::fisheye::initUndistortRectifyMap(K2_, D2_, R2_, P2_, new_size, CV_16SC2, rect_map_[1][0], rect_map_[1][1]);
#else
    double vfov_rad = vfov_now * CV_PI / 180.;
    double focal = new_size.height / 2. / tan(vfov_rad / 2.);
    P1_ = (cv::Mat_<double>(3, 3) << focal,
           0.,
           new_size.width / 2. - 0.5,
           0.,
           focal,
           new_size.height / 2. - 0.5,
           0.,
           0.,
           1.);
    P2_ = P1_.clone();
    R1_ = (cv::Mat_<double>(3, 3) << 1., 0., 0., 0., 1., 0., 0., 0., 1.);
    R2_ = R;
    InitUndistortRectifyMap(K1_, D1_, xi1_, R1_, P1_, new_size, rect_map_[0][0], rect_map_[0][1]);
    InitUndistortRectifyMap(K2_, D2_, xi2_, R2_, P2_, new_size, rect_map_[1][0], rect_map_[1][1]);
#endif

    std::cout << std::endl;
    std::cout << "K1_:\n" << K1_ << std::endl << std::endl;
    std::cout << "K2_:\n" << K2_ << std::endl << std::endl;
    std::cout << "D1_:\n" << D1_ << std::endl << std::endl;
    std::cout << "D2_:\n" << D2_ << std::endl << std::endl;
    std::cout << "R1_:\n" << R1_ << std::endl << std::endl;
    std::cout << "R2_:\n" << R2_ << std::endl << std::endl;
    std::cout << "P1_:\n" << P1_ << std::endl << std::endl;
    std::cout << "P2_:\n" << P2_ << std::endl << std::endl;
  }

  inline double MatRowMul(cv::Mat m, double x, double y, double z, int r) {
    return m.at<double>(r, 0) * x + m.at<double>(r, 1) * y + m.at<double>(r, 2) * z;
  }

  void InitUndistortRectifyMap(
      cv::Mat K, cv::Mat D, cv::Mat xi, cv::Mat R, cv::Mat P, cv::Size size, cv::Mat &map1, cv::Mat &map2) {
    map1 = cv::Mat(size, CV_32F);
    map2 = cv::Mat(size, CV_32F);

    double fx = K.at<double>(0, 0);
    double fy = K.at<double>(1, 1);
    double cx = K.at<double>(0, 2);
    double cy = K.at<double>(1, 2);
    double s = K.at<double>(0, 1);

    double xid = xi.at<double>(0, 0);

    double k1 = D.at<double>(0, 0);
    double k2 = D.at<double>(0, 1);
    double p1 = D.at<double>(0, 2);
    double p2 = D.at<double>(0, 3);

    cv::Mat KRi = (P * R).inv();

    for (int r = 0; r < size.height; ++r) {
      for (int c = 0; c < size.width; ++c) {
        double xc = MatRowMul(KRi, c, r, 1., 0);
        double yc = MatRowMul(KRi, c, r, 1., 1);
        double zc = MatRowMul(KRi, c, r, 1., 2);

        double rr = sqrt(xc * xc + yc * yc + zc * zc);
        double xs = xc / rr;
        double ys = yc / rr;
        double zs = zc / rr;

        double xu = xs / (zs + xid);
        double yu = ys / (zs + xid);

        double r2 = xu * xu + yu * yu;
        double r4 = r2 * r2;
        double xd = (1 + k1 * r2 + k2 * r4) * xu + 2 * p1 * xu * yu + p2 * (r2 + 2 * xu * xu);
        double yd = (1 + k1 * r2 + k2 * r4) * yu + 2 * p2 * xu * yu + p1 * (r2 + 2 * yu * yu);

        double u = fx * xd + s * yd + cx;
        double v = fy * yd + cy;

        map1.at<float>(r, c) = (float)u;
        map2.at<float>(r, c) = (float)v;
      }
    }
  }

  void stereo_callback(const sensor_msgs::ImageConstPtr &image_left,
                       const sensor_msgs::ImageConstPtr &image_right,
                       const sensor_msgs::CameraInfoConstPtr &cam_info_left,
                       const sensor_msgs::CameraInfoConstPtr &cam_info_right) {
    if (!(image_left->encoding.compare(sensor_msgs::image_encodings::MONO8) == 0 ||
          image_left->encoding.compare(sensor_msgs::image_encodings::MONO16) == 0 ||
          image_left->encoding.compare(sensor_msgs::image_encodings::BGR8) == 0 ||
          image_left->encoding.compare(sensor_msgs::image_encodings::RGB8) == 0) ||
        !(image_right->encoding.compare(sensor_msgs::image_encodings::MONO8) == 0 ||
          image_right->encoding.compare(sensor_msgs::image_encodings::MONO16) == 0 ||
          image_right->encoding.compare(sensor_msgs::image_encodings::BGR8) == 0 ||
          image_right->encoding.compare(sensor_msgs::image_encodings::RGB8) == 0)) {
      NODELET_ERROR("Input type must be image=mono8,mono16,rgb8,bgr8 (enc=%s)", image_left->encoding.c_str());
      return;
    }

    cv_bridge::CvImageConstPtr ptrLeftImage = cv_bridge::toCvShare(image_left, "mono8");
    cv_bridge::CvImageConstPtr ptrRightImage = cv_bridge::toCvShare(image_right, "mono8");

    const cv::Mat &mat_left = ptrLeftImage->image;
    const cv::Mat &mat_right = ptrRightImage->image;

    cv::Mat img_r_l, img_r_r;
    image_geometry::StereoCameraModel stereo_camera_model;
    stereo_camera_model.fromCameraInfo(*cam_info_left, *cam_info_right);

    if (is_rectified_) {
      stereo_camera_.camera_model_.baseline = stereo_camera_model.baseline();
      stereo_camera_.camera_model_.left.cx = stereo_camera_model.left().cx();
      stereo_camera_.camera_model_.left.cy = stereo_camera_model.left().cy();
      stereo_camera_.camera_model_.left.fx = stereo_camera_model.left().fx();
      stereo_camera_.camera_model_.right.cx = stereo_camera_model.right().cx();
      stereo_camera_.camera_model_.right.fx = stereo_camera_model.right().fx();

      img_r_l = mat_left.clone();
      img_r_r = mat_right.clone();
    } else {
      cv::remap(mat_left, img_r_l, rect_map_[0][0], rect_map_[0][1], cv::INTER_CUBIC);
      cv::remap(mat_right, img_r_r, rect_map_[1][0], rect_map_[1][1], cv::INTER_CUBIC);

      stereo_camera_.camera_model_.baseline = stereo_camera_model.baseline();
      stereo_camera_.camera_model_.left.cx = P1_.at<double>(0, 2);
      stereo_camera_.camera_model_.left.cy = P1_.at<double>(1, 2);
      stereo_camera_.camera_model_.left.fx = P1_.at<double>(0, 0);
      stereo_camera_.camera_model_.right.cx = stereo_camera_.camera_model_.left.cx;

      cv::Mat img_concat;
      cv::hconcat(img_r_l, img_r_r, img_concat);
      // cv::hconcat(mat_left, mat_right, img_concat);
      cv::cvtColor(img_concat, img_concat, cv::COLOR_GRAY2BGR);
      for (int i = 0; i < img_concat.rows; i += 32)
        cv::line(img_concat, cv::Point(0, i), cv::Point(img_concat.cols, i), cv::Scalar(0, 255, 0), 1, 8);
      cv::imshow("rect", img_concat);
      cv::waitKey(30);
    }

    std::cout << "================================== " << __LINE__ << std::endl;

    cv::Mat mat_disp;
    stereo_camera_.compute_disparity_map(img_r_l, img_r_r, mat_disp, 1);

    if (depth_frame_ == nullptr) depth_frame_ = new cv::Mat(mat_disp.size(), is_mm_ ? CV_16UC1 : CV_32FC1);
    stereo_camera_.disparity_to_depth_map(mat_disp, *depth_frame_);

    PointCloudTYPE::Ptr pcl_cloud(new PointCloudTYPE);
    stereo_camera_.depth_to_pointcloud(*depth_frame_, img_r_l, *pcl_cloud);

    if (is_use_colormap_) {
      cv::Mat colormap;
      cg::StereoCamera::get_colormap_ocv(*depth_frame_, colormap);
      cv::imshow("depth colormap", colormap);
      cv::waitKey(3);
    }

    publish_depth(*depth_frame_, cam_info_left, image_left->header.stamp);
    publish_cloud(pcl_cloud, image_left->header.stamp);
  }

  void publish_depth(cv::Mat &depth, const sensor_msgs::CameraInfoConstPtr &cam_info, ros::Time time_stamp) {
    std::string encoding = "";
    switch (depth.type()) {
      case CV_16UC1:
        encoding = sensor_msgs::image_encodings::TYPE_16UC1;
        break;
      case CV_32FC1:
        encoding = sensor_msgs::image_encodings::TYPE_32FC1;
        break;
    }

    sensor_msgs::Image depth_msg;
    std_msgs::Header depth_header;
    depth_header.frame_id = frame_id_depth_;
    depth_header.stamp = ros::Time::now();
    cv_bridge::CvImage(depth_header, encoding, depth).toImageMsg(depth_msg);

    sensor_msgs::CameraInfo depth_info;
    depth_info = *cam_info;
    depth_info.header = depth_msg.header;

    depth_pub_.publish(depth_msg, depth_info, time_stamp);
  }

  void publish_cloud(PointCloudTYPE::Ptr &pcl_cloud, ros::Time time_stamp) {
    sensor_msgs::PointCloud2 ros_cloud;
    pcl::toROSMsg(*pcl_cloud, ros_cloud);
    ros_cloud.header.stamp = time_stamp;
    ros_cloud.header.frame_id = frame_id_cloud_;

    cloud_pub_.publish(ros_cloud);
  }

 private:
  bool is_mm_;
  bool is_use_colormap_;

  cv::Mat *depth_frame_;

  ros::Publisher cloud_pub_;
  image_transport::CameraPublisher depth_pub_;

  image_transport::SubscriberFilter image_left_;
  image_transport::SubscriberFilter image_right_;
  message_filters::Subscriber<sensor_msgs::CameraInfo> camera_info_left_;
  message_filters::Subscriber<sensor_msgs::CameraInfo> camera_info_right_;

  typedef message_filters::sync_policies::
      ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::CameraInfo>
          MyApproxSyncStereoPolicy;
  message_filters::Synchronizer<MyApproxSyncStereoPolicy> *approx_sync_stereo_;

  typedef message_filters::sync_policies::
      ExactTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::CameraInfo>
          MyExactSyncStereoPolicy;
  message_filters::Synchronizer<MyExactSyncStereoPolicy> *exact_sync_stereo_;

  std::string frame_id_cloud_;
  std::string frame_id_depth_;

  cg::StereoCamera stereo_camera_;

  bool is_rectified_ = false;

  // stereo rectify
  cv::Mat rect_map_[2][2];
  cv::Mat K1_;
  cv::Mat K2_;
  cv::Mat D1_;
  cv::Mat D2_;
  cv::Mat xi1_, xi2_;
  cv::Mat R1_, R2_, P1_, P2_;
};

PLUGINLIB_EXPORT_CLASS(stereo_reconstruct::StereoNode, nodelet::Nodelet);
}  // namespace stereo_reconstruct
