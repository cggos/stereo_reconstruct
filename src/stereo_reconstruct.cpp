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
#include <pcl_conversions/pcl_conversions.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/image_encodings.h>

#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "stereo_camera.h"

#define WITH_DRAW 1

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
    {
      Eigen::Vector3d t1 = Eigen::Vector3d::Zero();
      Eigen::Matrix3d R1 = Eigen::Matrix3d::Identity();
      // Eigen::Map<Eigen::Matrix3d, Eigen::RowMajor> R2(reinterpret_cast<double*>(R.data));
      Eigen::Vector3d t2;
      Eigen::Matrix3d R2;
      cv::cv2eigen(t, t2);
      cv::cv2eigen(R, R2);

      // twist inputs to align on x axis
      Eigen::Vector3d x = t1 - t2;
      Eigen::Vector3d y = R1.col(2).cross(x);
      Eigen::Vector3d z = x.cross(y);

      Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
      T.topLeftCorner<3, 3>() << x.normalized(), y.normalized(), z.normalized();

      // took wrong camera as left (redo other way round)
      if (T(0, 0) < 0) {
        x = t2 - t1;
        y = R2.col(2).cross(x);
        z = x.cross(y);
        T.topLeftCorner<3, 3>() << x.normalized(), y.normalized(), z.normalized();
      }

      Eigen::Matrix3d Rinv1 = T.topLeftCorner<3, 3>().transpose() * R1;
      Eigen::Matrix3d Rinv2 = T.topLeftCorner<3, 3>().transpose() * R2;
      cv::eigen2cv(Rinv1, R1_);
      cv::eigen2cv(Rinv2, R2_);
    }

    cv::Mat Q;
    cv::Mat K1 = P1_.clone();
    cv::Mat K2 = P2_.clone();
    cv::Mat P1, P2;
    cv::Mat D1 = cv::Mat::zeros(4, 1, CV_32FC1);
    cv::Mat D2 = cv::Mat::zeros(4, 1, CV_32FC1);
    cv::stereoRectify(K1_, D1, K2_, D2, new_size, R, t, R1_, R2_, P1, P2, Q, CV_CALIB_ZERO_DISPARITY);

    init_undistort_rectify_map(K1_, D1_, xi1_, R1_, P1_, new_size, rect_map_[0][0], rect_map_[0][1]);
    init_undistort_rectify_map(K2_, D2_, xi2_, R2_, P2_, new_size, rect_map_[1][0], rect_map_[1][1]);
#endif

    // std::vector<cv::Vec<T2, 3>> epilines1, epilines2;
    // cv::computeCorrespondEpilines(points1, 1, F, epilines1);  // Index starts with 1
    // cv::computeCorrespondEpilines(points2, 2, F, epilines2);

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

  inline double mat_row_mul(cv::Mat m, double x, double y, double z, int r) {
    return m.at<double>(r, 0) * x + m.at<double>(r, 1) * y + m.at<double>(r, 2) * z;
  }

  void init_undistort_rectify_map(
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
        double xc = mat_row_mul(KRi, c, r, 1., 0);
        double yc = mat_row_mul(KRi, c, r, 1., 1);
        double zc = mat_row_mul(KRi, c, r, 1., 2);

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
    }

#if WITH_DRAW
    if (1) {
      cv::Mat img_concat;
      cv::hconcat(img_r_l, img_r_r, img_concat);
      cv::cvtColor(img_concat, img_concat, cv::COLOR_GRAY2BGR);
      for (int i = 0; i < img_concat.rows; i += 32)
        cv::line(img_concat, cv::Point(0, i), cv::Point(img_concat.cols, i), cv::Scalar(0, 255, 0), 1, 8);
      cv::imshow("rect", img_concat);
    } else {
      detect_match(img_r_l, img_r_r);
    }
    cv::waitKey(30);
#endif

    std::cout << "================================== " << __LINE__ << std::endl;

    cv::Mat mat_disp;
    stereo_camera_.compute_disparity_map(img_r_l, img_r_r, mat_disp, 1);

    // filter
    cv::medianBlur(mat_disp, mat_disp, 5);

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

  inline void detect_match(const cv::Mat &img_1, const cv::Mat &img_2) {
    if (!img_1.data || !img_2.data) {
      std::cout << " --(!) Error reading images " << std::endl;
      return;
    }

    cv::Ptr<cv::ORB> detector = cv::ORB::create(400);

    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
    cv::Mat descriptors_1, descriptors_2;
    detector->detectAndCompute(img_1, cv::Mat(), keypoints_1, descriptors_1);
    detector->detectAndCompute(img_2, cv::Mat(), keypoints_2, descriptors_2);

    cv::FlannBasedMatcher matcher(new cv::flann::LshIndexParams(20, 10, 2));
    std::vector<cv::DMatch> matches;
    matcher.match(descriptors_1, descriptors_2, matches);

    //-- Quick calculation of max and min distances between keypoints
    double max_dist = 0;
    double min_dist = 50;
    for (int i = 0; i < descriptors_1.rows; i++) {
      double dist = matches[i].distance;
      if (dist < min_dist) min_dist = dist;
      if (dist > max_dist) max_dist = dist;
    }
    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

    //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
    //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
    //-- small)
    //-- PS.- radiusMatch can also be used here.
    std::vector<cv::DMatch> good_matches;
    for (int i = 0; i < descriptors_1.rows; i++) {
      if (matches[i].distance <= std::max(2 * min_dist, 0.02)) good_matches.push_back(matches[i]);
    }

    //-- Draw only "good" matches
    cv::Mat img_matches;
    cv::drawMatches(img_1,
                    keypoints_1,
                    img_2,
                    keypoints_2,
                    good_matches,
                    img_matches,
                    cv::Scalar::all(-1),
                    cv::Scalar::all(-1),
                    std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    cv::imshow("FLANN Good Matches", img_matches);
    double mean_error = 0.0;
    for (int i = 0; i < (int)good_matches.size(); i++) {
      int idx1 = good_matches[i].queryIdx;
      int idx2 = good_matches[i].trainIdx;
      cv::Point2f pt1 = keypoints_1[idx1].pt;
      cv::Point2f pt2 = keypoints_2[idx2].pt;
      std::cout << "\"-- Good Match [" << std::setw(2) << i << "] Keypoint 1: " << std::setw(4) << idx1
                << "  -- Keypoint 2: " << std::setw(4) << idx2 << " --> " << pt1 << " <--> " << pt2 << std::endl;
      mean_error += std::abs(pt1.y - pt2.y);
    }
    mean_error /= good_matches.size();

    std::cout << "-- Mean Error (y): " << mean_error << std::endl;
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
