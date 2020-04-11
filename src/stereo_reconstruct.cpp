#include <ros/ros.h>
#include <pluginlib/class_list_macros.h>
#include <nodelet/nodelet.h>

#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/CameraInfo.h>

#include <image_geometry/stereo_camera_model.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>

#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/subscriber.h>

#include <cv_bridge/cv_bridge.h>

#include <pcl/pcl_base.h>
#include <pcl/point_types.h>
// #include <pcl/filters/voxel_grid.h>
// #include <pcl/registration/icp.h>

#include <pcl_conversions/pcl_conversions.h>

#include "stereo_camera.h"

namespace stereo_reconstruct {

    class StereoReconstruct : public nodelet::Nodelet {
    public:
        StereoReconstruct() :
                approx_sync_stereo_(0),
                exact_sync_stereo_(0),
                is_mm_(true),
                is_use_colormap_(false),
                frame_id_depth_("stereo_depth_optical_frame"),
                frame_id_cloud_("stereo_cloud_optical_frame"),
                depth_frame_(nullptr) {}

        virtual ~StereoReconstruct() {
            if (approx_sync_stereo_)
                delete approx_sync_stereo_;
            if (exact_sync_stereo_)
                delete exact_sync_stereo_;
            if (depth_frame_)
                delete depth_frame_;
        }

    private:
        virtual void onInit() {
            ros::NodeHandle &nh  = getNodeHandle();
            ros::NodeHandle &pnh = getPrivateNodeHandle();

            bool approx_sync = true;

            pnh.param("approx_sync", approx_sync, approx_sync);
            pnh.param("is_mm", is_mm_, is_mm_);
            pnh.param("is_use_colormap", is_use_colormap_, is_use_colormap_);
            pnh.param("frame_id_cloud", frame_id_cloud_, frame_id_cloud_);
            pnh.param("frame_id_depth", frame_id_depth_, frame_id_depth_);

            NODELET_INFO("Approximate time sync = %s", approx_sync ? "true" : "false");

            if (approx_sync) {
                approx_sync_stereo_ = new message_filters::Synchronizer<MyApproxSyncStereoPolicy>(
                        MyApproxSyncStereoPolicy(10), image_left_, image_right_, camera_info_left_,
                        camera_info_right_);
                approx_sync_stereo_->registerCallback(
                        boost::bind(&StereoReconstruct::stereo_callback, this, _1, _2, _3, _4));
            } else {
                exact_sync_stereo_ = new message_filters::Synchronizer<MyExactSyncStereoPolicy>(
                        MyExactSyncStereoPolicy(10), image_left_, image_right_, camera_info_left_, camera_info_right_);
                exact_sync_stereo_->registerCallback(
                        boost::bind(&StereoReconstruct::stereo_callback, this, _1, _2, _3, _4));
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
        }

        void stereo_callback(const sensor_msgs::ImageConstPtr &image_left,
                            const sensor_msgs::ImageConstPtr &image_right,
                            const sensor_msgs::CameraInfoConstPtr &cam_info_left,
                            const sensor_msgs::CameraInfoConstPtr &cam_info_right) {

            if (!(image_left->encoding.compare(sensor_msgs::image_encodings::MONO8)  == 0
               || image_left->encoding.compare(sensor_msgs::image_encodings::MONO16) == 0
               || image_left->encoding.compare(sensor_msgs::image_encodings::BGR8)   == 0
               || image_left->encoding.compare(sensor_msgs::image_encodings::RGB8)   == 0)
               ||
                !(image_right->encoding.compare(sensor_msgs::image_encodings::MONO8) == 0
               || image_right->encoding.compare(sensor_msgs::image_encodings::MONO16)== 0
               || image_right->encoding.compare(sensor_msgs::image_encodings::BGR8)  == 0
               || image_right->encoding.compare(sensor_msgs::image_encodings::RGB8)  == 0)) {
                NODELET_ERROR("Input type must be image=mono8,mono16,rgb8,bgr8 (enc=%s)", image_left->encoding.c_str());
                return;
            }

            if (cloud_pub_.getNumSubscribers() || depth_pub_.getNumSubscribers()) {

                cv_bridge::CvImageConstPtr ptrLeftImage  = cv_bridge::toCvShare(image_left,  "mono8");
                cv_bridge::CvImageConstPtr ptrRightImage = cv_bridge::toCvShare(image_right, "mono8");

                const cv::Mat &mat_left  = ptrLeftImage->image;
                const cv::Mat &mat_right = ptrRightImage->image;

                image_geometry::StereoCameraModel stereo_camera_model;
                stereo_camera_model.fromCameraInfo(*cam_info_left, *cam_info_right);

                stereo_camera_.camera_model_.baseline = stereo_camera_model.baseline();
                stereo_camera_.camera_model_.left.cx  = stereo_camera_model.left().cx();
                stereo_camera_.camera_model_.left.cy  = stereo_camera_model.left().cy();
                stereo_camera_.camera_model_.left.fx  = stereo_camera_model.left().fx();
                stereo_camera_.camera_model_.right.cx = stereo_camera_model.right().cx();
                stereo_camera_.camera_model_.right.fx = stereo_camera_model.right().fx();

                cv::Mat mat_disp;
                stereo_camera_.compute_disparity_map(mat_left, mat_right, mat_disp);

                if (depth_frame_ == nullptr)
                    depth_frame_ = new cv::Mat(mat_disp.size(), is_mm_ ? CV_16UC1 : CV_32FC1);
                stereo_camera_.disparity_to_depth_map(mat_disp, *depth_frame_);

                PointCloudTYPE::Ptr pcl_cloud(new PointCloudTYPE);
                stereo_camera_.depth_to_pointcloud(*depth_frame_, mat_left, *pcl_cloud);

                if(is_use_colormap_)
                {
                    cv::Mat colormap;
                    cg::StereoCamera::get_colormap_ocv(*depth_frame_, colormap);
                    cv::imshow("depth colormap", colormap);
                    cv::waitKey(3);
                }

                publish_depth(*depth_frame_, cam_info_left, image_left->header.stamp);

                publish_cloud(pcl_cloud, image_left->header.stamp);
            }
        }

        void publish_depth(
                cv::Mat &depth,
                const sensor_msgs::CameraInfoConstPtr &cam_info,
                ros::Time time_stamp) {

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
            depth_header.stamp    = ros::Time::now();
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

        typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::CameraInfo> MyApproxSyncStereoPolicy;
        message_filters::Synchronizer<MyApproxSyncStereoPolicy> *approx_sync_stereo_;

        typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::CameraInfo> MyExactSyncStereoPolicy;
        message_filters::Synchronizer<MyExactSyncStereoPolicy> *exact_sync_stereo_;

        std::string frame_id_cloud_;
        std::string frame_id_depth_;

        cg::StereoCamera stereo_camera_;
    };

    PLUGINLIB_EXPORT_CLASS(stereo_reconstruct::StereoReconstruct, nodelet::Nodelet);
}
