#include "stereo_camera.h"

#include <iostream>

#include <opencv2/calib3d/calib3d.hpp>

namespace cg {

    void StereoCamera::compute_disparity_map(const cv::Mat &mat_l, const cv::Mat &mat_r, cv::Mat &mat_disp) {

        if (mat_l.empty() || mat_r.empty()) {
            std::cerr << "[cgocv] " << __FUNCTION__ << " : mat_l or mat_r is empty !" << std::endl;
            return;
        }
        if (mat_l.channels() != 1 || mat_r.channels() != 1) {
            std::cerr << "[cgocv] " << __FUNCTION__ << " : mat_l or mat_r is NOT single-channel image !" << std::endl;
            return;
        }

        cv::Mat mat_d_bm;
        int method = 1;
        switch (method) {
        case 0:
        {
          int blockSize_ = 15;  //15
          int minDisparity_ = 0;   //0
          int numDisparities_ = 64;  //64
          int preFilterSize_ = 9;   //9
          int preFilterCap_ = 31;  //31
          int uniquenessRatio_ = 15;  //15
          int textureThreshold_ = 10;  //10
          int speckleWindowSize_ = 100; //100
          int speckleRange_ = 4;   //4

          cv::Ptr<cv::StereoBM> stereo = cv::StereoBM::create();
          stereo->setBlockSize(blockSize_);
          stereo->setMinDisparity(minDisparity_);
          stereo->setNumDisparities(numDisparities_);
          stereo->setPreFilterSize(preFilterSize_);
          stereo->setPreFilterCap(preFilterCap_);
          stereo->setUniquenessRatio(uniquenessRatio_);
          stereo->setTextureThreshold(textureThreshold_);
          stereo->setSpeckleWindowSize(speckleWindowSize_);
          stereo->setSpeckleRange(speckleRange_);
          stereo->compute(mat_l, mat_r, mat_d_bm);

          // stereoBM:
          // When disptype == CV_16S, the map is a 16-bit signed single-channel image,
          // containing disparity values scaled by 16
          mat_d_bm.convertTo(mat_disp, CV_32FC1, 1 / 16.f);
        }
          break;
        case 1:
        {
          enum { STEREO_BM = 0, STEREO_SGBM = 1, STEREO_HH = 2, STEREO_VAR = 3, STEREO_3WAY = 4 };

          cv::Size img_size = mat_l.size();
          int cn = mat_l.channels();

          int numberOfDisparities = ((img_size.width / 8) + 15) & -16;
          int SADWindowSize = 9;
          int sgbmWinSize = SADWindowSize > 0 ? SADWindowSize : 3;

          cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0, 16, 3);
          sgbm->setPreFilterCap(63);
          sgbm->setBlockSize(sgbmWinSize);
          sgbm->setP1(8  * cn * sgbmWinSize*sgbmWinSize);
          sgbm->setP2(32 * cn * sgbmWinSize*sgbmWinSize);
          sgbm->setMinDisparity(0);
          sgbm->setNumDisparities(numberOfDisparities);
          sgbm->setUniquenessRatio(10);
          sgbm->setSpeckleWindowSize(100);
          sgbm->setSpeckleRange(32);
          sgbm->setDisp12MaxDiff(1);

          int alg = STEREO_SGBM;
          if (alg == STEREO_HH)
            sgbm->setMode(cv::StereoSGBM::MODE_HH);
          else if (alg == STEREO_SGBM)
            sgbm->setMode(cv::StereoSGBM::MODE_SGBM);
          else if (alg == STEREO_3WAY)
            sgbm->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);

          sgbm->compute(mat_l, mat_r, mat_d_bm);          

          mat_d_bm.convertTo(mat_disp, CV_32FC1, 1 / 16.f);
        }
          break;
        }
    }

    void StereoCamera::disparity_to_depth_map(const cv::Mat &mat_disp, cv::Mat &mat_depth) {

        if (!(mat_depth.type() == CV_16UC1 || mat_depth.type() == CV_32FC1))
            return;

        double baseline = camera_model_.baseline;
        double left_cx = camera_model_.left.cx;
        double left_fx = camera_model_.left.fx;
        double right_cx = camera_model_.right.cx;
        double right_fx = camera_model_.right.fx;

        mat_depth = cv::Mat::zeros(mat_disp.size(), mat_depth.type());

        for (int h = 0; h < (int) mat_depth.rows; h++) {
            for (int w = 0; w < (int) mat_depth.cols; w++) {

                float disp = 0.f;

                switch (mat_disp.type()) {
                    case CV_16SC1:
                        disp = mat_disp.at<short>(h, w);
                        break;
                    case CV_32FC1:
                        disp = mat_disp.at<float>(h, w);
                        break;
                    case CV_8UC1:
                        disp = mat_disp.at<unsigned char>(h, w);
                        break;
                }

                float depth = 0.f;
                if (disp > 0.0f && baseline > 0.0f && left_fx > 0.0f) {
                    //Z = baseline * f / (d + cx1-cx0);
                    double c = 0.0f;
                    if (right_cx > 0.0f && left_cx > 0.0f)
                        c = right_cx - left_cx;
                    depth = float(left_fx * baseline / (disp + c));
                }

                switch (mat_depth.type()) {
                    case CV_16UC1: {
                        unsigned short depthMM = 0;
                        if (depth <= (float) USHRT_MAX)
                            depthMM = (unsigned short) depth;
                        mat_depth.at<unsigned short>(h, w) = depthMM;
                    }
                        break;
                    case CV_32FC1:
                        mat_depth.at<float>(h, w) = depth;
                        break;
                }
            }
        }
    }

    void StereoCamera::depth_to_pointcloud(
            const cv::Mat &mat_depth, const cv::Mat &mat_left,
            pcl::PointCloud<pcl::PointXYZRGB> &point_cloud, float max_depth) {

        point_cloud.height = (uint32_t) mat_depth.rows;
        point_cloud.width  = (uint32_t) mat_depth.cols;
        point_cloud.is_dense = false;
        point_cloud.resize(point_cloud.height * point_cloud.width);

        for (int h = 0; h < (int) mat_depth.rows; h++) {
            for (int w = 0; w < (int) mat_depth.cols; w++) {

                pcl::PointXYZRGB &pt = point_cloud.at(h * point_cloud.width + w);

                switch (mat_left.channels()) {
                    case 1: {
                        unsigned char v = mat_left.at<unsigned char>(h, w);
                        pt.b = v;
                        pt.g = v;
                        pt.r = v;
                    }
                        break;
                    case 3: {
                        cv::Vec3b v = mat_left.at<cv::Vec3b>(h, w);
                        pt.b = v[0];
                        pt.g = v[1];
                        pt.r = v[2];
                    }
                        break;
                }

                float depth = 0.f;
                switch (mat_depth.type()) {
                    case CV_16UC1: // unit is mm
                        depth = float(mat_depth.at<unsigned short>(h, w));
                        depth *= 0.001f; // convert to meter for pointcloud
                        break;
                    case CV_32FC1: // unit is meter
                        depth = mat_depth.at<float>(h, w);
                        break;
                }

                if (std::isfinite(depth) && depth >= 0 && depth < max_depth) {
                    double W = depth / camera_model_.left.fx;
                    pt.x = float((cv::Point2f(w, h).x - camera_model_.left.cx) * W);
                    pt.y = float((cv::Point2f(w, h).y - camera_model_.left.cy) * W);
                    pt.z = depth;
                } else
                    pt.x = pt.y = pt.z = std::numeric_limits<float>::quiet_NaN();
            }
        }
    }

    void StereoCamera::get_colormap_ocv(const cv::Mat &mat_in, cv::Mat &color_map, cv::ColormapTypes colortype) {
        double min, max;
        cv::minMaxLoc(mat_in, &min, &max);

        cv::Mat mat_scaled;
        if (min != max)
            mat_in.convertTo(mat_scaled, CV_8UC1, 255.0 / (max - min), 0); // -255.0 * min / (max - min)

        cv::applyColorMap(mat_scaled, color_map, int(colortype));
    }

}
