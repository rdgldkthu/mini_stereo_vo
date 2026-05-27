#ifndef SVO_VIEWER_H
#define SVO_VIEWER_H

#include <vector>

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include "svo/viewer_status.h"

namespace svo {

class Viewer {
public:
  struct Options {
    int image_wait_ms = 1;
    int trajectory_size = 600;
    double trajectory_scale = 8.0;
    bool center_on_current_pose = true;
  };

  explicit Viewer(const Options &options);

  bool update(const cv::Mat &left_img,
              const std::vector<cv::Point2f> &active_points,
              const std::vector<Eigen::Matrix4d> &poses,
              const std::vector<Eigen::Matrix4d> &gt_poses,
              const ViewerStatus &status);

private:
  cv::Mat drawImageView(const cv::Mat &left_img,
                        const std::vector<cv::Point2f> &active_points,
                        const ViewerStatus &status) const;

  cv::Mat drawTrajectoryView(const std::vector<Eigen::Matrix4d> &poses,
                             const std::vector<Eigen::Matrix4d> &gt_poses,
                             const ViewerStatus &status) const;

private:
  Options options_;
};

} // namespace svo

#endif // SVO_VIEWER_H
