#ifndef SVO_VIEWER_H
#define SVO_VIEWER_H

#include <vector>

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

namespace svo {

struct ViewerStatus {
  int frame_id = -1;
  int num_active_points = 0;
  int num_correspondences = 0;
  int num_inliers = 0;

  bool pose_accepted = false;
  bool reinitialized = false;
  bool inserted_keyframe = false;
  bool ran_local_ba = false;

  double delta_t = 0.0;
  double rmse_before = 0.0;
  double rmse_after = 0.0;
};

class Viewer {
public:
  struct Options {
    int image_wait_ms = 1;
    int trajectory_size = 600;
    double trajectory_scale = 8.0;
  };

  explicit Viewer(const Options &options);

  bool update(const cv::Mat &left_img,
              const std::vector<cv::Point2f> &active_points,
              const std::vector<Eigen::Matrix4d> &poses,
              const ViewerStatus &status);

private:
  cv::Mat drawImageView(const cv::Mat &left_img,
                        const std::vector<cv::Point2f> &active_points,
                        const ViewerStatus &status) const;

  cv::Mat drawTrajectoryView(const std::vector<Eigen::Matrix4d> &poses) const;

private:
  Options options_;
};

} // namespace svo

#endif // SVO_VIEWER_H
