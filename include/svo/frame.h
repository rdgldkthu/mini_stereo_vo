#ifndef SVO_FRAME_H
#define SVO_FRAME_H

#include <vector>

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

namespace svo {

struct Frame {
  int id = -1;
  double timestamp = 0.0;

  cv::Mat left_img;
  cv::Mat right_img;

  Eigen::Matrix4d pose_wc = Eigen::Matrix4d::Identity();

  bool is_keyframe = false;

  std::vector<cv::Point2f> tracked_points;
  std::vector<int> tracked_landmark_ids;
};

} // namespace svo

#endif // SVO_FRAME_H
