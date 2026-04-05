#ifndef SVO_FEATURE_H
#define SVO_FEATURE_H

#include <opencv2/opencv.hpp>

namespace svo {

struct Feature {
  cv::KeyPoint kp_left;
  cv::KeyPoint kp_right;
  int left_idx = -1;
  int right_idx = -1;
  float disparity = 0.0f;
};

} // namespace svo

#endif // SVO_FEATURE_H
