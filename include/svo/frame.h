#ifndef SVO_FRAME_H
#define SVO_FRAME_H

#include <opencv2/opencv.hpp>

namespace svo {

struct Frame {
  int id = -1;
  double timestamp = 0.0;
  cv::Mat left_img;
  cv::Mat right_img;
};

} // namespace svo

#endif // SVO_FRAME_H
