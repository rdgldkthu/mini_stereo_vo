#ifndef SVO_MAP_POINT_H
#define SVO_MAP_POINT_H

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

namespace svo {

struct MapPoint {
  int id = -1;
  Eigen::Vector3d p_cam = Eigen::Vector3d::Zero(); // left camera frame
  cv::Mat descriptor;                              // descriptor from left image
};

} // namespace svo

#endif // SVO_MAP_POINT_H
