#ifndef SVO_MAP_POINT_H
#define SVO_MAP_POINT_H

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

namespace svo {

struct MapPoint {
  int id = -1;

  Eigen::Vector3d p_w = Eigen::Vector3d::Zero();

  cv::Mat descriptor;

  int observed_times = 0;
  int tracked_times = 0;
  int missed_times = 0;

  bool is_outlier = false;
  bool is_active = true;
};

} // namespace svo

#endif // SVO_MAP_POINT_H
