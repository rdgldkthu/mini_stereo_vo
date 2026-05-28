#ifndef SVO_MAP_POINT_H
#define SVO_MAP_POINT_H

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

namespace svo {

struct MapPoint {
  int id = -1;

  Eigen::Vector3d p_w = Eigen::Vector3d::Zero();

  cv::Mat descriptor;

  // How many individual frames this landmark was successfully tracked by LK.
  // Used as a vitality signal for landmark priority when the map is at capacity.
  int tracked_frames = 0;

  // How many distinct keyframes have observed this landmark (set to 1 at
  // triangulation, incremented by Map::markKeyframeObservations at each later
  // keyframe where the landmark is still active). Pruning uses this counter so
  // that landmarks never established across more than one keyframe window are
  // cleaned up quickly, while durable landmarks are retained.
  int keyframe_observations = 0;

  // Consecutive frames in which the landmark was NOT tracked (reset to 0 on
  // successful tracking). Pruning evicts landmarks that exceed max_missed_times.
  int missed_times = 0;

  bool is_outlier = false;
  bool is_active = true;
};

} // namespace svo

#endif // SVO_MAP_POINT_H
