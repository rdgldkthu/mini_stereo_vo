#ifndef SVO_MAP_H
#define SVO_MAP_H

#include <vector>

#include "svo/frame.h"
#include "svo/map_point.h"

namespace svo {

class Map {
public:
  struct Options {
    int max_active_keyframes = 5;
    int max_active_landmarks = 2000;
    int min_observed_times = 2;
    int max_missed_times = 8;
  };

  explicit Map(const Options &options);

  void addKeyframe(const Frame &frame);

  void setActiveLandmarks(const std::vector<MapPoint> &landmarks);
  void addLandmarks(const std::vector<MapPoint> &landmarks);

  void assignNewLandmarkIds(std::vector<MapPoint> &landmarks);

  void markTrackedLandmarks(const std::vector<MapPoint> &tracked_landmarks);
  void markMissedLandmarks(const std::vector<int> &tracked_landmark_ids);

  void pruneLandmarks();

  const std::vector<Frame> &activeKeyframes() const;
  std::vector<Frame> &mutableActiveKeyframes();

  const std::vector<MapPoint> &activeLandmarks() const;
  std::vector<MapPoint> &mutableActiveLandmarks();

  int numActiveKeyframes() const;
  int numActiveLandmarks() const;

private:
  int findLandmarkIndexById(int landmark_id) const;

private:
  Options options_;
  std::vector<Frame> active_keyframes_;
  std::vector<MapPoint> active_landmarks_;

  int next_landmark_id = 0;
};

} // namespace svo

#endif // SVO_MAP_H
