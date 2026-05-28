#ifndef SVO_MAP_H
#define SVO_MAP_H

#include <deque>
#include <unordered_map>
#include <vector>

#include "svo/frame.h"
#include "svo/map_point.h"

namespace svo {

class Map {
public:
  struct Options {
    // Sliding-window size: oldest keyframe is evicted once this count is exceeded.
    int max_active_keyframes = 5;
    // Hard cap on the active landmark set; excess landmarks are pruned by
    // descending tracked_frames then ascending missed_times.
    int max_active_landmarks = 2000;
    // A landmark with fewer than this many keyframe_observations is considered
    // unestablished and is pruned when it also has missed_times > 2.
    // Requires a landmark to survive at least one full keyframe interval.
    int min_observed_times = 2;
    // Evict a landmark after it has been missed for this many consecutive frames.
    // At 10 Hz, 8 frames = 0.8 s — long enough to ride out brief occlusions.
    int max_missed_times = 8;
  };

  explicit Map(const Options &options);

  void addKeyframe(const Frame &frame);

  void setActiveLandmarks(const std::vector<MapPoint> &landmarks);
  void addLandmarks(const std::vector<MapPoint> &landmarks);

  void assignNewLandmarkIds(std::vector<MapPoint> &landmarks);

  void markTrackedLandmarks(const std::vector<MapPoint> &tracked_landmarks);
  void markMissedLandmarks(const std::vector<int> &tracked_landmark_ids);
  void markOutlierLandmarks(const std::vector<int> &outlier_ids);
  // Increment keyframe_observations for each landmark in landmark_ids.
  // Call once per keyframe insertion for the landmarks tracked at that keyframe.
  void markKeyframeObservations(const std::vector<int> &landmark_ids);

  void pruneLandmarks();

  const std::deque<Frame> &activeKeyframes() const;
  std::deque<Frame> &mutableActiveKeyframes();

  const std::vector<MapPoint> &activeLandmarks() const;
  std::vector<MapPoint> &mutableActiveLandmarks();

  int numActiveKeyframes() const;
  int numActiveLandmarks() const;

private:
  int findLandmarkIndexById(int landmark_id) const;
  void rebuildIndex();

private:
  Options options_;
  std::deque<Frame> active_keyframes_;
  std::vector<MapPoint> active_landmarks_;
  std::unordered_map<int, size_t> id_to_index_;

  int next_landmark_id = 0;
};

} // namespace svo

#endif // SVO_MAP_H
