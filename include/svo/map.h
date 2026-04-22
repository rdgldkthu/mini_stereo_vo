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
  };

  explicit Map(const Options &options);

  void addKeyframe(const Frame &frame);
  void setActiveLandmarks(const std::vector<MapPoint> &landmarks);

  const std::vector<Frame> &activeKeyframes() const;
  const std::vector<MapPoint> &activeLandmarks() const;

  int numActiveKeyframes() const;
  int numActiveLandmarks() const;

private:
  Options options_;
  std::vector<Frame> active_keyframes_;
  std::vector<MapPoint> active_landmarks_;
};

} // namespace svo

#endif // SVO_MAP_H
