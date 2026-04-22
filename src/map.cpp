#include "svo/map.h"

namespace svo {

Map::Map(const Options &options) : options_(options) {}

void Map::addKeyframe(const Frame &frame) {
  active_keyframes_.push_back(frame);

  if (static_cast<int>(active_keyframes_.size()) >
      options_.max_active_keyframes) {
    active_keyframes_.erase(active_keyframes_.begin());
  }
}

void Map::setActiveLandmarks(const std::vector<MapPoint> &landmarks) {
  active_landmarks_ = landmarks;
}

const std::vector<Frame> &Map::activeKeyframes() const {
  return active_keyframes_;
}

const std::vector<MapPoint> &Map::activeLandmarks() const {
  return active_landmarks_;
}

int Map::numActiveKeyframes() const {
  return static_cast<int>(active_keyframes_.size());
}

int Map::numActiveLandmarks() const {
  return static_cast<int>(active_landmarks_.size());
}

} // namespace svo