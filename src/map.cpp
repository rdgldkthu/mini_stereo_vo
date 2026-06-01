#include "svo/map.h"

namespace svo {

Map::Map(const Options &options) : options_(options) {}

void Map::addKeyframe(const Frame &frame) {
  active_keyframes_.push_back(frame);

  if (static_cast<int>(active_keyframes_.size()) >
      options_.max_active_keyframes) {
    active_keyframes_.pop_front();
  }
}

void Map::assignNewLandmarkIds(std::vector<MapPoint>& landmarks) {
  for (auto &landmark : landmarks) {
    landmark.id = next_landmark_id++;
  }
}

void Map::addLandmarks(const std::vector<MapPoint> &landmarks) {
  for (const auto &lm : landmarks) {
    MapPoint stored = lm;
    stored.descriptor = lm.descriptor.clone();
    landmarks_[lm.id] = std::move(stored);
  }
}

void Map::markTracked(const std::vector<int> &ids) {
  for (int id : ids) {
    auto it = landmarks_.find(id);
    if (it == landmarks_.end()) continue;
    it->second.is_outlier = false;
    it->second.tracked_frames += 1;
    it->second.missed_times = 0;
    it->second.is_active = true;
  }
}

void Map::markOutlierLandmarks(const std::vector<int> &outlier_ids) {
  for (int id : outlier_ids) {
    auto it = landmarks_.find(id);
    if (it != landmarks_.end()) {
      it->second.is_outlier = true;
      it->second.missed_times += 1;
    }
  }
}

void Map::markMissedLandmarks(const std::vector<int> &tracked_landmark_ids) {
  for (const auto &id : tracked_landmark_ids) {
    auto it = landmarks_.find(id);
    if (it != landmarks_.end()) {
      it->second.missed_times += 1;
    }
  }
}

void Map::markKeyframeObservations(const std::vector<int> &ids) {
  for (int id : ids) {
    auto it = landmarks_.find(id);
    if (it != landmarks_.end())
      it->second.keyframe_observations += 1;
  }
}

void Map::pruneLandmarks() {
  for (auto &[id, lm] : landmarks_) {
    if (lm.is_outlier) {
      lm.is_active = false;
    }
    if (lm.missed_times > options_.max_missed_times) {
      lm.is_active = false;
    }
    if (lm.keyframe_observations < options_.min_observed_times &&
        lm.missed_times > 2) {
      lm.is_active = false;
    }
  }
}

const std::deque<Frame> &Map::activeKeyframes() const {
  return active_keyframes_;
}

std::deque<Frame> &Map::mutableActiveKeyframes() {
  return active_keyframes_;
}

int Map::numActiveKeyframes() const {
  return static_cast<int>(active_keyframes_.size());
}

int Map::numActiveLandmarks() const {
  return static_cast<int>(landmarks_.size());
}

const MapPoint* Map::landmark(int id) const {
  auto it = landmarks_.find(id);
  return it != landmarks_.end() ? &it->second : nullptr;
}

MapPoint* Map::landmark(int id) {
  auto it = landmarks_.find(id);
  return it != landmarks_.end() ? &it->second : nullptr;
}

std::vector<int> Map::localMapLandmarkIds() const {
  std::vector<int> local_map_landmark_ids;
  for (const auto &[id, lm] : landmarks_) {
    if (!lm.is_outlier && lm.missed_times <= options_.max_missed_times) {
      local_map_landmark_ids.push_back(id);
    }
  }
  return local_map_landmark_ids;
}

} // namespace svo
