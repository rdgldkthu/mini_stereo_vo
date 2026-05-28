#include "svo/map.h"

#include <algorithm>
#include <unordered_set>

namespace svo {

Map::Map(const Options &options) : options_(options) {}

void Map::addKeyframe(const Frame &frame) {
  active_keyframes_.push_back(frame);

  if (static_cast<int>(active_keyframes_.size()) >
      options_.max_active_keyframes) {
    active_keyframes_.pop_front();
  }
}

void Map::setActiveLandmarks(const std::vector<MapPoint> &landmarks) {
  active_landmarks_ = landmarks;

  if (static_cast<int>(active_landmarks_.size()) >
      options_.max_active_landmarks) {
    active_landmarks_.resize(options_.max_active_landmarks);
  }

  rebuildIndex();
}

void Map::assignNewLandmarkIds(std::vector<MapPoint>& landmarks) {
  for (auto &landmark : landmarks) {
    landmark.id = next_landmark_id++;
  }
}

int Map::findLandmarkIndexById(int landmark_id) const {
  const auto it = id_to_index_.find(landmark_id);
  if (it != id_to_index_.end()) {
    return static_cast<int>(it->second);
  }
  return -1;
}

void Map::rebuildIndex() {
  id_to_index_.clear();
  id_to_index_.reserve(active_landmarks_.size());
  for (size_t i = 0; i < active_landmarks_.size(); ++i) {
    id_to_index_[active_landmarks_[i].id] = i;
  }
}

void Map::addLandmarks(const std::vector<MapPoint> &landmarks) {
  for (const auto &landmark : landmarks) {
    const int idx = findLandmarkIndexById(landmark.id);
    if (idx >= 0) {
      active_landmarks_[idx].p_w = landmark.p_w;
      active_landmarks_[idx].descriptor = landmark.descriptor.clone();
      active_landmarks_[idx].observed_times = landmark.observed_times;
      active_landmarks_[idx].tracked_times = landmark.tracked_times;
      active_landmarks_[idx].missed_times = landmark.missed_times;
      active_landmarks_[idx].is_active = landmark.is_active;
      active_landmarks_[idx].is_outlier = landmark.is_outlier;
    } else {
      id_to_index_[landmark.id] = active_landmarks_.size();
      active_landmarks_.push_back(landmark);
    }
  }

  if (static_cast<int>(active_landmarks_.size()) >
      options_.max_active_landmarks) {
    const int extra = static_cast<int>(active_landmarks_.size()) -
                      options_.max_active_landmarks;

    std::sort(active_landmarks_.begin(), active_landmarks_.end(),
              [](const MapPoint &a, const MapPoint &b) {
                if (a.observed_times != b.observed_times) {
                  return a.observed_times > b.observed_times;
                }
                return a.missed_times < b.missed_times;
              });

    active_landmarks_.resize(active_landmarks_.size() - extra);
    rebuildIndex();
  }
}

void Map::markTrackedLandmarks(const std::vector<MapPoint> &tracked_landmarks) {
  for (const auto &tracked : tracked_landmarks) {
    const int idx = findLandmarkIndexById(tracked.id);
    if (idx < 0) {
      continue;
    }

    active_landmarks_[idx].tracked_times += 1;
    active_landmarks_[idx].observed_times += 1;
    active_landmarks_[idx].missed_times = 0;
    active_landmarks_[idx].is_active = true;
    active_landmarks_[idx].is_outlier = false;
  }
}

void Map::markOutlierLandmarks(const std::vector<int> &outlier_ids) {
  for (int id : outlier_ids) {
    const int idx = findLandmarkIndexById(id);
    if (idx < 0) {
      continue;
    }
    active_landmarks_[idx].is_outlier = true;
    active_landmarks_[idx].missed_times += 1;
  }
}

void Map::markMissedLandmarks(const std::vector<int> &tracked_landmark_ids) {
  std::unordered_set<int> tracked_ids(tracked_landmark_ids.begin(),
                                      tracked_landmark_ids.end());

  for (auto &landmark : active_landmarks_) {
    if (tracked_ids.find(landmark.id) == tracked_ids.end()) {
      landmark.missed_times += 1;
    }
  }
}

void Map::pruneLandmarks() {
  std::vector<MapPoint> kept;
  kept.reserve(active_landmarks_.size());

  for (const auto &landmark : active_landmarks_) {
    if (landmark.is_outlier) {
      continue;
    }
    if (landmark.missed_times > options_.max_missed_times) {
      continue;
    }
    if (landmark.observed_times < options_.min_observed_times &&
        landmark.missed_times > 2) {
      continue;
    }
    kept.push_back(landmark);
  }

  active_landmarks_.swap(kept);
  rebuildIndex();
}

const std::deque<Frame> &Map::activeKeyframes() const {
  return active_keyframes_;
}

std::deque<Frame> &Map::mutableActiveKeyframes() {
  return active_keyframes_;
}

const std::vector<MapPoint> &Map::activeLandmarks() const {
  return active_landmarks_;
}

std::vector<MapPoint> &Map::mutableActiveLandmarks() {
  return active_landmarks_;
}

int Map::numActiveKeyframes() const {
  return static_cast<int>(active_keyframes_.size());
}

int Map::numActiveLandmarks() const {
  return static_cast<int>(active_landmarks_.size());
}

} // namespace svo
