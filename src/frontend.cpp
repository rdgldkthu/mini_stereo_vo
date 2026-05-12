#include "svo/frontend.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace svo {
namespace {

std::vector<cv::Point2f>
makeInitialActivePoints(const StereoInitResult &init_result) {
  std::vector<cv::Point2f> points;

  const size_t n =
      std::min(init_result.features.size(), init_result.landmarks.size());
  points.reserve(n);

  for (size_t i = 0; i < n; ++i) {
    points.push_back(init_result.features[i].kp_left.pt);
  }

  return points;
}

} // namespace

Frontend::Frontend(const Options &options) : options_(options) {}

void Frontend::initialize(const Frame &frame0,
                          const StereoInitResult &init_result,
                          std::vector<MapPoint> active_landmarks) {
  poses_.clear();
  poses_.push_back(Eigen::Matrix4d::Identity());

  prev_frame_ = frame0;
  prev_frame_.pose_wc = Eigen::Matrix4d::Identity();
  prev_frame_.is_keyframe = true;

  active_points_2d_ = makeInitialActivePoints(init_result);
  active_landmarks_ = std::move(active_landmarks);

  last_keyframe_frame_id_ = frame0.id;
  last_keyframe_pose_wc_ = Eigen::Matrix4d::Identity();

  last_init_frame_id_ = frame0.id;
  consecutive_rejected_poses_ = 0;
  dense_debug_center_ = -1;
  inserted_keyframes_since_last_ba_ = 0;
}

bool Frontend::acceptPose(int frame_id, int num_inliers,
                          int num_correspondences,
                          const Eigen::Matrix4d &candidate_pose,
                          FrontendFrameStats &stats) {
  const double inlier_ratio =
      static_cast<double>(num_inliers) / std::max(1, num_correspondences);

  const Eigen::Vector3d t_prev = poses_.back().block<3, 1>(0, 3);
  const Eigen::Vector3d t_curr = candidate_pose.block<3, 1>(0, 3);
  const double delta_t = (t_curr - t_prev).norm();

  stats.num_inliers = num_inliers;
  stats.inlier_ratio = inlier_ratio;
  stats.delta_t = delta_t;

  const bool accepted = num_inliers >= options_.min_pose_inliers &&
                        inlier_ratio >= options_.min_pose_inlier_ratio &&
                        delta_t <= options_.max_frame_translation_m;

  if (accepted) {
    poses_.push_back(candidate_pose);
    consecutive_rejected_poses_ = 0;
    stats.pose_accepted = true;
  } else {
    poses_.push_back(poses_.back());
    notePoseRejected(frame_id);
  }

  return accepted;
}

void Frontend::notePoseRejected(int frame_id) {
  consecutive_rejected_poses_++;
  dense_debug_center_ = frame_id;
}

bool Frontend::shouldReinitialize(int frame_id, bool pose_accepted,
                                  int num_active_tracks) const {
  if (frame_id - last_init_frame_id_ <= options_.min_reinit_frame_gap) {
    return false;
  }

  const bool weak_but_accepted =
      pose_accepted && num_active_tracks < options_.weak_track_threshold;

  const bool emergency_reinit =
      !pose_accepted &&
      (consecutive_rejected_poses_ >= options_.emergency_rejected_poses_count ||
       num_active_tracks < options_.weak_track_threshold);

  return weak_but_accepted || emergency_reinit;
}

void Frontend::noteReinitialized(int frame_id) {
  last_init_frame_id_ = frame_id;
  consecutive_rejected_poses_ = 0;
}

void Frontend::noteKeyframeInserted(int frame_id,
                                    const Eigen::Matrix4d &pose_wc) {
  last_keyframe_frame_id_ = frame_id;
  last_keyframe_pose_wc_ = pose_wc;
  inserted_keyframes_since_last_ba_++;
}

void Frontend::noteLocalBaAccepted() { inserted_keyframes_since_last_ba_ = 0; }

void Frontend::repeatLastPose() { poses_.push_back(poses_.back()); }

void Frontend::rejectPose(int frame_id, int /*num_correspondences*/,
                           FrontendFrameStats & /*stats*/) {
  poses_.push_back(poses_.back());
  notePoseRejected(frame_id);
}

void Frontend::setActiveTracks(const std::vector<cv::Point2f> &points,
                                const std::vector<MapPoint> &landmarks) {
  active_points_2d_ = points;
  active_landmarks_ = landmarks;
}

void Frontend::refreshActiveLandmarksFromMap(
    const std::vector<MapPoint> &map_landmarks) {
  std::unordered_map<int, const MapPoint *> id_to_lm;
  id_to_lm.reserve(map_landmarks.size());
  for (const auto &lm : map_landmarks) {
    id_to_lm[lm.id] = &lm;
  }
  for (auto &lm : active_landmarks_) {
    const auto it = id_to_lm.find(lm.id);
    if (it != id_to_lm.end()) {
      lm = *it->second;
    }
  }
}

bool Frontend::shouldSaveDenseDebug(int frame_id, int radius) const {
  return dense_debug_center_ >= 0 &&
         std::abs(frame_id - dense_debug_center_) <= radius;
}

bool Frontend::needNewKeyframe(const Eigen::Matrix4d &current_pose_wc,
                               int num_tracked_points,
                               int current_frame_id) const {
  if (current_frame_id - last_keyframe_frame_id_ <
      options_.keyframe_min_frame_gap) {
    return false;
  }
  const Eigen::Vector3d t_last = last_keyframe_pose_wc_.block<3, 1>(0, 3);
  const Eigen::Vector3d t_curr = current_pose_wc.block<3, 1>(0, 3);
  const double translation = (t_curr - t_last).norm();

  const Eigen::Matrix3d R_last = last_keyframe_pose_wc_.block<3, 3>(0, 0);
  const Eigen::Matrix3d R_curr = current_pose_wc.block<3, 3>(0, 0);
  const Eigen::Matrix3d R_rel = R_last.transpose() * R_curr;

  double trace_value = (R_rel.trace() - 1.0) * 0.5;
  trace_value = std::max(-1.0, std::min(1.0, trace_value));
  const double rotation_rad = std::acos(trace_value);
  const double rotation_deg = rotation_rad * 180.0 / M_PI;

  const bool translation_trigger =
      translation >= options_.keyframe_translation_threshold_m;
  const bool rotation_trigger =
      rotation_deg >= options_.keyframe_rotation_threshold_deg;
  const bool low_track_trigger =
      num_tracked_points < options_.keyframe_min_tracked_points &&
      translation >= options_.keyframe_low_track_translation_threshold_m;

  return translation_trigger || rotation_trigger || low_track_trigger;
}

} // namespace svo
