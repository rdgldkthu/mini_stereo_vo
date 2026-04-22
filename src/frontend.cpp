#include "svo/frontend.h"

#include <cmath>

namespace svo {

Frontend::Frontend(const Options &options) : options_(options) {}

bool Frontend::needNewKeyframe(const Eigen::Matrix4d &last_keyframe_pose_wc,
                               const Eigen::Matrix4d &current_pose_wc,
                               int num_tracked_points, int current_frame_id, int last_keyframe_frame_id) const {
  if (current_frame_id - last_keyframe_frame_id < options_.keyframe_min_frame_gap) {
    return false;
  }
  const Eigen::Vector3d t_last = last_keyframe_pose_wc.block<3, 1>(0, 3);
  const Eigen::Vector3d t_curr = current_pose_wc.block<3, 1>(0, 3);
  const double translation = (t_curr - t_last).norm();

  const Eigen::Matrix3d R_last = last_keyframe_pose_wc.block<3, 3>(0, 0);
  const Eigen::Matrix3d R_curr = current_pose_wc.block<3, 3>(0, 0);
  const Eigen::Matrix3d R_rel = R_last.transpose() * R_curr;

  double trace_value = (R_rel.trace() - 1.0) * 0.5;
  trace_value = std::max(-1.0, std::min(1.0, trace_value));
  const double rotation_rad = std::acos(trace_value);
  const double rotation_deg = rotation_rad * 180.0 / M_PI;

  const bool translation_trigger = translation >= options_.keyframe_translation_threshold_m;
  const bool rotation_trigger = rotation_deg >= options_.keyframe_rotation_threshold_deg;
  const bool low_track_trigger = num_tracked_points < options_.keyframe_min_tracked_points && translation >= options_.keyframe_low_track_translation_threshold_m;

  return translation_trigger || rotation_trigger || low_track_trigger;
}

} // namespace svo