#ifndef SVO_FRONTEND_H
#define SVO_FRONTEND_H

#include <Eigen/Core>

namespace svo {

class Frontend {
public:
  struct Options {
    double keyframe_translation_threshold_m = 1.0;
    double keyframe_rotation_threshold_deg = 10.0;
    int keyframe_min_tracked_points = 100;
    int keyframe_min_frame_gap = 15;
    double keyframe_low_track_translation_threshold_m = 0.5;
  };

  explicit Frontend(const Options &options);

  bool needNewKeyframe(const Eigen::Matrix4d &last_keyframe_pose_wc,
                       const Eigen::Matrix4d &current_pose_wc,
                       int num_tracked_points, int current_frame_id, int last_keyframe_frame_id) const;

private:
  Options options_;
};

} // namespace svo

#endif // SVO_FRONTEND_H
