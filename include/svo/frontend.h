#ifndef SVO_FRONTEND_H
#define SVO_FRONTEND_H

#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include "svo/frame.h"
#include "svo/map_point.h"

namespace svo {

class Camera;
class Estimator;
class Map;
class StereoInitializer;
class Tracker;

struct FrontendFrameStats {
  bool pose_success = false;
  bool pose_accepted = false;
  bool reinitialized = false;
  bool inserted_keyframe = false;

  int num_correspondences = 0;
  int num_inliers = 0;
  double inlier_ratio = 0.0;
  double delta_t = 0.0;
  double rmse_before = 0.0;
  double rmse_after = 0.0;
};

struct ProcessFrameResult {
  FrontendFrameStats stats;
  cv::Mat track_vis;       // non-empty only when save_debug=true
  bool should_exit = false;
};

class Frontend {
public:
  struct Options {
    // --- Keyframe insertion triggers ---
    // New keyframe when camera has moved this far since the last keyframe.
    // At 10 Hz KITTI frame rate, 1.5 m ≈ 54 km/h → triggers every 1–2 frames
    // at highway speed and every few frames in urban driving.
    double keyframe_translation_threshold_m = 1.0;
    // New keyframe when the camera has rotated this much since the last keyframe.
    double keyframe_rotation_threshold_deg = 10.0;
    // Also trigger a keyframe when tracked points drop below this count
    // (independently of translation/rotation), provided the camera has moved at
    // least keyframe_low_track_translation_threshold_m since the last keyframe.
    int keyframe_min_tracked_points = 100;
    // Minimum frame gap between successive keyframes to prevent thrashing when
    // the translation and point-count triggers fire on every frame.
    int keyframe_min_frame_gap = 15;
    // Companion to keyframe_min_tracked_points: the low-track trigger is
    // suppressed below this translation to avoid keyframes on static scenes.
    double keyframe_low_track_translation_threshold_m = 0.5;

    // --- Pose acceptance ---
    // Bootstrap and reinit require at least this many triangulated landmarks.
    int min_initial_landmarks = 20;
    // Minimum PnP inlier count to accept a pose estimate.
    int min_pose_inliers = 15;
    // Minimum fraction of correspondences that must be PnP inliers.
    // 10 % is intentionally permissive: with 200 tracks, 20 inliers still gives
    // a well-constrained pose; tighter values cause spurious rejections on turns.
    double min_pose_inlier_ratio = 0.10;
    // Reject poses that imply more than this much translation in one frame.
    // Derived from dataset frame rate × plausible max speed:
    //   KITTI 10 Hz, ~130 km/h ≈ 36 m/s → 3.6 m/frame theoretical max.
    //   2.0 m ≈ 72 km/h — a conservative bound that rejects tracking failures
    //   while accepting legitimate highway motion.
    double max_frame_translation_m = 2.0;

    // --- Reinitialization policy ---
    // Don't reinitialize more often than once per this many frames (prevents
    // thrashing when tracking is persistently poor; 10 frames = 1 s at 10 Hz).
    int min_reinit_frame_gap = 10;
    // Fewer than this many active tracks is considered "weak" and triggers
    // reinitialization (subject to the frame-gap guard above).
    int weak_track_threshold = 80;
    // Trigger emergency reinitialization after this many consecutive rejected
    // poses even if the frame-gap guard has not expired.
    int emergency_rejected_poses_count = 2;
  };

  explicit Frontend(const Options &options);

  // Stereo-initialise from frame 0 and seed the map.
  // Returns true on success; optionally writes the match visualisation to *init_vis.
  bool bootstrap(Frame &frame0, StereoInitializer &initializer, Map &map,
                 const Camera &camera, bool save_debug = false,
                 cv::Mat *init_vis = nullptr);

  // Run one frame through the full VO pipeline.
  // All module references are owned by the caller (main / SLAM orchestrator).
  ProcessFrameResult processFrame(int frame_id, Frame &curr_frame,
                                  Tracker &tracker, Estimator &estimator,
                                  StereoInitializer &initializer, Map &map,
                                  const Camera &camera, bool save_debug = false);

  bool needNewKeyframe(const Eigen::Matrix4d &current_pose_wc,
                       int num_tracked_points, int current_frame_id) const;

  bool acceptPose(int frame_id, int num_inliers, int num_correspondences,
                  const Eigen::Matrix4d &candidate_pose,
                  FrontendFrameStats &stats);

  void rejectPose(int frame_id, int num_correspondences,
                  FrontendFrameStats &stats);

  void repeatLastPose();

  bool shouldReinitialize(int frame_id, bool pose_accepted,
                          int num_active_tracks) const;

  void setActiveTracks(const std::vector<cv::Point2f> &points,
                       const std::vector<MapPoint> &landmarks);

  bool shouldSaveDenseDebug(int frame_id, int radius) const;

  void noteReinitialized(int frame_id);
  void notePoseRejected(int frame_id);
  void noteKeyframeInserted(int frame_id, const Eigen::Matrix4d &pose_wc);

  const std::vector<Eigen::Matrix4d> &poses() const { return poses_; }
  const Eigen::Matrix4d &currentPose() const { return poses_.back(); }

  std::vector<cv::Point2f> &activePoints() { return active_points_2d_; }
  const std::vector<cv::Point2f> &activePoints() const {
    return active_points_2d_;
  }

  std::vector<MapPoint> &activeLandmarks() { return active_landmarks_; }
  const std::vector<MapPoint> &activeLandmarks() const {
    return active_landmarks_;
  }

private:
  void initialize(const Frame &frame0, std::vector<cv::Point2f> active_points,
                  std::vector<MapPoint> active_landmarks);

  Options options_;

  // Authoritative output trajectory for the VO. One entry per processed frame,
  // world-from-camera (T_wc). Map keyframe poses are derived snapshots copied
  // from this vector at insertion time; they never diverge while BA is absent.
  // (The SLAM upgrade promotes the map to the single optimizable source and
  // makes this vector a view over it instead.)
  std::vector<Eigen::Matrix4d> poses_;
  Frame prev_frame_;

  std::vector<cv::Point2f> active_points_2d_;
  std::vector<MapPoint> active_landmarks_;

  int last_keyframe_frame_id_ = 0;
  Eigen::Matrix4d last_keyframe_pose_wc_ = Eigen::Matrix4d::Identity();

  int last_init_frame_id_ = 0;
  int consecutive_rejected_poses_ = 0;
  int dense_debug_center_ = -1;

  cv::Point2f motion_hint_{0.f, 0.f};
};

} // namespace svo

#endif // SVO_FRONTEND_H
