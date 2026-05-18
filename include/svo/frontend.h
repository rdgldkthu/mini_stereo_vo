#ifndef SVO_FRONTEND_H
#define SVO_FRONTEND_H

#include <vector>

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include "svo/frame.h"
#include "svo/map_point.h"

namespace svo {

struct FrontendFrameStats {
  bool pose_success = false;
  bool pose_accepted = false;
  bool reinitialized = false;
  bool inserted_keyframe = false;
  bool ran_local_ba = false;
  bool local_ba_accepted = false;
  bool local_ba_rejected = false;

  int num_inliers = 0;
  double inlier_ratio = 0.0;
  double delta_t = 0.0;
  double rmse_before = 0.0;
  double rmse_after = 0.0;
  double ba_rmse_before = 0.0;
  double ba_rmse_after = 0.0;
};

class Frontend {
public:
  struct Options {
    double keyframe_translation_threshold_m = 1.0;
    double keyframe_rotation_threshold_deg = 10.0;
    int keyframe_min_tracked_points = 100;
    int keyframe_min_frame_gap = 15;
    double keyframe_low_track_translation_threshold_m = 0.5;

    int min_initial_landmarks = 20;
    int min_pose_inliers = 15;
    double min_pose_inlier_ratio = 0.10;
    double max_frame_translation_m = 2.0;

    int min_reinit_frame_gap = 10;
    int weak_track_threshold = 80;
    int emergency_rejected_poses_count = 2;

    int local_ba_keyframe_interval = 2;
  };

  explicit Frontend(const Options &options);

  void initialize(const Frame &frame0, std::vector<cv::Point2f> active_points,
                  std::vector<MapPoint> active_landmarks);

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

  void refreshActiveLandmarksFromMap(const std::vector<MapPoint> &map_landmarks);

  bool shouldSaveDenseDebug(int frame_id, int radius) const;

  void noteReinitialized(int frame_id);
  void notePoseRejected(int frame_id);
  void noteKeyframeInserted(int frame_id, const Eigen::Matrix4d &pose_wc);
  void noteLocalBaAccepted();
  bool shouldRunLocalBA();

  const std::vector<Eigen::Matrix4d> &poses() const { return poses_; }
  const Eigen::Matrix4d &currentPose() const { return poses_.back(); }

  const Frame &previousFrame() const { return prev_frame_; }
  void setPreviousFrame(const Frame &frame) { prev_frame_ = frame; }

  std::vector<cv::Point2f> &activePoints() { return active_points_2d_; }
  const std::vector<cv::Point2f> &activePoints() const {
    return active_points_2d_;
  }

  std::vector<MapPoint> &activeLandmarks() { return active_landmarks_; }
  const std::vector<MapPoint> &activeLandmarks() const {
    return active_landmarks_;
  }

private:
  Options options_;

  std::vector<Eigen::Matrix4d> poses_;
  Frame prev_frame_;

  std::vector<cv::Point2f> active_points_2d_;
  std::vector<MapPoint> active_landmarks_;

  int last_keyframe_frame_id_ = 0;
  Eigen::Matrix4d last_keyframe_pose_wc_ = Eigen::Matrix4d::Identity();

  int last_init_frame_id_ = 0;
  int consecutive_rejected_poses_ = 0;
  int dense_debug_center_ = -1;
  int inserted_keyframes_since_last_ba_ = 0;
};

} // namespace svo

#endif // SVO_FRONTEND_H
