#include "svo/frontend.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <unordered_set>

#include "svo/camera.h"
#include "svo/estimator.h"
#include "svo/geometry.h"
#include "svo/map.h"
#include "svo/stereo_initializer.h"
#include "svo/tracker.h"

namespace {

std::vector<cv::Point2f>
makeInitialActivePoints(const svo::StereoInitResult &r) {
  const size_t n = std::min(r.features.size(), r.landmarks.size());
  std::vector<cv::Point2f> pts;
  pts.reserve(n);
  for (size_t i = 0; i < n; ++i)
    pts.push_back(r.features[i].kp_left.pt);
  return pts;
}

std::vector<svo::MapPoint>
makeInitialActiveLandmarks(const svo::StereoInitResult &r) {
  const size_t n = std::min(r.features.size(), r.landmarks.size());
  std::vector<svo::MapPoint> lms;
  lms.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    svo::MapPoint lm = r.landmarks[i];
    lm.tracked_frames        = 1;
    lm.keyframe_observations = 1;
    lm.missed_times          = 0;
    lm.is_outlier            = false;
    lm.is_active             = true;
    lms.push_back(lm);
  }
  return lms;
}


std::vector<svo::MapPoint>
transformLandmarksToWorld(std::vector<svo::MapPoint> lms,
                           const Eigen::Matrix4d &T_wc) {
  const Eigen::Matrix3d R = T_wc.block<3, 3>(0, 0);
  const Eigen::Vector3d t = T_wc.block<3, 1>(0, 3);
  for (auto &lm : lms)
    lm.p_w = R * lm.p_w + t;
  return lms;
}

void gatherPnPInliers(const std::vector<Eigen::Vector3d> &obj,
                       const std::vector<cv::Point2f> &img,
                       const std::vector<int> &inlier_idx,
                       std::vector<Eigen::Vector3d> &out_obj,
                       std::vector<cv::Point2f> &out_img) {
  out_obj.clear();
  out_img.clear();
  for (const int idx : inlier_idx) {
    if (idx < 0 || idx >= static_cast<int>(obj.size())) continue;
    out_obj.push_back(obj[idx]);
    out_img.push_back(img[idx]);
  }
}

void filterTrackedByInliers(const std::vector<cv::Point2f> &pts,
                              const std::vector<int> &ids,
                              const std::vector<int> &inlier_idx,
                              std::vector<cv::Point2f> &out_pts,
                              std::vector<int> &out_ids,
                              std::vector<int> &out_outlier_ids) {
  const std::unordered_set<int> inlier_set(inlier_idx.begin(), inlier_idx.end());
  out_pts.clear();
  out_ids.clear();
  out_outlier_ids.clear();
  for (int i = 0; i < static_cast<int>(pts.size()); ++i) {
    if (inlier_set.count(i)) {
      out_pts.push_back(pts[i]);
      out_ids.push_back(ids[i]);
    } else {
      out_outlier_ids.push_back(ids[i]);
    }
  }
}

// replace_active=true: setActiveLandmarks (reinit); false: addKeyframe+addLandmarks
// (keyframe) — caller must pre-populate frame.tracked_points/ids from active tracks.
bool createKeyframeFromStereo(svo::Frontend &frontend,
                               svo::Frame &frame,
                               const Eigen::Matrix4d &pose_wc,
                               svo::StereoInitializer &initializer,
                               svo::Map &map,
                               const svo::Camera &camera,
                               bool replace_active,
                               int min_init_landmarks) {
  const svo::StereoInitResult result = initializer.run(frame, camera, false);
  if (result.num_triangulated < min_init_landmarks) return false;

  auto lms  = makeInitialActiveLandmarks(result);
  map.assignNewLandmarkIds(lms);
  lms = transformLandmarksToWorld(lms, pose_wc);
  auto pts  = makeInitialActivePoints(result);

  std::vector<int> ids;
  for (auto &lm : lms) {
    ids.push_back(lm.id);
  }

  if (replace_active) {
    map.addLandmarks(lms);
  } else {
    frame.pose_wc = pose_wc;
    frame.tracked_points.insert(frame.tracked_points.end(), pts.begin(), pts.end());
    for (const auto &lm : lms)
      frame.tracked_landmark_ids.push_back(lm.id);
    map.addKeyframe(frame);
    map.addLandmarks(lms);
  }

  frontend.setActiveTracks(pts, ids);
  return true;
}

} // namespace

namespace svo {

Frontend::Frontend(const Options &options) : options_(options) {}

// ---------------------------------------------------------------------------
// Private: core state initialisation (called only from bootstrap)
// ---------------------------------------------------------------------------
void Frontend::initialize(const Frame &frame0,
                          std::vector<cv::Point2f> active_points,
                          std::vector<int> active_landmark_ids) {
  poses_.clear();
  poses_.push_back(Eigen::Matrix4d::Identity());

  prev_frame_          = frame0;
  prev_frame_.pose_wc  = Eigen::Matrix4d::Identity();
  prev_frame_.is_keyframe = true;

  active_points_2d_ = std::move(active_points);
  active_landmark_ids_ = std::move(active_landmark_ids);

  last_keyframe_frame_id_  = frame0.id;
  last_keyframe_pose_wc_   = Eigen::Matrix4d::Identity();

  last_init_frame_id_      = frame0.id;
  consecutive_rejected_poses_ = 0;
  dense_debug_center_      = -1;

  motion_model_.anchor(poses_.back());
}

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------
bool Frontend::bootstrap(Frame &frame0, StereoInitializer &initializer,
                          Map &map, const Camera &camera,
                          bool save_debug, cv::Mat *init_vis) {
  const StereoInitResult result = initializer.run(frame0, camera, save_debug);
  std::cout << "Bootstrap: " << result.num_triangulated
            << " landmarks triangulated\n";

  if (result.num_triangulated < options_.min_initial_landmarks) {
    std::cerr << "Bootstrap failed: too few landmarks\n";
    return false;
  }

  if (init_vis && save_debug)
    *init_vis = result.match_vis;

  auto lms = makeInitialActiveLandmarks(result);
  map.assignNewLandmarkIds(lms);

  std::vector<int> ids;
  for (auto& lm : lms) {
    ids.push_back(lm.id);
  }

  initialize(frame0, makeInitialActivePoints(result), ids);

  frame0.pose_wc   = Eigen::Matrix4d::Identity();
  frame0.is_keyframe = true;
  frame0.tracked_points = active_points_2d_;
  frame0.tracked_landmark_ids.clear();
  for (const auto &lm : lms)
    frame0.tracked_landmark_ids.push_back(lm.id);

  map.addKeyframe(frame0);
  map.addLandmarks(lms);

  return true;
}

// ---------------------------------------------------------------------------
// Per-frame VO pipeline
// ---------------------------------------------------------------------------
ProcessFrameResult Frontend::processFrame(int frame_id, Frame &curr_frame,
                                           Tracker &tracker,
                                           Estimator &estimator,
                                           StereoInitializer &initializer,
                                           Map &map, const Camera &camera,
                                           bool save_debug) {
  ProcessFrameResult result;
  FrontendFrameStats &fs = result.stats;

  const TrackResult tr = tracker.trackFrameToFrame(
      prev_frame_, curr_frame, active_points_2d_, active_landmark_ids_,
      save_debug, {0, 0});

  if (!tr.prev_points.empty()) {
    std::vector<float> du, dv;
    du.reserve(tr.prev_points.size());
    dv.reserve(tr.prev_points.size());
    for (size_t i = 0; i < tr.prev_points.size(); ++i) {
      du.push_back(tr.curr_points[i].x - tr.prev_points[i].x);
      dv.push_back(tr.curr_points[i].y - tr.prev_points[i].y);
    }
    const size_t mid = du.size() / 2;
    std::nth_element(du.begin(), du.begin() + mid, du.end());
    std::nth_element(dv.begin(), dv.begin() + mid, dv.end());
  }

  std::vector<Eigen::Vector3d> object_points;
  std::vector<int> valid_ids;
  std::vector<cv::Point2f> valid_curr_pts;
  for (size_t i = 0; i < tr.landmark_ids.size(); ++i) {
    const MapPoint *lm = map.landmark(tr.landmark_ids[i]);
    if (!lm) continue;
    object_points.push_back(lm->p_w);
    valid_ids.push_back(tr.landmark_ids[i]);
    valid_curr_pts.push_back(tr.curr_points[i]);
  }

  fs.num_correspondences = static_cast<int>(valid_ids.size());
  const auto &est_opts   = estimator.options();
  std::vector<int> pnp_inlier_idx;

  if (fs.num_correspondences >= est_opts.min_pnp_points) {
    Eigen::Matrix3d init_R_cw = Eigen::Matrix3d::Identity();
    Eigen::Vector3d init_t_cw = Eigen::Vector3d::Zero();

    const Eigen::Matrix4d T_pred = motion_model_.predict();
    svo::poseCwFromWc(T_pred, init_R_cw, init_t_cw);

    const PoseEstimateResult raw =
        estimator.estimatePosePnPRansac(object_points, valid_curr_pts,
                                        camera, init_R_cw, init_t_cw, true);

    if (raw.success) {
      fs.pose_success  = true;
      pnp_inlier_idx   = raw.inlier_indices;
      PoseEstimateResult final_pose = raw;

      if (raw.num_inliers >= est_opts.min_refine_inliers) {
        std::vector<Eigen::Vector3d> inlier_obj;
        std::vector<cv::Point2f>     inlier_img;
        gatherPnPInliers(object_points, valid_curr_pts,
                         raw.inlier_indices, inlier_obj, inlier_img);
        const PoseEstimateResult refined =
            estimator.refinePosePoseOnly(inlier_obj, inlier_img, camera,
                                         raw.rotation, raw.translation);
        if (refined.success)
          final_pose = refined;
      }

      fs.rmse_before = raw.reprojection_rmse_before;
      fs.rmse_after  = final_pose.reprojection_rmse_after;

      acceptPose(frame_id, raw.num_inliers, fs.num_correspondences,
                 svo::poseWcFromCw(final_pose.rotation, final_pose.translation),
                 fs);
    } else {
      rejectPose(frame_id, fs.num_correspondences, fs);
    }
  } else {
    rejectPose(frame_id, fs.num_correspondences, fs);
  }

  std::vector<cv::Point2f> culled_pts;
  std::vector<int>         culled_ids;

  if (!pnp_inlier_idx.empty()) {
    std::vector<int> outlier_ids;
    filterTrackedByInliers(valid_curr_pts, valid_ids, pnp_inlier_idx,
                           culled_pts, culled_ids, outlier_ids);
    map.markOutlierLandmarks(outlier_ids);
  } else {
    culled_pts = valid_curr_pts;
    culled_ids = valid_ids;
  }

  if (shouldReinitialize(frame_id, fs.pose_accepted,
                          static_cast<int>(culled_pts.size()))) {
    std::cout << "Reinitializing at frame " << frame_id << "\n";
    if (createKeyframeFromStereo(*this, curr_frame, poses_.back(),
                                  initializer, map, camera,
                                  /*replace_active=*/true,
                                  options_.min_initial_landmarks)) {
      noteReinitialized(frame_id);
      fs.reinitialized = true;
    } else {
      setActiveTracks(culled_pts, culled_ids);
    }
  } else {
    setActiveTracks(culled_pts, culled_ids);
  }

  if (!fs.reinitialized) {
    map.markTracked(culled_ids);

    const std::vector<int> local_ids = map.localMapLandmarkIds();
    const std::unordered_set<int> tracked_set(culled_ids.begin(), culled_ids.end());
    std::vector<int> missed_ids;
    missed_ids.reserve(local_ids.size());
    for (int id : local_ids) {
      if (!tracked_set.count(id)) missed_ids.push_back(id);
    }
    map.markMissedLandmarks(missed_ids);

    map.pruneLandmarks();
  }

  if (fs.pose_accepted) {
    curr_frame.pose_wc = poses_.back();
    curr_frame.tracked_points = active_points_2d_;
    curr_frame.tracked_landmark_ids = active_landmark_ids_;

    if (needNewKeyframe(curr_frame.pose_wc,
                        static_cast<int>(active_points_2d_.size()), frame_id)) {
      curr_frame.is_keyframe = true;
      map.markKeyframeObservations(culled_ids);
      if (!createKeyframeFromStereo(*this, curr_frame, curr_frame.pose_wc,
                                     initializer, map, camera,
                                     /*replace_active=*/false,
                                     options_.min_initial_landmarks)) {
        map.addKeyframe(curr_frame);
      }
      noteKeyframeInserted(frame_id, curr_frame.pose_wc);
      fs.inserted_keyframe = true;

      std::cout << "Inserted keyframe at frame " << frame_id
                << "  | active keyframes: " << map.numActiveKeyframes()
                << "  | active landmarks: " << map.numActiveLandmarks() << "\n";
    }
  }

  result.should_exit = static_cast<int>(active_points_2d_.size()) <
                        est_opts.min_pnp_points;
  if (save_debug)
    result.track_vis = tr.track_vis;

  prev_frame_ = curr_frame;
  return result;
}

// ---------------------------------------------------------------------------
bool Frontend::acceptPose(int frame_id, int num_inliers,
                          int num_correspondences,
                          const Eigen::Matrix4d &candidate_pose,
                          FrontendFrameStats &stats) {
  const double inlier_ratio =
      static_cast<double>(num_inliers) / std::max(1, num_correspondences);

  const Eigen::Vector3d t_prev = poses_.back().block<3, 1>(0, 3);
  const Eigen::Vector3d t_curr = candidate_pose.block<3, 1>(0, 3);
  const double delta_t = (t_curr - t_prev).norm();

  stats.num_inliers   = num_inliers;
  stats.inlier_ratio  = inlier_ratio;
  stats.delta_t       = delta_t;

  const bool accepted = num_inliers >= options_.min_pose_inliers &&
                        inlier_ratio >= options_.min_pose_inlier_ratio &&
                        delta_t <= options_.max_frame_translation_m;

  if (accepted) {
    poses_.push_back(candidate_pose);
    consecutive_rejected_poses_ = 0;
    stats.pose_accepted = true;
    motion_model_.update(poses_.back());
  } else {
    poses_.push_back(poses_.back());
    notePoseRejected(frame_id);
    motion_model_.onRejected();
  }

  return accepted;
}

void Frontend::notePoseRejected(int frame_id) {
  consecutive_rejected_poses_++;
  dense_debug_center_ = frame_id;
}

bool Frontend::shouldReinitialize(int frame_id, bool pose_accepted,
                                  int num_active_tracks) const {
  if (frame_id - last_init_frame_id_ <= options_.min_reinit_frame_gap)
    return false;

  const bool weak_but_accepted =
      pose_accepted && num_active_tracks < options_.weak_track_threshold;
  const bool emergency =
      !pose_accepted &&
      (consecutive_rejected_poses_ >= options_.emergency_rejected_poses_count ||
       num_active_tracks < options_.weak_track_threshold);

  return weak_but_accepted || emergency;
}

void Frontend::noteReinitialized(int frame_id) {
  last_init_frame_id_          = frame_id;
  consecutive_rejected_poses_  = 0;
}

void Frontend::noteKeyframeInserted(int frame_id,
                                    const Eigen::Matrix4d &pose_wc) {
  last_keyframe_frame_id_ = frame_id;
  last_keyframe_pose_wc_  = pose_wc;
}

void Frontend::repeatLastPose() { poses_.push_back(poses_.back()); }

void Frontend::rejectPose(int frame_id, int /*num_correspondences*/,
                           FrontendFrameStats & /*stats*/) {
  poses_.push_back(poses_.back());
  notePoseRejected(frame_id);
  motion_model_.onRejected();
}

void Frontend::setActiveTracks(const std::vector<cv::Point2f> &points,
                                const std::vector<int> &landmark_ids) {
  active_points_2d_ = points;
  active_landmark_ids_ = landmark_ids;
}

bool Frontend::shouldSaveDenseDebug(int frame_id, int radius) const {
  return dense_debug_center_ >= 0 &&
         std::abs(frame_id - dense_debug_center_) <= radius;
}

bool Frontend::needNewKeyframe(const Eigen::Matrix4d &current_pose_wc,
                               int num_tracked_points,
                               int current_frame_id) const {
  if (current_frame_id - last_keyframe_frame_id_ <
      options_.keyframe_min_frame_gap)
    return false;

  const Eigen::Vector3d t_last = last_keyframe_pose_wc_.block<3, 1>(0, 3);
  const Eigen::Vector3d t_curr = current_pose_wc.block<3, 1>(0, 3);
  const double translation = (t_curr - t_last).norm();

  const Eigen::Matrix3d R_last = last_keyframe_pose_wc_.block<3, 3>(0, 0);
  const Eigen::Matrix3d R_curr = current_pose_wc.block<3, 3>(0, 0);
  const Eigen::Matrix3d R_rel  = R_last.transpose() * R_curr;

  double trace_val = (R_rel.trace() - 1.0) * 0.5;
  trace_val = std::max(-1.0, std::min(1.0, trace_val));
  const double rotation_deg = std::acos(trace_val) * 180.0 / M_PI;

  const bool trans_trigger =
      translation >= options_.keyframe_translation_threshold_m;
  const bool rot_trigger =
      rotation_deg >= options_.keyframe_rotation_threshold_deg;
  const bool low_track_trigger =
      num_tracked_points < options_.keyframe_min_tracked_points &&
      translation >= options_.keyframe_low_track_translation_threshold_m;

  return trans_trigger || rot_trigger || low_track_trigger;
}

} // namespace svo
