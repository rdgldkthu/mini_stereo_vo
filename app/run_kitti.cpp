#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

#include "svo/camera.h"
#include "svo/dataset_kitti.h"
#include "svo/estimator.h"
#include "svo/frame.h"
#include "svo/frontend.h"
#include "svo/map.h"
#include "svo/map_point.h"
#include "svo/stereo_initializer.h"
#include "svo/tracker.h"

namespace fs = std::filesystem;

namespace {

std::vector<cv::Point2f>
makeInitialActivePoints(const svo::StereoInitResult &init_result) {
  std::vector<cv::Point2f> points;

  const size_t n =
      std::min(init_result.features.size(), init_result.landmarks.size());
  points.reserve(n);

  for (size_t i = 0; i < n; ++i) {
    points.push_back(init_result.features[i].kp_left.pt);
  }

  return points;
}

std::vector<svo::MapPoint>
makeInitialActiveLandmarks(const svo::StereoInitResult &init_result) {
  std::vector<svo::MapPoint> landmarks;

  const size_t n =
      std::min(init_result.features.size(), init_result.landmarks.size());
  landmarks.reserve(n);

  for (size_t i = 0; i < n; ++i) {
    svo::MapPoint landmark = init_result.landmarks[i];
    landmark.observed_times = 1;
    landmark.tracked_times = 1;
    landmark.missed_times = 0;
    landmark.is_outlier = false;
    landmark.is_active = true;
    landmarks.push_back(landmark);
  }

  return landmarks;
}

Eigen::Matrix4d makePoseWcFromPnP(const Eigen::Matrix3d &R_cw,
                                  const Eigen::Vector3d &t_cw) {
  Eigen::Matrix4d T_wc = Eigen::Matrix4d::Identity();

  const Eigen::Matrix3d R_wc = R_cw.transpose();
  const Eigen::Vector3d t_wc = -R_wc * t_cw;

  T_wc.block<3, 3>(0, 0) = R_wc;
  T_wc.block<3, 1>(0, 3) = t_wc;

  return T_wc;
}

void makePoseCwFromPoseWc(const Eigen::Matrix4d &T_wc, Eigen::Matrix3d &R_cw,
                          Eigen::Vector3d &t_cw) {
  const Eigen::Matrix3d R_wc = T_wc.block<3, 3>(0, 0);
  const Eigen::Vector3d t_wc = T_wc.block<3, 1>(0, 3);

  R_cw = R_wc.transpose();
  t_cw = -R_cw * t_wc;
}

std::vector<svo::MapPoint>
transformLandmarksToWorld(const std::vector<svo::MapPoint> &local_landmarks,
                          const Eigen::Matrix4d &T_wc) {
  std::vector<svo::MapPoint> world_landmarks = local_landmarks;

  const Eigen::Matrix3d R_wc = T_wc.block<3, 3>(0, 0);
  const Eigen::Vector3d t_wc = T_wc.block<3, 1>(0, 3);

  for (auto &landmark : world_landmarks) {
    landmark.p_w = R_wc * landmark.p_w + t_wc;
  }

  return world_landmarks;
}

void writeKittiTrajectory(const fs::path &out_path,
                          const std::vector<Eigen::Matrix4d> &poses) {
  fs::create_directories(out_path.parent_path());

  std::ofstream ofs(out_path);
  if (!ofs) {
    throw std::runtime_error("Failed to open output file: " +
                             out_path.string());
  }

  ofs << std::fixed << std::setprecision(9);

  for (const auto &T : poses) {
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 4; ++c) {
        ofs << T(r, c);
        if (!(r == 2 && c == 3)) {
          ofs << " ";
        }
      }
    }
    ofs << "\n";
  }
}

void gatherPnPInliers(const std::vector<Eigen::Vector3d> &object_points,
                      const std::vector<cv::Point2f> &image_points,
                      const cv::Mat &inlier_indices,
                      std::vector<Eigen::Vector3d> &inlier_object_points,
                      std::vector<cv::Point2f> &inlier_image_points) {
  inlier_object_points.clear();
  inlier_image_points.clear();

  for (int i = 0; i < inlier_indices.rows; ++i) {
    const int idx = inlier_indices.at<int>(i, 0);
    if (idx < 0 || idx >= static_cast<int>(object_points.size())) {
      continue;
    }
    inlier_object_points.push_back(object_points[idx]);
    inlier_image_points.push_back(image_points[idx]);
  }
}

bool isInDebugWindow(int frame_id, int center, int radius) {
  return std::abs(frame_id - center) <= radius;
}

std::vector<svo::MapPoint> refreshTrackedLandmarksFromMap(
    const std::vector<svo::MapPoint> &tracked_landmarks,
    const std::vector<svo::MapPoint> &map_landmarks) {
  std::unordered_map<int, svo::MapPoint> id_to_landmark;
  id_to_landmark.reserve(map_landmarks.size());

  for (const auto &landmark : map_landmarks) {
    id_to_landmark[landmark.id] = landmark;
  }

  std::vector<svo::MapPoint> refreshed;
  refreshed.reserve(tracked_landmarks.size());

  for (const auto &landmark : tracked_landmarks) {
    const auto it = id_to_landmark.find(landmark.id);
    if (it != id_to_landmark.end()) {
      refreshed.push_back(it->second);
    } else {
      refreshed.push_back(landmark);
    }
  }

  return refreshed;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 3 || argc > 4) {
    std::cerr << "Usage: " << argv[0]
              << " <kitti_root> <sequence> [output_pose_file]\n";
    std::cerr << "Example: " << argv[0]
              << " data/kitti 05 results/traj/05_vo.txt\n";
    return 1;
  }

  const fs::path kitti_root = argv[1];
  const std::string sequence = argv[2];
  const fs::path output_pose =
      (argc == 4) ? fs::path(argv[3])
                  : fs::path("results/traj") / (sequence + "_vo.txt");

  fs::create_directories("results/debug");
  fs::create_directories("results/traj");

  // -------------------------------------------------------------------------
  // Dataset + camera
  // -------------------------------------------------------------------------
  svo::DatasetKitti dataset;
  if (!dataset.open(kitti_root.string(), sequence)) {
    std::cerr << "Failed to open KITTI dataset.\n";
    return 1;
  }

  svo::Camera camera;
  if (!camera.loadFromKittiCalib(dataset.calibPath().string())) {
    std::cerr << "Failed to load KITTI calibration from: "
              << dataset.calibPath() << "\n";
    return 1;
  }

  std::cout << "Loaded calibration from: " << dataset.calibPath() << "\n";
  camera.print();

  // -------------------------------------------------------------------------
  // Modules
  // -------------------------------------------------------------------------
  svo::StereoInitializer::Options init_options;
  init_options.max_features = 1500;
  init_options.hamming_threshold = 40;
  init_options.row_tolerance_px = 2.0;
  init_options.min_disparity_px = 3.0;
  init_options.max_disparity_px = 120.0;
  init_options.max_depth_m = 80.0;
  init_options.image_border_px = 10;
  init_options.max_visualized_matches = 100;
  svo::StereoInitializer initializer(init_options);

  svo::Tracker::Options tracker_options;
  tracker_options.win_size = cv::Size(21, 21);
  tracker_options.max_level = 3;
  tracker_options.max_bidirectional_error_px = 1.5;
  tracker_options.image_border_px = 10;
  tracker_options.max_visualized_tracks = 150;
  svo::Tracker tracker(tracker_options);

  svo::Estimator::Options estimator_options;
  estimator_options.use_extrinsic_guess = false;
  estimator_options.iterations_count = 100;
  estimator_options.reprojection_error_px = 4.0f;
  estimator_options.confidence = 0.99;
  estimator_options.min_pnp_points = 6;
  estimator_options.pose_refine_iterations = 10;
  estimator_options.pose_refine_epsilon = 1e-6;
  estimator_options.pose_refine_huber_delta = 5.0;
  estimator_options.min_refine_inliers = 10;
  estimator_options.local_ba_iterations = 3;
  estimator_options.local_ba_epsilon = 1e-6;
  estimator_options.local_ba_huber_delta = 5.0;
  estimator_options.max_ba_keyframes = 3;
  estimator_options.max_ba_landmarks = 60;
  estimator_options.min_ba_observations = 25;
  svo::Estimator estimator(estimator_options);

  svo::Frontend::Options frontend_options;
  frontend_options.keyframe_translation_threshold_m = 1.5;
  frontend_options.keyframe_rotation_threshold_deg = 12.0;
  frontend_options.keyframe_min_tracked_points = 60;
  frontend_options.keyframe_min_frame_gap = 5;
  frontend_options.keyframe_low_track_translation_threshold_m = 0.5;
  svo::Frontend frontend(frontend_options);

  svo::Map::Options map_options;
  map_options.max_active_keyframes = 5;
  map_options.max_active_landmarks = 2000;
  map_options.min_observed_times = 2;
  map_options.max_missed_times = 8;
  svo::Map map(map_options);

  // -------------------------------------------------------------------------
  // Initial stereo bootstrapping
  // -------------------------------------------------------------------------
  svo::Frame frame0;
  if (!dataset.loadFrame(0, frame0)) {
    std::cerr << "Failed to load frame 0.\n";
    return 1;
  }

  const svo::StereoInitResult init_result = initializer.run(frame0, camera);

  std::cout << "Stereo initialization result:\n";
  std::cout << "  triangulated: " << init_result.num_triangulated << "\n";

  if (init_result.num_triangulated < 20) {
    std::cerr << "Too few initial landmarks.\n";
    return 1;
  }

  if (!init_result.match_vis.empty()) {
    cv::imwrite("results/debug/" + sequence + "_init_matches.png",
                init_result.match_vis);
  }

  std::vector<cv::Point2f> active_points_2d =
      makeInitialActivePoints(init_result);
  std::vector<svo::MapPoint> active_landmarks =
      makeInitialActiveLandmarks(init_result);
  map.assignNewLandmarkIds(active_landmarks);

  // frame 0 defines world frame
  frame0.pose_wc = Eigen::Matrix4d::Identity();
  frame0.is_keyframe = true;
  frame0.tracked_points = active_points_2d;
  frame0.tracked_landmark_ids.clear();
  frame0.tracked_landmark_ids.reserve(active_landmarks.size());
  for (const auto &landmark : active_landmarks) {
    frame0.tracked_landmark_ids.push_back(landmark.id);
  }

  map.addKeyframe(frame0);
  map.setActiveLandmarks(active_landmarks);

  int last_keyframe_frame_id = 0;
  Eigen::Matrix4d last_keyframe_pose_wc = frame0.pose_wc;

  // -------------------------------------------------------------------------
  // Runtime state
  // -------------------------------------------------------------------------
  std::vector<Eigen::Matrix4d> poses;
  poses.reserve(dataset.numFrames());
  poses.push_back(Eigen::Matrix4d::Identity());

  svo::Frame prev_frame = frame0;
  int last_init_frame_id = 0;
  int consecutive_rejected_poses = 0;
  int dense_debug_center = -1;

  // -------------------------------------------------------------------------
  // Logging
  // -------------------------------------------------------------------------
  std::ofstream stats("results/debug/" + sequence + "_vo_stats.csv");
  stats << "frame_id,num_active_points,num_correspondences,num_inliers,"
           "inlier_ratio,pose_success,pose_accepted,reinitialized,is_keyframe,"
           "num_keyframes,num_map_landmarks,local_ba,ba_rmse_before,ba_rmse_"
           "after,tx,ty,tz,delta_t,rmse_before,rmse_after\n";

  stats << "0," << active_points_2d.size() << ",0,0,0.0,1,1,0,1,"
        << map.numActiveKeyframes() << "," << map.numActiveLandmarks() << ","
        << "0,0,0,0,0,0,0,0,0\n";

  // -------------------------------------------------------------------------
  // Main VO loop
  // -------------------------------------------------------------------------
  for (int frame_id = 1; frame_id < dataset.numFrames(); ++frame_id) {
    svo::Frame curr_frame;
    if (!dataset.loadFrame(frame_id, curr_frame)) {
      std::cerr << "Failed to load frame " << frame_id << "\n";
      poses.push_back(poses.back());

      const Eigen::Vector3d t_out = poses.back().block<3, 1>(0, 3);
      stats << frame_id << ",0,0,0,0.0,0,0,0,0," << map.numActiveKeyframes()
            << "," << map.numActiveLandmarks() << "," << t_out(0) << ","
            << t_out(1) << "," << t_out(2) << ",0,0,0\n";
      continue;
    }

    const svo::TrackResult track_result = tracker.trackFrameToFrame(
        prev_frame, curr_frame, active_points_2d, active_landmarks);

    bool pose_success = false;
    bool pose_accepted = false;
    bool reinitialized = false;
    bool inserted_keyframe = false;
    bool ran_local_ba = false;

    int num_inliers = 0;
    double inlier_ratio = 0.0;
    double delta_t = 0.0;
    double rmse_before = 0.0;
    double rmse_after = 0.0;
    double ba_rmse_before = 0.0;
    double ba_rmse_after = 0.0;

    Eigen::Matrix4d candidate_pose = poses.back();

    // ---------------------------------------------------------------------
    // Raw PnP + pose-only refinement
    // ---------------------------------------------------------------------
    if (track_result.num_valid_correspondences >=
        estimator_options.min_pnp_points) {
      Eigen::Matrix3d init_R_cw = Eigen::Matrix3d::Identity();
      Eigen::Vector3d init_t_cw = Eigen::Vector3d::Zero();
      makePoseCwFromPoseWc(poses.back(), init_R_cw, init_t_cw);

      const svo::PoseEstimateResult raw_pose_result =
          estimator.estimatePosePnPRansac(track_result.object_points,
                                          track_result.image_points, camera,
                                          init_R_cw, init_t_cw, true);

      if (raw_pose_result.success) {
        pose_success = true;
        num_inliers = raw_pose_result.num_inliers;
        inlier_ratio = static_cast<double>(num_inliers) /
                       std::max(1, track_result.num_valid_correspondences);

        svo::PoseEstimateResult final_pose_result = raw_pose_result;

        if (raw_pose_result.num_inliers >=
            estimator_options.min_refine_inliers) {
          std::vector<Eigen::Vector3d> inlier_object_points;
          std::vector<cv::Point2f> inlier_image_points;
          gatherPnPInliers(track_result.object_points,
                           track_result.image_points,
                           raw_pose_result.inlier_indices, inlier_object_points,
                           inlier_image_points);

          const svo::PoseEstimateResult refined_pose_result =
              estimator.refinePosePoseOnly(
                  inlier_object_points, inlier_image_points, camera,
                  raw_pose_result.rotation, raw_pose_result.translation);

          if (refined_pose_result.success) {
            final_pose_result = refined_pose_result;
          }
        }

        rmse_before = raw_pose_result.reprojection_rmse_before;
        rmse_after = final_pose_result.reprojection_rmse_after;

        candidate_pose = makePoseWcFromPnP(final_pose_result.rotation,
                                           final_pose_result.translation);

        const Eigen::Vector3d t_prev = poses.back().block<3, 1>(0, 3);
        const Eigen::Vector3d t_curr = candidate_pose.block<3, 1>(0, 3);
        delta_t = (t_curr - t_prev).norm();

        const bool enough_inliers = (raw_pose_result.num_inliers >= 15);
        const bool enough_ratio = (inlier_ratio >= 0.10);
        const bool reasonable_jump = (delta_t <= 2.0);

        if (enough_inliers && enough_ratio && reasonable_jump) {
          pose_accepted = true;
          poses.push_back(candidate_pose);
          consecutive_rejected_poses = 0;
        } else {
          poses.push_back(poses.back());
          consecutive_rejected_poses++;
          dense_debug_center = frame_id;
        }
      } else {
        poses.push_back(poses.back());
        consecutive_rejected_poses++;
        dense_debug_center = frame_id;
      }
    } else {
      poses.push_back(poses.back());
      consecutive_rejected_poses++;
      dense_debug_center = frame_id;
    }

    // ---------------------------------------------------------------------
    // Reinitialization policy
    // ---------------------------------------------------------------------
    const bool weak_but_accepted =
        pose_accepted && (frame_id - last_init_frame_id > 10) &&
        (num_inliers < 12 || track_result.curr_points.size() < 80);

    const bool emergency_reinit = !pose_accepted &&
                                  (frame_id - last_init_frame_id > 10) &&
                                  (consecutive_rejected_poses >= 2 ||
                                   track_result.curr_points.size() < 80);

    if (weak_but_accepted || emergency_reinit) {
      std::cout << "Reinitializing at frame " << frame_id << "\n";

      const svo::StereoInitResult reinit_result =
          initializer.run(curr_frame, camera);

      if (reinit_result.num_triangulated >= 20) {
        active_points_2d = makeInitialActivePoints(reinit_result);

        std::vector<svo::MapPoint> new_landmarks = makeInitialActiveLandmarks(reinit_result);
        map.assignNewLandmarkIds(new_landmarks);

        active_landmarks = transformLandmarksToWorld(new_landmarks, poses.back());

        map.setActiveLandmarks(active_landmarks);

        last_init_frame_id = frame_id;
        reinitialized = true;
        consecutive_rejected_poses = 0;
      } else {
        active_points_2d = track_result.curr_points;
        active_landmarks = track_result.tracked_landmarks;
      }
    } else {
      active_points_2d = track_result.curr_points;
      active_landmarks = track_result.tracked_landmarks;
    }

    // ---------------------------------------------------------------------
    // Landmark bookkeeping
    // ---------------------------------------------------------------------
    map.markTrackedLandmarks(track_result.tracked_landmarks);
    map.markMissedLandmarks(track_result.landmark_ids);
    map.pruneLandmarks();

    // ---------------------------------------------------------------------
    // Keyframe insertion
    // ---------------------------------------------------------------------
    if (pose_accepted) {
      curr_frame.pose_wc = poses.back();
      curr_frame.tracked_points = active_points_2d;
      curr_frame.tracked_landmark_ids.clear();
      curr_frame.tracked_landmark_ids.reserve(active_landmarks.size());
      for (const auto &landmark : active_landmarks) {
        curr_frame.tracked_landmark_ids.push_back(landmark.id);
      }

      if (frontend.needNewKeyframe(last_keyframe_pose_wc, curr_frame.pose_wc,
                                   static_cast<int>(active_points_2d.size()),
                                   frame_id, last_keyframe_frame_id)) {
        curr_frame.is_keyframe = true;
        map.addKeyframe(curr_frame);

        const svo::StereoInitResult keyframe_init_result =
            initializer.run(curr_frame, camera);

        if (keyframe_init_result.num_triangulated >= 20) {
          std::vector<svo::MapPoint> new_landmarks = makeInitialActiveLandmarks(keyframe_init_result);
          map.assignNewLandmarkIds(new_landmarks);

          new_landmarks = transformLandmarksToWorld(new_landmarks, curr_frame.pose_wc);

          map.addLandmarks(new_landmarks);

          active_points_2d = makeInitialActivePoints(keyframe_init_result);
          active_landmarks = new_landmarks;
        }

        // -------------------------
        // Local bundle adjustment
        // -------------------------
        if (map.numActiveKeyframes() >= 2 && map.numActiveLandmarks() >= 20) {
          svo::LocalBAResult ba_result = estimator.runLocalBundleAdjustment(
              map.mutableActiveKeyframes(), map.mutableActiveLandmarks(),
              camera);

          if (ba_result.success) {
            ran_local_ba = true;
            ba_rmse_before = ba_result.rmse_before;
            ba_rmse_after = ba_result.rmse_after;

            // refresh the current active tracked landmarks using optimized map
            // positions
            active_landmarks = refreshTrackedLandmarksFromMap(
                active_landmarks, map.activeLandmarks());

            std::cout << "Local BA at frame " << frame_id
                      << " | keyframes: " << ba_result.num_keyframes
                      << " | landmarks: " << ba_result.num_landmarks
                      << " | observations: " << ba_result.num_observations
                      << " | rmse: " << ba_result.rmse_before << " -> "
                      << ba_result.rmse_after << "\n";
          }
        }

        last_keyframe_frame_id = frame_id;
        last_keyframe_pose_wc = curr_frame.pose_wc;
        inserted_keyframe = true;

        std::cout << "Inserted keyframe at frame " << frame_id
                  << " | active keyframes: " << map.numActiveKeyframes()
                  << " | active landmarks: " << map.numActiveLandmarks()
                  << "\n";
      }
    }

    // ---------------------------------------------------------------------
    // Logging + debug output
    // ---------------------------------------------------------------------
    const Eigen::Vector3d t_out = poses.back().block<3, 1>(0, 3);

    stats << frame_id << "," << active_points_2d.size() << ","
          << track_result.num_valid_correspondences << "," << num_inliers << ","
          << inlier_ratio << "," << (pose_success ? 1 : 0) << ","
          << (pose_accepted ? 1 : 0) << "," << (reinitialized ? 1 : 0) << ","
          << (inserted_keyframe ? 1 : 0) << "," << map.numActiveKeyframes()
          << "," << map.numActiveLandmarks() << "," << (ran_local_ba ? 1 : 0)
          << "," << ba_rmse_before << "," << ba_rmse_after << "," << t_out(0)
          << "," << t_out(1) << "," << t_out(2) << "," << delta_t << ","
          << rmse_before << "," << rmse_after << "\n";

    const bool save_sparse_debug = (frame_id % 10 == 0);
    const bool save_dense_debug =
        (dense_debug_center >= 0) &&
        isInDebugWindow(frame_id, dense_debug_center, 10);

    if (!track_result.track_vis.empty() &&
        (save_sparse_debug || save_dense_debug)) {
      const std::string image_path = "results/debug/" + sequence + "_track_" +
                                     cv::format("%06d", frame_id) + ".png";
      cv::imwrite(image_path, track_result.track_vis);
    }

    std::cout << "frame " << frame_id
              << " | active: " << active_points_2d.size()
              << " | corr: " << track_result.num_valid_correspondences
              << " | inliers: " << num_inliers << " | ratio: " << inlier_ratio
              << " | delta_t: " << delta_t
              << " | pose_success: " << pose_success
              << " | pose_accepted: " << pose_accepted
              << " | reinit: " << reinitialized
              << " | keyframe: " << inserted_keyframe << "\n";

    if (active_points_2d.size() < 20) {
      std::cout << "Tracking dropped below threshold at frame " << frame_id
                << ". Stopping early.\n";
      break;
    }

    prev_frame = curr_frame;
  }

  // -------------------------------------------------------------------------
  // Final trajectory write
  // -------------------------------------------------------------------------
  writeKittiTrajectory(output_pose, poses);
  std::cout << "Wrote VO trajectory to: " << output_pose << "\n";

  return 0;
}