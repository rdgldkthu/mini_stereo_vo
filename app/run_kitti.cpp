#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
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
#include "svo/pose_writer.h"
#include "svo/stereo_initializer.h"
#include "svo/tracker.h"
#include "svo/rerun_viewer.h"
#include "svo/viewer.h"

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

void gatherPnPInliers(const std::vector<Eigen::Vector3d> &object_points,
                      const std::vector<cv::Point2f> &image_points,
                      const std::vector<int> &inlier_indices,
                      std::vector<Eigen::Vector3d> &inlier_object_points,
                      std::vector<cv::Point2f> &inlier_image_points) {
  inlier_object_points.clear();
  inlier_image_points.clear();

  for (const int idx : inlier_indices) {
    if (idx < 0 || idx >= static_cast<int>(object_points.size())) {
      continue;
    }
    inlier_object_points.push_back(object_points[idx]);
    inlier_image_points.push_back(image_points[idx]);
  }
}


std::vector<Eigen::Matrix4d> loadKittiPoses(const fs::path &pose_path) {
  std::vector<Eigen::Matrix4d> poses;

  std::ifstream ifs(pose_path);
  if (!ifs) {
    std::cerr << "GT pose file not found: " << pose_path << "\n";
    return poses;
  }

  std::string line;
  while (std::getline(ifs, line)) {
    std::stringstream ss(line);

    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();

    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 4; ++c) {
        ss >> T(r, c);
      }
    }

    if (ss) {
      poses.push_back(T);
    }
  }

  std::cout << "Loaded " << poses.size() << " ground-truth poses from "
            << pose_path << "\n";

  return poses;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " <kitti_root> <sequence> [pose_keyword] [--save-debug] [--no-viewer]\n";
    std::cerr << "Example: " << argv[0] << " data/kitti 05 vo\n";
    return 1;
  }

  const fs::path kitti_root = argv[1];
  const std::string sequence = argv[2];

  bool save_debug = false;
  bool no_viewer = false;
  std::string pose_keyword;
  for (int i = 3; i < argc; ++i) {
    if (std::string(argv[i]) == "--save-debug") {
      save_debug = true;
    } else if (std::string(argv[i]) == "--no-viewer") {
      no_viewer = true;
    } else {
      pose_keyword = argv[i];
    }
  }

  std::time_t now = std::time(nullptr);
  char date_buf[7];
  std::strftime(date_buf, sizeof(date_buf), "%y%m%d", std::localtime(&now));
  const std::string date_str(date_buf);

  const std::string file_name_stem = pose_keyword.empty()
      ? sequence + "_" + date_str
      : sequence + "_" + date_str + "_" + pose_keyword;
  const fs::path output_pose = fs::path("results/traj") / (file_name_stem + ".txt");

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

  const fs::path gt_pose_path = kitti_root / "poses" / (sequence + ".txt");
  const std::vector<Eigen::Matrix4d> gt_poses = loadKittiPoses(gt_pose_path);

  // -------------------------------------------------------------------------
  // Modules
  // -------------------------------------------------------------------------
  svo::StereoInitializer::Options init_options;
  init_options.max_features = 1000;
  init_options.hamming_threshold = 40;
  init_options.row_tolerance_px = 2.0;
  init_options.min_disparity_px = 3.0;
  init_options.max_disparity_px = 120.0;
  init_options.max_depth_m = 80.0;
  init_options.image_border_px = 10;
  init_options.max_visualized_matches = 100;
  init_options.grid_rows = 4;
  init_options.grid_cols = 8;
  init_options.max_per_cell = 10;
  svo::StereoInitializer initializer(init_options);

  svo::Tracker::Options tracker_options;
  tracker_options.win_size = cv::Size(25, 25);
  tracker_options.max_level = 4;
  tracker_options.max_bidirectional_error_px = 1.5;
  tracker_options.image_border_px = 10;
  tracker_options.max_visualized_tracks = 150;
  svo::Tracker tracker(tracker_options);

  svo::Estimator::Options estimator_options;
  estimator_options.use_extrinsic_guess = false;
  estimator_options.iterations_count = 100;
  estimator_options.reprojection_error_px = 3.0f;
  estimator_options.confidence = 0.99;
  estimator_options.min_pnp_points = 6;
  estimator_options.pose_refine_iterations = 10;
  estimator_options.pose_refine_epsilon = 1e-6;
  estimator_options.pose_refine_huber_delta = 5.0;
  estimator_options.min_refine_inliers = 10;
  estimator_options.local_ba_iterations = 3;
  estimator_options.local_ba_epsilon = 1e-6;
  estimator_options.local_ba_huber_delta = 5.0;
  estimator_options.local_ba_damping = 1e-3;
  estimator_options.max_ba_keyframes = 3;
  estimator_options.max_ba_landmarks = 100;
  estimator_options.min_ba_observations = 20;
  estimator_options.min_ba_landmark_observations = 2;
  svo::Estimator estimator(estimator_options);

  svo::Frontend::Options frontend_options;
  frontend_options.keyframe_translation_threshold_m = 1.5;
  frontend_options.keyframe_rotation_threshold_deg = 8.0;
  frontend_options.keyframe_min_tracked_points = 60;
  frontend_options.keyframe_min_frame_gap = 5;
  frontend_options.keyframe_low_track_translation_threshold_m = 0.5;
  frontend_options.min_pose_inliers = 15;
  frontend_options.min_pose_inlier_ratio = 0.10;
  frontend_options.max_frame_translation_m = 2.0;
  frontend_options.min_reinit_frame_gap = 10;
  frontend_options.weak_track_threshold = 80;
  frontend_options.emergency_rejected_poses_count = 2;
  frontend_options.local_ba_keyframe_interval = 2;
  svo::Frontend frontend(frontend_options);

  svo::Map::Options map_options;
  map_options.max_active_keyframes = 5;
  map_options.max_active_landmarks = 2000;
  map_options.min_observed_times = 2;
  map_options.max_missed_times = 8;
  svo::Map map(map_options);

  svo::Viewer::Options viewer_options;
  viewer_options.image_wait_ms = 1;
  viewer_options.trajectory_size = 600;
  viewer_options.trajectory_scale = 0.5;
  svo::Viewer viewer(viewer_options);

  svo::RerunViewer::Options rerun_options;
  rerun_options.app_id = "stereo_slam";
  rerun_options.spawn_viewer = !no_viewer;
  svo::RerunViewer rerun_viewer(rerun_options);

  // -------------------------------------------------------------------------
  // Initial stereo bootstrapping
  // -------------------------------------------------------------------------
  svo::Frame frame0;
  if (!dataset.loadFrame(0, frame0)) {
    std::cerr << "Failed to load frame 0.\n";
    return 1;
  }

  const svo::StereoInitResult init_result = initializer.run(frame0, camera, save_debug);

  std::cout << "Stereo initialization result:\n";
  std::cout << "  triangulated: " << init_result.num_triangulated << "\n";

  if (init_result.num_triangulated < 20) {
    std::cerr << "Too few initial landmarks.\n";
    return 1;
  }

  if (save_debug && !init_result.match_vis.empty()) {
    cv::imwrite("results/debug/" + file_name_stem + "_init_matches.png",
                init_result.match_vis);
  }

  std::vector<svo::MapPoint> active_landmarks =
      makeInitialActiveLandmarks(init_result);
  map.assignNewLandmarkIds(active_landmarks);

  frontend.initialize(frame0, makeInitialActivePoints(init_result), active_landmarks);

  frame0.pose_wc = Eigen::Matrix4d::Identity();
  frame0.is_keyframe = true;
  frame0.tracked_points = frontend.activePoints();
  frame0.tracked_landmark_ids.clear();
  for (const auto &landmark : active_landmarks) {
    frame0.tracked_landmark_ids.push_back(landmark.id);
  }

  map.addKeyframe(frame0);
  map.setActiveLandmarks(active_landmarks);

  // -------------------------------------------------------------------------
  // Logging
  // -------------------------------------------------------------------------
  std::ofstream stats("results/debug/" + file_name_stem + "_stats.csv");
  stats << "frame_id,num_active_points,num_correspondences,num_inliers,"
           "inlier_ratio,pose_success,pose_accepted,reinitialized,is_keyframe,"
           "num_keyframes,num_map_landmarks,local_ba,local_ba_accepted,"
           "local_ba_rejected,ba_rmse_before,ba_rmse_after,"
           "tx,ty,tz,delta_t,rmse_before,rmse_after\n";

  stats << "0," << frontend.activePoints().size() << ",0,0,0.0,1,1,0,1,"
        << map.numActiveKeyframes() << "," << map.numActiveLandmarks() << ","
        << "0,0,0,0,0,0,0,0,0,0,0\n";

  // -------------------------------------------------------------------------
  // Main VO loop
  // -------------------------------------------------------------------------
  const auto loop_start = std::chrono::steady_clock::now();
  int frames_processed = 0;
  cv::Point2f motion_hint{0.0f, 0.0f};

  for (int frame_id = 1; frame_id < dataset.numFrames(); ++frame_id) {
    svo::Frame curr_frame;
    if (!dataset.loadFrame(frame_id, curr_frame)) {
      std::cerr << "Failed to load frame " << frame_id << "\n";
      frontend.repeatLastPose();

      const Eigen::Vector3d t_out = frontend.currentPose().block<3, 1>(0, 3);
      stats << frame_id << ",0,0,0,0.0,0,0,0,0," << map.numActiveKeyframes()
            << "," << map.numActiveLandmarks() << "," << "0,0,0,0,0," << t_out(0)
            << "," << t_out(1) << "," << t_out(2) << ",0,0,0\n";
      continue;
    }

    const svo::TrackResult track_result = tracker.trackFrameToFrame(
        frontend.previousFrame(), curr_frame, frontend.activePoints(),
        frontend.activeLandmarks(), save_debug, motion_hint);

    if (!track_result.prev_points.empty()) {
      std::vector<float> du, dv;
      du.reserve(track_result.prev_points.size());
      dv.reserve(track_result.prev_points.size());
      for (size_t i = 0; i < track_result.prev_points.size(); ++i) {
        du.push_back(track_result.curr_points[i].x - track_result.prev_points[i].x);
        dv.push_back(track_result.curr_points[i].y - track_result.prev_points[i].y);
      }
      const size_t mid = du.size() / 2;
      std::nth_element(du.begin(), du.begin() + mid, du.end());
      std::nth_element(dv.begin(), dv.begin() + mid, dv.end());
      motion_hint = {du[mid], dv[mid]};
    }

    svo::FrontendFrameStats frame_stats;
    Eigen::Matrix4d candidate_pose = frontend.currentPose();

    // ---------------------------------------------------------------------
    // Raw PnP + pose-only refinement
    // ---------------------------------------------------------------------
    if (track_result.num_valid_correspondences >=
        estimator_options.min_pnp_points) {
      Eigen::Matrix3d init_R_cw = Eigen::Matrix3d::Identity();
      Eigen::Vector3d init_t_cw = Eigen::Vector3d::Zero();
      makePoseCwFromPoseWc(frontend.currentPose(), init_R_cw, init_t_cw);

      const svo::PoseEstimateResult raw_pose_result =
          estimator.estimatePosePnPRansac(track_result.object_points,
                                          track_result.image_points, camera,
                                          init_R_cw, init_t_cw, true);

      if (raw_pose_result.success) {
        frame_stats.pose_success = true;

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

        frame_stats.rmse_before = raw_pose_result.reprojection_rmse_before;
        frame_stats.rmse_after = final_pose_result.reprojection_rmse_after;

        candidate_pose = makePoseWcFromPnP(final_pose_result.rotation,
                                           final_pose_result.translation);

        frontend.acceptPose(frame_id, raw_pose_result.num_inliers,
                            track_result.num_valid_correspondences,
                            candidate_pose, frame_stats);
      } else {
        frontend.rejectPose(frame_id, track_result.num_valid_correspondences,
                            frame_stats);
      }
    } else {
      frontend.rejectPose(frame_id, track_result.num_valid_correspondences,
                          frame_stats);
    }

    // ---------------------------------------------------------------------
    // Reinitialization policy
    // ---------------------------------------------------------------------
    if (frontend.shouldReinitialize(
            frame_id, frame_stats.pose_accepted,
            static_cast<int>(track_result.curr_points.size()))) {
      std::cout << "Reinitializing at frame " << frame_id << "\n";

      const svo::StereoInitResult reinit_result =
          initializer.run(curr_frame, camera, false);

      if (reinit_result.num_triangulated >= 20) {
        std::vector<svo::MapPoint> new_landmarks =
            makeInitialActiveLandmarks(reinit_result);
        map.assignNewLandmarkIds(new_landmarks);

        new_landmarks =
            transformLandmarksToWorld(new_landmarks, frontend.currentPose());

        frontend.setActiveTracks(makeInitialActivePoints(reinit_result), new_landmarks);
        map.setActiveLandmarks(new_landmarks);

        frontend.noteReinitialized(frame_id);
        frame_stats.reinitialized = true;
        motion_hint = {0.0f, 0.0f};
      } else {
        frontend.setActiveTracks(track_result.curr_points, track_result.tracked_landmarks);
      }
    } else {
      frontend.setActiveTracks(track_result.curr_points,
                               track_result.tracked_landmarks);
    }

    if (!frame_stats.reinitialized) {
      map.markTrackedLandmarks(track_result.tracked_landmarks);
      map.markMissedLandmarks(track_result.landmark_ids);
      map.pruneLandmarks();
    }

    // ---------------------------------------------------------------------
    // Keyframe insertion
    // ---------------------------------------------------------------------
    if (frame_stats.pose_accepted) {
      curr_frame.pose_wc = frontend.currentPose();
      curr_frame.tracked_points = frontend.activePoints();
      curr_frame.tracked_landmark_ids.clear();
      for (const auto& landmark : frontend.activeLandmarks()) {
        curr_frame.tracked_landmark_ids.push_back(landmark.id);
      }

      if (frontend.needNewKeyframe(curr_frame.pose_wc, static_cast<int>(frontend.activePoints().size()), frame_id)) {
        curr_frame.is_keyframe = true;

        const svo::StereoInitResult keyframe_init_result = initializer.run(curr_frame, camera, false);

        if (keyframe_init_result.num_triangulated >= 20) {
          std::vector<svo::MapPoint> new_landmarks = makeInitialActiveLandmarks(keyframe_init_result);
          map.assignNewLandmarkIds(new_landmarks);

          new_landmarks = transformLandmarksToWorld(new_landmarks, curr_frame.pose_wc);

          const auto new_points = makeInitialActivePoints(keyframe_init_result);
          curr_frame.tracked_points.insert(curr_frame.tracked_points.end(), new_points.begin(), new_points.end());
          for (const auto &lm : new_landmarks) {
            curr_frame.tracked_landmark_ids.push_back(lm.id);
          }

          map.addKeyframe(curr_frame);
          map.addLandmarks(new_landmarks);
          frontend.setActiveTracks(new_points, new_landmarks);
        } else {
          map.addKeyframe(curr_frame);
        }

        frontend.noteKeyframeInserted(frame_id, curr_frame.pose_wc);
        frame_stats.inserted_keyframe = true;

        // -------------------------
        // Local bundle adjustment
        // -------------------------
        if (map.numActiveKeyframes() >= 3 && map.numActiveLandmarks() >= 20 && frontend.shouldRunLocalBA()) {
          std::deque<svo::Frame> keyframe_backup = map.activeKeyframes();
          std::vector<svo::MapPoint> landmarks_backup = map.activeLandmarks();

          svo::LocalBAResult ba_result = estimator.runLocalBundleAdjustment(map.mutableActiveKeyframes(), map.mutableActiveLandmarks(), camera);

          frame_stats.ran_local_ba = true;
          if (ba_result.success && ba_result.rmse_after > 0.0 && ba_result.rmse_after <= ba_result.rmse_before) {
            frame_stats.local_ba_accepted = true;
            frame_stats.ba_rmse_before = ba_result.rmse_before;
            frame_stats.ba_rmse_after = ba_result.rmse_after;

            frontend.refreshActiveLandmarksFromMap(map.activeLandmarks());
            frontend.noteLocalBaAccepted();

            std::cout << "Local BA at frame " << frame_id
                      << "  | keyframes: " << ba_result.num_keyframes
                      << "  | landmarks: " << ba_result.num_landmarks
                      << "  | observations: " << ba_result.num_observations
                      << "  | rmse: " << ba_result.rmse_before
                      << " -> " << ba_result.rmse_after << "\n";
          } else {
            map.mutableActiveKeyframes() = keyframe_backup;
            map.mutableActiveLandmarks() = landmarks_backup;

            frame_stats.local_ba_rejected = true;
            if (ba_result.success) {
              frame_stats.ba_rmse_before = ba_result.rmse_before;
              frame_stats.ba_rmse_after = ba_result.rmse_after;
            }

            std::cout << "Rejected local BA at frame " << frame_id << "\n";
          }
        }

        std::cout << "Inserted keyframe at frame " << frame_id
        << "  | active keyframes: " << map.numActiveKeyframes()
        << "  | active landmarks: " << map.numActiveLandmarks() << "\n";
      }
    }

    // ---------------------------------------------------------------------
    // Logging + debug output
    // ---------------------------------------------------------------------
    const Eigen::Vector3d t_out = frontend.currentPose().block<3, 1>(0, 3);

    stats << frame_id << "," << frontend.activePoints().size() << ","
          << track_result.num_valid_correspondences << ","
          << frame_stats.num_inliers << "," << frame_stats.inlier_ratio << ","
          << (frame_stats.pose_success ? 1 : 0) << ","
          << (frame_stats.pose_accepted ? 1 : 0) << ","
          << (frame_stats.reinitialized ? 1 : 0) << ","
          << (frame_stats.inserted_keyframe ? 1 : 0) << ","
          << map.numActiveKeyframes() << "," << map.numActiveLandmarks() << ","
          << (frame_stats.ran_local_ba ? 1 : 0) << ","
          << (frame_stats.local_ba_accepted ? 1 : 0) << ","
          << (frame_stats.local_ba_rejected ? 1 : 0) << ","
          << frame_stats.ba_rmse_before << "," << frame_stats.ba_rmse_after
          << "," << t_out(0) << "," << t_out(1) << "," << t_out(2) << ","
          << frame_stats.delta_t << "," << frame_stats.rmse_before << ","
          << frame_stats.rmse_after << "\n";

    if (save_debug) {
      const bool save_sparse_debug = (frame_id % 10 == 0);
      const bool save_dense_debug = frontend.shouldSaveDenseDebug(frame_id, 5);
      if (!track_result.track_vis.empty() &&
          (save_sparse_debug || save_dense_debug)) {
        const std::string image_path = "results/debug/" + file_name_stem + "_track_" +
                                       cv::format("%06d", frame_id) + ".png";
        cv::imwrite(image_path, track_result.track_vis);
      }
    }

    std::cout << "frame " << frame_id
              << " | active: " << frontend.activePoints().size()
              << " | corr: " << track_result.num_valid_correspondences
              << " | inliers: " << frame_stats.num_inliers
              << " | ratio: " << frame_stats.inlier_ratio
              << " | delta_t: " << frame_stats.delta_t
              << " | pose_success: " << frame_stats.pose_success
              << " | pose_accepted: " << frame_stats.pose_accepted
              << " | reinit: " << frame_stats.reinitialized
              << " | keyframe: " << frame_stats.inserted_keyframe << "\n";

    if (frontend.activePoints().size() < 20) {
      std::cout << "Tracking dropped below threshold at frame " << frame_id
                << ". Stopping early.\n";
      break;
    }

    frontend.setPreviousFrame(curr_frame);

    svo::ViewerStatus viewer_status;
    viewer_status.frame_id = frame_id;
    viewer_status.num_active_points =
        static_cast<int>(frontend.activePoints().size());
    viewer_status.num_correspondences = track_result.num_valid_correspondences;
    viewer_status.num_inliers = frame_stats.num_inliers;
    viewer_status.pose_accepted = frame_stats.pose_accepted;
    viewer_status.reinitialized = frame_stats.reinitialized;
    viewer_status.inserted_keyframe = frame_stats.inserted_keyframe;
    viewer_status.ran_local_ba = frame_stats.ran_local_ba;
    viewer_status.delta_t = frame_stats.delta_t;
    viewer_status.rmse_before = frame_stats.rmse_before;
    viewer_status.rmse_after = frame_stats.rmse_after;

    if (!no_viewer)
      rerun_viewer.update(frame_id, curr_frame.left_img, frontend.activePoints(),
                          frontend.poses(), gt_poses, viewer_status);

    // Fallback: OpenCV 2D viewer (uncomment to use instead of Rerun)
    // if (!viewer.update(curr_frame.left_img, frontend.activePoints(), frontend.poses(), gt_poses,
    //                    viewer_status)) {
    //   std::cout << "Viewer requested exit.\n";
    //   break;
    // }

    ++frames_processed;
  }

  // -------------------------------------------------------------------------
  // Final trajectory write
  // -------------------------------------------------------------------------
  const auto loop_end = std::chrono::steady_clock::now();
  const double total_s =
      std::chrono::duration<double>(loop_end - loop_start).count();
  const double avg_ms =
      frames_processed > 0 ? (total_s / frames_processed) * 1000.0 : 0.0;
  std::cout << "Total time: " << total_s << " s"
            << "  |  frames: " << frames_processed
            << "  |  avg: " << avg_ms << " ms/frame\n";

  {
    const fs::path time_log_path = "results/time_log.csv";
    const bool write_header = !fs::exists(time_log_path);
    std::ofstream time_log(time_log_path, std::ios::app);
    if (write_header) {
      time_log << "date,sequence,keyword,frames,total_s,avg_ms_per_frame\n";
    }
    time_log << date_str << "," << sequence << "," << pose_keyword << ","
             << frames_processed << "," << total_s << "," << avg_ms << "\n";
  }

  svo::PoseWriter::writeKittiTrajectory(output_pose, frontend.poses());
  std::cout << "Wrote VO trajectory to: " << output_pose << "\n";

  return 0;
}
