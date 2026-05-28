#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include "svo/camera.h"
#include "svo/dataset_kitti.h"
#include "svo/estimator.h"
#include "svo/frame.h"
#include "svo/frontend.h"
#include "svo/map.h"
#include "svo/pose_writer.h"
#include "svo/rerun_viewer.h"
#include "svo/stereo_initializer.h"
#include "svo/tracker.h"
#include "svo/viewer_status.h"

namespace fs = std::filesystem;

namespace {

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
    for (int r = 0; r < 3; ++r)
      for (int c = 0; c < 4; ++c)
        ss >> T(r, c);
    if (ss) poses.push_back(T);
  }
  std::cout << "Loaded " << poses.size() << " ground-truth poses from "
            << pose_path << "\n";
  return poses;
}

} // namespace

int main(int argc, char **argv) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0]
              << " <kitti_root> <sequence> [pose_keyword]"
                 " [--save-debug] [--no-viewer]"
                 " [--max-frames N] [--output-file PATH]\n";
    return 1;
  }

  const fs::path   kitti_root = argv[1];
  const std::string sequence  = argv[2];

  bool save_debug = false;
  bool no_viewer  = false;
  int  max_frames = -1;
  std::string pose_keyword;
  std::string output_file_override;
  for (int i = 3; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--save-debug")  save_debug = true;
    else if (arg == "--no-viewer") no_viewer = true;
    else if (arg == "--max-frames" && i + 1 < argc)
      max_frames = std::stoi(argv[++i]);
    else if (arg == "--output-file" && i + 1 < argc)
      output_file_override = argv[++i];
    else pose_keyword = arg;
  }

  std::time_t now = std::time(nullptr);
  char date_buf[7];
  std::strftime(date_buf, sizeof(date_buf), "%y%m%d", std::localtime(&now));
  const std::string date_str(date_buf);

  const std::string file_name_stem = pose_keyword.empty()
      ? sequence + "_" + date_str
      : sequence + "_" + date_str + "_" + pose_keyword;

  fs::create_directories("results/debug");
  fs::create_directories("results/traj");

  const fs::path output_pose = output_file_override.empty()
      ? fs::path("results/traj") / (file_name_stem + ".txt")
      : fs::path(output_file_override);

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

  const auto gt_poses =
      loadKittiPoses(kitti_root / "poses" / (sequence + ".txt"));

  cv::setRNGSeed(42);

  // -------------------------------------------------------------------------
  // Modules — each independently accessible for future SLAM backend use
  // -------------------------------------------------------------------------
  svo::StereoInitializer::Options init_options;
  init_options.max_features          = 1000;
  init_options.hamming_threshold     = 40;
  init_options.row_tolerance_px      = 2.0;
  init_options.min_disparity_px      = 3.0;
  init_options.max_disparity_px      = 120.0;
  init_options.max_depth_m           = 80.0;
  init_options.image_border_px       = 10;
  init_options.max_visualized_matches = 100;
  init_options.grid_rows             = 4;
  init_options.grid_cols             = 8;
  init_options.max_per_cell          = 10;
  svo::StereoInitializer initializer(init_options);

  svo::Tracker::Options tracker_options;
  tracker_options.win_size                   = cv::Size(25, 25);
  tracker_options.max_level                  = 4;
  tracker_options.max_bidirectional_error_px = 1.5;
  tracker_options.image_border_px            = 10;
  tracker_options.max_visualized_tracks      = 150;
  svo::Tracker tracker(tracker_options);

  svo::Estimator::Options estimator_options;
  estimator_options.use_extrinsic_guess   = false;
  estimator_options.iterations_count      = 100;
  estimator_options.reprojection_error_px = 3.0f;
  estimator_options.confidence            = 0.99;
  estimator_options.min_pnp_points        = 6;
  estimator_options.pose_refine_iterations = 10;
  estimator_options.pose_refine_epsilon   = 1e-6;
  estimator_options.pose_refine_huber_delta = 5.0;
  estimator_options.min_refine_inliers    = 10;
  svo::Estimator estimator(estimator_options);

  svo::Frontend::Options frontend_options;
  frontend_options.keyframe_translation_threshold_m         = 1.5;
  frontend_options.keyframe_rotation_threshold_deg          = 8.0;
  frontend_options.keyframe_min_tracked_points              = 60;
  frontend_options.keyframe_min_frame_gap                   = 5;
  frontend_options.keyframe_low_track_translation_threshold_m = 0.5;
  frontend_options.min_pose_inliers                         = 15;
  frontend_options.min_pose_inlier_ratio                    = 0.10;
  frontend_options.max_frame_translation_m                  = 2.0;
  frontend_options.min_reinit_frame_gap                     = 10;
  frontend_options.weak_track_threshold                     = 80;
  frontend_options.emergency_rejected_poses_count           = 2;
  svo::Frontend frontend(frontend_options);

  svo::Map::Options map_options;
  map_options.max_active_keyframes = 5;
  map_options.max_active_landmarks = 2000;
  map_options.min_observed_times   = 2;
  map_options.max_missed_times     = 8;
  svo::Map map(map_options);

  std::optional<svo::RerunViewer> rerun_viewer;
  if (!no_viewer) {
    svo::RerunViewer::Options rerun_options;
    rerun_options.app_id = "stereo_slam";
    rerun_viewer.emplace(rerun_options);
  }

  // -------------------------------------------------------------------------
  // Bootstrap from frame 0
  // -------------------------------------------------------------------------
  svo::Frame frame0;
  if (!dataset.loadFrame(0, frame0)) {
    std::cerr << "Failed to load frame 0.\n";
    return 1;
  }

  cv::Mat init_vis;
  if (!frontend.bootstrap(frame0, initializer, map, camera, save_debug, &init_vis)) {
    std::cerr << "Bootstrap failed: too few initial landmarks.\n";
    return 1;
  }

  if (save_debug && !init_vis.empty())
    cv::imwrite("results/debug/" + file_name_stem + "_init_matches.png", init_vis);

  // -------------------------------------------------------------------------
  // Logging
  // -------------------------------------------------------------------------
  std::ofstream stats("results/debug/" + file_name_stem + "_stats.csv");
  stats << "frame_id,num_active_points,num_correspondences,num_inliers,"
           "inlier_ratio,pose_success,pose_accepted,reinitialized,is_keyframe,"
           "num_keyframes,num_map_landmarks,"
           "tx,ty,tz,delta_t,rmse_before,rmse_after\n";
  stats << "0," << frontend.activePoints().size() << ",0,0,0.0,1,1,0,1,"
        << map.numActiveKeyframes() << "," << map.numActiveLandmarks()
        << ",0,0,0,0,0,0\n";

  // -------------------------------------------------------------------------
  // Main VO loop
  // -------------------------------------------------------------------------
  const auto loop_start     = std::chrono::steady_clock::now();
  int        frames_processed = 0;

  for (int frame_id = 1; frame_id < dataset.numFrames(); ++frame_id) {
    if (max_frames > 0 && frame_id >= max_frames) break;
    svo::Frame curr_frame;
    if (!dataset.loadFrame(frame_id, curr_frame)) {
      std::cerr << "Failed to load frame " << frame_id << "\n";
      frontend.repeatLastPose();

      const Eigen::Vector3d t = frontend.currentPose().block<3, 1>(0, 3);
      stats << frame_id << ",0,0,0,0.0,0,0,0,0,"
            << map.numActiveKeyframes() << "," << map.numActiveLandmarks()
            << "," << t(0) << "," << t(1) << "," << t(2) << ",0,0,0\n";
      continue;
    }

    const svo::ProcessFrameResult r =
        frontend.processFrame(frame_id, curr_frame,
                              tracker, estimator, initializer,
                              map, camera, save_debug);

    const Eigen::Vector3d t = frontend.currentPose().block<3, 1>(0, 3);
    stats << frame_id << "," << frontend.activePoints().size() << ","
          << r.stats.num_correspondences << ","
          << r.stats.num_inliers << "," << r.stats.inlier_ratio << ","
          << (r.stats.pose_success     ? 1 : 0) << ","
          << (r.stats.pose_accepted    ? 1 : 0) << ","
          << (r.stats.reinitialized    ? 1 : 0) << ","
          << (r.stats.inserted_keyframe ? 1 : 0) << ","
          << map.numActiveKeyframes() << "," << map.numActiveLandmarks() << ","
          << t(0) << "," << t(1) << "," << t(2) << ","
          << r.stats.delta_t << "," << r.stats.rmse_before << ","
          << r.stats.rmse_after << "\n";

    std::cout << "frame " << frame_id
              << " | active: "       << frontend.activePoints().size()
              << " | corr: "         << r.stats.num_correspondences
              << " | inliers: "      << r.stats.num_inliers
              << " | ratio: "        << r.stats.inlier_ratio
              << " | delta_t: "      << r.stats.delta_t
              << " | pose_success: " << r.stats.pose_success
              << " | pose_accepted: "<< r.stats.pose_accepted
              << " | reinit: "       << r.stats.reinitialized
              << " | keyframe: "     << r.stats.inserted_keyframe << "\n";

    if (r.should_exit) {
      std::cout << "Tracking dropped below threshold at frame " << frame_id
                << ". Stopping early.\n";
      break;
    }

    if (!r.track_vis.empty()) {
      const bool save_sparse = (frame_id % 10 == 0);
      const bool save_dense  = frontend.shouldSaveDenseDebug(frame_id, 5);
      if (save_sparse || save_dense) {
        cv::imwrite("results/debug/" + file_name_stem + "_track_" +
                        cv::format("%06d", frame_id) + ".png",
                    r.track_vis);
      }
    }

    svo::ViewerStatus vs;
    vs.frame_id           = frame_id;
    vs.num_active_points  = static_cast<int>(frontend.activePoints().size());
    vs.num_correspondences = r.stats.num_correspondences;
    vs.num_inliers        = r.stats.num_inliers;
    vs.pose_accepted      = r.stats.pose_accepted;
    vs.reinitialized      = r.stats.reinitialized;
    vs.inserted_keyframe  = r.stats.inserted_keyframe;
    vs.delta_t            = r.stats.delta_t;
    vs.rmse_before        = r.stats.rmse_before;
    vs.rmse_after         = r.stats.rmse_after;

    if (rerun_viewer)
      rerun_viewer->update(frame_id, curr_frame.left_img,
                           frontend.activePoints(), frontend.poses(),
                           gt_poses, vs);

    ++frames_processed;
  }

  // -------------------------------------------------------------------------
  // Final trajectory + timing
  // -------------------------------------------------------------------------
  const auto   loop_end = std::chrono::steady_clock::now();
  const double total_s  =
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
    if (write_header)
      time_log << "date,sequence,keyword,frames,total_s,avg_ms_per_frame\n";
    time_log << date_str << "," << sequence << "," << pose_keyword << ","
             << frames_processed << "," << total_s << "," << avg_ms << "\n";
  }

  svo::PoseWriter::writeKittiTrajectory(output_pose, frontend.poses());
  std::cout << "Wrote VO trajectory to: " << output_pose << "\n";

  return 0;
}
