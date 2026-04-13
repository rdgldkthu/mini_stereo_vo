#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

#include "svo/camera.h"
#include "svo/dataset_kitti.h"
#include "svo/estimator.h"
#include "svo/frame.h"
#include "svo/map_point.h"
#include "svo/stereo_initializer.h"
#include "svo/tracker.h"

namespace fs = std::filesystem;

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
makeInitalActiveLandmarks(const svo::StereoInitResult &init_result) {
  std::vector<svo::MapPoint> landmarks;

  const size_t n =
      std::min(init_result.features.size(), init_result.landmarks.size());
  landmarks.reserve(n);

  for (size_t i = 0; i < n; ++i) {
    landmarks.push_back(init_result.landmarks[i]);
  }

  return landmarks;
}

Eigen::Matrix4d makePoseWcFromPnP(const Eigen::Matrix3d &R_cw,
                                  const Eigen::Vector3d &t_cw) {
  Eigen::Matrix4d T_wc = Eigen::Matrix4d::Identity();

  const Eigen::Matrix3d R_wc = R_cw.transpose();
  const Eigen::Vector3d t_wc = -R_cw * t_cw;

  T_wc.block<3, 3>(0, 0) = R_wc;
  T_wc.block<3, 1>(0, 3) = t_wc;

  return T_wc;
}

void writeKittiTrajectory(const std::filesystem::path &out_path,
                          const std::vector<Eigen::Matrix4d> &poses) {
  namespace fs = std::filesystem;

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

  svo::Frame frame0;
  if (!dataset.loadFrame(0, frame0)) {
    std::cerr << "Failed to load frame 0.\n";
    return 1;
  }

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
  svo::StereoInitResult init_result = initializer.run(frame0, camera);

  std::cout << "Stereo initialization result:\n";
  std::cout << "  triangulated: " << init_result.num_triangulated << "\n";

  if (init_result.num_triangulated < 20) {
    std::cerr << "Too few initial landmarks.\n";
    return 1;
  }

  if (!init_result.match_vis.empty()) {
    cv::imwrite("result/debug/" + sequence + "_init_matches.png",
                init_result.match_vis);
  }

  std::vector<cv::Point2f> active_points_2d =
      makeInitialActivePoints(init_result);
  std::vector<svo::MapPoint> active_landmarks =
      makeInitalActiveLandmarks(init_result);

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
  svo::Estimator estimator(estimator_options);

  std::vector<Eigen::Matrix4d> poses;
  poses.reserve(dataset.numFrames());
  poses.push_back(Eigen::Matrix4d::Identity());

  std::ofstream stats("results/debug/" + sequence + "_vo_stats.csv");
  stats << "frame_id,num_active_points,num_correspondences,num_inliers,pose_"
           "success\n";
  stats << "0," << active_points_2d.size() << ",0,0,1\n";

  svo::Frame prev_frame = frame0;
  int last_init_frame_id = 0;

  for (int frame_id = 1; frame_id < dataset.numFrames(); ++frame_id) {
    svo::Frame curr_frame;
    if (!dataset.loadFrame(frame_id, curr_frame)) {
      std::cerr << "Failed to load frame " << frame_id << "\n";
      poses.push_back(poses.back());
      stats << frame_id << ",0,0,0,0\n";
      continue;
    }

    const svo::TrackResult track_result = tracker.trackFrameToFrame(
        prev_frame, curr_frame, active_points_2d, active_landmarks);

    bool pose_success = false;
    int num_inliers = 0;

    if (track_result.num_valid_correspondences >=
        estimator_options.min_pnp_points) {
      const svo::PoseEstimateResult pose_result =
          estimator.estimatePosePnPRansac(track_result.object_points,
                                          track_result.image_points, camera);

      if (pose_result.success && pose_result.num_inliers >= 10) {
        poses.push_back(
            makePoseWcFromPnP(pose_result.rotation, pose_result.translation));
        pose_success = true;
        num_inliers = pose_result.num_inliers;
      } else {
        poses.push_back(poses.back());
      }
    } else {
      poses.push_back(poses.back());
    }

    if ((frame_id - last_init_frame_id > 10) && (num_inliers < 12 || pose_success == false || track_result.curr_points.size() < 80)) {
      std::cout << "Reinitializing at frame " << frame_id << "\n";
      svo::StereoInitResult reinit_result = initializer.run(curr_frame, camera);
      active_points_2d = makeInitialActivePoints(reinit_result);
      active_landmarks = makeInitalActiveLandmarks(reinit_result);
      last_init_frame_id = frame_id;
    } else {
      active_points_2d = track_result.curr_points;
      active_landmarks = track_result.tracked_landmarks;
    }

    stats << frame_id << "," << active_points_2d.size() << ","
          << track_result.num_valid_correspondences << "," << num_inliers << ","
          << (pose_success ? 1 : 0) << "\n";

    if (!track_result.track_vis.empty() && (frame_id % 10 == 0)) {
      const std::string image_path = "results/debug/" + sequence + "_track_" +
                                     cv::format("%06d", frame_id) + ".png";
      cv::imwrite(image_path, track_result.track_vis);
    }

    std::cout << "frame " << frame_id
              << " | active: " << active_points_2d.size()
              << " | corr: " << track_result.num_valid_correspondences
              << " | inliers: " << num_inliers
              << " | pose_success: " << pose_success << "\n";

    if (active_points_2d.size() < 20) {
      std::cout << "Tracking dropped below threshold at frame " << frame_id
                << ". Stopping early.\n";
      break;
    }

    prev_frame = curr_frame;
  }

  writeKittiTrajectory(output_pose, poses);
  std::cout << "Wrote VO trajectory to: " << output_pose << "\n";

  return 0;
}