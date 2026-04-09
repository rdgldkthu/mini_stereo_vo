#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/opencv.hpp>

#include "svo/camera.h"
#include "svo/dataset_kitti.h"
#include "svo/estimator.h"
#include "svo/frame.h"
#include "svo/stereo_initializer.h"
#include "svo/tracker.h"

namespace fs = std::filesystem;

void writeIdentityKitti(const fs::path &out_path, std::size_t num_frames) {
  fs::create_directories(out_path.parent_path());

  std::ofstream ofs(out_path);
  if (!ofs) {
    throw std::runtime_error("Failed to open output file: " +
                             out_path.string());
  }

  ofs << std::fixed << std::setprecision(9);
  for (std::size_t i = 0; i < num_frames; ++i) {
    ofs << "1 0 0 0 0 1 0 0 0 0 1 0\n";
  }
}

void writeTwoFrameKitti(const fs::path &out_path, const Eigen::Matrix3d &R1,
                        const Eigen::Vector3d &t1) {
  fs::create_directories(out_path.parent_path());

  std::ofstream ofs(out_path);
  if (!ofs) {
    throw std::runtime_error("Failed to open output file: " +
                             out_path.string());
  }

  ofs << std::fixed << std::setprecision(9);

  // frame 0: identity
  ofs << "1 0 0 0 0 1 0 0 0 0 1 0\n";

  // frame 1
  ofs << R1(0, 0) << " " << R1(0, 1) << " " << R1(0, 2) << " " << t1(0) << " "
      << R1(1, 0) << " " << R1(1, 1) << " " << R1(1, 2) << " " << t1(1) << " "
      << R1(2, 0) << " " << R1(2, 1) << " " << R1(2, 2) << " " << t1(2) << "\n";
}

int main(int argc, char **argv) {
  if (argc < 3 || argc > 4) {
    std::cerr << "Usage: " << argv[0]
              << " <kitti_root> <sequence> [output_pose_file]\n";
    std::cerr << "Example: " << argv[0]
              << " data/kitti 05 results/traj/05.txt\n";
    return 1;
  }

  const fs::path kitti_root = argv[1];
  const std::string sequence = argv[2];
  const fs::path output_pose =
      (argc == 4) ? fs::path(argv[3])
                  : fs::path("results/traj") / (sequence + ".txt");

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
  svo::Frame frame1;

  if (!dataset.loadFrame(0, frame0)) {
    std::cerr << "Failed to load frame 0.\n";
    return 1;
  }
  if (!dataset.loadFrame(1, frame1)) {
    std::cerr << "Failed to load frame 1.\n";
    return 1;
  }

  std::cout << "Frame 0 size: " << frame0.left_img.cols << " x "
            << frame0.left_img.rows << "\n";

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
  std::cout << "  left keypoints: " << init_result.num_left_keypoints << "\n";
  std::cout << "  right keypoints: " << init_result.num_right_keypoints << "\n";
  std::cout << "  raw matches: " << init_result.num_raw_matches << "\n";
  std::cout << "  distance filtered: " << init_result.num_distance_filtered
            << "\n";
  std::cout << "  row filtered: " << init_result.num_row_filtered << "\n";
  std::cout << "  disparity filtered: " << init_result.num_disparity_filtered
            << "\n";
  std::cout << "  triangulated: " << init_result.num_triangulated << "\n";

  fs::create_directories("results/debug");
  const std::string prefix = "results/debug/" + sequence + "_init";

  if (!init_result.match_vis.empty()) {
    cv::imwrite(prefix + "_matches.png", init_result.match_vis);
  }

  {
    std::ofstream ofs(prefix + "_stats.txt");
    ofs << "num_left_keypoints " << init_result.num_left_keypoints << "\n";
    ofs << "num_right_keypoints " << init_result.num_right_keypoints << "\n";
    ofs << "num_raw_matches " << init_result.num_raw_matches << "\n";
    ofs << "num_distance_filtered " << init_result.num_distance_filtered
        << "\n";
    ofs << "num_row_filtered " << init_result.num_row_filtered << "\n";
    ofs << "num_disparity_filtered " << init_result.num_disparity_filtered
        << "\n";
    ofs << "num_triangulated " << init_result.num_triangulated << "\n";
    ofs << "num_depth_gt_50 " << init_result.num_depth_gt_50 << "\n";
    ofs << "num_depth_gt_80 " << init_result.num_depth_gt_80 << "\n";
    ofs << "min_disparity " << init_result.min_disparity << "\n";
    ofs << "mean_disparity " << init_result.mean_disparity << "\n";
    ofs << "max_disparity " << init_result.max_disparity << "\n";
    ofs << "mean_row_error " << init_result.mean_row_error << "\n";
    ofs << "max_row_error " << init_result.max_row_error << "\n";
    ofs << "min_depth " << init_result.min_depth << "\n";
    ofs << "mean_depth " << init_result.mean_depth << "\n";
    ofs << "max_depth " << init_result.max_depth << "\n";
  }

  {
    std::ofstream ofs(prefix + "_points.txt");
    ofs << "# id x y z ul vl ur vr disparity\n";

    const size_t n =
        std::min(init_result.features.size(), init_result.landmarks.size());
    for (size_t i = 0; i < n; ++i) {
      const auto &f = init_result.features[i];
      const auto &p = init_result.landmarks[i];

      ofs << p.id << " " << p.p_cam.x() << " " << p.p_cam.y() << " "
          << p.p_cam.z() << " " << f.kp_left.pt.x << " " << f.kp_left.pt.y
          << " " << f.kp_right.pt.x << " " << f.kp_right.pt.y << " "
          << f.disparity << "\n";
    }
  }

  svo::Tracker::Options tracker_options;
  tracker_options.win_size = cv::Size(21, 21);
  tracker_options.max_level = 3;
  tracker_options.max_bidirectional_error_px = 1.5;
  tracker_options.image_border_px = 10;
  tracker_options.max_visualized_tracks = 150;

  svo::Tracker tracker(tracker_options);
  const svo::TrackResult track_result = tracker.trackFrameToFrame(
      frame0, frame1, init_result.features, init_result.landmarks);

  std::cout << "Tracking result:\n";
  std::cout << "  input tracks: " << track_result.num_input_tracks << "\n";
  std::cout << "  flow success: " << track_result.num_flow_success << "\n";
  std::cout << "  inside image: " << track_result.num_inside_image << "\n";
  std::cout << "  valid correspondences: "
            << track_result.num_valid_correspondences << "\n";

  const std::string track_prefix =
      "results/debug/" + sequence + "_track_000001";

  if (!track_result.track_vis.empty()) {
    cv::imwrite(track_prefix + ".png", track_result.track_vis);
  }

  {
    std::ofstream ofs("results/debug/" + sequence + "_track_stats.txt");
    ofs << "num_input_tracks " << track_result.num_input_tracks << "\n";
    ofs << "num_flow_success " << track_result.num_flow_success << "\n";
    ofs << "num_inside_image " << track_result.num_inside_image << "\n";
    ofs << "num_valid_correspondences "
        << track_result.num_valid_correspondences << "\n";
  }

  svo::Estimator::Options estimator_options;
  estimator_options.use_extrinsic_guess = false;
  estimator_options.iterations_count = 100;
  estimator_options.reprojection_error_px = 4.0f;
  estimator_options.confidence = 0.99;
  estimator_options.min_pnp_points = 6;

  svo::Estimator estimator(estimator_options);
  const svo::PoseEstimateResult pose_result = estimator.estimatePosePnPRansac(
      track_result.object_points, track_result.image_points, camera);

  std::cout << "PnP result:\n";
  std::cout << "  success: " << pose_result.success << "\n";
  std::cout << "  object points: " << pose_result.num_object_points << "\n";
  std::cout << "  image points: " << pose_result.num_image_points << "\n";
  std::cout << "  inliers: " << pose_result.num_inliers << "\n";

  {
    std::ofstream ofs("results/debug/" + sequence + "_pnp_stats.txt");
    ofs << "success " << pose_result.success << "\n";
    ofs << "num_object_points " << pose_result.num_object_points << "\n";
    ofs << "num_image_points " << pose_result.num_image_points << "\n";
    ofs << "num_inliers " << pose_result.num_inliers << "\n";

    if (pose_result.success) {
      ofs << "rotation\n" << pose_result.rotation << "\n";
      ofs << "translation\n" << pose_result.translation.transpose() << "\n";
    }
  }

  // keep full-length dummy export for format compatibility
  writeIdentityKitti(output_pose, dataset.numFrames());

  // write separate 2-frame pose estimate
  if (pose_result.success) {
    writeTwoFrameKitti("results/traj/" + sequence + "_two_frame.txt",
                       pose_result.rotation, pose_result.translation);
  }

  std::cout << "Wrote dummy KITTI trajectory to: " << output_pose << "\n";
  if (pose_result.success) {
    std::cout << "Wrote two-frame trajectory to: results/traj/" << sequence
              << "_two_frame.txt\n";
    std::cout << "Estimated rotation:\n" << pose_result.rotation << "\n";
    std::cout << "Estimated translation:\n"
              << pose_result.translation.transpose() << "\n";
  }

  if (!track_result.track_vis.empty()) {
    cv::imshow("tracking", track_result.track_vis);
    cv::waitKey(0);
  }

  return 0;
}