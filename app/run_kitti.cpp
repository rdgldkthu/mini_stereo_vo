#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

#include <opencv2/opencv.hpp>

#include "svo/camera.h"
#include "svo/dataset_kitti.h"
#include "svo/frame.h"
#include "svo/stereo_initializer.h"

namespace fs = std::filesystem;

void write_identity_kitti(const fs::path &out_path, std::size_t num_frames) {
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
  if (!dataset.loadFrame(0, frame0)) {
    std::cerr << "Failed to load frame 0.\n";
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
  svo::StereoInitResult result = initializer.run(frame0, camera);

  std::cout << "Stereo initialization result:\n";
  std::cout << "  left keypoints: " << result.num_left_keypoints << "\n";
  std::cout << "  right keypoints: " << result.num_right_keypoints << "\n";
  std::cout << "  raw matches: " << result.num_raw_matches << "\n";
  std::cout << "  distance filtered: " << result.num_distance_filtered << "\n";
  std::cout << "  row filtered: " << result.num_row_filtered << "\n";
  std::cout << "  disparity filtered: " << result.num_disparity_filtered
            << "\n";
  std::cout << "  triangulated: " << result.num_triangulated << "\n";
  std::cout << "  disparity min/mean/max: " << result.min_disparity << " / "
            << result.mean_disparity << " / " << result.max_disparity << "\n";
  std::cout << "  row error mean/max: " << result.mean_row_error << " / "
            << result.max_row_error << "\n";
  std::cout << "  depth min/mean/max: " << result.min_depth << " / "
            << result.mean_depth << " / " << result.max_depth << "\n";
  std::cout << "  depth > 50m: " << result.num_depth_gt_50 << "\n";
  std::cout << "  depth > 80m: " << result.num_depth_gt_80 << "\n";

  fs::create_directories("results/debug");
  const std::string prefix = "results/debug/" + sequence + "_init";

  if (!result.match_vis.empty()) {
    cv::imwrite(prefix + "_matches.png", result.match_vis);
    std::cout << "Saved match visualization to " << prefix << "_matches.png\n";
  }

  {
    std::ofstream ofs(prefix + "_stats.txt");
    ofs << "num_left_keypoints " << result.num_left_keypoints << "\n";
    ofs << "num_right_keypoints " << result.num_right_keypoints << "\n";
    ofs << "num_raw_matches " << result.num_raw_matches << "\n";
    ofs << "num_distance_filtered " << result.num_distance_filtered << "\n";
    ofs << "num_row_filtered " << result.num_row_filtered << "\n";
    ofs << "num_disparity_filtered " << result.num_disparity_filtered << "\n";
    ofs << "num_triangulated " << result.num_triangulated << "\n";
    ofs << "num_depth_gt_50 " << result.num_depth_gt_50 << "\n";
    ofs << "num_depth_gt_80 " << result.num_depth_gt_80 << "\n";
    ofs << "min_disparity " << result.min_disparity << "\n";
    ofs << "mean_disparity " << result.mean_disparity << "\n";
    ofs << "max_disparity " << result.max_disparity << "\n";
    ofs << "mean_row_error " << result.mean_row_error << "\n";
    ofs << "max_row_error " << result.max_row_error << "\n";
    ofs << "min_depth " << result.min_depth << "\n";
    ofs << "mean_depth " << result.mean_depth << "\n";
    ofs << "max_depth " << result.max_depth << "\n";
  }
  std::cout << "Saved stats to " << prefix << "_stats.txt\n";

  {
    std::ofstream ofs(prefix + "_points.txt");
    ofs << "# id x y z ul vl ur vr disparity\n";

    const size_t n = std::min(result.features.size(), result.landmarks.size());
    for (size_t i = 0; i < n; ++i) {
      const auto &f = result.features[i];
      const auto &p = result.landmarks[i];

      ofs << p.id << " " << p.p_cam.x() << " " << p.p_cam.y() << " "
          << p.p_cam.z() << " " << f.kp_left.pt.x << " " << f.kp_left.pt.y
          << " " << f.kp_right.pt.x << " " << f.kp_right.pt.y << " "
          << f.disparity << "\n";
    }
  }
  std::cout << "Saved points to " << prefix << "_points.txt\n";

  write_identity_kitti(output_pose, dataset.numFrames());
  std::cout << "Wrote dummy KITTI trajectory to: " << output_pose << "\n";

  if (!result.match_vis.empty()) {
    cv::imshow("stereo initializer", result.match_vis);
    cv::waitKey(0);
  }

  return 0;
}