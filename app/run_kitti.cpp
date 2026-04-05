#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "svo/camera.h"

namespace fs = std::filesystem;

std::vector<fs::path> sorted_pngs(const fs::path &dir) {
  std::vector<fs::path> files;
  if (!fs::exists(dir) || !fs::is_directory(dir)) {
    return files;
  }

  for (const auto &entry : fs::directory_iterator(dir)) {
    if (entry.is_regular_file() && entry.path().extension() == ".png") {
      files.push_back(entry.path());
    }
  }

  std::sort(files.begin(), files.end());
  return files;
}

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

  const fs::path seq_dir = kitti_root / "sequences" / sequence;
  const fs::path calib_path = seq_dir / "calib.txt";
  const fs::path left_dir = seq_dir / "image_0";
  const fs::path right_dir = seq_dir / "image_1";

  const auto left_files = sorted_pngs(left_dir);
  const auto right_files = sorted_pngs(right_dir);

  if (left_files.empty()) {
    std::cerr << "No left images found in: " << left_dir << "\n";
    return 1;
  }
  if (right_files.empty()) {
    std::cerr << "No right images found in: " << right_dir << "\n";
    return 1;
  }
  if (left_files.size() != right_files.size()) {
    std::cerr << "Left/right image count mismatch: " << left_files.size()
              << " vs " << right_files.size() << "\n";
    return 1;
  }

  // Load calibration
  svo::Camera camera;
  if (!camera.loadFromKittiCalib(calib_path.string())) {
    std::cerr << "Failed to load KITTI calibration from: " << calib_path
              << "\n";
    return 1;
  }

  std::cout << "Loaded calibration from: " << calib_path << "\n";
  camera.print();

  // Read first stereo pair for sanity checks
  cv::Mat first_left = cv::imread(left_files[0].string(), cv::IMREAD_GRAYSCALE);
  cv::Mat first_right =
      cv::imread(right_files[0].string(), cv::IMREAD_GRAYSCALE);

  if (first_left.empty() || first_right.empty()) {
    std::cerr << "Failed to read first stereo frame.\n";
    return 1;
  }

  std::cout << "First frame size: " << first_left.cols << " x "
            << first_left.rows << "\n";

  // Simple triangulation sanity test with a fake disparity
  {
    const double ul = 200.0;
    const double vl = 150.0;
    const double ur = 190.0; // disparity = 10 px

    Eigen::Vector3d p_c;
    if (camera.triangulateRectified(ul, vl, ur, p_c)) {
      std::cout << "Sample triangulated point from synthetic disparity:\n";
      std::cout << "  ul=" << ul << ", vl=" << vl << ", ur=" << ur << "\n";
      std::cout << "  p_c = [" << p_c.transpose() << "]\n";
    } else {
      std::cout << "Synthetic triangulation test failed.\n";
    }
  }

  write_identity_kitti(output_pose, left_files.size());

  std::cout << "Loaded " << left_files.size() << " stereo pairs.\n";
  std::cout << "Wrote dummy KITTI trajectory to: " << output_pose << "\n";
  std::cout << "Press q or ESC to quit.\n";

  for (std::size_t i = 0; i < left_files.size(); ++i) {
    cv::Mat left = cv::imread(left_files[i].string(), cv::IMREAD_GRAYSCALE);
    cv::Mat right = cv::imread(right_files[i].string(), cv::IMREAD_GRAYSCALE);

    if (left.empty() || right.empty()) {
      std::cerr << "Failed to read frame " << i << "\n";
      return 1;
    }

    const int canvas_h = std::max(left.rows, right.rows);
    const int canvas_w = left.cols + right.cols;
    cv::Mat canvas(canvas_h, canvas_w, CV_8UC1, cv::Scalar(0));

    left.copyTo(canvas(cv::Rect(0, 0, left.cols, left.rows)));
    right.copyTo(canvas(cv::Rect(left.cols, 0, right.cols, right.rows)));

    cv::putText(canvas, "KITTI seq " + sequence + " frame " + std::to_string(i),
                cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                cv::Scalar(255), 2);

    cv::imshow("stereo_vo day1 check", canvas);
    const int key = cv::waitKey(10);
    if (key == 'q' || key == 27) {
      break;
    }
  }

  return 0;
}
