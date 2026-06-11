#include "svo/dataset_kitti.h"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>

#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

namespace svo {

std::vector<fs::path> DatasetKitti::sortedPngs(const fs::path &dir) {
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

bool DatasetKitti::open(const std::string &kitti_root,
                        const std::string &sequence) {
  sequence_ = sequence;
  kitti_root_ = fs::path(kitti_root);
  seq_dir_ = kitti_root_ / "sequences" / sequence;
  calib_path_ = seq_dir_ / "calib.txt";

  const fs::path left_dir = seq_dir_ / "image_0";
  const fs::path right_dir = seq_dir_ / "image_1";

  left_images_ = sortedPngs(left_dir);
  right_images_ = sortedPngs(right_dir);

  if (left_images_.empty()) {
    std::cerr << "No left images found in: " << left_dir << "\n";
    return false;
  }
  if (right_images_.empty()) {
    std::cerr << "No right images found in: " << right_dir << "\n";
    return false;
  }
  if (left_images_.size() != right_images_.size()) {
    std::cerr << "Left/right image count mismatch: " << left_images_.size()
              << " vs " << right_images_.size() << "\n";
    return false;
  }
  if (!fs::exists(calib_path_)) {
    std::cerr << "Missing calib file: " << calib_path_ << "\n";
    return false;
  }

  return true;
}

bool DatasetKitti::loadFrame(int frame_id, Frame &frame) const {
  if (frame_id < 0 || frame_id >= numFrames()) {
    return false;
  }

  frame.id = frame_id;
  frame.left_img =
      cv::imread(left_images_[frame_id].string(), cv::IMREAD_GRAYSCALE);
  frame.right_img =
      cv::imread(right_images_[frame_id].string(), cv::IMREAD_GRAYSCALE);

  if (frame.left_img.empty() || frame.right_img.empty()) {
    return false;
  }

  return true;
}

std::vector<Eigen::Matrix4d> DatasetKitti::loadGtPoses() const {
  const fs::path pose_path = kitti_root_ / "poses" / (sequence_ + ".txt");
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

} // namespace svo
