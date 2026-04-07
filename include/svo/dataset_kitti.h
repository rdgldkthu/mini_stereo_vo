#ifndef SVO_DATASET_KITTI_H
#define SVO_DATASET_KITTI_H

#include <filesystem>
#include <string>
#include <vector>

#include "svo/frame.h"

namespace svo {

class DatasetKitti {
public:
  bool open(const std::string &kitti_root, const std::string &sequence);

  bool loadFrame(int frame_id, Frame &frame) const;

  int numFrames() const { return static_cast<int>(left_images_.size()); }

  const std::filesystem::path &calibPath() const { return calib_path_; }
  const std::string &sequence() const { return sequence_; }

private:
  static std::vector<std::filesystem::path>
  sortedPngs(const std::filesystem::path &dir);

private:
  std::string sequence_;
  std::filesystem::path seq_dir_;
  std::filesystem::path calib_path_;
  std::vector<std::filesystem::path> left_images_;
  std::vector<std::filesystem::path> right_images_;
};

} // namespace svo

#endif // SVO_DATASET_KITTI_H