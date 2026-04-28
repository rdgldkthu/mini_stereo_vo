#include "svo/pose_writer.h"

#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace fs = std::filesystem;

namespace svo {

void PoseWriter::writeKittiTrajectory(
    const fs::path &out_path, const std::vector<Eigen::Matrix4d> &poses) {
  fs::create_directories(out_path.parent_path());

  std::ofstream ofs(out_path);
  if (!ofs) {
    throw std::runtime_error("Failed to open trajectory file: " +
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

void PoseWriter::writeIdentityKittiTrajectory(const fs::path &out_path,
                                              int num_poses) {
  fs::create_directories(out_path.parent_path());

  std::ofstream ofs(out_path);
  if (!ofs) {
    throw std::runtime_error("Failed to open trajectory file: " +
                             out_path.string());
  }

  ofs << std::fixed << std::setprecision(9);

  for (int i = 0; i < num_poses; ++i) {
    ofs << "1 0 0 0 0 1 0 0 0 0 1 0\n";
  }
}

} // namespace svo
