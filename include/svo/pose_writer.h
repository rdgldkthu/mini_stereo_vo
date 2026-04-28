#ifndef SVO_POSE_WRITER_H
#define SVO_POSE_WRITER_H

#include <filesystem>
#include <vector>

#include <Eigen/Core>

namespace svo {

class PoseWriter {
public:
  static void writeKittiTrajectory(const std::filesystem::path &out_path,
                                   const std::vector<Eigen::Matrix4d> &poses);

  static void
  writeIdentityKittiTrajectory(const std::filesystem::path &out_path,
                               int num_poses);
};

} // namespace svo

#endif // SVO_POSE_WRITER_H
