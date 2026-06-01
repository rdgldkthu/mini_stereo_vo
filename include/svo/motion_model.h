#ifndef SVO_MOTION_MODEL_H
#define SVO_MOTION_MODEL_H

#include <Eigen/Core>
#include <sophus/se3.hpp>

namespace svo {

class MotionModel {
public:
  Eigen::Matrix4d predict() const;          // CV if ≥2 accepted, else last pose
  void update(const Eigen::Matrix4d &T_wc); // called on pose acceptance
  void onRejected();                        // decay velocity toward zero
  void anchor(const Eigen::Matrix4d &T_wc); // hard reset(loop-closure hook)

private:
  static Sophus::SE3d fromMatrix(const Eigen::Matrix4d &T) {
    return Sophus::SE3d(Sophus::SO3d(T.block<3, 3>(0, 0)), T.block<3, 1>(0, 3));
  }

private:
  Sophus::SE3d T_last_;   // last accepted pose, identity on construction
  Sophus::SE3d T_delta_; // velocity, identity = no motion
  bool has_velocity_ = false;
};

} // namespace svo
#endif