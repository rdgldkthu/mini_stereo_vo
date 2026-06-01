#include "svo/motion_model.h"

namespace svo {

Eigen::Matrix4d MotionModel::predict() const {
  if (has_velocity_)
    return (T_last_ * T_delta_).matrix();
  return T_last_.matrix();
}

void MotionModel::update(const Eigen::Matrix4d &T_wc) {
  const Sophus::SE3d T_new = fromMatrix(T_wc);
  T_delta_ = T_last_.inverse() * T_new;
  T_last_ = T_new;
  has_velocity_ = true;
}

void MotionModel::onRejected() {
  T_delta_ = Sophus::SE3d::exp(0.5 * T_delta_.log());
}

void MotionModel::anchor(const Eigen::Matrix4d &T_wc) {
  const Sophus::SE3d T_new = fromMatrix(T_wc);
  T_last_ = T_new;
  T_delta_ = Sophus::SE3d{};
  has_velocity_ = false;
}

} // namespace svo
