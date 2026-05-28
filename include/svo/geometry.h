#ifndef SVO_GEOMETRY_H
#define SVO_GEOMETRY_H

#include <Eigen/Core>

namespace svo {

// Build T_wc (world-from-camera, 4×4) from the PnP outputs (R_cw, t_cw).
inline Eigen::Matrix4d poseWcFromCw(const Eigen::Matrix3d &R_cw,
                                     const Eigen::Vector3d &t_cw) {
  const Eigen::Matrix3d R_wc = R_cw.transpose();
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.block<3, 3>(0, 0) = R_wc;
  T.block<3, 1>(0, 3) = -R_wc * t_cw;
  return T;
}

// Extract (R_cw, t_cw) from T_wc (world-from-camera, 4×4).
inline void poseCwFromWc(const Eigen::Matrix4d &T_wc, Eigen::Matrix3d &R_cw,
                          Eigen::Vector3d &t_cw) {
  R_cw = T_wc.block<3, 3>(0, 0).transpose();
  t_cw = -R_cw * T_wc.block<3, 1>(0, 3);
}

} // namespace svo

#endif // SVO_GEOMETRY_H
