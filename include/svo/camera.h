#ifndef SVO_CAMERA_H
#define SVO_CAMERA_H

#include <Eigen/Core>
#include <string>

namespace svo{

class Camera {
public:
  bool loadFromKittiCalib(const std::string& calib_path);

  Eigen::Vector3d pixel2Camera(double u, double v, double depth) const;
  bool triangulateRectified(double ul, double vl, double rt,
                            Eigen::Vector3d &p_c) const;

  void print() const;

public:
  double fx = 0.0;
  double fy = 0.0;
  double cx = 0.0;
  double cy = 0.0;
  double baseline = 0.0;

  Eigen::Matrix<double, 3, 4> P_left = Eigen::Matrix<double, 3, 4>::Zero();
  Eigen::Matrix<double, 3, 4> P_right = Eigen::Matrix<double, 3, 4>::Zero();
};

} // namespace svo

#endif // SVO_CAMERA_H