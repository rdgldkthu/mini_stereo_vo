#include "svo/camera.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace svo {
namespace {

bool parseProjectionLine(const std::string &line, const std::string &key,
                         Eigen::Matrix<double, 3, 4> &P) {
  if (line.rfind(key, 0) != 0) {
    return false;
  }

  std::string values = line.substr(key.size());
  std::istringstream ss(values);

  std::vector<double> nums;
  double value = 0.0;
  while (ss >> value) {
    nums.push_back(value);
  }

  if (nums.size() != 12) {
    throw std::runtime_error(
        "Projection matrix line does not contain 12 numbers: " + line);
  }

  for (int r = 0; r < 3; r++) {
    for (int c = 0; c < 4; c++) {
      P(r, c) = nums[r * 4 + c];
    }
  }

  return true;
}

} // namespace

bool Camera::loadFromKittiCalib(const std::string &calib_path) {
  std::ifstream ifs(calib_path);
  if (!ifs) {
    std::cerr << "Failed to open calib file: " << calib_path << std::endl;
    return false;
  }

  bool found_p0 = false;
  bool found_p1 = false;

  std::string line;
  while (std::getline(ifs, line)) {
    if (!found_p0 && parseProjectionLine(line, "P0:", P_left)) {
      found_p0 = true;
      continue;
    }
    if (!found_p1 && parseProjectionLine(line, "P1:", P_right)) {
      found_p1 = true;
      continue;
    }
  }

  if (!found_p0 || !found_p1) {
    std::cerr << "Failed to find P0/P1 in calib file." << std::endl;
    return false;
  }

  fx = P_left(0, 0);
  fy = P_left(1, 1);
  cx = P_left(0, 2);
  cy = P_left(1, 2);

  // For rectified stereo:
  // P_right(0,3) = -fx * baseline
  baseline = -P_right(0, 3) / P_right(0, 0);

  return true;
}

Eigen::Vector3d Camera::pixel2Camera(double u, double v, double depth) const {
  const double x = (u - cx) * depth / fx;
  const double y = (v - cy) * depth / fy;
  return Eigen::Vector3d(x, y, depth);
}

bool Camera::triangulateRectified(double ul, double vl, double ur,
                                  Eigen::Vector3d &p_c) const {
  const double disparity = ul - ur;
  if (disparity <= 1e-6) {
    return false;
  }

  const double depth = fx * baseline / disparity;
  if (!std::isfinite(depth) || depth <= 0.0) {
    return false;
  }

  p_c = pixel2Camera(ul, vl, depth);
  return true;
}

void Camera::print() const {
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "Camera intrinsics:\n";
  std::cout << "  fx: " << fx << "\n";
  std::cout << "  fy: " << fy << "\n";
  std::cout << "  cx: " << cx << "\n";
  std::cout << "  cy: " << cy << "\n";
  std::cout << "  baseline: " << baseline << "\n\n";

  std::cout << "P_left:\n" << P_left << "\n\n";
  std::cout << "P_right:\n" << P_right << "\n";
}

} // namespace svo