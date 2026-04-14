#include "svo/estimator.h"

#include <vector>

namespace svo {

Estimator::Estimator(const Options &options) : options_(options) {}

cv::Mat Estimator::makeCameraMatrix(const Camera &camera) const {
  return (cv::Mat_<double>(3, 3) << camera.fx, 0.0, camera.cx, 0.0, camera.fy,
          camera.cy, 0.0, 0.0, 1.0);
}

PoseEstimateResult Estimator::estimatePosePnPRansac(
    const std::vector<Eigen::Vector3d> &object_points,
    const std::vector<cv::Point2f> &image_points, const Camera &camera) const {
  PoseEstimateResult result;
  result.num_object_points = static_cast<int>(object_points.size());
  result.num_image_points = static_cast<int>(image_points.size());

  if (object_points.size() != image_points.size()) {
    return result;
  }
  if (static_cast<int>(object_points.size()) < options_.min_pnp_points) {
    return result;
  }

  std::vector<cv::Point3f> cv_object_points;
  cv_object_points.reserve(object_points.size());

  for (const auto &p : object_points) {
    cv_object_points.emplace_back(static_cast<float>(p.x()),
                                  static_cast<float>(p.y()),
                                  static_cast<float>(p.z()));
  }

  const cv::Mat K = makeCameraMatrix(camera);
  const cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, CV_64F);

  cv::Mat rvec, tvec, inliers;

  const bool ok = cv::solvePnPRansac(
      cv_object_points, image_points, K, dist_coeffs, rvec, tvec,
      options_.use_extrinsic_guess, options_.iterations_count,
      options_.reprojection_error_px, options_.confidence, inliers,
      cv::SOLVEPNP_ITERATIVE);

  if (!ok) {
    return result;
  }

  cv::Mat R_cv;
  cv::Rodrigues(rvec, R_cv);

  Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
  Eigen::Vector3d t = Eigen::Vector3d::Zero();

  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      R(r, c) = R_cv.at<double>(r, c);
    }
  }

  for (int r = 0; r < 3; ++r) {
    t(r) = tvec.at<double>(r, 0);
  }

  result.success = true;
  result.rvec = rvec;
  result.tvec = tvec;
  result.inlier_indices = inliers;
  result.rotation = R;
  result.translation = t;
  result.num_inliers = inliers.rows;

  return result;
}

} // namespace svo