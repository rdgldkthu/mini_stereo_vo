#include "svo/estimator.h"

#include <vector>

#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <opencv2/calib3d.hpp>
#include <sophus/ceres_manifold.hpp>
#include <sophus/se3.hpp>

namespace svo {
namespace {

struct ReprojectionCostFunctor {
  ReprojectionCostFunctor(const Eigen::Vector3d &p_w, const cv::Point2f &obs,
                          double fx, double fy, double cx, double cy)
      : p_w_(p_w), obs_(obs.x, obs.y), fx_(fx), fy_(fy), cx_(cx), cy_(cy) {}

  template <typename T>
  bool operator()(const T* const pose_data, T* residuals) const {
    const Eigen::Map<const Sophus::SE3<T>> T_cw(pose_data);
    const Eigen::Matrix<T, 3, 1> p_c = T_cw * p_w_.cast<T>();

    if (p_c[2] <= T(1e-8)) {
      residuals[0] = T(0);
      residuals[1] = T(0);
      return true;
    }

    residuals[0] = T(fx_) * p_c[0] / p_c[2] + T(cx_) - T(obs_[0]);
    residuals[1] = T(fy_) * p_c[1] / p_c[2] + T(cy_) - T(obs_[1]);

    return true;
  }

  static ceres::CostFunction *Create(const Eigen::Vector3d &p_w,
                                     const cv::Point2f &obs,
                                     const Camera &cam) {
    return new ceres::AutoDiffCostFunction<ReprojectionCostFunctor, 2,
                                           Sophus::SE3d::num_parameters>(
        new ReprojectionCostFunctor(p_w, obs, cam.fx, cam.fy, cam.cx, cam.cy));
  }

private:
  Eigen::Vector3d p_w_;
  Eigen::Vector2d obs_;
  double fx_, fy_, cx_, cy_;
};

bool projectPoint(const Eigen::Vector3d &p_w, const Eigen::Matrix3d &R_cw,
                  const Eigen::Vector3d &t_cw, const Camera &camera,
                  Eigen::Vector2d &pixel, Eigen::Vector3d &p_c) {
  p_c = R_cw * p_w + t_cw;
  if (p_c.z() <= 1e-8) {
    return false;
  }

  const double x = p_c.x();
  const double y = p_c.y();
  const double z = p_c.z();

  pixel(0) = camera.fx * x / z + camera.cx;
  pixel(1) = camera.fy * y / z + camera.cy;
  return true;
}

double
computeReprojectionRmse(const std::vector<Eigen::Vector3d> &object_points,
                        const std::vector<cv::Point2f> &image_points,
                        const Camera &camera, const Eigen::Matrix3d &R_cw,
                        const Eigen::Vector3d &t_cw) {
  if (object_points.empty() || object_points.size() != image_points.size()) {
    return 0.0;
  }

  double sum_sq = 0.0;
  int count = 0;

  for (size_t i = 0; i < object_points.size(); ++i) {
    Eigen::Vector2d proj;
    Eigen::Vector3d p_c;
    if (!projectPoint(object_points[i], R_cw, t_cw, camera, proj, p_c)) {
      continue;
    }

    const Eigen::Vector2d obs(image_points[i].x, image_points[i].y);
    const Eigen::Vector2d err = obs - proj;
    sum_sq += err.squaredNorm();
    count++;
  }

  if (count == 0) {
    return 0.0;
  }

  return std::sqrt(sum_sq / static_cast<double>(count));
}

} // namespace

Estimator::Estimator(const Options &options) : options_(options) {}

cv::Mat Estimator::makeCameraMatrix(const Camera &camera) const {
  return (cv::Mat_<double>(3, 3) << camera.fx, 0.0, camera.cx, 0.0, camera.fy,
          camera.cy, 0.0, 0.0, 1.0);
}

PoseEstimateResult Estimator::estimatePosePnPRansac(
    const std::vector<Eigen::Vector3d> &object_points,
    const std::vector<cv::Point2f> &image_points, const Camera &camera) const {
  return estimatePosePnPRansac(object_points, image_points, camera,
                               Eigen::Matrix3d::Identity(),
                               Eigen::Vector3d::Zero(), false);
}

PoseEstimateResult Estimator::estimatePosePnPRansac(
    const std::vector<Eigen::Vector3d> &object_points,
    const std::vector<cv::Point2f> &image_points, const Camera &camera,
    const Eigen::Matrix3d &initial_rotation_cw,
    const Eigen::Vector3d &initial_translation_cw,
    bool use_initial_guess) const {
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

  cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64F);
  cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64F);
  cv::Mat inliers;

  if (use_initial_guess) {
    cv::Mat R_init = (cv::Mat_<double>(3, 3) << initial_rotation_cw(0, 0),
                      initial_rotation_cw(0, 1), initial_rotation_cw(0, 2),
                      initial_rotation_cw(1, 0), initial_rotation_cw(1, 1),
                      initial_rotation_cw(1, 2), initial_rotation_cw(2, 0),
                      initial_rotation_cw(2, 1), initial_rotation_cw(2, 2));
    cv::Rodrigues(R_init, rvec);
    tvec.at<double>(0, 0) = initial_translation_cw(0);
    tvec.at<double>(1, 0) = initial_translation_cw(1);
    tvec.at<double>(2, 0) = initial_translation_cw(2);
  }

  const bool ok =
      cv::solvePnPRansac(cv_object_points, image_points, K, dist_coeffs, rvec,
                         tvec, use_initial_guess, options_.iterations_count,
                         options_.reprojection_error_px, options_.confidence,
                         inliers, cv::SOLVEPNP_ITERATIVE);

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
  result.inlier_indices.reserve(inliers.rows);
  for (int i = 0; i < inliers.rows; ++i) {
    result.inlier_indices.push_back(inliers.at<int>(i, 0));
  }
  result.rotation = R;
  result.translation = t;
  result.num_inliers = inliers.rows;
  result.reprojection_rmse_before =
      computeReprojectionRmse(object_points, image_points, camera, R, t);
  result.reprojection_rmse_after = result.reprojection_rmse_before;

  return result;
}

PoseEstimateResult Estimator::refinePosePoseOnly(
    const std::vector<Eigen::Vector3d> &object_points,
    const std::vector<cv::Point2f> &image_points, const Camera &camera,
    const Eigen::Matrix3d &initial_rotation_cw,
    const Eigen::Vector3d &initial_translation_cw) const {
  PoseEstimateResult result;
  result.num_object_points = static_cast<int>(object_points.size());
  result.num_image_points = static_cast<int>(image_points.size());

  if (object_points.size() != image_points.size()) {
    return result;
  }
  if (static_cast<int>(object_points.size()) < options_.min_refine_inliers) {
    return result;
  }

  result.reprojection_rmse_before =
      computeReprojectionRmse(object_points, image_points, camera, initial_rotation_cw, initial_translation_cw);

  Sophus::SE3d T_cw(Sophus::SO3d(initial_rotation_cw), initial_translation_cw);

  ceres::Problem problem;
  for (size_t i = 0; i < object_points.size(); ++i) {
    problem.AddResidualBlock(
        ReprojectionCostFunctor::Create(object_points[i], image_points[i],
                                        camera),
        new ceres::HuberLoss(options_.pose_refine_huber_delta), T_cw.data());
  }

  problem.SetManifold(T_cw.data(), new Sophus::Manifold<Sophus::SE3>());

  ceres::Solver::Options solver_options;
  solver_options.linear_solver_type = ceres::DENSE_QR;
  solver_options.max_num_iterations = options_.pose_refine_iterations;
  solver_options.function_tolerance = options_.pose_refine_epsilon;
  solver_options.gradient_tolerance = 1e-10;
  solver_options.logging_type = ceres::SILENT;

  ceres::Solver::Summary summary;
  ceres::Solve(solver_options, &problem, &summary);

  if (!summary.IsSolutionUsable()) {
    return result;
  }

  result.success = true;
  result.rotation = T_cw.rotationMatrix();
  result.translation = T_cw.translation();
  result.num_inliers = static_cast<int>(object_points.size());
  result.reprojection_rmse_after = computeReprojectionRmse(
      object_points, image_points, camera, result.rotation, result.translation);

  return result;
}

} // namespace svo
