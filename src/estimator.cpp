#include "svo/estimator.h"

#include <cmath>
#include <limits>
#include <vector>

#include <Eigen/Dense>
#include <opencv2/calib3d.hpp>

namespace svo {
namespace {

Eigen::Matrix3d hat(const Eigen::Vector3d &w) {
  Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
  W(0, 1) = -w(2);
  W(0, 2) = w(1);
  W(1, 0) = w(2);
  W(1, 2) = -w(0);
  W(2, 0) = -w(1);
  W(2, 1) = w(0);
  return W;
}

Eigen::Matrix3d expSO3(const Eigen::Vector3d &w) {
  const double theta = w.norm();
  if (theta < 1e-12) {
    return Eigen::Matrix3d::Identity() + hat(w);
  }

  const Eigen::Vector3d a = w / theta;
  const Eigen::Matrix3d A = hat(a);

  return Eigen::Matrix3d::Identity() + std::sin(theta) * A +
         (1.0 - std::cos(theta)) * A * A;
}

double huberWeight(double squared_error, double delta) {
  const double error = std::sqrt(std::max(0.0, squared_error));
  if (error <= delta) {
    return 1.0;
  }
  return delta / error;
}

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

struct LocalBAObservation {
  int keyframe_index = -1;
  int landmark_index = -1;
  Eigen::Vector2d pixel = Eigen::Vector2d::Zero();
};

double computeLocalBaRmse(const std::vector<Eigen::Matrix3d> &rotations_cw,
                          const std::vector<Eigen::Vector3d> &translations_cw,
                          const std::vector<Eigen::Vector3d> &points_w,
                          const std::vector<LocalBAObservation> &observations,
                          const Camera &camera) {
  if (observations.empty()) {
    return 0.0;
  }

  double sum_sq = 0.0;
  int count = 0;

  for (const auto &obs : observations) {
    Eigen::Vector2d proj;
    Eigen::Vector3d p_c;
    if (!projectPoint(points_w[obs.landmark_index],
                      rotations_cw[obs.keyframe_index],
                      translations_cw[obs.keyframe_index], camera, proj, p_c)) {
      continue;
    }

    const Eigen::Vector2d e = obs.pixel - proj;
    sum_sq += e.squaredNorm();
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
  result.rvec = rvec;
  result.tvec = tvec;
  result.inlier_indices = inliers;
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

  Eigen::Matrix3d R_cw = initial_rotation_cw;
  Eigen::Vector3d t_cw = initial_translation_cw;

  result.reprojection_rmse_before =
      computeReprojectionRmse(object_points, image_points, camera, R_cw, t_cw);

  double last_cost = std::numeric_limits<double>::max();

  for (int iter = 0; iter < options_.pose_refine_iterations; ++iter) {
    Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> b = Eigen::Matrix<double, 6, 1>::Zero();

    double cost = 0.0;
    int valid_count = 0;

    for (size_t i = 0; i < object_points.size(); ++i) {
      Eigen::Vector2d proj;
      Eigen::Vector3d p_c;
      if (!projectPoint(object_points[i], R_cw, t_cw, camera, proj, p_c)) {
        continue;
      }

      const Eigen::Vector2d obs(image_points[i].x, image_points[i].y);
      const Eigen::Vector2d e = obs - proj;

      const double x = p_c.x();
      const double y = p_c.y();
      const double z = p_c.z();
      const double z2 = z * z;

      Eigen::Matrix<double, 2, 3> J_proj;
      J_proj << camera.fy / z, 0.0, -camera.fx * x / z2, 0.0, camera.fy / z,
          -camera.fy * y / z2;

      Eigen::Matrix<double, 3, 6> J_pc_xi;
      J_pc_xi.block<3, 3>(0, 0) = -hat(p_c);
      J_pc_xi.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();

      const Eigen::Matrix<double, 2, 6> J = J_proj * J_pc_xi;

      const double w =
          huberWeight(e.squaredNorm(), options_.pose_refine_huber_delta);

      H += w * J.transpose() * J;
      b += w * J.transpose() * e;
      cost += w * e.squaredNorm();
      valid_count++;
    }

    if (valid_count < options_.min_refine_inliers) {
      return result;
    }

    const Eigen::Matrix<double, 6, 1> dx = H.ldlt().solve(b);

    if (!dx.allFinite()) {
      return result;
    }

    if (dx.norm() < options_.pose_refine_epsilon) {
      break;
    }

    const Eigen::Vector3d dtheta = dx.head<3>();
    const Eigen::Vector3d dt = dx.tail<3>();

    R_cw = expSO3(dtheta) * R_cw;
    t_cw = t_cw + dt;

    if (std::abs(last_cost - cost) < options_.pose_refine_epsilon) {
      break;
    }
    last_cost = cost;
  }

  result.success = true;
  result.rotation = R_cw;
  result.translation = t_cw;
  result.num_inliers = static_cast<int>(object_points.size());
  result.reprojection_rmse_after =
      computeReprojectionRmse(object_points, image_points, camera, R_cw, t_cw);

  cv::Mat R_cv =
      (cv::Mat_<double>(3, 3) << R_cw(0, 0), R_cw(0, 1), R_cw(0, 2), R_cw(1, 0),
       R_cw(1, 1), R_cw(1, 2), R_cw(2, 0), R_cw(2, 1), R_cw(2, 2));
  cv::Rodrigues(R_cv, result.rvec);

  result.tvec = cv::Mat::zeros(3, 1, CV_64F);
  result.tvec.at<double>(0, 0) = t_cw(0);
  result.tvec.at<double>(1, 0) = t_cw(1);
  result.tvec.at<double>(2, 0) = t_cw(2);

  return result;
}

LocalBAResult
Estimator::runLocalBundleAdjustment(std::vector<Frame> &keyframes,
                                    std::vector<MapPoint> &landmarks,
                                    const Camera &camera) const {
  LocalBAResult result;

  if (keyframes.size() < 2 || landmarks.empty()) {
    return result;
  }

  // -------------------------------------------------------------------------
  // Select recent keyframes
  // -------------------------------------------------------------------------
  const int kf_count =
      std::min(static_cast<int>(keyframes.size()), options_.max_ba_keyframes);

  std::vector<Frame *> ba_keyframes;
  ba_keyframes.reserve(kf_count);

  for (int i = static_cast<int>(keyframes.size()) - kf_count;
       i < static_cast<int>(keyframes.size()); ++i) {
    ba_keyframes.push_back(&keyframes[i]);
  }

  // -------------------------------------------------------------------------
  // Build landmark lookup
  // -------------------------------------------------------------------------
  std::unordered_map<int, int> landmark_id_to_global_index;
  landmark_id_to_global_index.reserve(landmarks.size());
  for (int i = 0; i < static_cast<int>(landmarks.size()); ++i) {
    landmark_id_to_global_index[landmarks[i].id] = i;
  }

  // -------------------------------------------------------------------------
  // Gather observations from keyframes
  // -------------------------------------------------------------------------
  std::unordered_map<int, int> global_landmark_to_local;
  std::vector<int> local_to_global_landmark;
  std::vector<LocalBAObservation> observations;

  for (int kf_idx = 0; kf_idx < static_cast<int>(ba_keyframes.size());
       ++kf_idx) {
    const Frame &frame = *ba_keyframes[kf_idx];
    const int n = std::min(static_cast<int>(frame.tracked_points.size()),
                           static_cast<int>(frame.tracked_landmark_ids.size()));

    for (int i = 0; i < n; ++i) {
      const int landmark_id = frame.tracked_landmark_ids[i];
      const auto it = landmark_id_to_global_index.find(landmark_id);
      if (it == landmark_id_to_global_index.end()) {
        continue;
      }

      const int global_idx = it->second;

      auto local_it = global_landmark_to_local.find(global_idx);
      int local_idx = -1;
      if (local_it == global_landmark_to_local.end()) {
        if (static_cast<int>(local_to_global_landmark.size()) >=
            options_.max_ba_landmarks) {
          continue;
        }
        local_idx = static_cast<int>(local_to_global_landmark.size());
        global_landmark_to_local[global_idx] = local_idx;
        local_to_global_landmark.push_back(global_idx);
      } else {
        local_idx = local_it->second;
      }

      LocalBAObservation obs;
      obs.keyframe_index = kf_idx;
      obs.landmark_index = local_idx;
      obs.pixel =
          Eigen::Vector2d(frame.tracked_points[i].x, frame.tracked_points[i].y);
      observations.push_back(obs);
    }
  }

  if (static_cast<int>(observations.size()) < options_.min_ba_observations) {
    return result;
  }

  // -------------------------------------------------------------------------
  // Initialize parameter blocks
  // -------------------------------------------------------------------------
  const int num_keyframes = static_cast<int>(ba_keyframes.size());
  const int num_landmarks = static_cast<int>(local_to_global_landmark.size());

  std::vector<Eigen::Matrix3d> rotations_cw(num_keyframes);
  std::vector<Eigen::Vector3d> translations_cw(num_keyframes);
  std::vector<Eigen::Vector3d> points_w(num_landmarks);

  for (int k = 0; k < num_keyframes; ++k) {
    const Eigen::Matrix4d &T_wc = ba_keyframes[k]->pose_wc;
    const Eigen::Matrix3d R_wc = T_wc.block<3, 3>(0, 0);
    const Eigen::Vector3d t_wc = T_wc.block<3, 1>(0, 3);

    rotations_cw[k] = R_wc.transpose();
    translations_cw[k] = -rotations_cw[k] * t_wc;
  }

  for (int j = 0; j < num_landmarks; ++j) {
    points_w[j] = landmarks[local_to_global_landmark[j]].p_w;
  }

  result.rmse_before = computeLocalBaRmse(rotations_cw, translations_cw,
                                          points_w, observations, camera);

  // -------------------------------------------------------------------------
  // Dense Gauss-Newton
  // Fix first keyframe pose to remove gauge freedom.
  // Param layout:
  //   poses 1..K-1 : 6 each
  //   landmarks 0..P-1 : 3 each
  // -------------------------------------------------------------------------
  const int pose_dim = 6 * (num_keyframes - 1);
  const int point_dim = 3 * num_landmarks;
  const int total_dim = pose_dim + point_dim;

  if (total_dim <= 0) {
    return result;
  }

  double last_cost = std::numeric_limits<double>::max();

  for (int iter = 0; iter < options_.local_ba_iterations; ++iter) {
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(total_dim, total_dim);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(total_dim);

    double cost = 0.0;
    int valid_obs = 0;

    for (const auto &obs : observations) {
      const int kf_idx = obs.keyframe_index;
      const int lm_idx = obs.landmark_index;

      Eigen::Vector2d proj;
      Eigen::Vector3d p_c;
      if (!projectPoint(points_w[lm_idx], rotations_cw[kf_idx],
                        translations_cw[kf_idx], camera, proj, p_c)) {
        continue;
      }

      const Eigen::Vector2d e = obs.pixel - proj;

      const double x = p_c.x();
      const double y = p_c.y();
      const double z = p_c.z();
      const double z2 = z * z;

      Eigen::Matrix<double, 2, 3> J_proj;
      J_proj << camera.fx / z, 0.0, -camera.fx * x / z2, 0.0, camera.fy / z,
          -camera.fy * y / z2;

      const double w =
          huberWeight(e.squaredNorm(), options_.local_ba_huber_delta);

      Eigen::Matrix<double, 2, 6> J_pose = Eigen::Matrix<double, 2, 6>::Zero();
      if (kf_idx > 0) {
        Eigen::Matrix<double, 3, 6> J_pc_xi;
        J_pc_xi.block<3, 3>(0, 0) = -hat(p_c);
        J_pc_xi.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
        J_pose = J_proj * J_pc_xi;
      }

      const Eigen::Matrix<double, 2, 3> J_point = J_proj * rotations_cw[kf_idx];

      if (kf_idx > 0) {
        const int pose_offset = 6 * (kf_idx - 1);
        H.block<6, 6>(pose_offset, pose_offset) +=
            w * J_pose.transpose() * J_pose;
        b.segment<6>(pose_offset) += w * J_pose.transpose() * e;

        const int point_offset = pose_dim + 3 * lm_idx;
        H.block(pose_offset, point_offset, 6, 3) +=
            w * J_pose.transpose() * J_point;
        H.block(point_offset, pose_offset, 3, 6) +=
            w * J_point.transpose() * J_pose;
      }

      {
        const int point_offset = pose_dim + 3 * lm_idx;
        H.block<3, 3>(point_offset, point_offset) +=
            w * J_point.transpose() * J_point;
        b.segment<3>(point_offset) += w * J_point.transpose() * e;
      }

      cost += w * e.squaredNorm();
      valid_obs++;
    }

    if (valid_obs < options_.min_ba_observations) {
      return result;
    }

    const Eigen::VectorXd dx = H.ldlt().solve(b);
    if (!dx.allFinite()) {
      return result;
    }

    if (dx.norm() < options_.local_ba_epsilon) {
      break;
    }

    // update poses (except first)
    for (int kf_idx = 1; kf_idx < num_keyframes; ++kf_idx) {
      const int pose_offset = 6 * (kf_idx - 1);
      const Eigen::Vector3d dtheta = dx.segment<3>(pose_offset);
      const Eigen::Vector3d dt = dx.segment<3>(pose_offset + 3);

      rotations_cw[kf_idx] = expSO3(dtheta) * rotations_cw[kf_idx];
      translations_cw[kf_idx] += dt;
    }

    // update points
    for (int lm_idx = 0; lm_idx < num_landmarks; ++lm_idx) {
      const int point_offset = pose_dim + 3 * lm_idx;
      points_w[lm_idx] += dx.segment<3>(point_offset);
    }

    if (std::abs(last_cost - cost) < options_.local_ba_epsilon) {
      break;
    }
    last_cost = cost;
  }

  result.rmse_after = computeLocalBaRmse(rotations_cw, translations_cw,
                                         points_w, observations, camera);

  // -------------------------------------------------------------------------
  // Write back optimized poses and landmarks
  // -------------------------------------------------------------------------
  for (int k = 0; k < num_keyframes; ++k) {
    const Eigen::Matrix3d R_wc = rotations_cw[k].transpose();
    const Eigen::Vector3d t_wc = -R_wc * translations_cw[k];

    ba_keyframes[k]->pose_wc.setIdentity();
    ba_keyframes[k]->pose_wc.block<3, 3>(0, 0) = R_wc;
    ba_keyframes[k]->pose_wc.block<3, 1>(0, 3) = t_wc;
  }

  for (int j = 0; j < num_landmarks; ++j) {
    landmarks[local_to_global_landmark[j]].p_w = points_w[j];
  }

  result.success = true;
  result.num_keyframes = num_keyframes;
  result.num_landmarks = num_landmarks;
  result.num_observations = static_cast<int>(observations.size());
  return result;
}

} // namespace svo
