#ifndef SVO_ESTIMATOR_H
#define SVO_ESTIMATOR_H

#include <vector>

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include "svo/camera.h"
#include "svo/frame.h"
#include "svo/map_point.h"

namespace svo {

struct PoseEstimateResult {
  bool success = false;

  cv::Mat rvec;
  cv::Mat tvec;
  cv::Mat inlier_indices;

  Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
  Eigen::Vector3d translation = Eigen::Vector3d::Zero();

  int num_object_points = 0;
  int num_image_points = 0;
  int num_inliers = 0;

  double reprojection_rmse_before = 0.0;
  double reprojection_rmse_after = 0.0;
};

struct LocalBAResult {
  bool success = false;
  int num_keyframes = 0;
  int num_landmarks = 0;
  int num_observations = 0;
  double rmse_before = 0.0;
  double rmse_after = 0.0;
};

class Estimator {
public:
  struct Options {
    bool use_extrinsic_guess = false;
    int iterations_count = 100;
    float reprojection_error_px = 4.0f;
    double confidence = 0.99;
    int min_pnp_points = 6;

    int pose_refine_iterations = 10;
    double pose_refine_epsilon = 1e-6;
    double pose_refine_huber_delta = 5.0;
    int min_refine_inliers = 10;

    int local_ba_iterations = 5;
    double local_ba_epsilon = 1e-6;
    double local_ba_huber_delta = 5.0;
    double local_ba_damping = 1e-3;

    int max_ba_keyframes = 5;
    int max_ba_landmarks = 200;
    int min_ba_observations = 30;
    int min_ba_landmark_observations = 2;
  };

  explicit Estimator(const Options &options);

  PoseEstimateResult
  estimatePosePnPRansac(const std::vector<Eigen::Vector3d> &object_points,
                        const std::vector<cv::Point2f> &image_points,
                        const Camera &camera) const;

  PoseEstimateResult
  estimatePosePnPRansac(const std::vector<Eigen::Vector3d> &object_points,
                        const std::vector<cv::Point2f> &image_points,
                        const Camera &camera,
                        const Eigen::Matrix3d &initial_rotation_cw,
                        const Eigen::Vector3d &initial_translation_cw,
                        bool use_initial_guess) const;

  PoseEstimateResult
  refinePosePoseOnly(const std::vector<Eigen::Vector3d> &object_points,
                     const std::vector<cv::Point2f> &image_points,
                     const Camera &camera,
                     const Eigen::Matrix3d &initial_rotation_cw,
                     const Eigen::Vector3d &initial_translation_cw) const;

  LocalBAResult runLocalBundleAdjustment(std::vector<Frame> &keyframes,
                                         std::vector<MapPoint> &landmarks,
                                         const Camera &camera) const;

private:
  cv::Mat makeCameraMatrix(const Camera &camera) const;

private:
  Options options_;
};

} // namespace svo

#endif // SVO_ESTIMATOR_H
