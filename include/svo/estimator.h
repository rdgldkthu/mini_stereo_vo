#ifndef SVO_ESTIMATOR_H_
#define SVO_ESTIMATOR_H_

#include <vector>

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include "svo/camera.h"

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
};

class Estimator {
public:
  struct Options {
    bool use_extrinsic_guess = false;
    int iterations_count = 100;
    float reprojection_error_px = 4.0f;
    double confidence = 0.99;
    int min_pnp_points = 6;
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

private:
  cv::Mat makeCameraMatrix(const Camera &camera) const;

private:
  Options options_;
};

} // namespace svo

#endif // SVO_ESTIMATOR_H_