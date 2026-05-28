#ifndef SVO_ESTIMATOR_H
#define SVO_ESTIMATOR_H

#include <vector>

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include "svo/camera.h"

namespace svo {

struct PoseEstimateResult {
  bool success = false;

  std::vector<int> inlier_indices;

  Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
  Eigen::Vector3d translation = Eigen::Vector3d::Zero();

  int num_object_points = 0;
  int num_image_points = 0;
  int num_inliers = 0;

  double reprojection_rmse_before = 0.0;
  double reprojection_rmse_after = 0.0;
};

class Estimator {
public:
  struct Options {
    // Whether solvePnPRansac should use the supplied initial rotation/translation
    // as an extrinsic guess (passed through to OpenCV's useExtrinsicGuess flag).
    bool use_extrinsic_guess = false;
    // Maximum RANSAC iterations. With 99 % confidence and a 60 % inlier rate,
    // the minimum-sample-set size of 4 needs ~16 iterations; 100 is conservative.
    int iterations_count = 100;
    // RANSAC inlier threshold: a 3D–2D correspondence is an inlier when its
    // reprojection error is below this many pixels. KITTI calibration gives
    // ~0.5 px reprojection on a well-matched point; 3 px is a generous bound
    // that tolerates sub-pixel LK drift without admitting large outliers.
    float reprojection_error_px = 4.0f;
    // RANSAC confidence (probability that at least one sample is outlier-free).
    double confidence = 0.99;
    // Minimum number of 3D–2D correspondences required to attempt PnP.
    // P3P needs 4 non-degenerate points; 6 provides a safety margin.
    int min_pnp_points = 6;

    // Gauss-Newton pose-only refinement parameters (run after RANSAC on inliers).
    int pose_refine_iterations = 10;
    double pose_refine_epsilon = 1e-6;
    // Huber loss transition (pixels): residuals below this are L2, above are L1.
    double pose_refine_huber_delta = 5.0;
    // Skip refinement when fewer than this many inliers survive RANSAC
    // (too few points make the Gauss-Newton system poorly conditioned).
    int min_refine_inliers = 10;
  };

  explicit Estimator(const Options &options);

  const Options &options() const { return options_; }

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

private:
  cv::Mat makeCameraMatrix(const Camera &camera) const;

private:
  Options options_;
};

} // namespace svo

#endif // SVO_ESTIMATOR_H
