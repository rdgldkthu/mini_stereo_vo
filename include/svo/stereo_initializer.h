#ifndef SVO_STEREO_INITIALIZER_H
#define SVO_STEREO_INITIALIZER_H

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "svo/camera.h"
#include "svo/feature.h"
#include "svo/frame.h"
#include "svo/map_point.h"

namespace svo {

struct StereoInitResult {
  std::vector<Feature> features;
  std::vector<MapPoint> landmarks;

  cv::Mat match_vis;

  int num_left_keypoints = 0;
  int num_right_keypoints = 0;
  int num_raw_matches = 0;
  int num_distance_filtered = 0;
  int num_row_filtered = 0;
  int num_disparity_filtered = 0;
  int num_triangulated = 0;

  int num_depth_gt_50 = 0;
  int num_depth_gt_80 = 0;

  double min_disparity = 0.0;
  double max_disparity = 0.0;
  double mean_disparity = 0.0;

  double mean_row_error = 0.0;
  double max_row_error = 0.0;

  double min_depth = 0.0;
  double max_depth = 0.0;
  double mean_depth = 0.0;
};

class StereoInitializer {
public:
  struct Options {
    // Maximum ORB keypoints detected per image before grid bucketing.
    int max_features = 1500;
    // Maximum Hamming distance for a stereo descriptor match to be accepted.
    // ORB uses 256-bit descriptors; 40 bits ≈ 16 % mismatch — tight enough to
    // reject random matches while tolerating minor illumination differences
    // between left and right cameras.
    int hamming_threshold = 40;
    // Epipolar constraint for rectified stereo: matched keypoints must lie
    // within this many pixels of the same image row.
    double row_tolerance_px = 2.0;
    // Minimum disparity accepted after matching. Below ~3 px the depth
    // uncertainty (∝ 1/disparity²) becomes too large to be useful for PnP.
    double min_disparity_px = 3.0;
    // Maximum disparity accepted. For KITTI (baseline ≈ 0.54 m, fx ≈ 710 px):
    //   depth = fx * baseline / disparity → 120 px gives depth ≈ 3.2 m (min).
    double max_disparity_px = 120.0;
    // Maximum triangulated depth. Beyond ~80 m, stereo depth uncertainty grows
    // quadratically and these points contribute little to PnP accuracy while
    // adding noise.
    double max_depth_m = 80.0;
    // Keypoints within this many pixels of the image border are suppressed;
    // LK tracking near edges is unreliable.
    int image_border_px = 10;
    int max_visualized_matches = 100;
    // Grid bucketing: image divided into grid_rows x grid_cols cells;
    // only the top max_per_cell highest-response keypoints per cell are kept.
    // Set max_per_cell <= 0 to disable.
    int grid_rows = 4;
    int grid_cols = 8;
    int max_per_cell = 10;
  };

  explicit StereoInitializer(const Options& options);

  StereoInitResult run(const Frame &frame, const Camera &camera,
                      bool build_visualization = false);

private:
  cv::Mat makeDetectionMask(const cv::Size& image_size) const;
  void bucketFeatures(StereoInitResult& result,
                      const cv::Size& image_size) const;

private:
  Options options_;
  cv::Ptr<cv::ORB> orb_;
};

} // namespace svo

#endif // SVO_STEREO_INITIALIZER_H
