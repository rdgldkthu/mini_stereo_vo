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
    int max_features = 1500;
    int hamming_threshold = 40;
    double row_tolerance_px = 2.0;
    double min_disparity_px = 3.0;
    double max_disparity_px = 120.0;
    double max_depth_m = 80.0;
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
