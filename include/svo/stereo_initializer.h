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
  };

  explicit StereoInitializer(const Options& options);

  StereoInitResult run(const Frame &frame, const Camera &camera);

private:
  cv::Mat makeDetectionMask(const cv::Size& image_size) const;

private:
  Options options_;
};

} // namespace svo

#endif // SVO_STEREO_INITIALIZER_H
