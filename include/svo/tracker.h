#ifndef SVO_TRACKER_H
#define SVO_TRACKER_H

#include <vector>

#include <Eigen/Core>
#include <opencv2/opencv.hpp>

#include "svo/frame.h"
#include "svo/map_point.h"

namespace svo {

struct TrackResult {
  std::vector<cv::Point2f> prev_points;
  std::vector<cv::Point2f> curr_points;

  std::vector<MapPoint> tracked_landmarks;

  std::vector<Eigen::Vector3d> object_points;
  std::vector<cv::Point2f> image_points;
  std::vector<int> landmark_ids;

  cv::Mat track_vis;

  int num_input_tracks = 0;
  int num_flow_success = 0;
  int num_inside_image = 0;
  int num_valid_correspondences = 0;
};

class Tracker {
public:
  struct Options {
    // LK optical-flow patch size. Larger windows handle faster inter-frame
    // motion but increase compute cost; 25×25 comfortably covers KITTI vehicle
    // speeds (~1–3 px/frame at lower pyramid levels).
    cv::Size win_size = cv::Size(21, 21);
    // Number of pyramid levels (0 = full-resolution only). Each additional level
    // doubles the effective capture range; 4 levels handle motion up to ~16 px
    // at the finest scale before the coarsest level takes over.
    int max_level = 3;
    cv::TermCriteria term_criteria = cv::TermCriteria(
        cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);
    // Forward-backward consistency check: a track is kept only when re-tracking
    // the predicted point back to the previous frame lands within this many
    // pixels of the original. 1.5 px ≈ 1σ for well-converged LK flow.
    double max_bidirectional_error_px = 1.5;
    // Tracks whose current or predicted position is within this many pixels of
    // the image boundary are dropped (LK unreliable near edges).
    int image_border_px = 10;
    int max_visualized_tracks = 150;
  };

  explicit Tracker(const Options &options);

  TrackResult
  trackFrameToFrame(const Frame &prev_frame, const Frame &curr_frame,
                    const std::vector<cv::Point2f> &prev_points,
                    const std::vector<MapPoint> &prev_landmarks,
                    bool build_visualization = false,
                    cv::Point2f motion_hint = {0.0f, 0.0f}) const;

private:
  bool isInsideImage(const cv::Point2f &pt, const cv::Size &image_size) const;

private:
  Options options_;
};

} // namespace svo

#endif // SVO_TRACKER_H