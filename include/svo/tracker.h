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
    cv::Size win_size = cv::Size(21, 21);
    int max_level = 3;
    cv::TermCriteria term_criteria = cv::TermCriteria(
        cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);
    double max_bidirectional_error_px = 1.5;
    int image_border_px = 10;
    int max_visualized_tracks = 150;
  };

  explicit Tracker(const Options &options);

  TrackResult
  trackFrameToFrame(const Frame &prev_frame, const Frame &curr_frame,
                    const std::vector<cv::Point2f> &prev_points,
                    const std::vector<MapPoint> &prev_landmarks) const;

private:
  bool isInsideImage(const cv::Point2f &pt, const cv::Size &image_size) const;

private:
  Options options_;
};

} // namespace svo

#endif // SVO_TRACKER_H