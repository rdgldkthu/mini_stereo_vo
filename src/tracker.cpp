#include "svo/tracker.h"

#include <algorithm>
#include <vector>

namespace svo {

Tracker::Tracker(const Options &options) : options_(options) {}

bool Tracker::isInsideImage(const cv::Point2f &pt,
                            const cv::Size &image_size) const {
  const int b = options_.image_border_px;
  return pt.x >= b && pt.y >= b && pt.x < image_size.width - b &&
         pt.y < image_size.height - b;
}

TrackResult
Tracker::trackFrameToFrame(const Frame &prev_frame, const Frame &curr_frame,
                           const std::vector<cv::Point2f> &prev_points,
                           const std::vector<MapPoint> &prev_landmarks) const {
  TrackResult result;

  if (prev_frame.left_img.empty() || curr_frame.left_img.empty()) {
    return result;
  }
  if (prev_points.empty() || prev_landmarks.empty()) {
    return result;
  }
  if (prev_points.size() != prev_landmarks.size()) {
    return result;
  }

  result.num_input_tracks = static_cast<int>(prev_points.size());

  std::vector<cv::Point2f> curr_points;
  std::vector<uchar> status_forward;
  std::vector<float> error_forward;

  cv::calcOpticalFlowPyrLK(prev_frame.left_img, curr_frame.left_img,
                           prev_points, curr_points, status_forward,
                           error_forward, options_.win_size, options_.max_level,
                           options_.term_criteria);

  std::vector<cv::Point2f> back_points;
  std::vector<uchar> status_backward;
  std::vector<float> error_backward;

  cv::calcOpticalFlowPyrLK(curr_frame.left_img, prev_frame.left_img,
                           curr_points, back_points, status_backward,
                           error_backward, options_.win_size,
                           options_.max_level, options_.term_criteria);

  cv::Mat vis_prev, vis_curr;
  cv::cvtColor(prev_frame.left_img, vis_prev, cv::COLOR_GRAY2BGR);
  cv::cvtColor(curr_frame.left_img, vis_curr, cv::COLOR_GRAY2BGR);

  const int canvas_h = std::max(vis_prev.rows, vis_curr.rows);
  const int canvas_w = vis_prev.cols + vis_curr.cols;
  cv::Mat canvas(canvas_h, canvas_w, CV_8UC3, cv::Scalar(0, 0, 0));
  vis_prev.copyTo(canvas(cv::Rect(0, 0, vis_prev.cols, vis_prev.rows)));
  vis_curr.copyTo(
      canvas(cv::Rect(vis_prev.cols, 0, vis_curr.cols, vis_curr.rows)));

  int num_visualized = 0;

  for (size_t i = 0; i < prev_points.size(); ++i) {
    if (!status_forward[i] || !status_backward[i]) {
      continue;
    }

    result.num_flow_success++;

    const double bidirectional_error = cv::norm(prev_points[i] - back_points[i]);
    if (bidirectional_error > options_.max_bidirectional_error_px) {
      continue;
    }

    if (!isInsideImage(curr_points[i], curr_frame.left_img.size())) {
      continue;
    }

    result.num_inside_image++;

    result.prev_points.push_back(prev_points[i]);
    result.curr_points.push_back(curr_points[i]);
    result.tracked_landmarks.push_back(prev_landmarks[i]);
    result.object_points.push_back(prev_landmarks[i].p_cam);
    result.image_points.push_back(curr_points[i]);
    result.landmark_ids.push_back(prev_landmarks[i].id);
    result.num_valid_correspondences++;

    if (num_visualized < options_.max_visualized_tracks) {
      const cv::Point2f p0 = prev_points[i];
      const cv::Point2f p1 =
          curr_points[i] + cv::Point2f(static_cast<float>(vis_prev.cols), 0.0f);

      cv::circle(canvas, p0, 2, cv::Scalar(0, 255, 0), -1);
      cv::circle(canvas, p1, 2, cv::Scalar(0, 255, 0), -1);
      cv::line(canvas, p0, p1, cv::Scalar(0, 255, 255), 1);

      num_visualized++;
    }
  }

  const std::string line1 =
      "input: " + std::to_string(result.num_input_tracks) +
      "  flow ok: " + std::to_string(result.num_flow_success) +
      "  inside: " + std::to_string(result.num_inside_image);

  const std::string line2 = "valid 3D-2D correspondences: " +
                            std::to_string(result.num_valid_correspondences);

  cv::putText(canvas, line1, cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
              cv::Scalar(0, 255, 0), 2);
  cv::putText(canvas, line2, cv::Point(20, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7,
              cv::Scalar(0, 255, 0), 2);

  result.track_vis = canvas;
  return result;
}

} // namespace svo
