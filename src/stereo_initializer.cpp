#include "svo/stereo_initializer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

namespace svo {

StereoInitializer::StereoInitializer(const Options &options)
    : options_(options) {}

cv::Mat StereoInitializer::makeDetectionMask(const cv::Size& image_size) const {
  cv::Mat mask(image_size, CV_8UC1, cv::Scalar(255));

  const int b = options_.image_border_px;
  if (b <= 0) {
    return mask;
  }

  cv::rectangle(
      mask, cv::Rect(0, 0, image_size.width, std::min(b, image_size.height)),
      cv::Scalar(0), cv::FILLED);
  cv::rectangle(mask,
                cv::Rect(0, std::max(0, image_size.height - b),
                         image_size.width, std::min(b, image_size.height)),
                cv::Scalar(0), cv::FILLED);
  cv::rectangle(
      mask, cv::Rect(0, 0, std::min(b, image_size.width), image_size.height),
      cv::Scalar(0), cv::FILLED);
  cv::rectangle(mask,
                cv::Rect(std::max(0, image_size.width - b), 0,
                         std::min(b, image_size.width), image_size.height),
                cv::Scalar(0), cv::FILLED);

  return mask;
}

StereoInitResult StereoInitializer::run(const Frame &frame,
                                        const Camera &camera) {
  StereoInitResult result;

  if (frame.left_img.empty() || frame.right_img.empty()) {
    return result;
  }

  cv::Ptr<cv::ORB> orb = cv::ORB::create(options_.max_features);

  const cv::Mat left_mask = makeDetectionMask(frame.left_img.size());
  const cv::Mat right_mask = makeDetectionMask(frame.right_img.size());

  std::vector<cv::KeyPoint> kps_left;
  std::vector<cv::KeyPoint> kps_right;
  cv::Mat desc_left, desc_right;

  orb->detectAndCompute(frame.left_img, left_mask, kps_left, desc_left);
  orb->detectAndCompute(frame.right_img, right_mask, kps_right, desc_right);

  result.num_left_keypoints = static_cast<int>(kps_left.size());
  result.num_right_keypoints = static_cast<int>(kps_right.size());

  if (desc_left.empty() || desc_right.empty()) {
    return result;
  }

  cv::BFMatcher matcher(cv::NORM_HAMMING, true);
  std::vector<cv::DMatch> raw_matches;
  matcher.match(desc_left, desc_right, raw_matches);

  result.num_raw_matches = static_cast<int>(raw_matches.size());

  std::vector<cv::DMatch> distance_filtered;
  distance_filtered.reserve(raw_matches.size());
  for (const auto &m : raw_matches) {
    if (m.distance <= options_.hamming_threshold) {
      distance_filtered.push_back(m);
    }
  }
  result.num_distance_filtered = static_cast<int>(distance_filtered.size());

  std::sort(distance_filtered.begin(), distance_filtered.end(),
            [](const cv::DMatch &a, const cv::DMatch &b) {
              return a.distance < b.distance;
            });

  std::vector<double> disparities;
  std::vector<double> row_errors;
  std::vector<double> depths;
  std::vector<cv::DMatch> vis_matches;
  vis_matches.reserve(options_.max_visualized_matches);

  int landmark_id = 0;

  for (const auto &m : distance_filtered) {
    const auto &kp_l = kps_left[m.queryIdx];
    const auto &kp_r = kps_right[m.trainIdx];

    const double ul = kp_l.pt.x;
    const double vl = kp_l.pt.y;
    const double ur = kp_r.pt.x;
    const double vr = kp_r.pt.y;

    const double row_error = std::abs(vl - vr);
    if (row_error > options_.row_tolerance_px) {
      continue;
    }
    result.num_row_filtered++;

    const double disparity = ul - ur;
    if (disparity < options_.min_disparity_px ||
        disparity > options_.max_disparity_px) {
      continue;
    }
    result.num_disparity_filtered++;

    Eigen::Vector3d p_c;
    if (!camera.triangulateRectified(ul, vl, ur, p_c)) {
      continue;
    }
    if (p_c.z() > options_.max_depth_m) {
      continue;
    }

    Feature feature;
    feature.kp_left = kp_l;
    feature.kp_right = kp_r;
    feature.left_idx = m.queryIdx;
    feature.right_idx = m.trainIdx;
    feature.disparity = static_cast<float>(disparity);
    result.features.push_back(feature);

    MapPoint mp;
    mp.id = landmark_id++;
    mp.p_cam = p_c;
    mp.descriptor = desc_left.row(m.queryIdx).clone();
    result.landmarks.push_back(mp);

    disparities.push_back(disparity);
    row_errors.push_back(row_error);
    depths.push_back(p_c.z());
    result.num_triangulated++;

    if (p_c.z() > 50.0) {
      result.num_depth_gt_50++;
    }
    if (p_c.z() > 80.0) {
      result.num_depth_gt_80++;
    }

    if (static_cast<int>(vis_matches.size()) <
        options_.max_visualized_matches) {
      vis_matches.push_back(m);
    }
  }

  if (!disparities.empty()) {
    result.min_disparity =
        *std::min_element(disparities.begin(), disparities.end());
    result.max_disparity =
        *std::max_element(disparities.begin(), disparities.end());
    result.mean_disparity =
        std::accumulate(disparities.begin(), disparities.end(), 0.0) /
        disparities.size();
  }

  if (!row_errors.empty()) {
    result.mean_row_error =
        std::accumulate(row_errors.begin(), row_errors.end(), 0.0) /
        row_errors.size();
    result.max_row_error =
        *std::max_element(row_errors.begin(), row_errors.end());
  }

  if (!depths.empty()) {
    result.min_depth = *std::min_element(depths.begin(), depths.end());
    result.max_depth = *std::max_element(depths.begin(), depths.end());
    result.mean_depth =
        std::accumulate(depths.begin(), depths.end(), 0.0) / depths.size();
  }

  cv::drawMatches(frame.left_img, kps_left, frame.right_img, kps_right,
                  vis_matches, result.match_vis, cv::Scalar::all(-1),
                  cv::Scalar::all(-1), std::vector<char>(),
                  cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

  if (!result.match_vis.empty()) {
    const std::string line1 =
        "L kp: " + std::to_string(result.num_left_keypoints) +
        "  R kp: " + std::to_string(result.num_right_keypoints);

    const std::string line2 =
        "raw: " + std::to_string(result.num_raw_matches) +
        "  dist: " + std::to_string(result.num_distance_filtered) +
        "  row: " + std::to_string(result.num_row_filtered) +
        "  disp: " + std::to_string(result.num_disparity_filtered) +
        "  tri: " + std::to_string(result.num_triangulated);

    const std::string line3 =
        "disp min/mean/max: " +
        cv::format("%.2f / %.2f / %.2f", result.min_disparity,
                   result.mean_disparity, result.max_disparity);

    const std::string line4 =
        "row err mean/max: " +
        cv::format("%.2f / %.2f", result.mean_row_error, result.max_row_error);

    const std::string line5 = "depth min/mean/max: " +
                              cv::format("%.2f / %.2f / %.2f", result.min_depth,
                                         result.mean_depth, result.max_depth);

    cv::putText(result.match_vis, line1, cv::Point(20, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 255, 0), 2);
    cv::putText(result.match_vis, line2, cv::Point(20, 60),
                cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 255, 0), 2);
    cv::putText(result.match_vis, line3, cv::Point(20, 90),
                cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 255, 0), 2);
    cv::putText(result.match_vis, line4, cv::Point(20, 120),
                cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 255, 0), 2);
    cv::putText(result.match_vis, line5, cv::Point(20, 150),
                cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(0, 255, 0), 2);
  }

  return result;
}

} // namespace svo
