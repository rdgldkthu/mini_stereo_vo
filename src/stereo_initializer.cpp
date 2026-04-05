#include "svo/stereo_initializer.h"

#include <algorithm>
#include <numeric>

namespace svo {

StereoInitializer::StereoInitializer() : options_{} {}

StereoInitializer::StereoInitializer(const Options &options)
    : options_(options) {}

StereoInitResult StereoInitializer::run(const Frame &frame,
                                        const Camera &camera) {
  StereoInitResult result;

  if (frame.left_img.empty() || frame.right_img.empty()) {
    return result;
  }

  cv::Ptr<cv::ORB> orb = cv::ORB::create(options_.max_features);

  std::vector<cv::KeyPoint> kps_left;
  std::vector<cv::KeyPoint> kps_right;
  cv::Mat desc_left, desc_right;

  orb->detectAndCompute(frame.left_img, cv::noArray(), kps_left, desc_left);
  orb->detectAndCompute(frame.right_img, cv::noArray(), kps_right, desc_right);

  result.num_left_keypoints = static_cast<int>(kps_left.size());
  result.num_right_keypoints = static_cast<int>(kps_right.size());

  if (desc_left.empty() || desc_right.empty()) {
    return result;
  }

  cv::BFMatcher matcher(cv::NORM_HAMMING, true);
  std::vector<cv::DMatch> raw_matches;
  matcher.match(desc_left, desc_right, raw_matches);

  result.num_raw_matches = static_cast<int>(raw_matches.size());

  // Optional distance threshold
  std::vector<cv::DMatch> distance_filtered;
  distance_filtered.reserve(raw_matches.size());
  for (const auto &m : raw_matches) {
    if (m.distance <= options_.hamming_threshold) {
      distance_filtered.push_back(m);
    }
  }

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

    // Row constraint
    if (std::abs(vl - vr) > options_.row_tolerance_px) {
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

    depths.push_back(p_c.z());
    result.num_triangulated++;

    if (static_cast<int>(vis_matches.size()) <
        options_.max_visualized_matches) {
      vis_matches.push_back(m);
    }
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
        "  row: " + std::to_string(result.num_row_filtered) +
        "  disp: " + std::to_string(result.num_disparity_filtered) +
        "  tri: " + std::to_string(result.num_triangulated);
    const std::string line3 = "depth min/mean/max: " +
                              cv::format("%.2f / %.2f / %.2f", result.min_depth,
                                         result.mean_depth, result.max_depth);

    cv::putText(result.match_vis, line1, cv::Point(20, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    cv::putText(result.match_vis, line2, cv::Point(20, 60),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    cv::putText(result.match_vis, line3, cv::Point(20, 90),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
  }

  return result;
}

} // namespace svo
