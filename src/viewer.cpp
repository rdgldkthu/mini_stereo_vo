#include "svo/viewer.h"

#include <iomanip>
#include <sstream>

namespace svo {

Viewer::Viewer(const Options &options) : options_(options) {}

cv::Mat Viewer::drawImageView(const cv::Mat &left_img,
                              const std::vector<cv::Point2f> &active_points,
                              const ViewerStatus &status) const {
  cv::Mat vis;

  if (left_img.channels() == 1) {
    cv::cvtColor(left_img, vis, cv::COLOR_GRAY2BGR);
  } else {
    vis = left_img.clone();
  }

  for (const auto &p : active_points) {
    cv::circle(vis, p, 2, cv::Scalar(0, 255, 0), -1);
  }

  std::ostringstream oss;
  oss << "frame: " << status.frame_id << " active: " << status.num_active_points
      << " corr: " << status.num_correspondences
      << " inliers: " << status.num_inliers;

  cv::putText(vis, oss.str(), cv::Point(20, 30), cv::FONT_HERSHEY_SIMPLEX, 0.65,
              cv::Scalar(0, 255, 0), 2);

  std::ostringstream oss2;
  oss2 << "accepted: " << status.pose_accepted
       << " reinit: " << status.reinitialized
       << " kf: " << status.inserted_keyframe << " ba: " << status.ran_local_ba;

  cv::putText(vis, oss2.str(), cv::Point(20, 60), cv::FONT_HERSHEY_SIMPLEX,
              0.65, cv::Scalar(0, 255, 0), 2);

  std::ostringstream oss3;
  oss3 << std::fixed << std::setprecision(3) << "delta_t: " << status.delta_t
       << " rmse: " << status.rmse_before << " -> " << status.rmse_after;

  cv::putText(vis, oss3.str(), cv::Point(20, 90), cv::FONT_HERSHEY_SIMPLEX,
              0.65, cv::Scalar(0, 255, 0), 2);

  return vis;
}

cv::Mat Viewer::drawTrajectoryView(const std::vector<Eigen::Matrix4d> &poses) const {
  const int size = options_.trajectory_size;
  cv::Mat traj(size, size, CV_8UC3, cv::Scalar(20, 20, 20));

  const int center_x = size / 2;
  const int center_y = size / 2;

  if (poses.empty()) {
    return traj;
  }

  const Eigen::Vector3d t_curr = poses.back().block<3, 1>(0, 3);

  for (size_t i = 1; i < poses.size(); ++i) {
    const Eigen::Vector3d t0 = poses[i - 1].block<3, 1>(0, 3) - t_curr;
    const Eigen::Vector3d t1 = poses[i].block<3, 1>(0, 3) - t_curr;

    const int x0 =
        static_cast<int>(center_x + options_.trajectory_scale * t0.x());
    const int y0 =
        static_cast<int>(center_y - options_.trajectory_scale * t0.z());

    const int x1 =
        static_cast<int>(center_x + options_.trajectory_scale * t1.x());
    const int y1 =
        static_cast<int>(center_y - options_.trajectory_scale * t1.z());

    if (x0 >= 0 && x0 < size && y0 >= 0 && y0 < size && x1 >= 0 && x1 < size &&
        y1 >= 0 && y1 < size) {
      cv::line(traj, cv::Point(x0, y0), cv::Point(x1, y1),
               cv::Scalar(255, 255, 255), 1);
    }
  }

  // current position stays at center
  cv::circle(traj, cv::Point(center_x, center_y), 3, cv::Scalar(0, 0, 255), -1);

  cv::putText(traj, "Trajectory (x-z)", cv::Point(20, 30),
              cv::FONT_HERSHEY_SIMPLEX, 0.65, cv::Scalar(255, 255, 255), 2);

  return traj;
}

bool Viewer::update(const cv::Mat &left_img,
                    const std::vector<cv::Point2f> &active_points,
                    const std::vector<Eigen::Matrix4d> &poses,
                    const ViewerStatus &status) {
  const cv::Mat image_view = drawImageView(left_img, active_points, status);
  const cv::Mat traj_view = drawTrajectoryView(poses);

  cv::imshow("stereo_vo: tracking", image_view);
  cv::imshow("stereo_vo: trajectory", traj_view);

  const int key = cv::waitKey(options_.image_wait_ms);
  if (key == 'q' || key == 27) {
    return false;
  }

  return true;
}

} // namespace svo
