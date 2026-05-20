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

cv::Mat Viewer::drawTrajectoryView(const std::vector<Eigen::Matrix4d> &poses,
                                   const std::vector<Eigen::Matrix4d> &gt_poses,
                                   const ViewerStatus &status) const {
  const int size = options_.trajectory_size;
  cv::Mat traj(size, size, CV_8UC3, cv::Scalar(20, 20, 20));

  const int center_x = size / 2;
  const int center_y = size / 2;

  if (poses.empty()) {
    return traj;
  }

  const int frame_id = status.frame_id;

  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  if (options_.center_on_current_pose) {
    center = poses.back().block<3, 1>(0, 3);
  }

  auto project = [&](const Eigen::Vector3d &t) {
    const Eigen::Vector3d rel = t - center;
    const int x =
        static_cast<int>(center_x + options_.trajectory_scale * rel.x());
    const int y =
        static_cast<int>(center_y - options_.trajectory_scale * rel.z());
    return cv::Point(x, y);
  };

  if (!gt_poses.empty()) {
    const int gt_end =
        std::min(frame_id, static_cast<int>(gt_poses.size()) - 1);

    for (int i = 1; i <= gt_end; ++i) {
      const Eigen::Vector3d t0 = gt_poses[i - 1].block<3, 1>(0, 3);
      const Eigen::Vector3d t1 = gt_poses[i].block<3, 1>(0, 3);

      const cv::Point p0 = project(t0);
      const cv::Point p1 = project(t1);

      if (p0.x >= 0 && p0.x < size && p0.y >= 0 && p0.y < size && p1.x >= 0 &&
          p1.x < size && p1.y >= 0 && p1.y < size) {
        cv::line(traj, p0, p1, cv::Scalar(0, 180, 0), 2);
      }
    }

    if (gt_end >= 0) {
      const Eigen::Vector3d t_gt = gt_poses[gt_end].block<3, 1>(0, 3);
      const cv::Point p_gt = project(t_gt);
      if (p_gt.x >= 0 && p_gt.x < size && p_gt.y >= 0 && p_gt.y < size) {
        cv::circle(traj, p_gt, 4, cv::Scalar(255, 0, 0), -1);
      }
    }
  }

  for (size_t i = 1; i < poses.size(); ++i) {
    const Eigen::Vector3d t0 = poses[i - 1].block<3, 1>(0, 3);
    const Eigen::Vector3d t1 = poses[i].block<3, 1>(0, 3);

    const cv::Point p0 = project(t0);
    const cv::Point p1 = project(t1);

    if (p0.x >= 0 && p0.x < size && p0.y >= 0 && p0.y < size && p1.x >= 0 &&
        p1.x < size && p1.y >= 0 && p1.y < size) {
      cv::line(traj, p0, p1, cv::Scalar(0, 255, 255), 2);
    }
  }

  const Eigen::Vector3d t_est = poses.back().block<3, 1>(0, 3);
  const cv::Point p_est = project(t_est);
  if (p_est.x >= 0 && p_est.x < size && p_est.y >= 0 && p_est.y < size) {
    cv::circle(traj, p_est, 5, cv::Scalar(0, 0, 255), -1);
  }

  cv::putText(traj, "Trajectory x-z: est=yellow, gt=green", cv::Point(20, 30),
              cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);

  cv::putText(traj, "press q or ESC to quit", cv::Point(20, size - 25),
              cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(180, 180, 180), 1);

  return traj;
}

bool Viewer::update(const cv::Mat &left_img,
                    const std::vector<cv::Point2f> &active_points,
                    const std::vector<Eigen::Matrix4d> &poses,
                    const std::vector<Eigen::Matrix4d> &gt_poses,
                    const ViewerStatus &status) {
  const cv::Mat image_view = drawImageView(left_img, active_points, status);
  const cv::Mat traj_view = drawTrajectoryView(poses, gt_poses, status);

  cv::imshow("stereo_vo: tracking", image_view);
  cv::imshow("stereo_vo: trajectory", traj_view);

  const int key = cv::waitKey(options_.image_wait_ms);
  return !(key == 'q' || key == 27);
}

} // namespace svo
