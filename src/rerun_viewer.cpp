#include "svo/rerun_viewer.h"

#include <rerun.hpp>
#include <rerun/archetypes/image.hpp>
#include <rerun/archetypes/line_strips3d.hpp>
#include <rerun/archetypes/points2d.hpp>
#include <rerun/archetypes/points3d.hpp>
#include <rerun/archetypes/scalar.hpp>
#include <rerun/archetypes/transform3d.hpp>
#include <rerun/archetypes/view_coordinates.hpp>
#include <rerun/components/line_strip3d.hpp>
#include <rerun/components/position2d.hpp>
#include <rerun/components/position3d.hpp>
#include <rerun/components/translation3d.hpp>

#include <opencv2/imgproc.hpp>

namespace svo {

namespace {

rerun::archetypes::Transform3D poseToTransform(const Eigen::Matrix4d& T_wc) {
  const Eigen::Vector3f t = T_wc.block<3, 1>(0, 3).cast<float>();
  const Eigen::Matrix3f R = T_wc.block<3, 3>(0, 0).cast<float>();
  const rerun::datatypes::Vec3D cols[3] = {
      {R(0, 0), R(1, 0), R(2, 0)},
      {R(0, 1), R(1, 1), R(2, 1)},
      {R(0, 2), R(1, 2), R(2, 2)}};
  return rerun::archetypes::Transform3D::from_translation_mat3x3(
      rerun::components::Translation3D{t.x(), t.y(), t.z()}, cols);
}

void logTrajectory(rerun::RecordingStream& rec,
                   const std::string& path,
                   const std::vector<Eigen::Matrix4d>& poses,
                   size_t max_idx) {
  if (max_idx < 2) return;
  std::vector<rerun::datatypes::Vec3D> pts;
  pts.reserve(max_idx);
  for (size_t i = 0; i < max_idx; ++i)
    pts.push_back({static_cast<float>(poses[i](0, 3)),
                   static_cast<float>(poses[i](1, 3)),
                   static_cast<float>(poses[i](2, 3))});
  rec.log(path, rerun::archetypes::LineStrips3D(
                    {rerun::components::LineStrip3D(std::move(pts))}));
}

} // namespace

RerunViewer::RerunViewer(const Options& options) : options_(options) {
  rec_ = std::make_unique<rerun::RecordingStream>(options.app_id);
  if (options.spawn_viewer) {
    rec_->spawn().exit_on_failure();
  }
  if (!options.rrd_path.empty()) {
    rec_->save(options.rrd_path).exit_on_failure();
  }
  // KITTI camera convention: Right-Down-Forward
  rec_->log_static("world", rerun::archetypes::ViewCoordinates::RDF);
}

RerunViewer::~RerunViewer() = default;

bool RerunViewer::update(int frame_id,
                         const cv::Mat& left_img,
                         const std::vector<cv::Point2f>& active_points,
                         const std::vector<Eigen::Matrix4d>& poses,
                         const std::vector<Eigen::Matrix4d>& gt_poses,
                         const ViewerStatus& status) {
  rec_->set_time_sequence("frame", frame_id);

  // Camera image (BGR -> RGB)
  if (!left_img.empty()) {
    cv::Mat rgb;
    cv::cvtColor(left_img, rgb, cv::COLOR_BGR2RGB);
    rec_->log("camera/image",
              rerun::archetypes::Image::from_rgb24(
                  rerun::Collection<uint8_t>::borrow(
                      rgb.data,
                      static_cast<size_t>(rgb.total() * rgb.elemSize())),
                  {static_cast<uint32_t>(rgb.cols),
                   static_cast<uint32_t>(rgb.rows)}));
  }

  // Tracked 2D points
  if (!active_points.empty()) {
    std::vector<rerun::components::Position2D> pts;
    pts.reserve(active_points.size());
    for (const auto& p : active_points)
      pts.push_back({p.x, p.y});
    rec_->log("camera/tracked_pts",
              rerun::archetypes::Points2D(std::move(pts)));
  }

  // Current camera pose
  if (!poses.empty())
    rec_->log("world/camera", poseToTransform(poses.back()));

  // Estimated trajectory
  logTrajectory(*rec_, "world/est_traj", poses, poses.size());

  // Ground-truth trajectory up to current frame
  if (!gt_poses.empty()) {
    const size_t n =
        std::min(gt_poses.size(), static_cast<size_t>(frame_id + 1));
    logTrajectory(*rec_, "world/gt_traj", gt_poses, n);
  }

  // Scalar metrics
  rec_->log("metrics/rmse_after",
            rerun::archetypes::Scalar(status.rmse_after));
  rec_->log("metrics/rmse_before",
            rerun::archetypes::Scalar(status.rmse_before));
  rec_->log("metrics/inliers",
            rerun::archetypes::Scalar(
                static_cast<double>(status.num_inliers)));
  rec_->log("metrics/active_tracks",
            rerun::archetypes::Scalar(
                static_cast<double>(status.num_active_points)));

  return true;
}

void RerunViewer::logKeyframe(int kf_id,
                              const Eigen::Matrix4d& pose_wc,
                              const cv::Mat& /*img*/) {
  rec_->log("world/kf/" + std::to_string(kf_id), poseToTransform(pose_wc));
}

void RerunViewer::logLandmarkCloud(int frame_id,
                                   const std::vector<MapPoint>& landmarks) {
  rec_->set_time_sequence("frame", frame_id);
  std::vector<rerun::components::Position3D> pts;
  pts.reserve(landmarks.size());
  for (const auto& lm : landmarks) {
    if (lm.is_active && !lm.is_outlier)
      pts.push_back({static_cast<float>(lm.p_w.x()),
                     static_cast<float>(lm.p_w.y()),
                     static_cast<float>(lm.p_w.z())});
  }
  if (!pts.empty())
    rec_->log("world/landmarks",
              rerun::archetypes::Points3D(std::move(pts)));
}

void RerunViewer::logGlobalTrajectory(
    int frame_id,
    const std::vector<Eigen::Matrix4d>& kf_poses) {
  rec_->set_time_sequence("frame", frame_id);
  logTrajectory(*rec_, "world/slam_traj", kf_poses, kf_poses.size());
}

} // namespace svo
