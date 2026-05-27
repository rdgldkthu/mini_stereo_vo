#ifndef SVO_RERUN_VIEWER_H
#define SVO_RERUN_VIEWER_H

#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include "svo/map_point.h"
#include "svo/viewer_status.h"

namespace rerun {
class RecordingStream;
}

namespace svo {

class RerunViewer {
public:
  struct Options {
    std::string app_id   = "stereo_slam";
    bool spawn_viewer    = true;
    std::string rrd_path = "";
  };

  explicit RerunViewer(const Options& options);
  ~RerunViewer();

  // Drop-in for Viewer::update; always returns true (no blocking keypress).
  bool update(int frame_id,
              const cv::Mat& left_img,
              const std::vector<cv::Point2f>& active_points,
              const std::vector<Eigen::Matrix4d>& poses,
              const std::vector<Eigen::Matrix4d>& gt_poses,
              const ViewerStatus& status);

  void logKeyframe(int kf_id, const Eigen::Matrix4d& pose_wc, const cv::Mat& img);
  void logLandmarkCloud(int frame_id, const std::vector<MapPoint>& landmarks);
  void logGlobalTrajectory(int frame_id, const std::vector<Eigen::Matrix4d>& kf_poses);

private:
  std::unique_ptr<rerun::RecordingStream> rec_;
  Options options_;
};

} // namespace svo

#endif // SVO_RERUN_VIEWER_H
