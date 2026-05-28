#ifndef SVO_VIEWER_STATUS_H
#define SVO_VIEWER_STATUS_H

namespace svo {

struct ViewerStatus {
  int frame_id = -1;
  int num_active_points = 0;
  int num_correspondences = 0;
  int num_inliers = 0;

  bool pose_accepted = false;
  bool reinitialized = false;
  bool inserted_keyframe = false;

  double delta_t = 0.0;
  double rmse_before = 0.0;
  double rmse_after = 0.0;
};

} // namespace svo

#endif // SVO_VIEWER_STATUS_H
