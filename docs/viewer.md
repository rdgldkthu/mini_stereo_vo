# RerunViewer

**Header:** `include/svo/rerun_viewer.h`  
**Header:** `include/svo/viewer_status.h`  
**Source:** `src/rerun_viewer.cpp`

## Role

`RerunViewer` is the primary real-time 3D visualiser for the pipeline. It uses the Rerun SDK to log camera images, active 2D track positions, camera pose (Transform3D), estimated and ground-truth trajectory polylines, scalar metrics (RMSE, inlier count), keyframe images, and active landmark point clouds. The Rerun viewer process is spawned automatically unless `--no-viewer` is passed, in which case the `RerunViewer` is never constructed.

The viewer runs synchronously via `update()` and always returns `true` — there is no blocking keypress check.

---

## ViewerStatus

A lightweight per-frame snapshot populated in `run_kitti.cpp` and passed to `update()`, defined in `include/svo/viewer_status.h`:

```cpp
struct ViewerStatus {
    int  frame_id             = -1;
    int  num_active_points    = 0;
    int  num_correspondences  = 0;
    int  num_inliers          = 0;

    bool pose_accepted     = false;
    bool reinitialized     = false;
    bool inserted_keyframe = false;

    double delta_t     = 0.0;
    double rmse_before = 0.0;
    double rmse_after  = 0.0;
};
```

This is a lean subset of `FrontendFrameStats`. The separation keeps `RerunViewer` independent of `Frontend`.

---

## Options

```cpp
struct Options {
    std::string app_id   = "stereo_slam";
    bool spawn_viewer    = true;
    std::string rrd_path = "";
};
```

| Option | Default | Description |
|---|---|---|
| `app_id` | `"stereo_slam"` | Application ID shown in the Rerun viewer |
| `spawn_viewer` | `true` | Spawn the Rerun viewer process; `false` records only |
| `rrd_path` | `""` | Optional path to write a `.rrd` recording file |

At runtime, `run_kitti.cpp` constructs the viewer with `app_id = "stereo_slam"` and `spawn_viewer = true`. No `rrd_path` is set.

---

## Public Methods

### `bool update(int frame_id, const cv::Mat& left_img, const std::vector<cv::Point2f>& active_points, const std::vector<Eigen::Matrix4d>& poses, const std::vector<Eigen::Matrix4d>& gt_poses, const ViewerStatus& status)`

Called once per frame from `run_kitti.cpp`. Logs to the Rerun stream:

- Current left image.
- Active 2D track positions overlaid on the image.
- Camera pose as `Transform3D` (KITTI Right-Down-Forward world convention).
- Full estimated and ground-truth trajectory polylines.
- Scalar time-series for `rmse_before`, `rmse_after`, and `num_inliers`.

Always returns `true`.

---

### `void logKeyframe(int kf_id, const Eigen::Matrix4d& pose_wc, const cv::Mat& img)`

Logs a keyframe image and its pose as a separate entity in the Rerun timeline.

---

### `void logLandmarkCloud(int frame_id, const std::vector<MapPoint>& landmarks)`

Logs the active landmark set as a 3D point cloud at the given frame time.

---

### `void logGlobalTrajectory(int frame_id, const std::vector<Eigen::Matrix4d>& kf_poses)`

Logs the keyframe trajectory as a polyline in the 3D world view.

---

## World Convention

All poses are logged using KITTI's Right-Down-Forward convention. The internal `T_wc` (Eigen `Matrix4d`) is converted to `rerun::Transform3D` before logging.

## See Also

- [`Frontend`](frontend.md) — provides `poses()` and `activePoints()`
- `app/run_kitti.cpp` — constructs `RerunViewer`, fills `ViewerStatus`, calls `update()`
