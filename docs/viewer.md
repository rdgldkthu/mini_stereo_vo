# Viewer

**Header:** `include/svo/viewer.h`  
**Source:** `src/viewer.cpp`

## Role

`Viewer` provides real-time visual feedback using two OpenCV windows: a tracking overlay on the current left image and a top-down trajectory plot. It runs synchronously inside the main loop and doubles as a soft exit mechanism — pressing `q` or ESC signals the loop to stop.

---

## ViewerStatus

A lightweight snapshot of per-frame pipeline state, populated in `run_kitti.cpp` and passed to `update()`:

```cpp
struct ViewerStatus {
    int  frame_id            = -1;
    int  num_active_points   = 0;
    int  num_correspondences = 0;
    int  num_inliers         = 0;

    bool pose_accepted    = false;
    bool reinitialized    = false;
    bool inserted_keyframe = false;
    bool ran_local_ba     = false;

    double delta_t    = 0.0;   // translation from last accepted pose (m)
    double rmse_before = 0.0;  // reprojection RMSE before pose refinement (px)
    double rmse_after  = 0.0;  // reprojection RMSE after pose refinement (px)
};
```

This is a subset of `FrontendFrameStats`. The separation exists to keep `Viewer` independent of `Frontend`.

---

## Options

```cpp
struct Options {
    int    image_wait_ms         = 1;      // cv::waitKey delay
    int    trajectory_size       = 600;    // canvas size in pixels (square)
    double trajectory_scale      = 0.5;    // world metres → canvas pixels (runtime value)
    bool   center_on_current_pose = true;  // pan to follow latest pose
};
```

`trajectory_scale = 0.5` is set in `run_kitti.cpp` (default in the header is 8.0; runtime overrides it).

---

## Public Method

### `bool update(const cv::Mat& left_img, const std::vector<cv::Point2f>& active_points, const std::vector<Eigen::Matrix4d>& poses, const std::vector<Eigen::Matrix4d>& gt_poses, const ViewerStatus& status)`

Draws both views, calls `cv::imshow` + `cv::waitKey`, and returns `false` if the user pressed `q` (ASCII 113) or ESC (27).

---

## Image View (`drawImageView`)

Converts `left_img` to BGR, overlays green dots (radius 2) at each active track position, then writes three HUD lines at the top-left:

```
Line 1:  frame: <id>  active: <n>  corr: <n>  inliers: <n>
Line 2:  accepted: 0/1  reinit: 0/1  kf: 0/1  ba: 0/1
Line 3:  delta_t: 0.123  rmse: 1.23 -> 0.87
```

All text is rendered in green, font size 0.65, weight 2.

---

## Trajectory View (`drawTrajectoryView`)

Renders a 600×600 dark-grey canvas (`Scalar(20,20,20)`) showing the X-Z plane (horizontal = X, vertical = −Z so forward motion goes up the canvas).

**Projection:**

$$\text{canvas\_x} = \frac{W}{2} + s \cdot (t_x - \text{center}_x)$$

$$\text{canvas\_y} = \frac{H}{2} - s \cdot (t_z - \text{center}_z)$$

where `s = trajectory_scale = 0.5` px/m and `center` is the latest estimated pose translation (when `center_on_current_pose = true`).

**Colour coding:**

| Element | Colour |
|---|---|
| Estimated trajectory (lines) | Yellow (0, 255, 255) |
| Ground-truth trajectory (lines) | Green (0, 180, 0) |
| Current estimated position (dot) | Red (0, 0, 255), radius 5 |
| Current GT position (dot) | Blue (255, 0, 0), radius 4 |

Points outside the canvas bounds are clipped (not drawn). The legend `"Trajectory x-z: est=yellow, gt=green"` and quit hint are overlaid as text.

---

## Exit Condition

```cpp
const int key = cv::waitKey(options_.image_wait_ms);
return !(key == 'q' || key == 27);
```

`image_wait_ms = 1` keeps the display responsive while not blocking the loop significantly.

## See Also

- [`Frontend`](frontend.md) — provides `poses()` and `activePoints()`
- [`FrontendFrameStats`](frontend.md) — parent struct of the data in `ViewerStatus`
- `app/run_kitti.cpp` — fills `ViewerStatus` and calls `viewer.update()`
