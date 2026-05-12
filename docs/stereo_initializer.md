# StereoInitializer

**Header:** `include/svo/stereo_initializer.h`  
**Source:** `src/stereo_initializer.cpp`

## Role

`StereoInitializer` bootstraps the visual odometry system by detecting ORB features in a stereo pair, matching them across left and right images, and triangulating valid matches into 3D landmarks. It is called at two points in the pipeline:

1. **Frame 0** — to seed the initial set of tracked landmarks.
2. **On keyframe insertion** — to replenish landmarks that have thinned out.
3. **On reinitialization** — to recover from tracking failures.

Each call returns a `StereoInitResult` containing the matched features, triangulated `MapPoint` objects, diagnostic statistics, and a debug visualisation.

---

## Options

```cpp
struct Options {
    int    max_features         = 1500;   // ORB: max keypoints per image
    int    hamming_threshold    = 40;     // max Hamming distance to accept a match
    double row_tolerance_px     = 2.0;    // |v_left - v_right| ≤ this (px)
    double min_disparity_px     = 3.0;    // minimum disparity (px)
    double max_disparity_px     = 120.0;  // maximum disparity (px)
    double max_depth_m          = 80.0;   // maximum depth (m) after triangulation
    int    image_border_px      = 10;     // exclude this border from detection
    int    max_visualized_matches = 100;  // max matches drawn in match_vis
};
```

Values used in `run_kitti.cpp` match the struct defaults.

---

## StereoInitResult

```cpp
struct StereoInitResult {
    std::vector<Feature>   features;    // matched stereo pairs (parallel to landmarks)
    std::vector<MapPoint>  landmarks;   // triangulated 3D points (camera frame)

    cv::Mat match_vis;                  // debug visualisation image

    // Filtering pipeline counts
    int num_left_keypoints      = 0;
    int num_right_keypoints     = 0;
    int num_raw_matches         = 0;
    int num_distance_filtered   = 0;
    int num_row_filtered        = 0;
    int num_disparity_filtered  = 0;
    int num_triangulated        = 0;

    // Depth stats
    int num_depth_gt_50 = 0;
    int num_depth_gt_80 = 0;

    // Disparity statistics
    double min_disparity  = 0.0;
    double max_disparity  = 0.0;
    double mean_disparity = 0.0;

    // Row-error statistics
    double mean_row_error = 0.0;
    double max_row_error  = 0.0;

    // Depth statistics
    double min_depth  = 0.0;
    double max_depth  = 0.0;
    double mean_depth = 0.0;
};
```

`features[i]` and `landmarks[i]` are always in sync: `features[i].kp_left.pt` is the 2D track start, and `landmarks[i].p_w` (in camera frame at this stage) is the corresponding 3D point.

---

## Filtering Pipeline

```
ORB detect (left + right, border mask)
        │
        ▼
BFMatcher (HAMMING, cross-check = true)
        │
        ▼
Distance filter:  match.distance ≤ hamming_threshold (40)
        │  sort by distance ascending
        ▼
Row filter:       |v_left - v_right| ≤ 2.0 px      (epipolar constraint)
        │
        ▼
Disparity filter: 3.0 ≤ (u_left - u_right) ≤ 120.0 px
        │
        ▼
Triangulate:      camera.triangulateRectified(ul, vl, ur, p_c)
        │
        ▼
Depth filter:     p_c.z() ≤ 80.0 m
        │
        ▼
→ Feature + MapPoint added to result
```

### Why the Row Filter?

For rectified stereo, corresponding points lie on the same horizontal scan line. Any vertical offset is due to imperfect rectification or noise. A 2 px tolerance is tight enough to reject false positives while accommodating real KITTI sequences.

### Disparity → Depth Math

$$d = u_l - u_r \quad \text{(disparity, pixels)}$$

$$Z = \frac{f_x \cdot b}{d}$$

$$\mathbf{p}_c = \begin{pmatrix} (u_l - c_x) Z / f_x \\ (v_l - c_y) Z / f_y \\ Z \end{pmatrix}$$

Minimum disparity 3 px corresponds to a maximum triangulatable depth of approximately `fx·b/3`. Maximum 120 px corresponds to a minimum depth of approximately `fx·b/120 ≈ 0.6 m` for KITTI seq 05.

---

## Detection Mask

```cpp
cv::Mat StereoInitializer::makeDetectionMask(const cv::Size& image_size) const;
```

Creates a binary mask (255 = detect, 0 = ignore) that blacks out a `image_border_px = 10` pixel border on all four sides. This prevents keypoints near image edges where LK flow subsequently degrades.

---

## Landmark Initialisation

Triangulated points are placed in **camera frame** with temporary IDs (`0, 1, 2, …`). The caller (`run_kitti.cpp`) is responsible for:

1. Calling `Map::assignNewLandmarkIds` to assign globally unique IDs.
2. Calling `transformLandmarksToWorld(landmarks, T_wc)` to convert from camera frame to world frame before inserting into the map.

```cpp
// In run_kitti.cpp (keyframe insertion):
new_landmarks = transformLandmarksToWorld(new_landmarks, curr_frame.pose_wc);
map.addLandmarks(new_landmarks);
```

---

## Visualisation

`match_vis` is produced by `cv::drawMatches` (up to 100 matches) and annotated with five text lines covering keypoint counts, filtering stage counts, and disparity/depth statistics. Written to `results/debug/<stem>_init_matches.png` when `--save-debug` is passed.

---

## Minimum Landmark Threshold

The system requires at least 20 triangulated landmarks (`num_triangulated >= 20`) to accept an initialisation result. If this is not met:

- At startup: fatal error, program exits.
- On reinit / keyframe insertion: result is discarded; tracking continues with existing points.

## See Also

- [`Camera`](camera.md) — provides `triangulateRectified`
- [`Feature`](feature.md) — struct produced by this module
- [`MapPoint`](map_point.md) — struct produced by this module
- [`Map`](map.md) — receives landmarks via `setActiveLandmarks` / `addLandmarks`
- [`Frontend`](frontend.md) — receives initial points via `initialize` and `setActiveTracks`
