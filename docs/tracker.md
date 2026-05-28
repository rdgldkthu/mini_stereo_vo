# Tracker

**Header:** `include/svo/tracker.h`  
**Source:** `src/tracker.cpp`

## Role

`Tracker` propagates a set of 2D point tracks from the previous frame to the current frame using pyramidal Lucas-Kanade optical flow. It applies a forward-backward consistency check to reject unreliable tracks and produces the 3D–2D point correspondences that `Estimator::estimatePosePnPRansac` needs to estimate the camera pose.

---

## Options

```cpp
struct Options {
    cv::Size win_size                  = cv::Size(21, 21);
    int      max_level                 = 3;
    cv::TermCriteria term_criteria     = cv::TermCriteria(
                                            COUNT + EPS, 30, 0.01);
    double   max_bidirectional_error_px = 1.5;
    int      image_border_px           = 10;
    int      max_visualized_tracks     = 150;
};
```

| Option | Default | Runtime | Meaning |
|---|---|---|---|
| `win_size` | 21×21 | **25×25** | Search window per pyramid level for LK |
| `max_level` | 3 | **4** | Pyramid depth (levels 0–N) |
| `term_criteria` | 30 iter / 0.01 px EPS | — | Convergence criterion for LK |
| `max_bidirectional_error_px` | 1.5 | 1.5 | Max allowable back-projection error in forward-backward check |
| `image_border_px` | 10 | 10 | Points tracked to within this border are discarded |
| `max_visualized_tracks` | 150 | 150 | Max tracks drawn in `track_vis` |

Runtime values differ from struct defaults for `win_size` (25×25) and `max_level` (4); set in `run_kitti.cpp`.

---

## TrackResult

```cpp
struct TrackResult {
    std::vector<cv::Point2f>      prev_points;            // source points (kept for debug)
    std::vector<cv::Point2f>      curr_points;            // tracked 2D positions in curr frame
    std::vector<MapPoint>         tracked_landmarks;       // 3D landmarks parallel to curr_points

    std::vector<Eigen::Vector3d>  object_points;          // 3D for PnP (= tracked_landmarks[i].p_w)
    std::vector<cv::Point2f>      image_points;           // 2D for PnP (= curr_points[i])
    std::vector<int>              landmark_ids;            // MapPoint IDs (for map bookkeeping)

    cv::Mat track_vis;                                    // side-by-side debug image

    int num_input_tracks          = 0;  // input tracks attempted
    int num_flow_success          = 0;  // forward + backward LK both succeeded
    int num_inside_image          = 0;  // passed border check (= num_valid_correspondences)
    int num_valid_correspondences = 0;  // final count passed to PnP
};
```

`object_points` and `image_points` are the direct inputs to `Estimator::estimatePosePnPRansac`. They are parallel: `object_points[i]` is the world-frame 3D position of the point tracked to `image_points[i]`.

---

## Algorithm: Forward-Backward LK

The full signature is:

```cpp
TrackResult trackFrameToFrame(
    const Frame& prev_frame, const Frame& curr_frame,
    const std::vector<cv::Point2f>& prev_points,
    const std::vector<MapPoint>& prev_landmarks,
    bool build_visualization = false,
    cv::Point2f motion_hint = {0.0f, 0.0f}) const;
```

`motion_hint` is the median 2D optical flow from the previous frame (computed by `Frontend::processFrame`). It pre-shifts the predicted track positions before the forward LK call, improving convergence on fast-moving scenes.

### Algorithm

```cpp
// 1. Forward flow: prev → curr
cv::calcOpticalFlowPyrLK(prev_frame.left_img, curr_frame.left_img,
                         prev_points, curr_points,
                         status_forward, error_forward,
                         options_.win_size, options_.max_level,
                         options_.term_criteria);

// 2. Backward flow: curr → prev
cv::calcOpticalFlowPyrLK(curr_frame.left_img, prev_frame.left_img,
                         curr_points, back_points,
                         status_backward, error_backward, ...);

// 3. Consistency check
const double err = cv::norm(prev_points[i] - back_points[i]);
if (err > max_bidirectional_error_px) continue;  // reject
```

A track is accepted only when:

1. Both the forward and backward LK calls report success (`status_forward[i] && status_backward[i]`).
2. The round-trip error `‖p_{\text{prev}} - p_{\text{back}}‖ \leq 1.5\,\text{px}`.
3. The predicted current position is at least 10 px from every image border.

This eliminates drifted tracks and tracks that flow off the image edge.

---

## Filtering Funnel

```
num_input_tracks
    │  (forward LK fails or backward LK fails)
    ▼
num_flow_success
    │  (bidirectional error > 1.5 px)
    ▼
num_inside_image  =  num_valid_correspondences
```

Typical values on KITTI seq 05: input ~300, valid ~250.

---

## Visualisation

`track_vis` is a side-by-side BGR composite of `prev_frame.left_img` and `curr_frame.left_img`. For up to 150 accepted tracks:
- A green circle (radius 2) is drawn at each endpoint.
- A yellow line connects the previous position (left panel) to the current position (right panel, offset by image width).

Two text overlays report `num_input_tracks`, `num_flow_success`, `num_inside_image`, and `num_valid_correspondences`.

Written to `results/debug/<stem>_track_<NNNNNN>.png` when `--save-debug` is active (every 10th frame, plus a window around rejected poses).

---

## Pre-conditions

- `prev_points.size() == prev_landmarks.size()` (enforced by early return).
- Both images must be non-empty single-channel (grayscale).

## See Also

- [`Frame`](frame.md) — provides `left_img` for both previous and current frame
- [`MapPoint`](map_point.md) — `tracked_landmarks` carry the 3D positions
- [`Estimator`](estimator.md) — consumes `object_points` / `image_points` for PnP
- [`Frontend`](frontend.md) — provides `activePoints()` and `activeLandmarks()` as input
