# Frame

**Header:** `include/svo/frame.h`  
**Source:** *(header-only struct, no separate `.cpp`)*

## Role

`Frame` is the central data carrier that flows through the pipeline. It bundles the raw stereo images, the estimated camera pose, a keyframe flag, and the 2D track positions used for BA observation lookup. Both the `Frontend` (which owns `prev_frame_`) and the `Map` (which stores active keyframes) hold instances of `Frame`.

---

## Struct Definition

```cpp
struct Frame {
  int id = -1;
  double timestamp = 0.0;

  cv::Mat left_img;
  cv::Mat right_img;

  Eigen::Matrix4d pose_wc = Eigen::Matrix4d::Identity();

  bool is_keyframe = false;

  std::vector<cv::Point2f> tracked_points;
  std::vector<int>         tracked_landmark_ids;
};
```

---

## Fields

| Field | Type | Description |
|---|---|---|
| `id` | `int` | Zero-based frame index within the sequence |
| `timestamp` | `double` | Timestamp in seconds (currently unused; set to 0) |
| `left_img` | `cv::Mat` | Grayscale left image (loaded by `DatasetKitti::loadFrame`) |
| `right_img` | `cv::Mat` | Grayscale right image |
| `pose_wc` | `Eigen::Matrix4d` | World-from-camera transform T_wc (see Pose Convention below) |
| `is_keyframe` | `bool` | Set to `true` when the frame is promoted to a keyframe and added to `Map` |
| `tracked_points` | `vector<Point2f>` | 2D pixel positions of active tracks **at the time this frame was processed** |
| `tracked_landmark_ids` | `vector<int>` | Parallel to `tracked_points`; `tracked_landmark_ids[i]` is the `MapPoint::id` for `tracked_points[i]` |

---

## Pose Convention

All poses are stored as `T_wc` (world-from-camera / camera-in-world):

$$T_{wc} = \begin{pmatrix} R_{wc} & \mathbf{t}_{wc} \\ \mathbf{0}^\top & 1 \end{pmatrix}$$

- The camera origin in world coordinates is `t_wc = pose_wc.block<3,1>(0,3)`.
- PnP solvers return `(R_cw, t_cw)`; conversion to `T_wc`:

$$R_{wc} = R_{cw}^\top, \qquad \mathbf{t}_{wc} = -R_{wc} \cdot \mathbf{t}_{cw}$$

Frame 0 is always at identity: `pose_wc = I₄`.

---

## `tracked_points` / `tracked_landmark_ids` Invariant

These two vectors must always be the same length and represent the same set of active observations in this frame. They are used by `Estimator::runLocalBundleAdjustment` to look up pixel observations per keyframe:

```cpp
// In Estimator::runLocalBundleAdjustment:
const int n = std::min(frame.tracked_points.size(),
                       frame.tracked_landmark_ids.size());
for (int i = 0; i < n; ++i) {
    const int landmark_id = frame.tracked_landmark_ids[i];
    // ... build LocalBAObservation using frame.tracked_points[i]
}
```

---

## Lifecycle

1. **Created** by `DatasetKitti::loadFrame` (images only; `id` set, rest defaulted).
2. **Pose set** by `Frontend::acceptPose` / `run_kitti.cpp` after successful PnP.
3. **Track data written** in `run_kitti.cpp` just before keyframe insertion.
4. **Stored in `Map`** via `Map::addKeyframe` when `is_keyframe = true`.
5. **Stored in `Frontend`** as `prev_frame_` for the next tracking iteration.

## See Also

- [`DatasetKitti`](dataset_kitti.md) — loads images into `Frame`
- [`Frontend`](frontend.md) — owns `prev_frame_`; reads `pose_wc`
- [`Map`](map.md) — stores `Frame` objects as active keyframes
- [`Estimator`](estimator.md) — reads `tracked_points` / `tracked_landmark_ids` for BA
