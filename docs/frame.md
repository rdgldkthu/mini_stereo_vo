# Frame

**Header:** `include/svo/frame.h`  
**Source:** *(header-only struct, no separate `.cpp`)*

## Role

`Frame` is the central data carrier that flows through the pipeline. It bundles the raw stereo images, the estimated camera pose, a keyframe flag, and the 2D track positions at the time of processing. Both the `Frontend` (which owns `prev_frame_`) and the `Map` (which stores active keyframes) hold instances of `Frame`.

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

These two vectors must always be the same length and represent the same set of active observations at the time the frame was processed. They are populated by `Frontend::processFrame` just before keyframe insertion and stored in `Map` for future reference:

```cpp
// In Frontend::processFrame (keyframe path):
curr_frame.tracked_points = active_points_2d_;
curr_frame.tracked_landmark_ids.clear();
for (const auto& lm : active_landmarks_)
    curr_frame.tracked_landmark_ids.push_back(lm.id);
```

---

## Lifecycle

1. **Created** by `DatasetKitti::loadFrame` (images only; `id` set, rest defaulted).
2. **Pose set** by `Frontend::acceptPose` inside `processFrame` after successful PnP.
3. **Track data written** by `Frontend::processFrame` on the keyframe path.
4. **Stored in `Map`** via `Map::addKeyframe` when `is_keyframe = true`.
5. **Stored in `Frontend`** as `prev_frame_` for the next tracking iteration.

## See Also

- [`DatasetKitti`](dataset_kitti.md) — loads images into `Frame`
- [`Frontend`](frontend.md) — owns `prev_frame_`; writes `pose_wc` and track data
- [`Map`](map.md) — stores `Frame` objects as active keyframes
