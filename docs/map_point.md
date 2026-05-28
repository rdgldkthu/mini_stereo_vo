# MapPoint

**Header:** `include/svo/map_point.h`  
**Source:** *(header-only struct, no separate `.cpp`)*

## Role

`MapPoint` represents a single 3D landmark. It is triangulated by `StereoInitializer` in camera-frame coordinates and then transformed into the world frame. Once in the world frame it is stored in `Map::active_landmarks_` and also mirrored in `Frontend::active_landmarks_` so that the tracker can supply 3D–2D correspondences for PnP without going through the map each frame.

---

## Struct Definition

```cpp
struct MapPoint {
  int id = -1;

  Eigen::Vector3d p_w = Eigen::Vector3d::Zero();

  cv::Mat descriptor;

  int tracked_frames        = 0;
  int keyframe_observations = 0;
  int missed_times          = 0;

  bool is_outlier = false;
  bool is_active  = true;
};
```

---

## Fields

| Field | Type | Description |
|---|---|---|
| `id` | `int` | Globally unique landmark ID, assigned monotonically by `Map::assignNewLandmarkIds`. Initialised to `-1` until assigned. |
| `p_w` | `Eigen::Vector3d` | 3D position in the world frame (metres). Initially in camera frame at triangulation; transformed to world frame inside `Frontend` before insertion into the map. |
| `descriptor` | `cv::Mat` | 1×32 `CV_8U` ORB descriptor from the left image at triangulation time. Stored for potential future descriptor-based re-matching. |
| `tracked_frames` | `int` | Total number of individual frames in which this landmark has been successfully tracked by LK flow (incremented by `Map::markTrackedLandmarks`). Used for landmark priority when the map is at capacity. |
| `keyframe_observations` | `int` | Number of distinct keyframes that have observed this landmark (set to 1 at triangulation, incremented by `Map::markKeyframeObservations`). Used by the pruning logic to distinguish well-established landmarks from ephemeral ones. |
| `missed_times` | `int` | Number of consecutive frames in which LK flow failed to find this landmark (incremented by `Map::markMissedLandmarks`, reset to 0 on successful tracking or on being flagged as a PnP outlier). |
| `is_outlier` | `bool` | Set `true` by `Map::markOutlierLandmarks` when the PnP RANSAC classifies the corresponding track as an outlier. Causes immediate removal in `pruneLandmarks`. |
| `is_active` | `bool` | Set `true` on successful tracking. |

---

## Lifecycle

```
StereoInitializer::run()
  → MapPoint created with id=-1, p_w in camera frame
  → descriptor = desc_left.row(m.queryIdx).clone()

makeInitialActiveLandmarks()    ← inside frontend.cpp
  → tracked_frames = 1, keyframe_observations = 1, missed_times = 0

Map::assignNewLandmarkIds()
  → id assigned from monotone counter

transformLandmarksToWorld()     ← inside frontend.cpp
  → p_w = R_wc * p_w_cam + t_wc

Map::markTrackedLandmarks()     ← each frame, if tracked by LK
  → tracked_frames++, missed_times = 0

Map::markOutlierLandmarks()     ← each frame, for PnP outliers
  → is_outlier = true, missed_times++

Map::markMissedLandmarks()      ← each frame, if not tracked
  → missed_times++

Map::markKeyframeObservations() ← on keyframe insertion
  → keyframe_observations++

Map::pruneLandmarks()           ← called each frame
  → removes if is_outlier OR missed_times > 8 OR
              (keyframe_observations < 2 AND missed_times > 2)
```

---

## Pruning Criteria (from `Map::pruneLandmarks`)

A landmark is removed when **any** of the following holds:

| Condition | Meaning |
|---|---|
| `is_outlier == true` | Flagged as PnP outlier this frame |
| `missed_times > 8` | Lost for more than 8 consecutive frames |
| `keyframe_observations < 2 && missed_times > 2` | Never observed by a second keyframe and already losing track |

---

## Notes on `p_w` Precision

`p_w` is stored as `double` (Eigen) but PnP operates with `float` (`cv::Point3f`). The cast in `Estimator::estimatePosePnPRansac` introduces up to ~1 cm error at 100 m depth.

## See Also

- [`Map`](map.md) — manages collections of `MapPoint`
- [`StereoInitializer`](stereo_initializer.md) — creates `MapPoint` objects
- [`Tracker`](tracker.md) — propagates `MapPoint` through `TrackResult::tracked_landmarks`
- [`Estimator`](estimator.md) — reads `p_w` for reprojection
