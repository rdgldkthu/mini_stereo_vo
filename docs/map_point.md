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

  int observed_times = 0;
  int tracked_times  = 0;
  int missed_times   = 0;

  bool is_outlier = false;
  bool is_active  = true;
};
```

---

## Fields

| Field | Type | Description |
|---|---|---|
| `id` | `int` | Globally unique landmark ID, assigned monotonically by `Map::assignNewLandmarkIds`. Initialised to `-1` until assigned. |
| `p_w` | `Eigen::Vector3d` | 3D position in the world frame (metres). Initially in camera frame; transformed to world frame in `run_kitti.cpp` before insertion into the map. |
| `descriptor` | `cv::Mat` | 1×32 `CV_8U` ORB descriptor from the left image at the time of triangulation. Currently stored but not used for descriptor-based matching in the tracking loop. |
| `observed_times` | `int` | Total number of frames in which this landmark has been tracked (incremented by `Map::markTrackedLandmarks`). |
| `tracked_times` | `int` | Same semantics as `observed_times` in the current implementation (both incremented together). |
| `missed_times` | `int` | Number of consecutive frames in which LK flow failed to find this landmark (incremented by `Map::markMissedLandmarks`, reset to 0 on successful tracking). |
| `is_outlier` | `bool` | Marked `true` if the landmark should be immediately pruned (currently set externally; reset to `false` on successful tracking). |
| `is_active` | `bool` | Set to `true` on successful tracking; intended for downstream filtering but not yet used as a hard prune criterion. |

---

## Lifecycle

```
StereoInitializer::run()
  → MapPoint created with id=-1, p_w in camera frame
  → descriptor = desc_left.row(m.queryIdx).clone()

run_kitti.cpp: makeInitialActiveLandmarks()
  → observed_times = tracked_times = 1, missed_times = 0

Map::assignNewLandmarkIds()
  → id assigned from monotone counter

run_kitti.cpp: transformLandmarksToWorld()
  → p_w = R_wc * p_w_cam + t_wc

Map::markTrackedLandmarks()      ← each frame, if tracked
  → observed_times++, tracked_times++, missed_times=0

Map::markMissedLandmarks()       ← each frame, if not tracked
  → missed_times++

Map::pruneLandmarks()            ← called each frame
  → removes if is_outlier OR missed_times>8 OR (observed_times<2 AND missed_times>2)
```

---

## Pruning Criteria (from `Map::pruneLandmarks`)

A landmark is removed when **any** of the following holds:

| Condition | Meaning |
|---|---|
| `is_outlier == true` | Externally flagged for removal |
| `missed_times > 8` | Lost for more than 8 consecutive frames |
| `observed_times < 2 && missed_times > 2` | Never well-established and already losing track |

---

## Notes on `p_w` Precision

`p_w` is stored as `double` (Eigen) but PnP operates with `float` (`cv::Point3f`). The cast in `Estimator::estimatePosePnPRansac` introduces up to ~1 cm error at 100 m depth. After local BA, `p_w` is refined in double precision.

## See Also

- [`Map`](map.md) — manages collections of `MapPoint`
- [`StereoInitializer`](stereo_initializer.md) — creates `MapPoint` objects
- [`Tracker`](tracker.md) — propagates `MapPoint` through `TrackResult::tracked_landmarks`
- [`Estimator`](estimator.md) — reads `p_w` for reprojection and refines it in BA
