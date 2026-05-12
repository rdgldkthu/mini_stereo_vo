# Feature

**Header:** `include/svo/feature.h`  
**Source:** *(header-only struct, no separate `.cpp`)*

## Role

`Feature` is a plain data struct representing a single matched stereo keypoint pair produced by `StereoInitializer`. It pairs a left-image keypoint with the corresponding right-image keypoint, plus the precomputed disparity. Features are short-lived: they exist inside `StereoInitResult` and are consumed immediately to build `MapPoint` objects; they are not stored in `Map` or passed through the tracking loop.

---

## Struct Definition

```cpp
struct Feature {
  cv::KeyPoint kp_left;
  cv::KeyPoint kp_right;
  int left_idx  = -1;
  int right_idx = -1;
  float disparity = 0.0f;
};
```

---

## Fields

| Field | Type | Description |
|---|---|---|
| `kp_left` | `cv::KeyPoint` | ORB keypoint in the left rectified image |
| `kp_right` | `cv::KeyPoint` | Matching ORB keypoint in the right rectified image |
| `left_idx` | `int` | Index of `kp_left` in the ORB detector's keypoint array (used to retrieve descriptor) |
| `right_idx` | `int` | Index of `kp_right` in the ORB detector's keypoint array |
| `disparity` | `float` | Pixel disparity: `kp_left.pt.x − kp_right.pt.x` |

---

## Relationship to Other Types

```
StereoInitResult
  └─ features : vector<Feature>     ← one entry per triangulated stereo match
  └─ landmarks : vector<MapPoint>   ← parallel; landmarks[i] ↔ features[i]
```

`kp_left.pt` is taken directly as the initial 2D track position passed to `Frontend::initialize` and later to `Tracker::trackFrameToFrame`.

The descriptor for each `Feature` is stored on the corresponding `MapPoint` (from `desc_left.row(m.queryIdx)`), not on the `Feature` itself.

## See Also

- [`StereoInitializer`](stereo_initializer.md) — creates `Feature` objects
- [`MapPoint`](map_point.md) — parallel struct storing the 3D position and descriptor
- [`Frame`](frame.md) — `tracked_points` are seeded from `feature.kp_left.pt`
