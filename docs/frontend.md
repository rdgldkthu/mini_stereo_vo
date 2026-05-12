# Frontend

**Header:** `include/svo/frontend.h`  
**Source:** `src/frontend.cpp`

## Role

`Frontend` is the stateful gatekeeper of the VO pipeline. It accumulates the pose history, maintains the active 2D track positions and corresponding 3D landmarks, and implements all decision logic: whether to accept or reject a candidate pose, whether to insert a keyframe, and whether to trigger reinitialization. `run_kitti.cpp` is the orchestrator; `Frontend` supplies the policy.

---

## FrontendFrameStats

Filled by `Frontend` and `run_kitti.cpp` on every frame. Used for logging and for constructing `ViewerStatus`.

```cpp
struct FrontendFrameStats {
    // Decisions
    bool pose_success      = false;  // PnP converged
    bool pose_accepted     = false;  // passed Frontend acceptance criteria
    bool reinitialized     = false;  // stereo reinit ran this frame
    bool inserted_keyframe = false;  // new keyframe added to Map
    bool ran_local_ba      = false;  // local BA was attempted
    bool local_ba_accepted = false;  // BA improved RMSE
    bool local_ba_rejected = false;  // BA failed or worsened RMSE

    // Metrics
    int    num_inliers   = 0;    // PnP inliers
    double inlier_ratio  = 0.0;  // num_inliers / num_correspondences
    double delta_t       = 0.0;  // ‖t_curr - t_prev‖ (m)
    double rmse_before   = 0.0;  // reprojection RMSE before refinement (px)
    double rmse_after    = 0.0;  // reprojection RMSE after refinement (px)
    double ba_rmse_before = 0.0; // local BA RMSE before (px)
    double ba_rmse_after  = 0.0; // local BA RMSE after (px)
};
```

---

## Options

```cpp
struct Options {
    // Keyframe insertion criteria
    double keyframe_translation_threshold_m        = 1.0;   // min translation to trigger KF
    double keyframe_rotation_threshold_deg         = 10.0;  // min rotation to trigger KF
    int    keyframe_min_tracked_points             = 100;   // low-track threshold for KF
    int    keyframe_min_frame_gap                  = 15;    // min frames between KFs
    double keyframe_low_track_translation_threshold_m = 0.5; // secondary translation threshold

    // Pose acceptance
    int    min_initial_landmarks  = 20;   // minimum landmarks to start system
    int    min_pose_inliers       = 15;   // PnP inliers required to accept pose
    double min_pose_inlier_ratio  = 0.10; // fraction of correspondences that must be inliers
    double max_frame_translation_m = 2.0; // max allowed per-frame displacement (outlier guard)

    // Reinitialization
    int    min_reinit_frame_gap          = 10;  // min frames after last reinit
    int    weak_track_threshold          = 80;  // active tracks below this → consider reinit
    int    emergency_rejected_poses_count = 2;  // consecutive rejects → emergency reinit

    // Local BA
    int    local_ba_keyframe_interval = 2;  // run BA every N new keyframes
};
```

Values used at runtime (from `run_kitti.cpp`):
- `keyframe_translation_threshold_m = 1.5`
- `keyframe_rotation_threshold_deg = 12.0`
- `keyframe_min_tracked_points = 60`
- `keyframe_min_frame_gap = 5`
- `min_pose_inliers = 15`, `min_pose_inlier_ratio = 0.10`, `max_frame_translation_m = 2.0`
- `weak_track_threshold = 80`, `emergency_rejected_poses_count = 2`

---

## State

| Private field | Type | Description |
|---|---|---|
| `poses_` | `vector<Matrix4d>` | Full pose history, one T_wc per frame |
| `prev_frame_` | `Frame` | Previous frame (image + pose), used as LK source |
| `active_points_2d_` | `vector<Point2f>` | Current 2D track positions |
| `active_landmarks_` | `vector<MapPoint>` | Parallel 3D landmarks |
| `last_keyframe_frame_id_` | `int` | Frame ID of last inserted keyframe |
| `last_keyframe_pose_wc_` | `Matrix4d` | Pose of last inserted keyframe |
| `last_init_frame_id_` | `int` | Frame ID of last (re)initialization |
| `consecutive_rejected_poses_` | `int` | Counter reset on acceptance |
| `inserted_keyframes_since_last_ba_` | `int` | Trigger for local BA |
| `dense_debug_center_` | `int` | Frame ID of last pose rejection (for debug image saving) |

---

## Methods

### `void initialize(const Frame& frame0, const StereoInitResult& init_result, std::vector<MapPoint> active_landmarks)`

Sets up state for frame 0:
- `poses_` = `[I₄]`
- `prev_frame_` = frame0 with `is_keyframe = true`
- `active_points_2d_` = left keypoint positions from `init_result.features`
- `active_landmarks_` = moved from `active_landmarks` parameter
- All counters reset to 0

---

### `bool acceptPose(int frame_id, int num_inliers, int num_correspondences, const Eigen::Matrix4d& candidate_pose, FrontendFrameStats& stats)`

Computes `inlier_ratio = num_inliers / max(1, num_correspondences)` and `delta_t = ‖t_curr - t_prev‖`.

**Acceptance condition (all three must hold):**

| Criterion | Threshold |
|---|---|
| `num_inliers >= min_pose_inliers` | 15 |
| `inlier_ratio >= min_pose_inlier_ratio` | 0.10 |
| `delta_t <= max_frame_translation_m` | 2.0 m |

On acceptance: appends `candidate_pose` to `poses_`, resets `consecutive_rejected_poses_`.  
On rejection: appends the **previous** pose (pose stands still), calls `notePoseRejected`.

---

### `bool needNewKeyframe(const Eigen::Matrix4d& current_pose_wc, int num_tracked_points, int current_frame_id) const`

Returns `true` when the camera has moved or rotated enough since the last keyframe, or when too few points are tracked. Gated by a minimum frame gap.

**Translation trigger:**

$$\|\mathbf{t}_{\text{curr}} - \mathbf{t}_{\text{kf}}\| \geq 1.5 \text{ m}$$

**Rotation trigger:**

$$\theta = \arccos\!\left(\frac{\text{trace}(R_{\text{last}}^\top R_{\text{curr}}) - 1}{2}\right) \geq 12°$$

Implemented as:

```cpp
const Eigen::Matrix3d R_rel = R_last.transpose() * R_curr;
double trace_value = (R_rel.trace() - 1.0) * 0.5;
trace_value = std::max(-1.0, std::min(1.0, trace_value));  // clamp for numerical safety
const double rotation_deg = std::acos(trace_value) * 180.0 / M_PI;
```

**Low-track trigger:**

$$\text{num\_tracked\_points} < 60 \;\text{ AND }\; \|\Delta \mathbf{t}\| \geq 0.5 \text{ m}$$

---

### `bool shouldReinitialize(int frame_id, bool pose_accepted, int num_active_tracks) const`

Returns `true` when one of two conditions holds, after enforcing a minimum gap of 10 frames since the last reinit:

| Condition | Trigger |
|---|---|
| **Weak-but-accepted** | `pose_accepted && num_active_tracks < 80` |
| **Emergency** | `!pose_accepted && (consecutive_rejected >= 2 || num_active_tracks < 80)` |

---

### `void refreshActiveLandmarksFromMap(const std::vector<MapPoint>& map_landmarks)`

After successful local BA, landmark positions in `Map` have been refined. This method propagates those updates into `active_landmarks_` by matching IDs:

```cpp
std::unordered_map<int, const MapPoint*> id_to_lm;
for (const auto& lm : map_landmarks) id_to_lm[lm.id] = &lm;
for (auto& lm : active_landmarks_) {
    const auto it = id_to_lm.find(lm.id);
    if (it != id_to_lm.end()) lm = *it->second;
}
```

---

### `insertedKeyframesSinceLastBa()` / `noteKeyframeInserted()` / `noteLocalBaAccepted()`

Counter protocol for scheduling local BA:

```
noteKeyframeInserted() → inserted_keyframes_since_last_ba_++
noteLocalBaAccepted()  → inserted_keyframes_since_last_ba_ = 0
```

`run_kitti.cpp` checks `insertedKeyframesSinceLastBa() >= localBaKeyframeInterval() (2)` to decide when to run BA.

---

### Other Methods

| Method | Description |
|---|---|
| `repeatLastPose()` | Appends last pose again (used when frame fails to load) |
| `rejectPose(...)` | Appends last pose and calls `notePoseRejected` (used when PnP fails to converge) |
| `setActiveTracks(points, landmarks)` | Replaces active 2D/3D track state |
| `shouldSaveDenseDebug(frame_id, radius)` | Returns `true` within `radius` frames of the last pose rejection |
| `noteReinitialized(frame_id)` | Records reinit frame; resets `consecutive_rejected_poses_` |
| `poses()` | Const reference to full pose history |
| `currentPose()` | Last pose in history |
| `activePoints()` / `activeLandmarks()` | Access to active track state |

## See Also

- [`Estimator`](estimator.md) — produces `PoseEstimateResult` consumed by `acceptPose`
- [`StereoInitializer`](stereo_initializer.md) — provides `StereoInitResult` for `initialize`
- [`Map`](map.md) — stores keyframes; provides refined landmarks for `refreshActiveLandmarksFromMap`
- [`Viewer`](viewer.md) — reads `poses()` and `activePoints()`
- [`PoseWriter`](pose_writer.md) — writes `poses()` at the end of the run
