# Frontend

**Header:** `include/svo/frontend.h`  
**Source:** `src/frontend.cpp`

## Role

`Frontend` is the stateful gatekeeper of the VO pipeline. It owns the pose history and active track state, implements all decision logic (accept/reject pose, keyframe insertion, reinitialization), and encapsulates the full per-frame pipeline behind a single `processFrame` call. `run_kitti.cpp` calls `bootstrap` once, then `processFrame` per frame; all other module interactions happen inside `Frontend`.

---

## FrontendFrameStats

Returned inside `ProcessFrameResult` on every frame:

```cpp
struct FrontendFrameStats {
    // Decisions
    bool pose_success      = false;  // PnP converged
    bool pose_accepted     = false;  // passed acceptance criteria
    bool reinitialized     = false;  // stereo reinit ran this frame
    bool inserted_keyframe = false;  // new keyframe added to Map

    // Metrics
    int    num_correspondences = 0;
    int    num_inliers   = 0;
    double inlier_ratio  = 0.0;
    double delta_t       = 0.0;   // ‖t_curr - t_prev‖ (m)
    double rmse_before   = 0.0;   // reprojection RMSE before refinement (px)
    double rmse_after    = 0.0;   // reprojection RMSE after refinement (px)
};
```

---

## ProcessFrameResult

```cpp
struct ProcessFrameResult {
    FrontendFrameStats stats;
    cv::Mat track_vis;       // non-empty only when save_debug=true
    bool should_exit = false;
};
```

`should_exit` is set when `active_points < min_pnp_points (6)`, signalling catastrophic tracking failure.

---

## Options

```cpp
struct Options {
    // Keyframe insertion criteria
    double keyframe_translation_threshold_m          = 1.0;
    double keyframe_rotation_threshold_deg           = 10.0;
    int    keyframe_min_tracked_points               = 100;
    int    keyframe_min_frame_gap                    = 15;
    double keyframe_low_track_translation_threshold_m = 0.5;

    // Pose acceptance
    int    min_initial_landmarks  = 20;
    int    min_pose_inliers       = 15;
    double min_pose_inlier_ratio  = 0.10;
    double max_frame_translation_m = 2.0;

    // Reinitialization
    int    min_reinit_frame_gap           = 10;
    int    weak_track_threshold           = 80;
    int    emergency_rejected_poses_count = 2;
};
```

Values used at runtime (from `run_kitti.cpp`):
- `keyframe_translation_threshold_m = 1.5`
- `keyframe_rotation_threshold_deg = 8.0`
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
| `dense_debug_center_` | `int` | Frame ID of last pose rejection (for debug image saving) |
| `motion_hint_` | `cv::Point2f` | Median optical flow from the previous frame, fed to tracker |

---

## Methods

### `bool bootstrap(Frame& frame0, StereoInitializer& initializer, Map& map, const Camera& camera, bool save_debug, cv::Mat* init_vis)`

Bootstraps the system from frame 0:

1. Runs `StereoInitializer::run(frame0)`.
2. Returns `false` if `num_triangulated < min_initial_landmarks`.
3. Assigns IDs, initialises pose history with `I₄`, seeds the map (`addKeyframe`, `setActiveLandmarks`).
4. Optionally writes the match visualisation to `*init_vis`.

---

### `ProcessFrameResult processFrame(int frame_id, Frame& curr_frame, Tracker& tracker, Estimator& estimator, StereoInitializer& initializer, Map& map, const Camera& camera, bool save_debug)`

The full per-frame VO pipeline in one call:

1. **Track** — `Tracker::trackFrameToFrame` with `motion_hint_` seeded from the median optical flow of the previous frame.
2. **Pose estimate** — `Estimator::estimatePosePnPRansac` seeded with a constant-velocity prediction; followed by `refinePosePoseOnly` on inliers.
3. **Accept/reject** — `acceptPose` or `rejectPose`; outlier tracks filtered by PnP inlier mask.
4. **Reinit check** — `shouldReinitialize` → `createKeyframeFromStereo` (replaces active landmarks).
5. **Map update** — `markTrackedLandmarks`, `markMissedLandmarks`, `pruneLandmarks`.
6. **Keyframe** — `needNewKeyframe` → `createKeyframeFromStereo` (adds keyframe + new landmarks to Map).
7. Returns `ProcessFrameResult`; sets `should_exit` if too few active points remain.

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

Returns `true` when the camera has moved or rotated enough since the last keyframe, or when too few points are tracked. Gated by `keyframe_min_frame_gap`.

**Translation trigger:**

$$\|\mathbf{t}_{\text{curr}} - \mathbf{t}_{\text{kf}}\| \geq 1.5 \text{ m}$$

**Rotation trigger:**

$$\theta = \arccos\!\left(\frac{\text{trace}(R_{\text{last}}^\top R_{\text{curr}}) - 1}{2}\right) \geq 8°$$

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
| **Emergency** | `!pose_accepted && (consecutive_rejected >= 2 \|\| num_active_tracks < 80)` |

---

### Other Methods

| Method | Description |
|---|---|
| `repeatLastPose()` | Appends last pose again (used when a frame fails to load) |
| `rejectPose(...)` | Appends last pose and calls `notePoseRejected` (used when PnP fails to converge) |
| `setActiveTracks(points, landmarks)` | Replaces active 2D/3D track state |
| `shouldSaveDenseDebug(frame_id, radius)` | Returns `true` within `radius` frames of the last pose rejection |
| `noteReinitialized(frame_id)` | Records reinit frame; resets `consecutive_rejected_poses_` |
| `noteKeyframeInserted(frame_id, pose)` | Updates `last_keyframe_frame_id_` and `last_keyframe_pose_wc_` |
| `poses()` | Const reference to full pose history |
| `currentPose()` | Last pose in history |
| `activePoints()` / `activeLandmarks()` | Access to active track state |

## See Also

- [`Estimator`](estimator.md) — produces `PoseEstimateResult` consumed inside `processFrame`
- [`StereoInitializer`](stereo_initializer.md) — provides `StereoInitResult` for bootstrap and reinit
- [`Map`](map.md) — stores keyframes and landmarks; updated from within `processFrame`
- [`RerunViewer`](viewer.md) — reads `poses()` and `activePoints()`
- [`PoseWriter`](pose_writer.md) — writes `poses()` at the end of the run
