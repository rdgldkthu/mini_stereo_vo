# mini_stereo_vo — Project Overview

A minimal stereo visual odometry system written in C++17. It runs on KITTI odometry sequences and estimates a 6-DoF camera trajectory using ORB-based stereo initialisation, pyramidal LK optical flow tracking, and PnP-RANSAC pose estimation with Ceres/Sophus SE(3) pose refinement.

---

## Repository Layout

```
mini_stereo_vo/
├── app/
│   └── run_kitti.cpp           Main entry point (303 lines)
├── include/svo/                Public headers (svo namespace)
│   ├── camera.h
│   ├── dataset_kitti.h
│   ├── estimator.h
│   ├── feature.h
│   ├── frame.h
│   ├── frontend.h
│   ├── geometry.h
│   ├── map.h
│   ├── map_point.h
│   ├── pose_writer.h
│   ├── rerun_viewer.h
│   ├── stereo_initializer.h
│   ├── tracker.h
│   └── viewer_status.h
├── src/                        Implementations
│   ├── camera.cpp
│   ├── dataset_kitti.cpp
│   ├── estimator.cpp           (257 lines)
│   ├── frontend.cpp            (462 lines — largest file)
│   ├── map.cpp
│   ├── pose_writer.cpp
│   ├── rerun_viewer.cpp
│   ├── stereo_initializer.cpp
│   └── tracker.cpp
├── scripts/
│   ├── bootstrap_ubuntu2404.sh
│   ├── eval_kitti.sh
│   └── generate_vis.py
├── docs/                       ← you are here
├── results/
│   ├── traj/                   Estimated trajectories (.txt)
│   ├── debug/                  Per-frame stats CSV + optional PNGs
│   └── tables/                 evo evaluation plots and tables
└── data/kitti/                 Dataset (not tracked)
```

---

## Build

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

**Dependencies:** Eigen3, OpenCV (core, imgcodecs, imgproc, highgui, features2d, video, calib3d), rerun_sdk (v0.22.1, fetched automatically via CMake FetchContent), C++17.

Install on Ubuntu 24.04:

```bash
scripts/bootstrap_ubuntu2404.sh
```

The `svo_core` static library is linked into the `run_kitti` executable. All headers are under `include/svo/`; all sources under `src/`.

---

## Run

```bash
OPENCV_TRACE=0 ./build/run_kitti data/kitti 05 [pose_keyword] [--save-debug] [--no-viewer] [--max-frames N] [--output-file PATH]
```

| Argument | Description |
|---|---|
| `data/kitti` | KITTI odometry root (contains `sequences/` and `poses/`) |
| `05` | Sequence number |
| `pose_keyword` | Optional label appended to output filenames (e.g. `vo`) |
| `--save-debug` | Save init-match PNG and tracking-overlay PNGs to `results/debug/` |
| `--no-viewer` | Skip the Rerun visualisation window |
| `--max-frames N` | Stop after N frames (useful for regression tests) |
| `--output-file PATH` | Override the default trajectory output path |

**Outputs:**

| File | Description |
|---|---|
| `results/traj/<seq>_<YYMMDD>[_<kw>].txt` | KITTI-format trajectory (one 3×4 matrix per line) |
| `results/debug/<stem>_stats.csv` | Per-frame pipeline statistics (always written) |
| `results/time_log.csv` | Timing summary appended per run |
| `results/debug/<stem>_init_matches.png` | Stereo match visualisation (with `--save-debug`) |
| `results/debug/<stem>_track_<NNNNNN>.png` | Tracking overlay (with `--save-debug`, sampled) |

---

## Evaluate

```bash
scripts/eval_kitti.sh 05 results/traj/05_260512_vo.txt
```

Uses `evo` (installed into `.venv`). Outputs APE/RPE plots and tables under `results/tables/<seq>/` and appends a summary row to `results/metrics_log.csv`.

---

## System Architecture

### Module Graph

```
DatasetKitti ──── loadFrame() ──────────────────────────────────────────────┐
                                                                             │
Camera ──── loadFromKittiCalib() ──────────────────────────────────────┐    │
                                                                        │    │
                                              Bootstrap (frame 0):     │    │
                                              frontend.bootstrap(...)  │    │
                                                └─ StereoInitializer   │    │
                                                └─ Map::addKeyframe    │    │
                                                └─ Map::setActiveLm    │    │
                                                                        │    │
                              ┌────────────────────────────────────────▼────▼──┐
                              │            Main loop (frames 1..N)             │
                              │                                                │
                              │  r = frontend.processFrame(frame_id, frame,    │
                              │          tracker, estimator, initializer,      │
                              │          map, camera)                          │
                              │                                                │
                              │    Tracker::trackFrameToFrame → TrackResult    │
                              │        │                                       │
                              │        ▼                                       │
                              │    Estimator::estimatePosePnPRansac            │
                              │        │                                       │
                              │        ▼                                       │
                              │    Estimator::refinePosePoseOnly               │
                              │        │                                       │
                              │        ▼                                       │
                              │    acceptPose / rejectPose                     │
                              │        │                                       │
                              │        ├── shouldReinitialize?                 │
                              │        │     └─ StereoInitializer::run()       │
                              │        │     └─ Map::setActiveLandmarks        │
                              │        │                                       │
                              │        └── needNewKeyframe?                    │
                              │              └─ Map::markKeyframeObservations  │
                              │              └─ StereoInitializer::run()       │
                              │              └─ Map::addKeyframe + addLandmarks│
                              │                                                │
                              └────────────────────────────────────────────────┘
                                          │
                                          ▼
                              RerunViewer::update (if not --no-viewer)
                              PoseWriter::writeKittiTrajectory (final)
```

---

## Pose Convention

All poses are stored as **T_wc** (world-from-camera / camera-in-world):

$$T_{wc} = \begin{pmatrix} R_{wc} & \mathbf{t}_{wc} \\ \mathbf{0}^\top & 1 \end{pmatrix} \in SE(3)$$

where $\mathbf{t}_{wc}$ is the camera origin expressed in world coordinates.

`solvePnPRansac` returns `(R_cw, t_cw)`. Conversion is provided by `geometry.h`:

```cpp
// Build T_wc from PnP outputs
Eigen::Matrix4d poseWcFromCw(const Eigen::Matrix3d& R_cw,
                              const Eigen::Vector3d& t_cw);

// Extract (R_cw, t_cw) from T_wc
void poseCwFromWc(const Eigen::Matrix4d& T_wc,
                  Eigen::Matrix3d& R_cw, Eigen::Vector3d& t_cw);
```

---

## Pipeline Walkthrough

### Bootstrap (frame 0)

```
frontend.bootstrap(frame0, initializer, map, camera, save_debug)
  StereoInitializer::run(frame0)   → features[], landmarks[] (camera frame)
  makeInitialActiveLandmarks()     → tracked_frames=1, keyframe_observations=1
  Map::assignNewLandmarkIds()      → ids 0..N-1
  Frontend::initialize()           → poses=[I₄], active_points, active_landmarks
  Map::addKeyframe(frame0)
  Map::setActiveLandmarks(lms)
→ Write stats CSV row 0
```

Frame 0's landmarks are in **camera frame** (which coincides with world frame since `T_wc = I₄`).

### Main Loop (frames 1..N)

```
1. DatasetKitti::loadFrame(i) → curr_frame
   (on failure: frontend.repeatLastPose(), write CSV row, continue)

2. r = frontend.processFrame(frame_id, curr_frame,
                               tracker, estimator, initializer,
                               map, camera, save_debug)

   ── Inside processFrame ─────────────────────────────────────
   a. Tracker::trackFrameToFrame(prev_frame, curr_frame,
          active_points, active_landmarks, motion_hint)
      → TrackResult: object_points (3D), image_points (2D)

   b. Constant-velocity seed:
        T_pred = T_curr * T_prev⁻¹ * T_curr  (when ≥2 accepted poses)
        Falls back to last pose otherwise.

   c. If num_valid_correspondences ≥ 6:
        Estimator::estimatePosePnPRansac(...)   → R_cw, t_cw, inliers
        If success && inliers ≥ 10:
          Estimator::refinePosePoseOnly(inliers) → refined R_cw, t_cw
        acceptPose(...)   → T_wc appended to poses_
      Else:
        rejectPose(...)   → last pose repeated

   d. Filter tracked points to PnP inliers; mark outliers in Map.

   e. shouldReinitialize(frame_id, pose_accepted, num_tracks)?
        If yes: StereoInitializer::run(curr_frame)
                transformLandmarksToWorld(new_lm, current_pose)
                Map::setActiveLandmarks(new_lm)
                Frontend::setActiveTracks(new_pts, new_lm)
        Else:   Frontend::setActiveTracks(culled_pts, culled_lms)
                Map::markTracked/Missed/pruneLandmarks

   f. If pose_accepted && needNewKeyframe:
        Map::markKeyframeObservations(culled_ids)
        StereoInitializer::run(curr_frame)   → new_lm (camera frame)
        transformLandmarksToWorld(new_lm)
        Map::addKeyframe + addLandmarks
        Frontend::setActiveTracks(new_pts, new_lm)
        Frontend::noteKeyframeInserted(frame_id, pose)

   g. result.should_exit = (active_points < 6)
   ── End processFrame ────────────────────────────────────────

3. Write CSV row; save debug PNGs if --save-debug
4. Build ViewerStatus; call RerunViewer::update (if not --no-viewer)
5. Check r.should_exit; break if true
```

### Final Output

```
PoseWriter::writeKittiTrajectory(output_pose, frontend.poses())
```

---

## Key Data Flows

### 3D Landmark Lifecycle

```
StereoInitializer::run()
  p_w = pixel2Camera(ul, vl, depth)       ← camera frame

transformLandmarksToWorld(lms, T_wc)      ← inside frontend.cpp
  p_w = R_wc * p_w + t_wc                 ← world frame

Map::addLandmarks / setActiveLandmarks
  p_w stored in Map::active_landmarks_

Frontend::setActiveTracks
  p_w mirrored in Frontend::active_landmarks_

Tracker::trackFrameToFrame
  object_points[i] = lm[i].p_w            ← fed to PnP
```

### Pose History

`Frontend::poses_` grows by one entry per frame:
- Accepted frames: new `T_wc`
- Rejected frames: repeat of `poses_.back()`

At sequence end, `poses_` has exactly `N` entries (one per frame processed), matching the KITTI ground-truth file line count.

---

## Tuned Parameters (from `run_kitti.cpp`)

| Module | Parameter | Value |
|---|---|---|
| StereoInitializer | `max_features` | 1000 |
| StereoInitializer | `hamming_threshold` | 40 |
| StereoInitializer | `row_tolerance_px` | 2.0 |
| StereoInitializer | `min_disparity_px` | 3.0 |
| StereoInitializer | `max_disparity_px` | 120.0 |
| StereoInitializer | `max_depth_m` | 80.0 |
| StereoInitializer | `grid_rows` | 4 |
| StereoInitializer | `grid_cols` | 8 |
| StereoInitializer | `max_per_cell` | 10 |
| Tracker | `win_size` | 25×25 |
| Tracker | `max_level` | 4 |
| Tracker | `max_bidirectional_error_px` | 1.5 |
| Estimator | `iterations_count` (RANSAC) | 100 |
| Estimator | `reprojection_error_px` | 3.0 |
| Estimator | `confidence` | 0.99 |
| Estimator | `pose_refine_iterations` | 10 |
| Estimator | `pose_refine_huber_delta` | 5.0 |
| Frontend | `keyframe_translation_threshold_m` | 1.5 |
| Frontend | `keyframe_rotation_threshold_deg` | 8.0 |
| Frontend | `keyframe_min_tracked_points` | 60 |
| Frontend | `keyframe_min_frame_gap` | 5 |
| Frontend | `min_pose_inliers` | 15 |
| Frontend | `min_pose_inlier_ratio` | 0.10 |
| Frontend | `max_frame_translation_m` | 2.0 |
| Frontend | `weak_track_threshold` | 80 |
| Frontend | `emergency_rejected_poses_count` | 2 |
| Map | `max_active_keyframes` | 5 |
| Map | `max_active_landmarks` | 2000 |
| Map | `max_missed_times` | 8 |

---

## Stats CSV Schema

Written to `results/debug/<stem>_stats.csv` for every run:

```
frame_id, num_active_points, num_correspondences, num_inliers,
inlier_ratio, pose_success, pose_accepted, reinitialized, is_keyframe,
num_keyframes, num_map_landmarks,
tx, ty, tz, delta_t, rmse_before, rmse_after
```

---

## Module Documentation Index

| Module | File | Description |
|---|---|---|
| Camera | [camera.md](camera.md) | Intrinsics, projection matrices, triangulation |
| DatasetKitti | [dataset_kitti.md](dataset_kitti.md) | KITTI sequence I/O |
| Estimator | [estimator.md](estimator.md) | PnP RANSAC and Ceres/Sophus SE(3) pose refinement |
| Feature | [feature.md](feature.md) | Stereo keypoint pair struct |
| Frame | [frame.md](frame.md) | Per-frame data carrier |
| Frontend | [frontend.md](frontend.md) | Pipeline gatekeeper, pose history, reinit logic |
| Map | [map.md](map.md) | Sliding-window keyframe and landmark storage |
| MapPoint | [map_point.md](map_point.md) | 3D landmark struct |
| PoseWriter | [pose_writer.md](pose_writer.md) | KITTI trajectory file output |
| StereoInitializer | [stereo_initializer.md](stereo_initializer.md) | ORB stereo matching and triangulation |
| Tracker | [tracker.md](tracker.md) | Pyramidal LK optical flow with FB check |
| RerunViewer | [viewer.md](viewer.md) | Real-time 3D visualisation using Rerun SDK |
