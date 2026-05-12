# mini_stereo_vo — Project Overview

A minimal stereo visual odometry system written in C++17. It runs on KITTI odometry sequences and estimates a 6-DoF camera trajectory using ORB-based stereo initialisation, pyramidal LK optical flow tracking, PnP-RANSAC pose estimation, and local bundle adjustment.

---

## Repository Layout

```
mini_stereo_vo/
├── app/
│   └── run_kitti.cpp           Main entry point (611 lines)
├── include/svo/                Public headers (svo namespace)
│   ├── camera.h
│   ├── dataset_kitti.h
│   ├── estimator.h
│   ├── feature.h
│   ├── frame.h
│   ├── frontend.h
│   ├── map.h
│   ├── map_point.h
│   ├── pose_writer.h
│   ├── stereo_initializer.h
│   ├── tracker.h
│   └── viewer.h
├── src/                        Implementations
│   ├── camera.cpp
│   ├── dataset_kitti.cpp
│   ├── estimator.cpp           (671 lines — largest file)
│   ├── frontend.cpp
│   ├── map.cpp
│   ├── pose_writer.cpp
│   ├── stereo_initializer.cpp
│   ├── tracker.cpp
│   └── viewer.cpp
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
cmake -S . -B build -G Ninja
cmake --build build -j
```

**Dependencies:** Eigen3, OpenCV (core, imgcodecs, imgproc, highgui, features2d, video, calib3d), C++17.

Install on Ubuntu 24.04:

```bash
scripts/bootstrap_ubuntu2404.sh
```

The `svo_core` static library is linked into the `run_kitti` executable. All headers are under `include/svo/`; all sources under `src/`.

---

## Run

```bash
./build/run_kitti data/kitti 05 [keyword] [--save-debug]
```

| Argument | Description |
|---|---|
| `data/kitti` | KITTI odometry root (contains `sequences/` and `poses/`) |
| `05` | Sequence number |
| `keyword` | Optional label appended to output filenames (e.g. `vo`) |
| `--save-debug` | Save init-match PNG and tracking-overlay PNGs to `results/debug/` |

**Outputs:**

| File | Description |
|---|---|
| `results/traj/<seq>_<YYMMDD>[_<kw>].txt` | KITTI-format trajectory (one 3×4 matrix per line) |
| `results/debug/<stem>_stats.csv` | Per-frame pipeline statistics (always written) |
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
DatasetKitti ──── loadFrame() ──────────────────────────────────────┐
                                                                     │
Camera ──── loadFromKittiCalib() ──────────────────────────────┐    │
                                                               │    │
                                                               ▼    ▼
                                                         StereoInitializer
                                                               │
                                              features[]  ◄───┤
                                              landmarks[] ◄───┘
                                                    │
                              ┌─────────────────────▼─────────────────────┐
                              │               Frontend                     │
                              │  poses[]  active_points[]  active_landmarks│
                              └────────────────────┬──────────────────────┘
                                                   │
                        ┌──────────────────────────▼──────────────────────────┐
                        │   run_kitti.cpp (main loop, frame_id = 1..N)        │
                        │                                                     │
                        │  Tracker ──→ TrackResult (object_points, image_pts) │
                        │      │                                              │
                        │      ▼                                              │
                        │  Estimator::estimatePosePnPRansac                  │
                        │      │                                              │
                        │      ▼                                              │
                        │  Estimator::refinePosePoseOnly                     │
                        │      │                                              │
                        │      ▼                                              │
                        │  Frontend::acceptPose / rejectPose                 │
                        │      │                                              │
                        │      ├── shouldReinitialize? → StereoInitializer   │
                        │      │                                              │
                        │      └── needNewKeyframe? ──────────────────────┐  │
                        │                                                  │  │
                        │                          Map::addKeyframe        │  │
                        │                          StereoInitializer       │  │
                        │                          Map::addLandmarks        │  │
                        │                          Estimator::runLocalBA   │  │
                        │                          Frontend::refresh...    │  │
                        │                                                  │  │
                        └──────────────────────────────────────────────────┘  │
                                                                              │
                              Viewer ──────────────────────────────────────── │
                              PoseWriter (final write) ──────────────────────-┘
```

---

## Pose Convention

All poses are stored as **T_wc** (world-from-camera / camera-in-world):

$$T_{wc} = \begin{pmatrix} R_{wc} & \mathbf{t}_{wc} \\ \mathbf{0}^\top & 1 \end{pmatrix} \in SE(3)$$

where $\mathbf{t}_{wc}$ is the camera origin expressed in world coordinates.

`solvePnPRansac` returns `(R_cw, t_cw)`. Conversion:

$$R_{wc} = R_{cw}^\top, \qquad \mathbf{t}_{wc} = -R_{wc} \cdot \mathbf{t}_{cw}$$

Implemented in `run_kitti.cpp`:

```cpp
Eigen::Matrix4d makePoseWcFromPnP(const Eigen::Matrix3d& R_cw,
                                   const Eigen::Vector3d& t_cw) {
    Eigen::Matrix4d T_wc = Eigen::Matrix4d::Identity();
    T_wc.block<3,3>(0,0) = R_cw.transpose();
    T_wc.block<3,1>(0,3) = -R_cw.transpose() * t_cw;
    return T_wc;
}
```

---

## Pipeline Walkthrough

### Bootstrap (frame 0)

```
loadFrame(0)
StereoInitializer::run()        → features[], landmarks[] (camera frame)
makeInitialActiveLandmarks()    → observed=1, tracked=1, missed=0
Map::assignNewLandmarkIds()     → ids 0..N-1
Frontend::initialize()          → poses=[I₄], active_points, active_landmarks
Map::addKeyframe(frame0)
Map::setActiveLandmarks()
→ Write stats CSV row 0
```

Frame 0's landmarks are in **camera frame** (which coincides with world frame since `T_wc = I₄`).

### Main Loop (frames 1..N)

For each frame:

```
1. DatasetKitti::loadFrame(i)

2. Tracker::trackFrameToFrame(prev_frame, curr_frame,
       frontend.activePoints(), frontend.activeLandmarks())
   → TrackResult: object_points (3D), image_points (2D), landmark_ids

3. If num_valid_correspondences >= 6:
     Estimator::estimatePosePnPRansac(...)   → R_cw, t_cw, inliers
     If success && inliers >= 10:
       Estimator::refinePosePoseOnly(inliers) → refined R_cw, t_cw
     Frontend::acceptPose(...)               → T_wc appended to poses_
   Else:
     Frontend::rejectPose(...)               → last pose repeated

4. Reinitialization check (Frontend::shouldReinitialize):
     If yes: StereoInitializer::run(curr_frame)
             transformLandmarksToWorld(new_lm, current_pose)
             Frontend::setActiveTracks(new_points, new_lm)
             Map::setActiveLandmarks(new_lm)
     Else:   Frontend::setActiveTracks(track_result.curr_points, ...)
             Map::markTrackedLandmarks / markMissedLandmarks / pruneLandmarks

5. Keyframe insertion (if pose_accepted && Frontend::needNewKeyframe):
     Map::addKeyframe(curr_frame)
     StereoInitializer::run(curr_frame)      → new_lm (camera frame)
     transformLandmarksToWorld(new_lm)
     Map::addLandmarks(new_lm)
     Frontend::setActiveTracks(new_points, new_lm)

     Local BA (if insertedKeyframesSinceLastBa >= 2 && kf >= 3 && lm >= 20):
       backup kf/lm vectors
       Estimator::runLocalBundleAdjustment(map.kf, map.lm)
       If rmse_after <= rmse_before:
         Frontend::refreshActiveLandmarksFromMap(map.lm)
         Frontend::noteLocalBaAccepted()
       Else:
         restore from backup
     Frontend::noteKeyframeInserted(frame_id, pose)

6. Log stats to CSV, optionally save debug PNGs
7. Update Viewer; check for exit key

8. Frontend::setPreviousFrame(curr_frame)
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

transformLandmarksToWorld(lm, T_wc)
  p_w = R_wc * p_w + t_wc                 ← world frame

Map::addLandmarks / setActiveLandmarks
  p_w stored in Map::active_landmarks_

Frontend::setActiveTracks / refreshActiveLandmarksFromMap
  p_w mirrored in Frontend::active_landmarks_

Tracker::trackFrameToFrame
  object_points[i] = lm[i].p_w            ← fed to PnP

Estimator::runLocalBundleAdjustment
  p_w refined and written back to Map
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
| StereoInitializer | `max_features` | 1500 |
| StereoInitializer | `hamming_threshold` | 40 |
| StereoInitializer | `row_tolerance_px` | 2.0 |
| StereoInitializer | `min_disparity_px` | 3.0 |
| StereoInitializer | `max_disparity_px` | 120.0 |
| StereoInitializer | `max_depth_m` | 80.0 |
| Tracker | `win_size` | 21×21 |
| Tracker | `max_level` | 3 |
| Tracker | `max_bidirectional_error_px` | 1.5 |
| Estimator | `iterations_count` (RANSAC) | 100 |
| Estimator | `reprojection_error_px` | 4.0 |
| Estimator | `confidence` | 0.99 |
| Estimator | `pose_refine_iterations` | 10 |
| Estimator | `pose_refine_huber_delta` | 5.0 |
| Estimator | `local_ba_iterations` | 3 |
| Estimator | `local_ba_damping` | 1e-3 |
| Estimator | `max_ba_keyframes` | 3 |
| Estimator | `max_ba_landmarks` | 100 |
| Estimator | `min_ba_observations` | 20 |
| Frontend | `keyframe_translation_threshold_m` | 1.5 |
| Frontend | `keyframe_rotation_threshold_deg` | 12.0 |
| Frontend | `keyframe_min_tracked_points` | 60 |
| Frontend | `keyframe_min_frame_gap` | 5 |
| Frontend | `min_pose_inliers` | 15 |
| Frontend | `min_pose_inlier_ratio` | 0.10 |
| Frontend | `max_frame_translation_m` | 2.0 |
| Frontend | `weak_track_threshold` | 80 |
| Frontend | `emergency_rejected_poses_count` | 2 |
| Frontend | `local_ba_keyframe_interval` | 2 |
| Map | `max_active_keyframes` | 5 |
| Map | `max_active_landmarks` | 2000 |
| Map | `max_missed_times` | 8 |
| Viewer | `trajectory_scale` | 0.5 px/m |

---

## Stats CSV Schema

Written to `results/debug/<stem>_stats.csv` for every run:

```
frame_id, num_active_points, num_correspondences, num_inliers,
inlier_ratio, pose_success, pose_accepted, reinitialized, is_keyframe,
num_keyframes, num_map_landmarks, local_ba, local_ba_accepted,
local_ba_rejected, ba_rmse_before, ba_rmse_after,
tx, ty, tz, delta_t, rmse_before, rmse_after
```

---

## Module Documentation Index

| Module | File | Description |
|---|---|---|
| Camera | [camera.md](camera.md) | Intrinsics, projection matrices, triangulation |
| DatasetKitti | [dataset_kitti.md](dataset_kitti.md) | KITTI sequence I/O |
| Estimator | [estimator.md](estimator.md) | PnP RANSAC, Gauss-Newton refinement, local BA |
| Feature | [feature.md](feature.md) | Stereo keypoint pair struct |
| Frame | [frame.md](frame.md) | Per-frame data carrier |
| Frontend | [frontend.md](frontend.md) | Pipeline gatekeeper, pose history, reinit logic |
| Map | [map.md](map.md) | Sliding-window keyframe and landmark storage |
| MapPoint | [map_point.md](map_point.md) | 3D landmark struct |
| PoseWriter | [pose_writer.md](pose_writer.md) | KITTI trajectory file output |
| StereoInitializer | [stereo_initializer.md](stereo_initializer.md) | ORB stereo matching and triangulation |
| Tracker | [tracker.md](tracker.md) | Pyramidal LK optical flow with FB check |
| Viewer | [viewer.md](viewer.md) | Real-time OpenCV visualisation |
