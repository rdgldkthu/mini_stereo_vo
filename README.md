# mini_stereo_vo

![Tracking overlay](assets/tracking.gif)

Stereo visual odometry built from scratch in C++. Tracks a moving camera through KITTI sequences using stereo triangulation, pyramidal LK tracking, and PnP with Gauss-Newton pose refinement — no SLAM library, no shortcuts. A sliding-window map manages active landmarks and keyframes; a constant-velocity motion model seeds each PnP solve. The pipeline is visualized in real time with [Rerun](https://www.rerun.io/). A regression test gates every build on APE against KITTI seq 05.

`C++ 17` · `Eigen3` · `OpenCV` · `Rerun` · `CMake / Ninja` · `KITTI` · `evo`

![Trajectory](assets/trajectory.png)

---

## Pipeline

```
  Stereo image pair
         │
         ▼
  [Stereo Init]   ── ORB detect + match, row/disparity filter, triangulate ──▶  3D landmarks
         │
         ▼
   [Tracker]      ── pyramidal LK optical flow + forward-backward check ──────▶  2D tracks
         │
         ▼
  [Estimator]     ── PnP RANSAC → Gauss-Newton refinement (Huber loss) ───────▶  pose T_wc
         │
         ▼
  [Frontend]      ── quality gate, keyframe decision, stereo re-triangulation ▶  keyframe
         │
         ▼
    [Map]          ── sliding window: 5 keyframes · 2 000 landmarks ──────────▶  pruned map
         │
         ▼
[RerunViewer]     ── real-time 3D trajectory, landmarks, and metrics ─────────▶  display
```

Each stage is a self-contained module (`include/svo/`, `src/`). The main loop in `app/run_kitti.cpp` orchestrates them — no hidden global state.

**Pose refinement** — after PnP RANSAC, a custom Gauss-Newton optimizer re-solves pose on inliers only, using Huber loss to suppress residual outliers. The refined pose replaces the RANSAC result only if reprojection RMSE improves.

**Motion prediction** — the frontend seeds each PnP solve with a constant-velocity prediction computed from the two most recent accepted poses, falling back to the last known pose when needed.

**Failure recovery** — the frontend gates each pose on inlier count, inlier ratio, and motion magnitude. Consecutive rejections trigger stereo reinitialization: new landmarks are triangulated from the current frame and transformed into the world frame using the last valid pose, so the system recovers without losing global position.

---

## Results

| Tracking overlay | Inlier ratio over time |
|:---:|:---:|
| ![Tracking](assets/tracking.png) | ![Inlier ratio](assets/inlier_ratio.png) |

Runs through all 2 761 frames of KITTI seq 05. Median inlier ratio 0.91. Trajectory follows ground truth with expected long-range drift — pure VO without loop closure or global optimization.

**KITTI seq 05 — evo metrics**

| Metric | RMSE | Mean | Median |
|:---|---:|---:|---:|
| APE (m) | 8.17 | 7.22 | 6.75 |
| RPE (m/frame) | 0.069 | 0.041 | 0.028 |

**Runtime** — 13.8 ms/frame avg (Release build, `OPENCV_TRACE=0`, KITTI seq 05 on CPU)

---

## Quick Start

**Dependencies** — Eigen3, OpenCV (core, imgcodecs, imgproc, highgui, features2d, video, calib3d). Install everything including `evo` with:

```bash
bash scripts/bootstrap_ubuntu2404.sh
```

**Build and run:**

```bash
# build
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build -j

# run (KITTI seq 05) — trajectory written to results/traj/05_<YYMMDD>[_<keyword>].txt
OPENCV_TRACE=0 ./build/run_kitti data/kitti 05 [pose_keyword] [--save-debug] [--no-viewer] [--max-frames N] [--output-file PATH]

# run tests (unit geometry + regression against KITTI seq 05)
ctest --test-dir build

# evaluate APE / RPE — outputs plots to results/tables/05/
scripts/eval_kitti.sh 05 results/traj/<output>.txt

# regenerate README images
source .venv/bin/activate && python3 scripts/generate_vis.py --seq 05 --traj results/traj/<output>.txt

# regenerate tracking GIF (requires a prior --save-debug run)
source .venv/bin/activate && python3 scripts/make_tracking_gif.py --seq 05
```

---

## Repo Layout

```
mini_stereo_vo/
├── app/run_kitti.cpp          # entry point and pipeline orchestration
├── include/svo/               # module headers
├── src/                       # implementations
│   ├── camera.cpp
│   ├── dataset_kitti.cpp
│   ├── stereo_initializer.cpp
│   ├── tracker.cpp
│   ├── estimator.cpp
│   ├── frontend.cpp
│   ├── map.cpp
│   ├── pose_writer.cpp
│   └── rerun_viewer.cpp
├── tests/
│   └── test_geometry.cpp      # unit tests for triangulation and pose conversions
├── scripts/
│   ├── generate_vis.py        # generate README assets
│   ├── make_tracking_gif.py   # assemble debug frames into tracking GIF
│   ├── eval_kitti.sh          # APE / RPE evaluation via evo
│   ├── run_regression.sh      # regression gate: seq 05 APE threshold
│   └── bootstrap_ubuntu2404.sh
├── assets/                    # images used in this README
└── results/
    ├── traj/                  # output trajectory files
    ├── debug/                 # per-frame tracking PNGs + stats CSV
    └── tables/                # evo evaluation plots
```
