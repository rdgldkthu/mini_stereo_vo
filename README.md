# mini_stereo_vo

![Trajectory](assets/trajectory.png)

Stereo visual odometry built from scratch in C++. Tracks a moving camera through the KITTI benchmark — no SLAM library, no shortcuts.

`C++ 17` · `Eigen3` · `OpenCV` · `CMake / Ninja` · `KITTI` · `evo`

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
  [Local BA]      ── joint pose + landmark optimization every 2 keyframes ────▶  refined poses
```

Each stage is a self-contained module (`include/svo/`, `src/`). The main loop in `app/run_kitti.cpp` orchestrates them — no hidden global state.

**Pose refinement** — after PnP RANSAC, a custom Gauss-Newton optimizer re-solves pose on inliers only, using Huber loss to suppress residual outliers. The refined pose replaces the RANSAC result only if reprojection RMSE improves.

**Failure recovery** — the frontend gates each pose on inlier count, inlier ratio, and motion magnitude. Consecutive rejections trigger stereo reinitialization: new landmarks are triangulated from the current frame and transformed into the world frame using the last valid pose, so the system recovers without losing global position.

---

## Results

| Tracking overlay | Inlier ratio over time |
|:---:|:---:|
| ![Tracking](assets/tracking.png) | ![Inlier ratio](assets/inlier_ratio.png) |

Runs through all 1 117 frames of KITTI seq 05. Inlier ratio stays above 0.9 for the majority of the sequence. Trajectory follows ground truth with expected long-range drift — pure VO without loop closure or global optimization.

---

## Quick Start

**Dependencies** — Eigen3, OpenCV (core, imgcodecs, imgproc, highgui, features2d, video, calib3d). Install everything including `evo` with:

```bash
bash scripts/bootstrap_ubuntu2404.sh
```

**Build and run:**

```bash
# build
cmake -S . -B build -G Ninja && cmake --build build -j

# run (KITTI seq 05)
./build/run_kitti data/kitti 05 results/traj/05_vo.txt

# evaluate APE / RPE — outputs plots to results/tables/05/
scripts/eval_kitti.sh 05 results/traj/05_vo.txt

# regenerate README images
source .venv/bin/activate && python3 scripts/generate_vis.py --seq 05 --traj results/traj/05_vo.txt
```

---

## Repo Layout

```
mini_stereo_vo/
├── app/run_kitti.cpp          # entry point and pipeline orchestration
├── include/svo/               # module headers
├── src/                       # implementations
│   ├── stereo_initializer.cpp
│   ├── tracker.cpp
│   ├── estimator.cpp
│   ├── frontend.cpp
│   ├── map.cpp
│   └── viewer.cpp
├── scripts/
│   ├── generate_vis.py        # generate README assets
│   ├── eval_kitti.sh          # APE / RPE evaluation via evo
│   └── bootstrap_ubuntu2404.sh
├── assets/                    # images used in this README
└── results/
    ├── traj/                  # output trajectory files
    ├── debug/                 # per-frame tracking PNGs + stats CSV
    └── tables/                # evo evaluation plots
```
