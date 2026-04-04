# mini_stereo_vo

A from-scratch stereo visual odometry project in C++17 on the KITTI odometry dataset.

## Current status

Day 1 milestone:

- Ubuntu 24.04 build works
- KITTI stereo pairs load correctly
- Left/right image viewer works
- A valid KITTI-format trajectory text file is written

The current trajectory output is a dummy identity trajectory used only to verify:

1. dataset loading
2. file writing
3. KITTI format compatibility with evaluation tools

## Scope

This project aims to implement a small, understandable stereo visual odometry system from scratch.

Planned components:

- dataset loader
- stereo triangulation
- temporal tracking
- pose estimation with PnP
- keyframe insertion
- sparse local map
- trajectory visualization
- quantitative evaluation

Non-goals for v1:

- loop closure
- ROS/ROS2
- IMU fusion
- global bundle adjustment
- dense mapping

## Dependencies

System packages:

- build-essential
- cmake
- ninja-build
- OpenCV
- Eigen

Vendored:

- Sophus (planned for pose representation, not yet linked in day 1 build)

Python:

- evo

## Dataset layout

Expected KITTI directory layout:

data/kitti/
├── poses/
│   └── 05.txt
└── sequences/
    └── 05/
        ├── calib.txt
        ├── times.txt
        ├── image_0/
        └── image_1/

## Build

```bash
cmake -S . -B build -G Ninja
cmake --build build -j
```

## Run

```bash
./build/run_kitti data/kitti 05 results/traj/05.txt
```

## Evaluate trajectory file format

```bash
source .venv/bin/activate
evo_traj kitti results/traj/05.txt --ref=data/kitti/poses/05.txt -p --plot_mode=xz
```

Note:

- The current output trajectory is not meaningful yet.
- This check is only to verify that the file format is valid and aligned frame-by-frame.

## Next step

Day2:

- parse callibration
- create stereo correspondences
- triangulate initial landmarks
