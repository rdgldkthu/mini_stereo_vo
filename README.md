# mini_stereo_vo

A from-scratch stereo visual odometry project in C++17 using the KITTI odometry dataset.

## Goal

Build a small and understandable stereo visual odometry system from scratch, with clean code structure, visual debugging, and quantitative evaluation.

This project focuses on:

- understanding the full VO pipeline by implementing it directly
- keeping the system small enough to finish
- producing a clean portfolio-quality demo and repo

## Current status

Implemented:

- KITTI stereo dataset loader
- KITTI calibration parsing
- rectified stereo triangulation helper
- ORB-based stereo feature detection and matching
- row/disparity filtering for stereo correspondences
- initial sparse landmark triangulation from a stereo pair
- debug visualization and statistics export
- KITTI-format trajectory file export scaffold

Not implemented yet:

- temporal tracking between frames
- pose estimation with PnP
- keyframe insertion
- active map management
- local optimization / bundle adjustment
- loop closure

## Repository structure

```text
stereo-vo/
├── app/
│   └── run_kitti.cpp
├── include/
│   └── svo/
│       ├── camera.h
│       ├── dataset_kitti.h
│       ├── feature.h
│       ├── frame.h
│       ├── map_point.h
│       └── stereo_initializer.h
├── src/
│   ├── camera.cpp
│   ├── dataset_kitti.cpp
│   └── stereo_initializer.cpp
├── results/
│   ├── debug/
│   └── traj/
├── data/
├── third_party/
├── CMakeLists.txt
└── README.md
```

## Dependencies

System packages:

- build-essential
- cmake
- ninja-build
- OpenCV
- Eigen

Optional Python tools:

- evo

## Environment

Target platform:

- Ubuntu 24.04
- C++17
- CMake
- OpenCV 4
- Eigen 3

## Dataset

This project currently uses the KITTI odometry dataset with grayscale stereo images.

Expected directory layout:

```text
data/kitti/
├── poses/
│   └── 05.txt
└── sequences/
    └── 05/
        ├── calib.txt
        ├── times.txt
        ├── image_0/
        └── image_1/
```

## Build

```bash
cmake -S . -B build -G Ninja
cmake --build build -j
```

## Run

```bash
./build/run_kitti data/kitti 05 results/traj/05.txt
```

## Current runtime behavior

The current executable does the following:

1. opens a KITTI stereo sequence
2. loads calibration from `calib.txt`
3. loads the first stereo pair
4. detects ORB features in left and right images
5. matches stereo correspondences
6. filters matches using row and disparity constraints
7. triangulates initial 3D landmarks
8. saves debug outputs
9. writes a KITTI-format trajectory file scaffold

## Output files

After running, the following files are generated:

```text
results/debug/05_init_matches.png
results/debug/05_init_stats.txt
results/debug/05_init_points.txt
results/traj/05.txt
```

### `05_init_matches.png`

Stereo match visualization with summary statistics overlaid.

### `05_init_stats.txt`

Contains:

- number of keypoints
- number of raw matches
- number of filtered matches
- number of triangulated landmarks
- disparity statistics
- row error statistics
- depth statistics

### `05_init_points.txt`

Contains triangulated landmark data in the form:

```text
id x y z ul vl ur vr disparity
```

### `05.txt`

Current trajectory output is a placeholder identity trajectory written in KITTI pose format to keep the export/evaluation path ready.

## Current implementation notes

### Calibration

The system parses KITTI projection matrices and extracts:

- `fx`
- `fy`
- `cx`
- `cy`
- stereo baseline

### Stereo geometry

Because the KITTI stereo images are rectified, depth is recovered from disparity using:

```text
Z = fx * baseline / disparity
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
```

### Stereo initialization

The current stereo initializer:

- uses ORB for feature detection and descriptor extraction
- uses brute-force Hamming matching with cross-check
- enforces row consistency
- rejects invalid disparity ranges
- triangulates a sparse set of valid landmarks

## Limitations of the current version

This is not yet full visual odometry.

The system currently initializes landmarks from a single stereo pair, but it does not yet:

- track them across time
- estimate camera motion
- maintain an active map
- recover trajectory from frame-to-frame motion

## Next steps

Planned next components:

- temporal feature tracking between consecutive left frames
- 3D-2D correspondence construction
- pose estimation with PnP + RANSAC
- keyframe insertion policy
- local map maintenance
- trajectory visualization and evaluation

## Notes

This project intentionally prioritizes:

- clarity over feature completeness
- direct implementation over wrapping a large SLAM framework
- a finishable VO system over a broad but shallow keyword-driven project
