# Estimator

**Header:** `include/svo/estimator.h`  
**Source:** `src/estimator.cpp`

## Role

`Estimator` is the numerical core of the pipeline. It provides two operations:

1. **`estimatePosePnPRansac`** — robust pose estimation from 3D–2D correspondences via OpenCV's `solvePnPRansac`.
2. **`refinePosePoseOnly`** — Ceres-based pose-only refinement on the inlier set, using Sophus SE(3) manifold parameterisation with Huber loss and autodiff Jacobians.

Both methods return result structs and do not mutate shared state. `Frontend::processFrame` decides whether to accept the results.

---

## Result Struct

### `PoseEstimateResult`

```cpp
struct PoseEstimateResult {
    bool success = false;

    std::vector<int> inlier_indices;  // indices into object/image_points

    Eigen::Matrix3d rotation    = Eigen::Matrix3d::Identity();  // R_cw
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();      // t_cw

    int    num_object_points = 0;
    int    num_image_points  = 0;
    int    num_inliers       = 0;

    double reprojection_rmse_before = 0.0;  // px, on inliers before refinement
    double reprojection_rmse_after  = 0.0;  // px, after Gauss-Newton refinement
};
```

`rotation` and `translation` encode `(R_cw, t_cw)` — camera-from-world. The caller converts to `T_wc` via `geometry.h::poseWcFromCw`.

---

## Options

```cpp
struct Options {
    // PnP RANSAC
    bool   use_extrinsic_guess   = false;
    int    iterations_count      = 100;
    float  reprojection_error_px = 4.0f;
    double confidence            = 0.99;
    int    min_pnp_points        = 6;

    // Pose-only refinement
    int    pose_refine_iterations  = 10;
    double pose_refine_epsilon     = 1e-6;
    double pose_refine_huber_delta = 5.0;
    int    min_refine_inliers      = 10;
};
```

Runtime values set in `run_kitti.cpp` (all others match struct defaults):
- `reprojection_error_px = 3.0f` (struct default: 4.0f)

---

## `estimatePosePnPRansac`

### Signatures

```cpp
// No initial guess
PoseEstimateResult estimatePosePnPRansac(
    const std::vector<Eigen::Vector3d>& object_points,
    const std::vector<cv::Point2f>&     image_points,
    const Camera&                        camera) const;

// With initial guess (used by Frontend::processFrame via constant-velocity model)
PoseEstimateResult estimatePosePnPRansac(
    const std::vector<Eigen::Vector3d>& object_points,
    const std::vector<cv::Point2f>&     image_points,
    const Camera&                        camera,
    const Eigen::Matrix3d& initial_rotation_cw,
    const Eigen::Vector3d& initial_translation_cw,
    bool use_initial_guess) const;
```

The no-guess overload calls the second form with identity and `use_initial_guess = false`.

### Algorithm

1. Convert `Eigen::Vector3d` → `cv::Point3f` (float precision).
2. Build `K` from `camera.fx, fy, cx, cy`; `dist_coeffs = 0` (rectified).
3. If `use_initial_guess`: convert `initial_rotation_cw` to Rodrigues and fill `rvec`/`tvec`.
4. Call `cv::solvePnPRansac(..., cv::SOLVEPNP_ITERATIVE)`.
5. Convert output `rvec` back to `Eigen::Matrix3d` via `cv::Rodrigues`.
6. Compute `reprojection_rmse_before` over all inlier 3D–2D pairs.

**Returns** with `success = false` if fewer than `min_pnp_points = 6` correspondences or if `solvePnPRansac` returns `false`.

---

## `refinePosePoseOnly`

Ceres-based pose-only optimisation over the PnP inlier set. The pose is parameterised as a `Sophus::SE3d` with the `Sophus::Manifold<Sophus::SE3>` manifold, so Ceres handles the retraction automatically. Jacobians are computed via **autodiff** — no hand-rolled Lie algebra derivatives.

### Cost Functor

```cpp
struct ReprojectionCostFunctor {
    // Residual: [fx*(px/pz)+cx - obs_x, fy*(py/pz)+cy - obs_y]  (2 components)
    template <typename T>
    bool operator()(const T* const pose_data, T* residuals) const;
};
```

The functor maps `Sophus::SE3<T>` (7 parameters, internal Sophus layout) to a 2-component pixel residual. Points behind the camera (`p_c[2] <= 1e-8`) return zero residuals.

### Solver Configuration

| Parameter | Value |
|---|---|
| Loss function | `ceres::HuberLoss(pose_refine_huber_delta)` (default 5.0 px) |
| Linear solver | `ceres::DENSE_QR` |
| Max iterations | `pose_refine_iterations` (default 10) |
| Function tolerance | `pose_refine_epsilon` (default 1e-6) |
| Manifold | `Sophus::Manifold<Sophus::SE3>` |

Refinement is skipped when `object_points.size() < min_refine_inliers` (default 10).

On failure (`!summary.IsSolutionUsable()`), returns an unsuccessful `PoseEstimateResult`.

---

## Internal Helper Functions (anonymous namespace)

| Function | Description |
|---|---|
| `ReprojectionCostFunctor` | Ceres AutoDiff cost functor; maps `Sophus::SE3<T>` to 2-D pixel residual |
| `projectPoint(p_w, R_cw, t_cw, cam, pixel, p_c)` | Projects 3D point; returns `false` if behind camera |
| `computeReprojectionRmse(...)` | RMSE over a set of 3D–2D correspondences |

## See Also

- [`Camera`](camera.md) — provides `fx, fy, cx, cy` for Jacobian computation
- [`Tracker`](tracker.md) — supplies `object_points` / `image_points` for PnP
- [`Frontend`](frontend.md) — calls `estimatePosePnPRansac` and `refinePosePoseOnly` inside `processFrame`
