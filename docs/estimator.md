# Estimator

**Header:** `include/svo/estimator.h`  
**Source:** `src/estimator.cpp`

## Role

`Estimator` is the numerical core of the pipeline. It provides two operations:

1. **`estimatePosePnPRansac`** — robust pose estimation from 3D–2D correspondences via OpenCV's `solvePnPRansac`.
2. **`refinePosePoseOnly`** — Gauss-Newton pose-only refinement on the inlier set, using a left-perturbation SE(3) parameterisation with Huber loss.

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

Runtime values set in `run_kitti.cpp`:
- `reprojection_error_px = 3.0f`
- All other values match struct defaults.

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

Gauss-Newton pose-only optimisation over the PnP inlier set. Operates in double precision. The pose is parameterised on the Lie algebra of SE(3) using a **left perturbation** model.

### SE(3) Left Perturbation

The update is applied as:

$$T_{cw} \leftarrow \exp(\hat{\xi}) \cdot T_{cw}$$

where $\xi = [\boldsymbol{\omega}^\top, \boldsymbol{\rho}^\top]^\top \in \mathbb{R}^6$ is the Lie algebra element (rotation first, translation second).

The implemented exponential map:

$$\exp(\hat{\omega}) = I + \sin\theta \cdot [\hat{a}] + (1-\cos\theta) \cdot [\hat{a}]^2 \quad \text{(Rodrigues)}$$

$$\Delta t = J_l(\omega) \cdot \rho$$

where $J_l$ is the left Jacobian of SO(3):

$$J_l(\omega) = I + \frac{1-\cos\theta}{\theta^2} [\omega]_\times + \frac{\theta - \sin\theta}{\theta^3} [\omega]_\times^2$$

### Per-observation Jacobian

For a 3D point $\mathbf{p}_w$ projected to camera frame $\mathbf{p}_c = R_{cw}\mathbf{p}_w + \mathbf{t}_{cw}$, the reprojection is:

$$\pi(\mathbf{p}_c) = \begin{pmatrix} f_x \cdot p_{cx}/p_{cz} + c_x \\ f_y \cdot p_{cy}/p_{cz} + c_y \end{pmatrix}$$

**Projection Jacobian** (2×3):

$$J_\pi = \begin{pmatrix} f_x/z & 0 & -f_x x/z^2 \\ 0 & f_y/z & -f_y y/z^2 \end{pmatrix}$$

**Camera-point Jacobian w.r.t. left SE(3) perturbation** (3×6):

$$J_{\mathbf{p}_c,\xi} = \begin{pmatrix} -[\mathbf{p}_c]_\times & I_3 \end{pmatrix}$$

where $[\mathbf{p}_c]_\times$ is the skew-symmetric matrix of $\mathbf{p}_c$.

**Full Jacobian** (2×6):

$$J = J_\pi \cdot J_{\mathbf{p}_c,\xi}$$

### Huber Loss

$$w = \begin{cases} 1 & \text{if } \|\mathbf{e}\| \leq \delta \\ \delta / \|\mathbf{e}\| & \text{otherwise} \end{cases} \quad \delta = 5.0 \text{ px}$$

### Normal Equations

$$H = \sum_i w_i J_i^\top J_i, \quad \mathbf{b} = \sum_i w_i J_i^\top \mathbf{e}_i$$

$$H \cdot \Delta\xi = \mathbf{b} \quad \text{(LDLT solve)}$$

Iteration stops when `‖Δξ‖ < 1e-6` or cost change `< 1e-6`, up to 10 iterations.

```cpp
const Eigen::Matrix<double, 6, 1> dx = H.ldlt().solve(b);
applyLeftSE3Increment(dx, R_cw, t_cw);
```

---

## Internal Helper Functions (anonymous namespace)

| Function | Description |
|---|---|
| `hat(w)` | 3×3 skew-symmetric matrix from `Vector3d` |
| `expSO3(w)` | Rodrigues exponential map on SO(3) |
| `leftJacobianSO3(w)` | Left Jacobian of SO(3) |
| `applyLeftSE3Increment(dx, R, t)` | Updates `(R_cw, t_cw)` by left-multiplying `exp(dx)` |
| `huberWeight(sq_err, delta)` | Returns Huber weight scalar |
| `projectPoint(p_w, R_cw, t_cw, cam, pixel, p_c)` | Projects 3D point; returns `false` if behind camera |
| `computeReprojectionRmse(...)` | RMSE over a set of 3D–2D correspondences |

## See Also

- [`Camera`](camera.md) — provides `fx, fy, cx, cy` for Jacobian computation
- [`Tracker`](tracker.md) — supplies `object_points` / `image_points` for PnP
- [`Frontend`](frontend.md) — calls `estimatePosePnPRansac` and `refinePosePoseOnly` inside `processFrame`
