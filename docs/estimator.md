# Estimator

**Header:** `include/svo/estimator.h`  
**Source:** `src/estimator.cpp`

## Role

`Estimator` is the numerical core of the pipeline. It provides three operations:

1. **`estimatePosePnPRansac`** — robust pose estimation from 3D–2D correspondences via OpenCV's `solvePnPRansac`.
2. **`refinePosePoseOnly`** — Gauss-Newton pose-only refinement on the inlier set, using a left-perturbation SE(3) parameterisation with Huber loss.
3. **`runLocalBundleAdjustment`** — dense Gauss-Newton joint optimisation of keyframe poses and landmark positions over a sliding window.

All three methods return result structs; they do not mutate any shared state directly. `run_kitti.cpp` decides whether to accept results.

---

## Result Structs

### `PoseEstimateResult`

```cpp
struct PoseEstimateResult {
    bool success = false;

    cv::Mat rvec;            // Rodrigues rotation vector (OpenCV, 3×1 CV_64F)
    cv::Mat tvec;            // translation vector (OpenCV, 3×1 CV_64F)
    cv::Mat inlier_indices;  // N×1 CV_32S indices into object/image_points

    Eigen::Matrix3d rotation    = Eigen::Matrix3d::Identity();  // R_cw
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();      // t_cw

    int    num_object_points = 0;
    int    num_image_points  = 0;
    int    num_inliers       = 0;
    double reprojection_rmse_before = 0.0;  // px, computed on all inliers before refinement
    double reprojection_rmse_after  = 0.0;  // px, after Gauss-Newton refinement
};
```

`rotation` and `translation` encode `(R_cw, t_cw)` — camera-from-world. The caller converts to `T_wc` via `makePoseWcFromPnP` in `run_kitti.cpp`.

### `LocalBAResult`

```cpp
struct LocalBAResult {
    bool success        = false;
    int  num_keyframes  = 0;
    int  num_landmarks  = 0;
    int  num_observations = 0;
    double rmse_before  = 0.0;  // px, before optimisation
    double rmse_after   = 0.0;  // px, after optimisation
};
```

---

## Options

```cpp
struct Options {
    // PnP RANSAC
    bool   use_extrinsic_guess  = false;
    int    iterations_count     = 100;
    float  reprojection_error_px = 4.0f;
    double confidence           = 0.99;
    int    min_pnp_points       = 6;

    // Pose-only refinement
    int    pose_refine_iterations  = 10;
    double pose_refine_epsilon     = 1e-6;
    double pose_refine_huber_delta = 5.0;
    int    min_refine_inliers      = 10;

    // Local BA
    int    local_ba_iterations        = 3;     // (5 in header default; 3 at runtime)
    double local_ba_epsilon           = 1e-6;
    double local_ba_huber_delta       = 5.0;
    double local_ba_damping           = 1e-3;
    int    max_ba_keyframes           = 3;     // (5 in header default; 3 at runtime)
    int    max_ba_landmarks           = 100;   // (200 in header default; 100 at runtime)
    int    min_ba_observations        = 20;    // (30 in header default; 20 at runtime)
    int    min_ba_landmark_observations = 2;
};
```

---

## `estimatePosePnPRansac`

### Signature (with initial guess)

```cpp
PoseEstimateResult estimatePosePnPRansac(
    const std::vector<Eigen::Vector3d>& object_points,
    const std::vector<cv::Point2f>&     image_points,
    const Camera&                        camera,
    const Eigen::Matrix3d& initial_rotation_cw,
    const Eigen::Vector3d& initial_translation_cw,
    bool use_initial_guess) const;
```

The no-guess overload calls this with identity and `use_initial_guess = false`.

### Algorithm

1. Convert `Eigen::Vector3d` → `cv::Point3f` (note: **float** precision).
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

## `runLocalBundleAdjustment`

Joint optimisation over the `max_ba_keyframes = 3` most recent keyframes and up to `max_ba_landmarks = 100` landmarks that appear in at least 2 of those keyframes.

### Observation Gathering

```
For each of the K most recent keyframes:
  For each tracked_landmark_id in frame.tracked_landmark_ids:
    Count how many keyframes see this landmark.
    Accept only landmarks seen in >= min_ba_landmark_observations (2) keyframes.
    Cap at max_ba_landmarks (100).
```

Minimum `min_ba_observations = 20` total observations required to run.

### Parameter Layout

The optimisation variable is:

$$\mathbf{x} = \underbrace{\xi_1, \xi_2, \ldots, \xi_{K-1}}_{6(K-1) \text{ dof}},\; \underbrace{\mathbf{p}_{w,1}, \ldots, \mathbf{p}_{w,P}}_{3P \text{ dof}}$$

The **first keyframe pose is fixed** to remove gauge freedom. This means observations from keyframe 0 contribute only to the point Jacobian block, not the pose Jacobian block.

### Dense Gauss-Newton with LM Damping

$$H = \begin{pmatrix} H_{pp} & H_{pl} \\ H_{lp} & H_{ll} \end{pmatrix} + \lambda I, \quad \mathbf{b} = \begin{pmatrix} \mathbf{b}_p \\ \mathbf{b}_l \end{pmatrix}$$

Per-observation Jacobians:

- **Pose Jacobian** $J_{\text{pose}}$ (2×6): same formula as in `refinePosePoseOnly`; set to zero for the fixed keyframe 0.
- **Point Jacobian** $J_{\text{point}}$ (2×3):

$$J_{\text{point}} = J_\pi \cdot R_{cw}$$

Damping factor $\lambda = 10^{-3}$ is added to the full diagonal before each LDLT solve.

The same Huber weight (`delta = 5.0 px`) is applied per observation.

### Write-back

After convergence, the optimised `(R_cw, t_cw)` are converted back to `T_wc` and written into `ba_keyframes[k]->pose_wc`. Refined `p_w` are written back into the `landmarks` vector.

The caller (`run_kitti.cpp`) accepts these only if `rmse_after <= rmse_before`; otherwise it restores from a backup.

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
| `computeLocalBaRmse(...)` | RMSE over `LocalBAObservation` set |

## See Also

- [`Camera`](camera.md) — provides `fx, fy, cx, cy` for Jacobian computation
- [`Tracker`](tracker.md) — supplies `object_points` / `image_points` for PnP
- [`Map`](map.md) — supplies `activeKeyframes()` / `activeLandmarks()` for BA
- [`Frontend`](frontend.md) — receives refined pose via `acceptPose`; refreshes landmarks after BA
