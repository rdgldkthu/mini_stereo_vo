# Camera

**Header:** `include/svo/camera.h`  
**Source:** `src/camera.cpp`

## Role

`Camera` holds the intrinsic parameters and rectified projection matrices for the stereo rig. It is loaded once from a KITTI `calib.txt` file and then passed read-only to every module that needs to project or triangulate points. All geometry in the pipeline assumes rectified images, so no distortion coefficients are stored or used.

---

## Public Fields

| Field | Type | Description |
|---|---|---|
| `fx` | `double` | Focal length in x (pixels) |
| `fy` | `double` | Focal length in y (pixels) |
| `cx` | `double` | Principal point x (pixels) |
| `cy` | `double` | Principal point y (pixels) |
| `baseline` | `double` | Stereo baseline (metres, positive) |
| `P_left` | `Eigen::Matrix<double,3,4>` | Left projection matrix (from KITTI `P0:`) |
| `P_right` | `Eigen::Matrix<double,3,4>` | Right projection matrix (from KITTI `P1:`) |

---

## Public Methods

### `bool loadFromKittiCalib(const std::string& calib_path)`

Parses a KITTI-format `calib.txt` file and populates all fields. Returns `true` on success.

**File format expected:**

```
P0: fx 0 cx 0  0 fy cy 0  0 0 1 0
P1: fx 0 cx tx  0 fy cy 0  0 0 1 0
...
```

Each `P0:` / `P1:` line contains 12 whitespace-separated doubles that are read row-major into the 3×4 matrices. After loading:

- `fx = P_left(0,0)`, `fy = P_left(1,1)`, `cx = P_left(0,2)`, `cy = P_left(1,2)`
- Baseline extracted from the right camera's x-translation term:

$$\text{baseline} = \frac{-P_{\text{right}}(0,3)}{P_{\text{right}}(0,0)}$$

For KITTI, `P_right(0,3) = -f_x \cdot b < 0`, so the result is positive.

```cpp
baseline = -P_right(0, 3) / P_right(0, 0);
```

---

### `Eigen::Vector3d pixel2Camera(double u, double v, double depth) const`

Back-projects a pixel `(u, v)` with known `depth` (Z in camera frame) into a 3D camera-frame point.

$$\mathbf{p}_c = \begin{pmatrix} (u - c_x) \cdot d / f_x \\ (v - c_y) \cdot d / f_y \\ d \end{pmatrix}$$

```cpp
const double x = (u - cx) * depth / fx;
const double y = (v - cy) * depth / fy;
return Eigen::Vector3d(x, y, depth);
```

---

### `bool triangulateRectified(double ul, double vl, double ur, Eigen::Vector3d& p_c) const`

Triangulates a point from a rectified stereo match `(ul, vl)` ↔ `(ur, *)` using disparity.

**Math:**

$$d = u_l - u_r \qquad \text{(disparity, pixels)}$$

$$Z = \frac{f_x \cdot \text{baseline}}{d}$$

Then calls `pixel2Camera(ul, vl, Z)` to form `p_c`.

Guards:
- Returns `false` if `d ≤ 1e-6` (degenerate / negative disparity)
- Returns `false` if `Z` is not finite or non-positive

```cpp
const double disparity = ul - ur;
if (disparity <= 1e-6) return false;
const double depth = fx * baseline / disparity;
p_c = pixel2Camera(ul, vl, depth);
return true;
```

---

### `void print() const`

Prints intrinsics and both projection matrices to `stdout`. Used at startup for a quick sanity check.

---

## Notes

- The camera model assumes pre-rectified images (no radial/tangential distortion).
- `P_left` has the form `K [I | 0]`; `P_right` has the form `K [I | t_x]` where `t_x = -f_x · b`.
- The same `fx`, `fy` apply to both cameras after rectification (KITTI guarantee).

## See Also

- [`StereoInitializer`](stereo_initializer.md) — uses `triangulateRectified` to build initial landmarks
- [`Estimator`](estimator.md) — uses `fx`, `fy`, `cx`, `cy` to form the reprojection Jacobian
- [`DatasetKitti`](dataset_kitti.md) — provides `calibPath()` for loading
