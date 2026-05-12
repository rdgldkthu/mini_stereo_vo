# PoseWriter

**Header:** `include/svo/pose_writer.h`  
**Source:** `src/pose_writer.cpp`

## Role

`PoseWriter` is a stateless utility class with two static methods for writing pose trajectories in the KITTI format. It is called once at the end of a run to serialise `Frontend::poses()` to disk. The output file is then consumed by `scripts/eval_kitti.sh` for APE/RPE evaluation with the `evo` tool.

---

## Static Methods

### `static void writeKittiTrajectory(const fs::path& out_path, const std::vector<Eigen::Matrix4d>& poses)`

Writes one pose per line. Each line contains 12 space-separated `double` values (9 decimal places) representing the top three rows of the 4×4 world-from-camera matrix `T_wc`:

```
r00 r01 r02 tx  r10 r11 r12 ty  r20 r21 r22 tz
```

This is the standard KITTI trajectory format. The fourth row `[0 0 0 1]` is omitted.

```cpp
PoseWriter::writeKittiTrajectory("results/traj/05_260512_vo.txt",
                                 frontend.poses());
```

Parent directories are created automatically (`fs::create_directories`). Throws `std::runtime_error` if the file cannot be opened.

---

### `static void writeIdentityKittiTrajectory(const fs::path& out_path, int num_poses)`

Writes `num_poses` lines of `1 0 0 0 0 1 0 0 0 0 1 0` — an all-identity trajectory. Useful as a fallback when the VO system fails to initialise.

---

## Output Format Example

```
1.000000000 0.000000000 0.000000000 0.000000000 0.000000000 1.000000000 0.000000000 0.000000000 0.000000000 0.000000000 1.000000000 0.000000000
0.999982834 0.003821045 -0.004213918 0.051234567 ...
```

---

## File Naming Convention

The output path is constructed in `run_kitti.cpp` as:

```
results/traj/<seq>_<YYMMDD>[_<keyword>].txt
```

Example: `results/traj/05_260512_vo.txt`

## See Also

- [`Frontend`](frontend.md) — provides `poses()` (the `vector<Matrix4d>` written here)
- `scripts/eval_kitti.sh` — consumes this file for `evo ape` / `evo rpe` evaluation
