# DatasetKitti

**Header:** `include/svo/dataset_kitti.h`  
**Source:** `src/dataset_kitti.cpp`

## Role

`DatasetKitti` is the dataset I/O layer. It opens a KITTI odometry sequence directory, discovers and sorts the stereo image pairs, and loads individual frames on demand as grayscale `cv::Mat` images. It also exposes the path to the calibration file so `Camera::loadFromKittiCalib` can be called independently.

---

## Expected Directory Layout

```
<kitti_root>/
  sequences/
    <seq>/               e.g. "05"
      image_0/           left camera PNGs  (000000.png, 000001.png, …)
      image_1/           right camera PNGs
      calib.txt
  poses/
    <seq>.txt            ground-truth poses (optional, for evaluation)
```

---

## Public Interface

### `bool open(const std::string& kitti_root, const std::string& sequence)`

Initialises the dataset. Scans `image_0/` and `image_1/` for `.png` files, sorts them lexicographically, and verifies that:

- Both directories contain at least one image.
- Left and right counts match exactly.
- `calib.txt` exists.

Returns `false` and prints to `stderr` on any failure.

```cpp
dataset.open("data/kitti", "05");
```

---

### `bool loadFrame(int frame_id, Frame& frame) const`

Loads the stereo pair at index `frame_id` into `frame.left_img` and `frame.right_img` as grayscale (`cv::IMREAD_GRAYSCALE`). Sets `frame.id = frame_id`. Returns `false` if `frame_id` is out of range or either image fails to load.

```cpp
svo::Frame f;
dataset.loadFrame(42, f);  // f.left_img, f.right_img now populated
```

---

### Accessors

| Method | Return | Description |
|---|---|---|
| `numFrames()` | `int` | Total number of frames (= number of left PNGs found) |
| `calibPath()` | `const fs::path&` | Path to `<seq_dir>/calib.txt` |
| `sequence()` | `const std::string&` | Sequence string (e.g. `"05"`) |

---

## Private Helpers

### `static std::vector<fs::path> sortedPngs(const fs::path& dir)`

Iterates `dir` for regular files with extension `.png`, collects their paths, and sorts them lexicographically (which corresponds to numeric order for zero-padded KITTI filenames).

---

## Notes

- Images are loaded lazily: `open()` only scans paths; pixel data is read in `loadFrame()`.
- Timestamps are not extracted from KITTI `times.txt`; `Frame::timestamp` remains 0.
- The grayscale format is intentional: all processing (ORB detection, LK flow) operates on single-channel images.

## See Also

- [`Camera`](camera.md) — loaded from `calibPath()`
- [`Frame`](frame.md) — populated by `loadFrame`
