# Map

**Header:** `include/svo/map.h`  
**Source:** `src/map.cpp`

## Role

`Map` is the sliding-window storage for active keyframes and active landmarks. It enforces capacity limits, tracks landmark visibility statistics (observed / missed counts), and prunes landmarks that have degraded too much to be useful. It is the authoritative source of landmark positions after local BA, and its keyframe vector is passed directly to `Estimator::runLocalBundleAdjustment`.

---

## Options

```cpp
struct Options {
    int max_active_keyframes  = 5;    // oldest keyframe evicted when exceeded
    int max_active_landmarks  = 2000; // capacity cap; overflow sorted and trimmed
    int min_observed_times    = 2;    // used in pruning condition
    int max_missed_times      = 8;    // hard prune after this many consecutive misses
};
```

Runtime values from `run_kitti.cpp` match the defaults.

---

## Public Methods

### `void addKeyframe(const Frame& frame)`

Appends `frame` to `active_keyframes_`. If the count exceeds `max_active_keyframes`, the oldest entry (front) is erased:

```cpp
active_keyframes_.push_back(frame);
if (active_keyframes_.size() > max_active_keyframes)
    active_keyframes_.erase(active_keyframes_.begin());
```

Only frames with `is_keyframe = true` are passed here from `run_kitti.cpp`.

---

### `void setActiveLandmarks(const std::vector<MapPoint>& landmarks)`

Replaces the entire landmark set. Trims to `max_active_landmarks` if necessary (simple resize, keeps first N).

Used at initialisation and after reinitialization events.

---

### `void addLandmarks(const std::vector<MapPoint>& landmarks)`

Merges new landmarks into the active set by ID:

- If a landmark with the same `id` already exists: updates `p_w`, `descriptor`, and all counters in-place.
- Otherwise: appends it.

If the merged set exceeds `max_active_landmarks`, the set is sorted by:

1. `observed_times` descending (well-seen landmarks stay)
2. `missed_times` ascending (recently-seen landmarks stay)

and the tail is trimmed.

```cpp
std::sort(active_landmarks_.begin(), active_landmarks_.end(),
          [](const MapPoint& a, const MapPoint& b) {
              if (a.observed_times != b.observed_times)
                  return a.observed_times > b.observed_times;
              return a.missed_times < b.missed_times;
          });
active_landmarks_.resize(max_active_landmarks);
```

---

### `void assignNewLandmarkIds(std::vector<MapPoint>& landmarks)`

Assigns globally unique, monotonically increasing IDs from an internal counter `next_landmark_id`:

```cpp
for (auto& lm : landmarks)
    lm.id = next_landmark_id++;
```

Must be called before landmarks are inserted into the map or handed to the frontend.

---

### `void markTrackedLandmarks(const std::vector<MapPoint>& tracked_landmarks)`

For each successfully tracked landmark (identified by `id`):

- `tracked_times += 1`
- `observed_times += 1`
- `missed_times = 0`
- `is_active = true`, `is_outlier = false`

---

### `void markMissedLandmarks(const std::vector<int>& tracked_landmark_ids)`

For every landmark in `active_landmarks_` whose `id` is **not** in `tracked_landmark_ids`:

- `missed_times += 1`

This uses an `unordered_set` for O(1) membership tests.

---

### `void pruneLandmarks()`

Removes landmarks matching any prune condition:

| Condition | Effect |
|---|---|
| `is_outlier == true` | Remove immediately |
| `missed_times > max_missed_times (8)` | Remove (lost track) |
| `observed_times < min_observed_times (2) && missed_times > 2` | Remove (never established) |

---

### Accessors

| Method | Description |
|---|---|
| `activeKeyframes()` / `mutableActiveKeyframes()` | const/non-const reference to keyframe vector |
| `activeLandmarks()` / `mutableActiveLandmarks()` | const/non-const reference to landmark vector |
| `numActiveKeyframes()` | current count of keyframes |
| `numActiveLandmarks()` | current count of landmarks |

The mutable accessors are needed by `Estimator::runLocalBundleAdjustment`, which modifies poses and landmark positions in-place, and by `run_kitti.cpp` for the backup/restore pattern around BA.

---

## BA Backup/Restore Pattern

Local BA modifies `mutableActiveKeyframes()` and `mutableActiveLandmarks()` in-place. If BA increases RMSE or fails, `run_kitti.cpp` restores from a pre-BA snapshot:

```cpp
// Before BA
std::vector<Frame>    kf_backup = map.activeKeyframes();
std::vector<MapPoint> lm_backup = map.activeLandmarks();

estimator.runLocalBundleAdjustment(map.mutableActiveKeyframes(),
                                   map.mutableActiveLandmarks(), camera);

if (!ba_result.success || ba_result.rmse_after > ba_result.rmse_before) {
    map.mutableActiveKeyframes() = kf_backup;
    map.mutableActiveLandmarks() = lm_backup;
}
```

---

## Private Helper

### `int findLandmarkIndexById(int landmark_id) const`

Linear search through `active_landmarks_` for a matching `id`. Returns `-1` if not found. Used by `addLandmarks` and `markTrackedLandmarks`.

## See Also

- [`Frame`](frame.md) — stored in `active_keyframes_`
- [`MapPoint`](map_point.md) — stored in `active_landmarks_`
- [`Estimator`](estimator.md) — reads and writes both containers during local BA
- [`Frontend`](frontend.md) — receives updated landmarks via `refreshActiveLandmarksFromMap`
