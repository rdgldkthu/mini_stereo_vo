# Map

**Header:** `include/svo/map.h`  
**Source:** `src/map.cpp`

## Role

`Map` is the sliding-window storage for active keyframes and active landmarks. It enforces capacity limits, tracks landmark visibility statistics, and prunes landmarks that have degraded too much to be useful. It is the authoritative store of landmark positions; `Frontend::processFrame` reads and updates it on every keyframe insertion.

---

## Options

```cpp
struct Options {
    int max_active_keyframes  = 5;    // oldest keyframe evicted when exceeded
    int max_active_landmarks  = 2000; // capacity cap; overflow sorted and trimmed
    int min_observed_times    = 2;    // minimum keyframe_observations for pruning
    int max_missed_times      = 8;    // hard prune after this many consecutive misses
};
```

Runtime values from `run_kitti.cpp` match the defaults.

---

## Public Methods

### `void addKeyframe(const Frame& frame)`

Appends `frame` to `active_keyframes_` (a `std::deque`). If the count exceeds `max_active_keyframes`, the oldest entry is evicted from the front:

```cpp
active_keyframes_.push_back(frame);
if (active_keyframes_.size() > max_active_keyframes)
    active_keyframes_.pop_front();
```

---

### `void setActiveLandmarks(const std::vector<MapPoint>& landmarks)`

Replaces the entire landmark set and trims to `max_active_landmarks` if necessary. Rebuilds the internal `id_to_index_` hash map.

Used at bootstrap and after reinitialization.

---

### `void addLandmarks(const std::vector<MapPoint>& landmarks)`

Merges new landmarks into the active set by ID:

- If a landmark with the same `id` already exists: updates `p_w`, `descriptor`, and all counters in-place.
- Otherwise: appends it.

If the merged set exceeds `max_active_landmarks`, the set is sorted by:

1. `tracked_frames` descending (high-vitality landmarks stay)
2. `missed_times` ascending (recently-seen landmarks stay)
3. `id` ascending (tie-break)

and the tail is trimmed.

---

### `void assignNewLandmarkIds(std::vector<MapPoint>& landmarks)`

Assigns globally unique, monotonically increasing IDs from an internal counter:

```cpp
for (auto& lm : landmarks)
    lm.id = next_landmark_id++;
```

Must be called before landmarks are inserted into the map or handed to the frontend.

---

### `void markTrackedLandmarks(const std::vector<MapPoint>& tracked_landmarks)`

For each successfully tracked landmark (identified by `id`):

- `tracked_frames += 1`
- `missed_times = 0`
- `is_active = true`, `is_outlier = false`

---

### `void markMissedLandmarks(const std::vector<int>& tracked_landmark_ids)`

For every landmark in `active_landmarks_` whose `id` is **not** in `tracked_landmark_ids`:

- `missed_times += 1`

Uses an `unordered_set` for O(1) membership tests.

---

### `void markOutlierLandmarks(const std::vector<int>& outlier_ids)`

For each ID in `outlier_ids` (PnP RANSAC outliers):

- `is_outlier = true`
- `missed_times += 1`

---

### `void markKeyframeObservations(const std::vector<int>& landmark_ids)`

For each ID in `landmark_ids`, increments `keyframe_observations` by 1. Called once per keyframe insertion for the landmarks that were active at the time of insertion.

---

### `void pruneLandmarks()`

Removes landmarks matching any prune condition:

| Condition | Effect |
|---|---|
| `is_outlier == true` | Remove immediately |
| `missed_times > max_missed_times (8)` | Remove (lost track) |
| `keyframe_observations < min_observed_times (2) && missed_times > 2` | Remove (never established) |

Rebuilds the `id_to_index_` hash map after pruning.

---

### Accessors

| Method | Description |
|---|---|
| `activeKeyframes()` / `mutableActiveKeyframes()` | const/non-const reference to keyframe deque |
| `activeLandmarks()` / `mutableActiveLandmarks()` | const/non-const reference to landmark vector |
| `numActiveKeyframes()` | current count of keyframes |
| `numActiveLandmarks()` | current count of landmarks |

---

## Private Helpers

### `int findLandmarkIndexById(int landmark_id) const`

O(1) lookup via `id_to_index_` hash map. Returns `-1` if not found.

### `void rebuildIndex()`

Rebuilds `id_to_index_` from scratch. Called after `setActiveLandmarks`, `addLandmarks` (when trimming occurs), and `pruneLandmarks`.

## See Also

- [`Frame`](frame.md) — stored in `active_keyframes_`
- [`MapPoint`](map_point.md) — stored in `active_landmarks_`
- [`Frontend`](frontend.md) — calls all Map methods from within `processFrame`
