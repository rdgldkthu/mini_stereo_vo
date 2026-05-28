#!/usr/bin/env bash
# CTest regression gate for stereo VO.
# Exits 77 (CTest SKIP) when data or tooling is absent.
# Usage: run_regression.sh <seq> <max_frames> <ape_threshold_m>
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

SEQ="${1:-05}"
MAX_FRAMES="${2:-100}"
APE_THRESHOLD="${3:-2.0}"

RUN_KITTI="$ROOT/build/run_kitti"
DATA_DIR="$ROOT/data/kitti"
GT_FILE="$DATA_DIR/poses/${SEQ}.txt"
TRAJ_FILE="/tmp/svo_regression_${SEQ}.txt"
APE_ZIP="/tmp/svo_regression_${SEQ}_ape.zip"

if [ ! -f "$RUN_KITTI" ]; then
  echo "SKIP: run_kitti not found at $RUN_KITTI"
  exit 77
fi
if [ ! -d "$DATA_DIR/sequences/${SEQ}" ] || [ ! -f "$GT_FILE" ]; then
  echo "SKIP: KITTI sequence $SEQ not found under $DATA_DIR"
  exit 77
fi

# Activate venv if evo_ape is not already on PATH
if ! command -v evo_ape &>/dev/null; then
  if [ -f "$ROOT/.venv/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$ROOT/.venv/bin/activate"
  fi
fi
if ! command -v evo_ape &>/dev/null; then
  echo "SKIP: evo_ape not found (install evo or run bootstrap_ubuntu2404.sh)"
  exit 77
fi

echo "Running VO: seq=$SEQ  max_frames=$MAX_FRAMES"
cd "$ROOT"
OPENCV_TRACE=0 "$RUN_KITTI" "$DATA_DIR" "$SEQ" \
  --no-viewer \
  --max-frames "$MAX_FRAMES" \
  --output-file "$TRAJ_FILE"

if [ ! -f "$TRAJ_FILE" ]; then
  echo "FAIL: trajectory file not produced"
  exit 1
fi

# Truncate GT to the number of poses actually written (tracking may stop early)
N_POSES=$(wc -l < "$TRAJ_FILE")
GT_TMP=$(mktemp)
trap 'rm -f "$GT_TMP"' EXIT
head -n "$N_POSES" "$GT_FILE" > "$GT_TMP"

echo "Evaluating APE (${N_POSES} poses, threshold ${APE_THRESHOLD} m)"
evo_ape kitti "$GT_TMP" "$TRAJ_FILE" --save_results "$APE_ZIP"

python3 - "$APE_THRESHOLD" "$APE_ZIP" <<'PYEOF'
import sys, zipfile, json
threshold = float(sys.argv[1])
with zipfile.ZipFile(sys.argv[2]) as z:
    with z.open('stats.json') as f:
        stats = json.load(f)
rmse = stats['rmse']
print(f"APE RMSE: {rmse:.4f} m  (threshold: {threshold} m)")
if rmse > threshold:
    print(f"FAIL: exceeds threshold")
    sys.exit(1)
print("PASS")
PYEOF
