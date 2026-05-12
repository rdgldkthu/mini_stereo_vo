#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <sequence> <estimated_traj>"
  echo "Example: $0 05 results/traj/05_vo.txt"
  exit 1
fi

SEQ="$1"
EST="$2"
EST_BASE=$(basename "$EST" .txt)
GT="data/kitti/poses/${SEQ}.txt"
OUT_DIR="results/tables/${SEQ}"

mkdir -p "$OUT_DIR"

if [ ! -f "$GT" ]; then
  echo "Ground-truth file not found: $GT"
  exit 1
fi

if [ ! -f "$EST" ]; then
  echo "Estimated trajectory file not found: $EST"
  exit 1
fi

echo "Evaluating sequence $SEQ"
echo "GT : $GT"
echo "EST: $EST"

# Truncate GT to match estimated trajectory length (old commits may produce fewer frames)
GT_EVAL="$GT"
GT_TMP=""
EST_LINES=$(wc -l < "$EST")
GT_LINES=$(wc -l < "$GT")
if [ "$EST_LINES" -ne "$GT_LINES" ]; then
  echo "Note: EST has ${EST_LINES} poses, GT has ${GT_LINES} — truncating GT to ${EST_LINES}"
  GT_TMP=$(mktemp)
  head -n "$EST_LINES" "$GT" > "$GT_TMP"
  GT_EVAL="$GT_TMP"
fi
trap '[ -n "$GT_TMP" ] && rm -f "$GT_TMP"' EXIT

# evo appends figure names to PNG paths (e.g. _trajectories, _xz) — keep base simple
evo_traj kitti "$EST" --ref="$GT_EVAL" -p --plot_mode=xz \
  --save_plot "${OUT_DIR}/${EST_BASE}_traj.png"

evo_ape kitti "$GT_EVAL" "$EST" -va --plot --plot_mode=xz \
  --save_plot "${OUT_DIR}/${EST_BASE}_ape.png" \
  --save_results "${OUT_DIR}/${EST_BASE}_ape.zip"

evo_rpe kitti "$GT_EVAL" "$EST" -va --plot --plot_mode=xz \
  --save_plot "${OUT_DIR}/${EST_BASE}_rpe.png" \
  --save_results "${OUT_DIR}/${EST_BASE}_rpe.zip"

echo "Saved results to $OUT_DIR"

python3 - "$SEQ" "$EST" "${OUT_DIR}/${EST_BASE}_ape.zip" "${OUT_DIR}/${EST_BASE}_rpe.zip" "results/metrics_log.csv" <<'PYEOF'
import sys, zipfile, json, csv, os
from datetime import datetime

seq, est, ape_zip, rpe_zip, csv_path = sys.argv[1:]

def load_stats(path):
    with zipfile.ZipFile(path) as z:
        with z.open('stats.json') as f:
            return json.load(f)

ape = load_stats(ape_zip)
rpe = load_stats(rpe_zip)

fields = ['datetime', 'sequence', 'est_traj',
          'ape_rmse', 'ape_mean', 'ape_median', 'ape_std', 'ape_min', 'ape_max',
          'rpe_rmse', 'rpe_mean', 'rpe_median', 'rpe_std', 'rpe_min', 'rpe_max']

row = [datetime.now().strftime('%Y-%m-%d %H:%M:%S'), seq, est,
       ape['rmse'], ape['mean'], ape['median'], ape['std'], ape['min'], ape['max'],
       rpe['rmse'], rpe['mean'], rpe['median'], rpe['std'], rpe['min'], rpe['max']]

new_file = not os.path.isfile(csv_path) or os.path.getsize(csv_path) == 0
with open(csv_path, 'a', newline='') as f:
    w = csv.writer(f)
    if new_file:
        w.writerow(fields)
    w.writerow(row)

print(f"Logged metrics -> {csv_path}")
PYEOF