#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <sequence> <estimated_traj>"
  echo "Example: $0 05 results/traj/05_vo.txt"
  exit 1
fi

SEQ="$1"
EST="$2"
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

evo_traj kitti "$EST" --ref="$GT" -p --plot_mode=xz \
  --save_plot "${OUT_DIR}/traj_xz.pdf"

evo_ape kitti "$GT" "$EST" -va --plot --plot_mode=xz \
  --save_plot "${OUT_DIR}/ape_xz.pdf" \
  --save_results "${OUT_DIR}/ape.zip"

evo_rpe kitti "$GT" "$EST" -va --plot --plot_mode=xz \
  --save_plot "${OUT_DIR}/rpe_xz.pdf" \
  --save_results "${OUT_DIR}/rpe.zip"

echo "Saved results to $OUT_DIR"