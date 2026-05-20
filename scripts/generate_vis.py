#!/usr/bin/env python3
"""Generate PNG assets for the README from a completed pipeline run.

Usage:
    python3 scripts/generate_vis.py [--seq 05] [--traj results/traj/05_vo.txt] [--kitti data/kitti]

Outputs (written to assets/):
    trajectory.png    — VO estimate vs ground truth, x-z view
    tracking.png      — representative tracking overlay frame
    init_matches.png  — stereo feature match visualization
    inlier_ratio.png  — inlier ratio over time with keyframe events
"""

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path


def load_poses_kitti(path):
    """Return (x, z) arrays from a KITTI-format 3×4 pose file."""
    xs, zs = [], []
    with open(path) as f:
        for line in f:
            vals = list(map(float, line.split()))
            if len(vals) != 12:
                continue
            xs.append(vals[3])   # tx
            zs.append(vals[11])  # tz
    return xs, zs


def plot_trajectory(vo_traj, gt_traj, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vo_x, vo_z = load_poses_kitti(vo_traj)
    gt_x, gt_z = load_poses_kitti(gt_traj)

    fig, ax = plt.subplots(figsize=(8, 6), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")

    ax.plot(gt_x, gt_z, color="#56b4e9", linewidth=1.8, label="Ground Truth")
    ax.plot(vo_x, vo_z, color="#e69f00", linewidth=1.4, label="Estimated VO", alpha=0.85)

    ax.set_xlabel("x (m)", color="#c9d1d9", fontsize=11)
    ax.set_ylabel("z (m)", color="#c9d1d9", fontsize=11)
    ax.set_title("Trajectory — KITTI seq 05", color="#e6edf3", fontsize=13, pad=12)
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=10)
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_inlier_ratio(stats_csv, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frames, ratios, kf_frames, reinit_frames = [], [], [], []
    with open(stats_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fid = int(row["frame_id"])
            frames.append(fid)
            ratios.append(float(row["inlier_ratio"]))
            if row["is_keyframe"].strip() == "1":
                kf_frames.append(fid)
            if row["reinitialized"].strip() == "1":
                reinit_frames.append(fid)

    fig, ax = plt.subplots(figsize=(10, 3.5), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")

    ax.plot(frames, ratios, color="#58a6ff", linewidth=1.0, alpha=0.9, label="Inlier ratio")

    for kf in kf_frames:
        ax.axvline(kf, color="#3fb950", linewidth=0.5, alpha=0.35)
    for ri in reinit_frames:
        ax.axvline(ri, color="#f78166", linewidth=0.9, alpha=0.6)

    # Proxy artists for legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="#58a6ff", linewidth=1.5, label="Inlier ratio"),
        Line2D([0], [0], color="#3fb950", linewidth=1.5, label="Keyframe"),
        Line2D([0], [0], color="#f78166", linewidth=1.5, label="Reinitialization"),
    ]
    ax.legend(handles=handles, facecolor="#161b22", edgecolor="#30363d",
              labelcolor="#c9d1d9", fontsize=9)

    ax.set_xlim(frames[0], frames[-1])
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Frame", color="#c9d1d9", fontsize=11)
    ax.set_ylabel("Inlier ratio", color="#c9d1d9", fontsize=11)
    ax.set_title("Tracking health — KITTI seq 05", color="#e6edf3", fontsize=13, pad=10)
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  wrote {out_path}")


def copy_tracking_frame(debug_dir, seq, out_path):
    """Pick a mid-sequence tracking frame and copy it."""
    candidates = sorted(Path(debug_dir).glob(f"{seq}_track_*.png"))
    if not candidates:
        print(f"  warning: no tracking frames found in {debug_dir}", file=sys.stderr)
        return
    # Pick roughly frame 300 (or mid-point if fewer frames)
    idx = min(len(candidates) // 3, len(candidates) - 1)
    shutil.copy2(candidates[idx], out_path)
    print(f"  wrote {out_path}  (from {candidates[idx].name})")


def copy_init_matches(debug_dir, seq, out_path):
    src = Path(debug_dir) / f"{seq}_init_matches.png"
    if not src.exists():
        print(f"  warning: {src} not found", file=sys.stderr)
        return
    shutil.copy2(src, out_path)
    print(f"  wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate README visual assets.")
    parser.add_argument("--seq", default="05", help="KITTI sequence ID (default: 05)")
    parser.add_argument("--traj", default=None,
                        help="VO trajectory file (default: results/traj/<seq>.txt)")
    parser.add_argument("--kitti", default="data/kitti",
                        help="KITTI dataset root (default: data/kitti)")
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    seq = args.seq
    traj_path = Path(args.traj) if args.traj else repo_root / "results" / "traj" / f"{seq}.txt"
    gt_path = Path(args.kitti) / "poses" / f"{seq}.txt"
    debug_dir = repo_root / "results" / "debug"
    stats_csv = debug_dir / f"{traj_path.stem}_stats.csv"
    assets_dir = repo_root / "assets"
    assets_dir.mkdir(exist_ok=True)

    print(f"Generating assets for seq {seq} ...")

    missing = []
    if not traj_path.exists():
        missing.append(f"VO trajectory: {traj_path}")
    if not gt_path.exists():
        missing.append(f"Ground truth: {gt_path}")
    if missing:
        print("ERROR — required files not found:", file=sys.stderr)
        for m in missing:
            print(f"  {m}", file=sys.stderr)
        print("\nRun the pipeline first:", file=sys.stderr)
        print(f"  ./build/run_kitti {args.kitti} {seq} {traj_path}", file=sys.stderr)
        sys.exit(1)

    plot_trajectory(traj_path, gt_path, assets_dir / "trajectory.png")

    if stats_csv.exists():
        plot_inlier_ratio(stats_csv, assets_dir / "inlier_ratio.png")
    else:
        print(f"  skipping inlier_ratio.png — {stats_csv} not found")

    copy_tracking_frame(debug_dir, seq, assets_dir / "tracking.png")
    copy_init_matches(debug_dir, seq, assets_dir / "init_matches.png")

    print("Done. Assets written to assets/")


if __name__ == "__main__":
    main()
