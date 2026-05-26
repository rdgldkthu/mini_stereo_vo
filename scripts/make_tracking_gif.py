#!/usr/bin/env python3
"""Assemble consecutive debug tracking frames into an animated GIF.

Requires tracking PNGs produced by running the pipeline with --save-debug:
    ./build/run_kitti data/kitti <seq> [keyword] --save-debug

Usage:
    python3 scripts/make_tracking_gif.py [options]

Options:
    --seq      KITTI sequence ID (default: 05)
    --stem     Exact file stem to match, e.g. 05_260512_vo
               (default: most recent matching stem in results/debug/)
    --start    Index into the sorted frame list to begin the window (default: 1/4 through)
    --count    Number of consecutive frames to include (default: 25)
    --fps      Playback speed in frames per second (default: 8)
    --width    Output width in pixels; height scaled proportionally (default: 640)
    --out      Output path (default: assets/tracking.gif)
    --debug    results/debug/ directory (default: results/debug)
"""

import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Make a tracking GIF from consecutive debug frames.")
    parser.add_argument("--seq", default="05")
    parser.add_argument("--stem", default=None,
                        help="File stem prefix (e.g. 05_260512_vo). "
                             "Default: most recent matching stem.")
    parser.add_argument("--start", type=int, default=None,
                        help="Index into the sorted frame list to start the window "
                             "(default: 1/4 through the available frames)")
    parser.add_argument("--count", type=int, default=25,
                        help="Number of consecutive frames (default: 25)")
    parser.add_argument("--fps", type=float, default=8,
                        help="GIF playback speed in fps (default: 8)")
    parser.add_argument("--width", type=int, default=640,
                        help="Output width in pixels (default: 640)")
    parser.add_argument("--out", default=None,
                        help="Output GIF path (default: assets/tracking.gif)")
    parser.add_argument("--debug", default="results/debug",
                        help="Debug output directory (default: results/debug)")
    args = parser.parse_args()

    try:
        from PIL import Image
        import imageio.v3 as iio
    except ImportError as e:
        sys.exit(f"Missing dependency: {e}\nRun: pip install Pillow imageio")

    repo_root = Path(__file__).parent.parent
    debug_dir = repo_root / args.debug
    out_path = Path(args.out) if args.out else repo_root / "assets" / "tracking.gif"

    all_frames = sorted(debug_dir.glob(f"{args.seq}_*_track_*.png"))
    if not all_frames:
        sys.exit(
            f"No tracking frames found in {debug_dir} matching '{args.seq}_*_track_*.png'.\n"
            "Run the pipeline with --save-debug first:\n"
            f"  ./build/run_kitti data/kitti {args.seq} [keyword] --save-debug"
        )

    # Filter to a specific stem if requested, otherwise use the most recent stem.
    if args.stem:
        all_frames = [f for f in all_frames if f.name.startswith(args.stem + "_track_")]
        if not all_frames:
            sys.exit(f"No frames found for stem '{args.stem}'.")
    else:
        latest = max(all_frames, key=lambda p: p.stat().st_mtime)
        stem = latest.name.split("_track_")[0]
        all_frames = [f for f in all_frames if f.name.startswith(stem + "_track_")]
        print(f"Using stem: {stem}  ({len(all_frames)} frames available)")

    start = args.start if args.start is not None else len(all_frames) // 4
    selected = all_frames[start : start + args.count]
    if not selected:
        sys.exit(f"--start {start} is past the end of the frame list ({len(all_frames)} frames).")

    print(f"Frames {start}–{start + len(selected) - 1} of {len(all_frames)} → {out_path}")

    duration_ms = int(1000 / args.fps)
    frames_out = []
    for path in selected:
        img = Image.open(path).convert("RGB")
        orig_w, orig_h = img.size
        new_h = round(orig_h * args.width / orig_w)
        img = img.resize((args.width, new_h), Image.LANCZOS)
        frames_out.append(img)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    iio.imwrite(
        str(out_path),
        frames_out,
        duration=duration_ms,
        loop=0,
        plugin="pillow",
        format="GIF",
    )

    size_kb = out_path.stat().st_size // 1024
    print(f"Wrote {out_path}  ({size_kb} KB, {len(frames_out)} frames, {args.fps} fps)")


if __name__ == "__main__":
    main()
