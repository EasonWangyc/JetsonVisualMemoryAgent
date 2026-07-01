from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frames from a video into data/images.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--out-dir", default="data/images")
    parser.add_argument("--every", type=int, default=30, help="Save one frame every N frames.")
    parser.add_argument("--prefix", default="frame")
    return parser.parse_args()


def main() -> None:
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("opencv-python is required for frame extraction.") from exc

    args = parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")
    frame_id = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_id % args.every == 0:
            path = out / f"{args.prefix}_{saved:05d}.jpg"
            cv2.imwrite(str(path), frame)
            saved += 1
        frame_id += 1
    cap.release()
    print(f"Saved {saved} frames to {out}")


if __name__ == "__main__":
    main()
