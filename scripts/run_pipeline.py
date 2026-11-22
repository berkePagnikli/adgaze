"""Run the pipeline (YOLO26 + ByteTrack + YuNet + 6DRepNet + L2CS-Net) on videos or a webcam.

    python scripts/run_pipeline.py                                # every video in scripts/data
    python scripts/run_pipeline.py --source data/man.mp4          # one video
    python scripts/run_pipeline.py --source 0 --show              # webcam
    python scripts/run_pipeline.py --set gaze_yaw_max=15 --out runs/yaw15

Results: scripts/runs/<video>/{frames.csv, persons.csv, summary.json, annotated.mp4}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import DATA_DIR, RUNS_DIR, SCRIPTS_DIR  # noqa: E402
from pipeline.config import Config  # noqa: E402
from pipeline.pipeline import Pipeline  # noqa: E402


def resolve(path: str) -> Path:
    """Paths may be given relative to the current folder or to scripts/."""
    p = Path(path)
    return p if p.is_absolute() or p.exists() else SCRIPTS_DIR / p


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", action="append", help="video file or webcam index (repeatable); "
                                                      "default: all videos in scripts/data")
    ap.add_argument("--out", default=str(RUNS_DIR), help="output root (default scripts/runs)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config override")
    ap.add_argument("--no-video", action="store_true", help="do not write annotated.mp4")
    ap.add_argument("--show", action="store_true", help="show the annotated frames live")
    ap.add_argument("--max-frames", type=int)
    args = ap.parse_args()

    sources = args.source or sorted(str(p) for p in DATA_DIR.glob("*.mp4"))
    if not sources:
        raise SystemExit(f"No videos found in {DATA_DIR}")
    cfg = Config().update(args.set)
    pipe = Pipeline(cfg)
    out_root = resolve(args.out)
    for src in sources:
        src = src if src.isdigit() else str(resolve(src))
        name = f"webcam{src}" if src.isdigit() else Path(src).stem
        s = pipe.run(src, out_root / name, save_video=not args.no_video, max_frames=args.max_frames, show=args.show)
        print(f"[{name}] {s['n_frames']} frames @ {s['processing_fps']} fps | persons {s['persons']} | "
              f"viewers {s['viewers']} | total watch {s['total_watch_s']} s -> {out_root / name}")


if __name__ == "__main__":
    main()
