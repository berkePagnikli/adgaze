"""Main entry point for running the AdGaze pipeline.

Usage:
    python -m adgaze                     # webcam, default config
    python -m adgaze --source video.mp4  # video file
    python -m adgaze --no-display        # headless (logging only)
"""

import argparse
import logging
import sys

from .pipeline.cascade import VideoProcessor
from .storage.db import Storage


def main():
    parser = argparse.ArgumentParser(description="AdGaze Engagement Pipeline")
    parser.add_argument("--config", default="adgaze/config/pipeline.yaml",
                        help="Path to pipeline config YAML")
    parser.add_argument("--source", default=None,
                        help="Video source (camera index or file path). Overrides config.")
    parser.add_argument("--no-display", action="store_true",
                        help="Run headless without visualization window")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Stop after N frames (0 = unlimited)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # Parse source — numeric string → int (camera index)
    source = args.source
    if source is not None:
        try:
            source = int(source)
        except ValueError:
            pass

    processor = VideoProcessor(args.config)
    processor.storage = Storage("data/adgaze.db")

    print("AdGaze Pipeline — press 'q' to quit")
    print(f"Config: {args.config}")
    print(f"Source: {source or 'default webcam'}")
    print(f"Display: {'off' if args.no_display else 'on'}")
    print()

    frames = processor.run(
        source=source,
        display=not args.no_display,
        max_frames=args.max_frames,
    )
    print(f"\nProcessed {frames} frames.")


if __name__ == "__main__":
    main()
