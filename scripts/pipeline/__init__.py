"""Audience analytics pipeline: YOLO26 + ByteTrack + YuNet + 6DRepNet + L2CS-Net.

Step 1-2  tracking.py   person detection (YOLO26) + tracking (ByteTrack)
Step 3    face.py       face detection (YuNet), face -> person assignment
Step 4A   head_pose.py  head pose (6DRepNet)
Step 4B   gaze.py       gaze direction (L2CS-Net)
Step 5    decision.py   looking / not looking / uncertain per frame
Step 6    temporal.py   temporal filter + dwell / watch time per track
Step 7    analytics.py  per-person table and video summary
          pipeline.py   runs the steps above frame by frame
"""

from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = SCRIPTS_DIR / "weights"
RUNS_DIR = SCRIPTS_DIR / "runs"
DATA_DIR = SCRIPTS_DIR / "data"
GT_DIR = SCRIPTS_DIR / "gt"
