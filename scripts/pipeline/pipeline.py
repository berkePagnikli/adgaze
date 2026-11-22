"""Runs steps 1-7 frame by frame on a video file or webcam and writes the results of one run.

Outputs in ``out_dir``:
    frames.csv    one row per (frame, tracked person): boxes, head pose, gaze, raw + filtered decision
    persons.csv   one row per person (track): dwell time, watch time, viewer flag
    summary.json  video-level analytics + config + speed
    annotated.mp4 visualisation (optional)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import pandas as pd
import torch

from .analytics import person_table, summarize
from .config import Config
from .decision import LOOKING, UNCERTAIN, decide
from .face import FaceDetector
from .gaze import GazeEstimator, gaze_arrow
from .head_pose import HeadPoseEstimator
from .temporal import WatchTimeCounter
from .tracking import PersonTracker

COLORS = {LOOKING: (60, 200, 60), 0: (60, 60, 230)}


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class Pipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = select_device(cfg.device)
        print(f"[pipeline] device: {self.device}")
        self.face = FaceDetector(cfg)
        self.head = HeadPoseEstimator(cfg, self.device)
        self.gaze = GazeEstimator(cfg, self.device)

    def run(self, source: str, out_dir: Path, save_video: bool = True, max_frames: int | None = None,
            show: bool = False) -> dict:
        cfg = self.cfg
        is_cam = str(source).isdigit()
        name = f"webcam{source}" if is_cam else Path(source).stem
        cap = cv2.VideoCapture(int(source) if is_cam else str(source))
        if not cap.isOpened():
            raise SystemExit(f"[pipeline] cannot open video: {source}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_dir.mkdir(parents=True, exist_ok=True)
        writer = None
        if save_video:
            writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                     (width, height))

        tracker = PersonTracker(cfg, self.device)  # new tracker (fresh IDs) for every video
        counter = WatchTimeCounter(cfg, fps)
        rows = []
        frame_idx = 0
        t0 = time.perf_counter()
        while max_frames is None or frame_idx < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            persons = tracker(frame)                                     # step 1-2
            self.face(frame, persons)                                    # step 3
            for p in persons:
                head = gaze = None
                if p["face"] is not None:
                    head = self.head(frame, p["face"]["box"])            # step 4A
                    gaze = self.gaze(frame, p["face"]["box"])            # step 4B
                p["head"], p["gaze"] = head, gaze
                p["raw"] = decide(head, gaze, cfg)                       # step 5
                p["looking"] = counter.update(p["track_id"], frame_idx, p["raw"])  # step 6
                rows.append(_row(name, frame_idx, fps, p))
            if writer is not None or show:
                vis = self._draw(frame, persons, counter, frame_idx)
                if writer is not None:
                    writer.write(vis)
                if show:
                    cv2.imshow("pipeline (q: quit)", vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            frame_idx += 1
        elapsed = time.perf_counter() - t0
        cap.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

        persons_df = person_table(counter, cfg)                          # step 7
        pd.DataFrame(rows, columns=FRAME_COLUMNS).to_csv(out_dir / "frames.csv", index=False)
        persons_df.to_csv(out_dir / "persons.csv", index=False)
        summary = {
            "video": name,
            "fps": fps,
            "n_frames": frame_idx,
            "duration_s": round(frame_idx / fps, 2),
            "processing_fps": round(frame_idx / elapsed, 2) if elapsed else 0.0,
            **summarize(persons_df),
            "config": cfg.to_dict(),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    @staticmethod
    def _draw(frame, persons, counter, frame_idx):
        vis = frame.copy()
        for p in persons:
            x1, y1, x2, y2 = map(int, p["box"])
            col = COLORS[p["looking"]]
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
            tid = p["track_id"]
            label = f"ID{tid} {'LOOKING' if p['looking'] else 'not'}{' ?' if p['raw'] == UNCERTAIN else ''}"
            cv2.putText(vis, label, (x1, max(y1 - 24, 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            cv2.putText(vis, f"dwell {counter.dwell_s(tid):.1f}s watch {counter.watch_s(tid):.1f}s",
                        (x1, max(y1 - 6, 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
            if p["face"] is not None:
                fx1, fy1, fx2, fy2 = map(int, p["face"]["box"])
                cv2.rectangle(vis, (fx1, fy1), (fx2, fy2), (255, 255, 255), 1)
                if p["gaze"] is not None:
                    cx, cy = (fx1 + fx2) // 2, (fy1 + fy2) // 2
                    dx, dy = gaze_arrow(p["gaze"]["yaw"], p["gaze"]["pitch"], 2.0 * (fx2 - fx1))
                    cv2.arrowedLine(vis, (cx, cy), (int(cx + dx), int(cy + dy)), (0, 220, 255), 2, tipLength=0.2)
        cv2.putText(vis, f"frame {frame_idx}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return vis


FRAME_COLUMNS = ["video", "frame", "time_s", "track_id", "px1", "py1", "px2", "py2", "person_conf",
                 "fx1", "fy1", "fx2", "fy2", "face_score", "head_yaw", "head_pitch", "head_roll",
                 "gaze_yaw", "gaze_pitch", "raw", "looking"]


def _row(video: str, frame_idx: int, fps: float, p: dict) -> dict:
    face, head, gaze = p["face"], p["head"], p["gaze"]
    fb = face["box"] if face is not None else [None] * 4
    r1 = lambda v: None if v is None else round(float(v), 1)  # noqa: E731
    return {
        "video": video, "frame": frame_idx, "time_s": round(frame_idx / fps, 3), "track_id": p["track_id"],
        "px1": r1(p["box"][0]), "py1": r1(p["box"][1]), "px2": r1(p["box"][2]), "py2": r1(p["box"][3]),
        "person_conf": round(p["conf"], 3),
        "fx1": r1(fb[0]), "fy1": r1(fb[1]), "fx2": r1(fb[2]), "fy2": r1(fb[3]),
        "face_score": round(face["score"], 3) if face is not None else None,
        "head_yaw": r1(head and head["yaw"]), "head_pitch": r1(head and head["pitch"]),
        "head_roll": r1(head and head["roll"]),
        "gaze_yaw": r1(gaze and gaze["yaw"]), "gaze_pitch": r1(gaze and gaze["pitch"]),
        "raw": p["raw"], "looking": p["looking"],
    }
