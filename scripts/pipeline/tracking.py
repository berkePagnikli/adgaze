"""Step 1-2: person detection (YOLO26) + multi-object tracking (ByteTrack) via Ultralytics."""

from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from . import WEIGHTS_DIR

PERSON_CLASS = 0


class PersonTracker:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = str(device)
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        # A missing official checkpoint (e.g. yolo26s.pt) is downloaded by Ultralytics to this path.
        self.model = YOLO(str(WEIGHTS_DIR / cfg.yolo_weights))

    def __call__(self, frame: np.ndarray) -> list[dict]:
        """Returns [{'track_id', 'box' (x1,y1,x2,y2), 'conf'}] for the tracked persons of this frame."""
        res = self.model.track(frame, persist=True, tracker=self.cfg.tracker, classes=[PERSON_CLASS],
                               conf=self.cfg.person_conf, imgsz=self.cfg.imgsz, device=self.device,
                               verbose=False)[0]
        boxes = res.boxes
        if boxes is None or boxes.id is None:
            return []
        ids = boxes.id.int().cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        return [{"track_id": int(t), "box": b.astype(float), "conf": float(c)} for t, b, c in zip(ids, xyxy, conf)]
