"""Step 3: face detection (YuNet, OpenCV Zoo) and assignment of faces to tracked persons."""

from __future__ import annotations

import cv2
import numpy as np

from .weights import weight_path


class FaceDetector:
    def __init__(self, cfg):
        self.cfg = cfg
        # Loaded from memory: OpenCV's ONNX reader fails on non-ASCII paths on Windows ("Masaüstü").
        data = np.frombuffer(weight_path("face_detection_yunet_2023mar.onnx").read_bytes(), dtype=np.uint8)
        self.det = cv2.FaceDetectorYN.create("onnx", data, np.array([], dtype=np.uint8), (320, 320),
                                             cfg.face_score, 0.3, 5000)

    def detect(self, frame: np.ndarray) -> list[dict]:
        h, w = frame.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(frame)
        if faces is None:
            return []
        return [{"box": np.array([f[0], f[1], f[0] + f[2], f[1] + f[3]], dtype=float), "score": float(f[14])}
                for f in faces]

    def __call__(self, frame: np.ndarray, persons: list[dict]) -> None:
        """Detects faces on the full frame and sets ``person['face']`` (dict or None) for every person.

        A face belongs to a person if its centre lies inside the top ``face_region`` part of the
        person box. Highest-scoring faces are assigned first; each person gets at most one face and
        competing persons are resolved by the horizontal distance to the box centre.
        """
        for p in persons:
            p["face"] = None
        if not persons:
            return
        for face in sorted(self.detect(frame), key=lambda f: -f["score"]):
            cx = (face["box"][0] + face["box"][2]) / 2
            cy = (face["box"][1] + face["box"][3]) / 2
            best, best_dx = None, np.inf
            for p in persons:
                x1, y1, x2, y2 = p["box"]
                inside = x1 <= cx <= x2 and y1 <= cy <= y1 + self.cfg.face_region * (y2 - y1)
                dx = abs(cx - (x1 + x2) / 2) / max(x2 - x1, 1.0)
                if inside and p["face"] is None and dx < best_dx:
                    best, best_dx = p, dx
            if best is not None:
                best["face"] = face


def crop(frame: np.ndarray, box) -> np.ndarray:
    """Crop ``box`` clamped to the image (empty array if the box is outside)."""
    h, w = frame.shape[:2]
    x1, y1 = max(0, int(box[0])), max(0, int(box[1]))
    x2, y2 = min(w, int(round(box[2]))), min(h, int(round(box[3])))
    return frame[y1:y2, x1:x2]
