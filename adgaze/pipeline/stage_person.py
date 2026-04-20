"""Stage 1: Person Detection Gate.

Uses YOLOv8n (ONNX) to detect people in the frame.
"""

import cv2
import numpy as np
import onnxruntime as ort


class PersonDetector:
    # COCO class id for 'person'
    PERSON_CLASS_ID = 0

    def __init__(self, config: dict):
        cfg = config["pipeline"]["person_detection"]
        self.enabled = cfg.get("enabled", True)
        self.input_size = cfg.get("input_size", 640)
        self.conf_threshold = cfg.get("confidence_threshold", 0.5)
        self.nms_threshold = cfg.get("nms_threshold", 0.45)

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if cfg.get("device") == "cuda" else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(cfg["model_path"], providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, float, int, int]:
        h, w = frame.shape[:2]
        scale = min(self.input_size / w, self.input_size / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(frame, (new_w, new_h))

        padded = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w] = resized

        blob = padded.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[np.newaxis, ...]
        return blob, scale, scale, new_w, new_h

    def detect(self, frame: np.ndarray) -> dict:
        """Return dict with 'persons_detected': bool and 'person_boxes': list of [x1,y1,x2,y2,conf]."""
        if not self.enabled:
            return {"persons_detected": True, "person_boxes": [], "person_count": 0}

        h_orig, w_orig = frame.shape[:2]
        blob, sx, sy, new_w, new_h = self._preprocess(frame)

        outputs = self.session.run(None, {self.input_name: blob})
        # YOLOv8 output shape: (1, 84, N) — transpose to (N, 84)
        preds = outputs[0][0].T  # (N, 84)

        boxes, scores = [], []
        for pred in preds:
            cx, cy, bw, bh = pred[:4]
            class_scores = pred[4:]
            class_id = int(np.argmax(class_scores))
            conf = float(class_scores[class_id])

            if class_id != self.PERSON_CLASS_ID or conf < self.conf_threshold:
                continue

            x1 = (cx - bw / 2) / sx
            y1 = (cy - bh / 2) / sy
            x2 = (cx + bw / 2) / sx
            y2 = (cy + bh / 2) / sy

            # Clamp to frame bounds
            x1 = max(0, min(x1, w_orig))
            y1 = max(0, min(y1, h_orig))
            x2 = max(0, min(x2, w_orig))
            y2 = max(0, min(y2, h_orig))

            boxes.append([x1, y1, x2 - x1, y2 - y1])
            scores.append(conf)

        if not boxes:
            return {"persons_detected": False, "person_boxes": [], "person_count": 0}

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.nms_threshold)
        if len(indices) == 0:
            return {"persons_detected": False, "person_boxes": [], "person_count": 0}

        result_boxes = []
        indices = indices.flatten()
        for idx in indices:
            x, y, w, h = boxes[idx]
            result_boxes.append({
                "bbox": [int(x), int(y), int(x + w), int(y + h)],
                "confidence": round(scores[idx], 3),
            })

        return {
            "persons_detected": True,
            "person_boxes": result_boxes,
            "person_count": len(result_boxes),
        }
