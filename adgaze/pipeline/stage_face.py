"""Stage 2: Face Detection Gate.

Uses OpenCV's YuNet face detector to detect faces within person bounding boxes.
YuNet outputs face bounding boxes + 5 facial landmarks (eyes, nose, mouth corners).
Operates on cropped person regions to save compute.
"""

import cv2
import numpy as np


class FaceDetector:
    def __init__(self, config: dict):
        cfg = config["pipeline"]["face_detection"]
        self.enabled = cfg.get("enabled", True)
        self.input_size = cfg.get("input_size", 320)
        self.conf_threshold = cfg.get("confidence_threshold", 0.5)
        self.nms_threshold = cfg.get("nms_threshold", 0.3)
        self._model_path = cfg["model_path"]

        # YuNet detector — we'll recreate per-crop with the right input size
        self._create_detector(self.input_size, self.input_size)

    def _create_detector(self, width: int, height: int):
        self._detector = cv2.FaceDetectorYN.create(
            self._model_path,
            "",
            (width, height),
            self.conf_threshold,
            self.nms_threshold,
        )

    def detect(self, frame: np.ndarray, person_boxes: list[dict]) -> dict:
        """Detect faces within each person bounding box.

        Returns dict with 'faces_detected': bool and 'faces': list of face dicts.
        Each face dict includes 'bbox' in full-frame coords and 'landmarks' (5 points).
        """
        if not self.enabled:
            return {"faces_detected": True, "faces": []}

        all_faces: list[dict] = []
        for person in person_boxes:
            x1, y1, x2, y2 = person["bbox"]
            # Add padding to person crop
            pw, ph = x2 - x1, y2 - y1
            pad_x, pad_y = int(pw * 0.1), int(ph * 0.05)
            cx1 = max(0, x1 - pad_x)
            cy1 = max(0, y1 - pad_y)
            cx2 = min(frame.shape[1], x2 + pad_x)
            cy2 = min(frame.shape[0], y2 + pad_y)

            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue

            ch, cw = crop.shape[:2]
            self._create_detector(cw, ch)

            _, detections = self._detector.detect(crop)

            if detections is None:
                continue

            # YuNet output per detection: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rcm, y_rcm, x_lcm, y_lcm, score]
            # re=right eye, le=left eye, nt=nose tip, rcm=right corner mouth, lcm=left corner mouth
            for det in detections:
                conf = float(det[14])
                if conf < self.conf_threshold:
                    continue

                fx, fy, fw, fh = int(det[0]), int(det[1]), int(det[2]), int(det[3])

                # Landmarks in full-frame coordinates
                landmarks = [
                    (float(det[4]) + cx1, float(det[5]) + cy1),   # right eye
                    (float(det[6]) + cx1, float(det[7]) + cy1),   # left eye
                    (float(det[8]) + cx1, float(det[9]) + cy1),   # nose tip
                    (float(det[10]) + cx1, float(det[11]) + cy1), # right mouth corner
                    (float(det[12]) + cx1, float(det[13]) + cy1), # left mouth corner
                ]

                all_faces.append({
                    "bbox": [fx + cx1, fy + cy1, fx + fw + cx1, fy + fh + cy1],
                    "confidence": round(conf, 3),
                    "landmarks": landmarks,
                    "person_bbox": person["bbox"],
                })

        return {
            "faces_detected": len(all_faces) > 0,
            "faces": all_faces,
            "face_count": len(all_faces),
        }
