"""Stage 0: Motion Detection Gate.

Uses background subtraction (MOG2) or frame differencing to detect motion.
Near-zero compute cost — pure CPU, no neural network.
"""

import cv2
import numpy as np


class MotionDetector:
    def __init__(self, config: dict):
        cfg = config["pipeline"]["motion"]
        self.enabled = cfg.get("enabled", True)
        self.method = cfg.get("method", "mog2")
        self.pixel_change_threshold = cfg.get("pixel_change_threshold", 0.02)
        self.consecutive_idle_frames = cfg.get("consecutive_idle_frames", 10)
        self._idle_count = 0

        if self.method == "mog2":
            self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=cfg.get("mog2_history", 500),
                varThreshold=cfg.get("mog2_var_threshold", 16),
                detectShadows=cfg.get("mog2_detect_shadows", False),
            )
        else:
            self._prev_gray = None

    def detect(self, frame: np.ndarray) -> dict:
        """Return dict with 'motion_detected' bool and 'pixel_change_ratio' float."""
        if not self.enabled:
            return {"motion_detected": True, "pixel_change_ratio": 1.0}

        if self.method == "mog2":
            fg_mask = self._bg_subtractor.apply(frame)
            ratio = np.count_nonzero(fg_mask) / fg_mask.size
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            if self._prev_gray is None:
                self._prev_gray = gray
                return {"motion_detected": True, "pixel_change_ratio": 1.0}
            diff = cv2.absdiff(self._prev_gray, gray)
            self._prev_gray = gray
            _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            ratio = np.count_nonzero(thresh) / thresh.size

        motion_detected = ratio >= self.pixel_change_threshold
        if motion_detected:
            self._idle_count = 0
        else:
            self._idle_count += 1

        # Gate: only truly idle after N consecutive no-motion frames
        effectively_idle = self._idle_count >= self.consecutive_idle_frames

        return {
            "motion_detected": not effectively_idle,
            "pixel_change_ratio": float(ratio),
        }
