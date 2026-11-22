"""Step 4A: head pose (6DRepNet, official ``sixdrepnet`` package, 300W-LP weights)."""

from __future__ import annotations

import torch

from .face import crop
from .weights import weight_path


class HeadPoseEstimator:
    def __init__(self, cfg, device: torch.device):
        from sixdrepnet import SixDRepNet

        self.margin = cfg.head_crop_margin
        gpu_id = (device.index or 0) if device.type == "cuda" else -1
        self.model = SixDRepNet(gpu_id=gpu_id, dict_path=str(weight_path("6DRepNet_300W_LP_AFLW2000.pth")))

    @torch.no_grad()
    def __call__(self, frame, face_box) -> dict | None:
        """Returns {'yaw', 'pitch', 'roll'} in degrees (0, 0, 0 = frontal face)."""
        x1, y1, x2, y2 = face_box
        w, h = x2 - x1, y2 - y1
        # Same enlargement as the official 6DRepNet demo (margin of the other side length).
        img = crop(frame, (x1 - self.margin * h, y1 - self.margin * w, x2 + self.margin * h, y2 + self.margin * w))
        if img.size == 0:
            return None
        pitch, yaw, roll = self.model.predict(img)
        return {"yaw": float(yaw[0]), "pitch": float(pitch[0]), "roll": float(roll[0])}
