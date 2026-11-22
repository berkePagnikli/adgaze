"""Step 4B: gaze direction (L2CS-Net, ResNet50 trained on Gaze360)."""

from __future__ import annotations

import cv2
import numpy as np
import torch

from .face import crop
from .weights import weight_path


class GazeEstimator:
    def __init__(self, cfg, device: torch.device):
        from l2cs import getArch
        from l2cs.utils import prep_input_numpy
        from safetensors.torch import load_file

        self.cfg = cfg
        self.device = device
        self.prep = prep_input_numpy
        self.model = getArch("ResNet50", 90)
        self.model.load_state_dict(load_file(str(weight_path("l2cs_gaze360_resnet50.safetensors"))))
        self.model.to(device).eval()
        self.bins = torch.arange(90, dtype=torch.float32, device=device)

    @torch.no_grad()
    def __call__(self, frame, face_box) -> dict | None:
        """Returns {'yaw', 'pitch'} in degrees (0, 0 = looking straight into the camera)."""
        x1, y1, x2, y2 = face_box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        # Square crop: the official code stretches the (taller than wide) face box to 224x224,
        # which distorts the face; a square crop keeps the aspect ratio.
        r = max(x2 - x1, y2 - y1) * self.cfg.gaze_crop_scale / 2
        img = crop(frame, (cx - r, cy - r, cx + r, cy + r))
        if img.size == 0:
            return None
        img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (224, 224))
        yaw_logits, pitch_logits = self.model(self.prep(img, self.device))  # network order: yaw, pitch
        yaw = (torch.softmax(yaw_logits, 1) * self.bins).sum(1) * 4 - 180
        pitch = (torch.softmax(pitch_logits, 1) * self.bins).sum(1) * 4 - 180
        return {"yaw": float(yaw[0]), "pitch": float(pitch[0])}


def gaze_arrow(yaw_deg: float, pitch_deg: float, length: float) -> tuple[float, float]:
    """2D image displacement of the gaze direction (same formula as the L2CS demo ``draw_gaze``)."""
    y, p = np.radians(yaw_deg), np.radians(pitch_deg)
    return -length * np.sin(y) * np.cos(p), -length * np.sin(p)
