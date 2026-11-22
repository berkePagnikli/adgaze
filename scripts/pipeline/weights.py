"""Model weights: stored in scripts/weights and downloaded on first use."""

from __future__ import annotations

import urllib.request
from pathlib import Path

from . import WEIGHTS_DIR

URLS = {
    "face_detection_yunet_2023mar.onnx":
        "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "6DRepNet_300W_LP_AFLW2000.pth":
        "https://cloud.ovgu.de/s/Q67RnLDy6JKLRWm/download/6DRepNet_300W_LP_AFLW2000.pth",
    # Official L2CSNet_gaze360.pkl (Google Drive) re-hosted unchanged as safetensors by py-feat.
    "l2cs_gaze360_resnet50.safetensors":
        "https://huggingface.co/py-feat/l2cs/resolve/main/l2cs_gaze360_resnet50.safetensors",
}


def weight_path(name: str) -> Path:
    path = WEIGHTS_DIR / name
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[weights] downloading {name} ...")
    tmp = path.with_suffix(path.suffix + ".part")
    req = urllib.request.Request(URLS[name], headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as fh:
        fh.write(resp.read())
    if tmp.stat().st_size < 10_000:
        tmp.unlink()
        raise RuntimeError(f"Download of {name} failed (file too small). Put it manually into {WEIGHTS_DIR}")
    tmp.replace(path)
    return path
