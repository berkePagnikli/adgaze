"""All pipeline parameters in one place. Override from the command line with ``--set key=value``."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields


@dataclass
class Config:
    device: str = "auto"                 # auto | cpu | cuda:0

    # Step 1-2: YOLO26 person detection + ByteTrack
    yolo_weights: str = "yolo26s.pt"     # downloaded to scripts/weights on first use
    tracker: str = "bytetrack.yaml"      # Ultralytics tracker config
    person_conf: float = 0.25
    imgsz: int = 640

    # Step 3: YuNet face detection
    face_score: float = 0.6
    face_region: float = 0.5             # face centre must lie in the top part of the person box

    # Step 4A: 6DRepNet (face box enlarged like the official demo)
    head_crop_margin: float = 0.2

    # Step 4B: L2CS-Net (square crop around the face box)
    gaze_crop_scale: float = 1.2

    # Step 5: looking decision. Camera is mounted on the screen -> "looking at the screen" means
    # gaze and head both point roughly at the camera. Offsets shift the target direction (degrees).
    gaze_yaw_max: float = 20.0
    gaze_pitch_max: float = 20.0
    gaze_yaw_offset: float = 0.0
    gaze_pitch_offset: float = 0.0
    head_yaw_max: float = 35.0
    head_pitch_max: float = 30.0

    # Step 6: temporal filter
    smooth_window: int = 5               # majority vote over the last N certain decisions
    hold_frames: int = 6                 # frames without a face that keep the last state

    # Step 7: analytics
    min_dwell_s: float = 0.5             # shorter tracks are ignored (false detections / passers-by)
    min_watch_s: float = 0.5             # a person counts as a viewer above this watch time

    def to_dict(self) -> dict:
        return asdict(self)

    def update(self, overrides: list[str]) -> "Config":
        types = {f.name: f.type for f in fields(self)}
        for item in overrides:
            key, _, value = item.partition("=")
            key = key.strip()
            if key not in types:
                raise SystemExit(f"Unknown config key '{key}'. Valid keys: {', '.join(types)}")
            current = getattr(self, key)
            setattr(self, key, type(current)(value) if not isinstance(current, str) else value)
        return self
