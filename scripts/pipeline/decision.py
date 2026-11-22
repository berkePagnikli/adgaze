"""Step 5: per-frame "looking at the screen" decision from head pose + gaze.

The camera is mounted on the screen, so the screen direction is (yaw, pitch) ~ (0, 0) plus an
optional offset (e.g. a camera above the screen -> people look slightly below the camera).
"""

from __future__ import annotations

LOOKING, NOT_LOOKING, UNCERTAIN = 1, 0, -1


def decide(head: dict | None, gaze: dict | None, cfg) -> int:
    """LOOKING if the gaze points at the screen and the head is turned towards it.

    The head-pose check is a sanity gate: L2CS is unreliable for profile / turned-away faces.
    Frames without a face (or without estimates) are UNCERTAIN and handled by the temporal filter.
    """
    if head is None or gaze is None:
        return UNCERTAIN
    gaze_ok = (abs(gaze["yaw"] - cfg.gaze_yaw_offset) <= cfg.gaze_yaw_max
               and abs(gaze["pitch"] - cfg.gaze_pitch_offset) <= cfg.gaze_pitch_max)
    head_ok = abs(head["yaw"]) <= cfg.head_yaw_max and abs(head["pitch"]) <= cfg.head_pitch_max
    return LOOKING if gaze_ok and head_ok else NOT_LOOKING
