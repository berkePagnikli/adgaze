"""Step 6: temporal filtering of the per-frame decisions and dwell / watch time per track ID."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from .decision import LOOKING, UNCERTAIN


@dataclass
class TrackState:
    first_frame: int
    last_frame: int = 0
    frames: int = 0                 # frames the person was tracked  -> dwell time
    looking_frames: int = 0         # frames with filtered state LOOKING -> watch time
    face_frames: int = 0
    recent: deque = field(default_factory=deque)  # last certain raw decisions
    state: int = 0                  # filtered state (0/1)
    since_certain: int = 0          # consecutive uncertain frames


class WatchTimeCounter:
    def __init__(self, cfg, fps: float):
        self.cfg = cfg
        self.fps = fps
        self.tracks: dict[int, TrackState] = {}

    def update(self, track_id: int, frame_idx: int, raw: int) -> int:
        """Feeds one raw decision (1/0/-1) of a track and returns its filtered state (1/0).

        Certain decisions: majority vote over the last ``smooth_window`` certain decisions.
        Uncertain frames (no face): keep the last state for ``hold_frames``, then NOT_LOOKING.
        """
        t = self.tracks.get(track_id)
        if t is None:
            t = self.tracks[track_id] = TrackState(first_frame=frame_idx, recent=deque(maxlen=self.cfg.smooth_window))
        t.frames += 1
        t.last_frame = frame_idx
        if raw == UNCERTAIN:
            t.since_certain += 1
            if t.since_certain > self.cfg.hold_frames:
                t.state = 0
                t.recent.clear()
        else:
            t.face_frames += 1
            t.since_certain = 0
            t.recent.append(raw)
            t.state = int(2 * sum(1 for r in t.recent if r == LOOKING) > len(t.recent))
        t.looking_frames += t.state
        return t.state

    def dwell_s(self, track_id: int) -> float:
        return self.tracks[track_id].frames / self.fps

    def watch_s(self, track_id: int) -> float:
        return self.tracks[track_id].looking_frames / self.fps
