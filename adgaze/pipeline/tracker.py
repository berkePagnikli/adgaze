"""Simple IoU-based multi-person tracker.

Associates detections across frames using IoU matching.
Maintains per-person state for temporal analytics.
"""

import time
from dataclasses import dataclass, field


@dataclass
class Track:
    track_id: int
    bbox: list[int]                  # [x1, y1, x2, y2]
    confidence: float
    first_seen: float                # timestamp
    last_seen: float                 # timestamp
    frames_seen: int = 1
    frames_lost: int = 0
    active: bool = True
    # Head pose history
    face_visible_frames: int = 0
    facing_ad_frames: int = 0
    # Extra metadata accumulated over lifetime
    metadata: dict = field(default_factory=dict)


def _iou(box_a: list[int], box_b: list[int]) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)

    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


class Tracker:
    def __init__(self, config: dict):
        cfg = config.get("tracker", {})
        self.max_lost_frames = cfg.get("max_lost_frames", 30)
        self.iou_threshold = cfg.get("iou_threshold", 0.3)
        self._next_id = 1
        self.tracks: dict[int, Track] = {}

    def update(self, detections: list[dict], timestamp: float | None = None) -> list[Track]:
        """Match detections to existing tracks via greedy IoU assignment.

        Args:
            detections: list of dicts with 'bbox' key ([x1,y1,x2,y2]).
            timestamp: current time (defaults to time.time()).

        Returns:
            List of active tracks after update.
        """
        now = timestamp or time.time()

        if not detections:
            # Mark all tracks as lost for this frame
            for track in self.tracks.values():
                if track.active:
                    track.frames_lost += 1
                    if track.frames_lost > self.max_lost_frames:
                        track.active = False
            return [t for t in self.tracks.values() if t.active]

        det_boxes = [d["bbox"] for d in detections]
        active_tracks = [t for t in self.tracks.values() if t.active]

        # Compute IoU matrix
        matched_det = set()
        matched_track = set()

        if active_tracks:
            iou_pairs = []
            for ti, track in enumerate(active_tracks):
                for di, det_box in enumerate(det_boxes):
                    score = _iou(track.bbox, det_box)
                    if score >= self.iou_threshold:
                        iou_pairs.append((score, ti, di))

            # Greedy matching — best IoU first
            iou_pairs.sort(key=lambda x: x[0], reverse=True)
            for score, ti, di in iou_pairs:
                if ti in matched_track or di in matched_det:
                    continue
                # Update track
                track = active_tracks[ti]
                track.bbox = det_boxes[di]
                track.confidence = detections[di].get("confidence", 0.0)
                track.last_seen = now
                track.frames_seen += 1
                track.frames_lost = 0
                matched_track.add(ti)
                matched_det.add(di)

        # Increment lost counter for unmatched active tracks
        for ti, track in enumerate(active_tracks):
            if ti not in matched_track:
                track.frames_lost += 1
                if track.frames_lost > self.max_lost_frames:
                    track.active = False

        # Create new tracks for unmatched detections
        for di, det in enumerate(detections):
            if di not in matched_det:
                track = Track(
                    track_id=self._next_id,
                    bbox=det["bbox"],
                    confidence=det.get("confidence", 0.0),
                    first_seen=now,
                    last_seen=now,
                )
                self.tracks[self._next_id] = track
                self._next_id += 1

        return [t for t in self.tracks.values() if t.active]

    def get_track(self, track_id: int) -> Track | None:
        return self.tracks.get(track_id)

    def get_completed_tracks(self) -> list[Track]:
        """Return tracks that are no longer active (for final logging)."""
        return [t for t in self.tracks.values() if not t.active]

    def cleanup(self):
        """Remove inactive tracks from memory."""
        self.tracks = {tid: t for tid, t in self.tracks.items() if t.active}
