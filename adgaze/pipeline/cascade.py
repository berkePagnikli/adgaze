"""Cascade Orchestrator.

Runs stages 0–3 in sequence with gating. Stages 4–5 are stubbed
(disabled by default) pending model training and data collection.
"""

import logging
import time

import cv2
import numpy as np
import yaml

from .stage_motion import MotionDetector
from .stage_person import PersonDetector
from .stage_face import FaceDetector
from .stage_headpose import HeadPoseEstimator
from .tracker import Tracker

logger = logging.getLogger("adgaze.cascade")


class CascadeResult:
    """Container for single-frame pipeline output."""

    def __init__(self):
        self.timestamp: float = 0.0
        self.frame_index: int = 0
        self.stage_reached: int = -1  # highest stage executed
        self.motion: dict = {}
        self.person: dict = {}
        self.face: dict = {}
        self.head_pose: dict = {}
        self.active_tracks: list = []
        self.latencies: dict = {}  # per-stage latency in ms

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "frame_index": self.frame_index,
            "stage_reached": self.stage_reached,
            "motion": self.motion,
            "person": self.person,
            "face": self.face,
            "head_pose": self.head_pose,
            "active_tracks": len(self.active_tracks),
            "latencies_ms": self.latencies,
        }


class Cascade:
    """Orchestrates the gated cascade pipeline."""

    def __init__(self, config_path: str | None = None, config: dict | None = None):
        if config is None:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
        self.config = config

        self.motion_detector = MotionDetector(config)
        self.person_detector = PersonDetector(config)
        self.face_detector = FaceDetector(config)
        self.head_pose_estimator = HeadPoseEstimator(config)
        self.tracker = Tracker(config)

        self._frame_index = 0
        # Temporal smoothing: require N consecutive not-facing frames to gate
        self._facing_smooth_count = 0
        self._facing_smooth_threshold = 5  # frames

    def process_frame(self, frame: np.ndarray) -> CascadeResult:
        """Run the cascade pipeline on a single frame.

        Returns a CascadeResult with all intermediate outputs and timing.
        """
        result = CascadeResult()
        result.timestamp = time.time()
        result.frame_index = self._frame_index
        self._frame_index += 1

        # ---------- Stage 0: Motion Detection ----------
        t0 = time.perf_counter()
        result.motion = self.motion_detector.detect(frame)
        result.latencies["stage0_motion"] = round((time.perf_counter() - t0) * 1000, 2)
        result.stage_reached = 0

        if not result.motion["motion_detected"]:
            logger.debug("Frame %d: idle (no motion)", result.frame_index)
            return result

        # ---------- Stage 1: Person Detection ----------
        t0 = time.perf_counter()
        result.person = self.person_detector.detect(frame)
        result.latencies["stage1_person"] = round((time.perf_counter() - t0) * 1000, 2)
        result.stage_reached = 1

        if not result.person["persons_detected"]:
            logger.debug("Frame %d: no person detected", result.frame_index)
            # Update tracker with empty detections
            result.active_tracks = self.tracker.update([], result.timestamp)
            return result

        # Update tracker with person detections
        result.active_tracks = self.tracker.update(
            result.person["person_boxes"], result.timestamp
        )

        # ---------- Stage 2: Face Detection ----------
        t0 = time.perf_counter()
        result.face = self.face_detector.detect(frame, result.person["person_boxes"])
        result.latencies["stage2_face"] = round((time.perf_counter() - t0) * 1000, 2)
        result.stage_reached = 2

        if not result.face["faces_detected"]:
            logger.debug("Frame %d: person present but no face (not facing)", result.frame_index)
            return result

        # ---------- Stage 3: Head Pose Estimation ----------
        t0 = time.perf_counter()
        result.head_pose = self.head_pose_estimator.estimate(frame, result.face["faces"])
        result.latencies["stage3_headpose"] = round((time.perf_counter() - t0) * 1000, 2)

        # Log head pose for all detected faces
        for fp in result.head_pose.get("face_poses", []):
            logger.info(
                "Frame %d: yaw=%.1f pitch=%.1f facing=%s",
                result.frame_index,
                fp.get("yaw", 0), fp.get("pitch", 0), fp.get("facing_ad"),
            )

        # Temporal smoothing: require several consecutive not-facing frames
        if result.head_pose["facing_ad"]:
            self._facing_smooth_count = 0
            result.stage_reached = 3
        else:
            self._facing_smooth_count += 1
            if self._facing_smooth_count >= self._facing_smooth_threshold:
                # Confirmed not facing — gate at stage 2
                result.stage_reached = 2
                return result
            else:
                # Not enough consecutive frames — treat as still facing
                result.stage_reached = 3

        # Update tracker face/head stats
        for fp in result.head_pose.get("face_poses", []):
            if fp["facing_ad"]:
                # Find the matching track by IoU with face's person_bbox
                for track in result.active_tracks:
                    if track.bbox == fp.get("face_bbox"):
                        track.facing_ad_frames += 1
                        break

        # Stages 4 & 5 are disabled — would run gaze estimation and engagement scoring here.
        logger.debug(
            "Frame %d: person facing ad (yaw=%.1f, pitch=%.1f) — stages 4-5 skipped",
            result.frame_index,
            result.head_pose["face_poses"][0].get("yaw", 0),
            result.head_pose["face_poses"][0].get("pitch", 0),
        )

        return result


class VideoProcessor:
    """Process a video source (webcam or file) through the cascade pipeline."""

    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.cascade = Cascade(config=self.config)
        self.storage = None  # Set externally if needed

    def run(self, source=None, display: bool = True, max_frames: int = 0):
        """Run the pipeline on a video source.

        Args:
            source: Camera index or video file path. None = use config.
            display: Show annotated frames in a window.
            max_frames: Stop after N frames (0 = unlimited).
        """
        if source is None:
            source = self.config["camera"]["source"]

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config["camera"].get("width", 1280))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config["camera"].get("height", 720))

        frame_count = 0
        fps_start = time.time()
        fps_frames = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                result = self.cascade.process_frame(frame)

                # Store event if storage is configured
                if self.storage is not None:
                    self.storage.log_frame_event(result)

                # Display
                if display:
                    annotated = self._annotate_frame(frame, result)

                    # FPS counter
                    fps_frames += 1
                    elapsed = time.time() - fps_start
                    if elapsed >= 1.0:
                        fps = fps_frames / elapsed
                        fps_start = time.time()
                        fps_frames = 0
                    else:
                        fps = fps_frames / max(elapsed, 0.001)

                    cv2.putText(annotated, f"FPS: {fps:.1f} | Stage: {result.stage_reached}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    cv2.imshow("AdGaze Pipeline", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                frame_count += 1
                if max_frames and frame_count >= max_frames:
                    break
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()

        return frame_count

    def _annotate_frame(self, frame: np.ndarray, result: CascadeResult) -> np.ndarray:
        """Draw bounding boxes and info on the frame."""
        annotated = frame.copy()

        # Person boxes (green)
        for person in result.person.get("person_boxes", []):
            x1, y1, x2, y2 = person["bbox"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"person {person['confidence']:.2f}",
                        (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Face boxes (blue)
        for face in result.face.get("faces", []):
            x1, y1, x2, y2 = face["bbox"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Draw landmarks
            if face.get("landmarks"):
                for lx, ly in face["landmarks"]:
                    cv2.circle(annotated, (int(lx), int(ly)), 2, (0, 0, 255), -1)

        # Head pose info
        for fp in result.head_pose.get("face_poses", []):
            x1, y1, _, _ = fp.get("face_bbox", [0, 0, 0, 0])
            color = (0, 255, 0) if fp["facing_ad"] else (0, 0, 255)
            label = f"Y:{fp.get('yaw', '?')} P:{fp.get('pitch', '?')}"
            cv2.putText(annotated, label, (x1, y1 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Latency info
        y_offset = 60
        for stage, ms in result.latencies.items():
            cv2.putText(annotated, f"{stage}: {ms:.1f}ms",
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            y_offset += 20

        return annotated
