"""Stage 3: Head Pose Estimation Gate.

Estimates yaw and pitch directly from facial landmark geometry.
No neural network — pure 2D landmark analysis.

YuNet landmark order:
  [0] right eye, [1] left eye, [2] nose tip,
  [3] right mouth corner, [4] left mouth corner

Key insight: when the head turns, the nose shifts away from the
midpoint between the eyes. The ratio of that offset to the
inter-eye distance gives a robust yaw estimate.
"""

import numpy as np


class HeadPoseEstimator:
    def __init__(self, config: dict):
        cfg = config["pipeline"]["head_pose"]
        self.enabled = cfg.get("enabled", True)
        self.yaw_threshold = cfg.get("yaw_threshold", 30.0)
        self.pitch_threshold = cfg.get("pitch_threshold", 30.0)

    @staticmethod
    def _estimate_yaw_pitch(landmarks: list[tuple[float, float]]) -> tuple[float, float]:
        """Estimate yaw and pitch from 5 facial landmarks using geometry.

        Yaw:  nose horizontal offset from eye midpoint, normalized by eye distance.
        Pitch: nose vertical position relative to eyes-to-mouth span.
        """
        r_eye = np.array(landmarks[0])
        l_eye = np.array(landmarks[1])
        nose = np.array(landmarks[2])
        r_mouth = np.array(landmarks[3])
        l_mouth = np.array(landmarks[4])

        eye_center = (r_eye + l_eye) / 2.0
        eye_dist = np.linalg.norm(r_eye - l_eye)

        if eye_dist < 1.0:
            # Eyes too close together (degenerate detection)
            return 0.0, 0.0

        # --- Yaw ---
        # Nose offset from eye center, normalized by inter-eye distance.
        # The nose depth relative to the eye plane is ~0.35 of inter-eye distance,
        # so we use 0.38 as normalizer to map the 2D offset to an angle estimate.
        nose_offset_x = nose[0] - eye_center[0]
        yaw_ratio = np.clip(nose_offset_x / (eye_dist * 0.38), -1.0, 1.0)
        yaw = np.degrees(np.arcsin(yaw_ratio))

        # --- Pitch ---
        mouth_center = (r_mouth + l_mouth) / 2.0
        face_height = mouth_center[1] - eye_center[1]

        if face_height < 1.0:
            return yaw, 0.0

        # Expected nose Y: about 45% of the way from eyes to mouth
        expected_nose_y = eye_center[1] + face_height * 0.45
        nose_offset_y = nose[1] - expected_nose_y
        pitch_ratio = np.clip(nose_offset_y / (face_height * 0.5), -1.0, 1.0)
        pitch = np.degrees(np.arcsin(pitch_ratio))

        return float(yaw), float(pitch)

    def estimate(self, frame: np.ndarray, faces: list[dict]) -> dict:
        """Estimate head pose for each face with landmarks.

        Returns dict with 'facing_ad': bool and 'face_poses': list of pose dicts.
        """
        if not self.enabled:
            return {"facing_ad": True, "face_poses": []}

        face_poses = []

        for face in faces:
            landmarks = face.get("landmarks")
            if landmarks is None or len(landmarks) < 5:
                face_poses.append({
                    "face_bbox": face["bbox"],
                    "yaw": None, "pitch": None,
                    "facing_ad": False,
                    "reason": "no_landmarks",
                })
                continue

            yaw, pitch = self._estimate_yaw_pitch(landmarks)
            facing = (abs(yaw) <= self.yaw_threshold and abs(pitch) <= self.pitch_threshold)

            face_poses.append({
                "face_bbox": face["bbox"],
                "landmarks": landmarks,
                "yaw": round(yaw, 1),
                "pitch": round(pitch, 1),
                "facing_ad": facing,
            })

        any_facing = any(fp["facing_ad"] for fp in face_poses)

        return {
            "facing_ad": any_facing,
            "face_poses": face_poses,
        }
