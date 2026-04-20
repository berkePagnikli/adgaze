"""SQLite storage for engagement events.

Stores per-frame and per-session analytics data.
No raw images or biometric data are ever stored (privacy by design).
"""

import sqlite3
import json
import threading
from pathlib import Path

import numpy as np


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _dumps(obj):
    return json.dumps(obj, cls=_NumpyEncoder)


class Storage:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db_path = db_path
        self._local = threading.local()
        self._init_db()

    @property
    def _conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self._db_path)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn

    def _init_db(self):
        conn = sqlite3.connect(self._db_path)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS frame_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                frame_index INTEGER NOT NULL,
                stage_reached INTEGER NOT NULL,
                motion_detected INTEGER,
                person_count INTEGER DEFAULT 0,
                face_count INTEGER DEFAULT 0,
                any_facing_ad INTEGER DEFAULT 0,
                latencies_json TEXT,
                details_json TEXT
            );

            CREATE TABLE IF NOT EXISTS track_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id INTEGER NOT NULL,
                first_seen REAL NOT NULL,
                last_seen REAL NOT NULL,
                duration_sec REAL NOT NULL,
                frames_seen INTEGER NOT NULL,
                face_visible_frames INTEGER DEFAULT 0,
                facing_ad_frames INTEGER DEFAULT 0,
                metadata_json TEXT
            );

            CREATE TABLE IF NOT EXISTS session_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_start REAL NOT NULL,
                session_end REAL NOT NULL,
                total_frames INTEGER NOT NULL,
                avg_fps REAL,
                total_persons INTEGER DEFAULT 0,
                total_faces INTEGER DEFAULT 0,
                total_facing_ad INTEGER DEFAULT 0,
                avg_stage_reached REAL,
                stage_distribution_json TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_frame_ts ON frame_events(timestamp);
            CREATE INDEX IF NOT EXISTS idx_track_id ON track_events(track_id);
        """)
        conn.commit()
        conn.close()

    def log_frame_event(self, result) -> None:
        """Log a single CascadeResult to the database."""
        self._conn.execute(
            """INSERT INTO frame_events
               (timestamp, frame_index, stage_reached, motion_detected,
                person_count, face_count, any_facing_ad, latencies_json, details_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                result.timestamp,
                result.frame_index,
                result.stage_reached,
                int(result.motion.get("motion_detected", False)),
                result.person.get("person_count", 0),
                result.face.get("face_count", 0),
                int(result.head_pose.get("facing_ad", False)),
                _dumps(result.latencies),
                _dumps(result.to_dict()),
            ),
        )
        self._conn.commit()

    def log_track(self, track) -> None:
        """Log a completed track."""
        self._conn.execute(
            """INSERT INTO track_events
               (track_id, first_seen, last_seen, duration_sec,
                frames_seen, face_visible_frames, facing_ad_frames, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                track.track_id,
                track.first_seen,
                track.last_seen,
                track.last_seen - track.first_seen,
                track.frames_seen,
                track.face_visible_frames,
                track.facing_ad_frames,
                _dumps(track.metadata),
            ),
        )
        self._conn.commit()

    def log_session_summary(self, start: float, end: float, total_frames: int,
                            avg_fps: float, stats: dict) -> None:
        self._conn.execute(
            """INSERT INTO session_summary
               (session_start, session_end, total_frames, avg_fps,
                total_persons, total_faces, total_facing_ad,
                avg_stage_reached, stage_distribution_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                start, end, total_frames, avg_fps,
                stats.get("total_persons", 0),
                stats.get("total_faces", 0),
                stats.get("total_facing_ad", 0),
                stats.get("avg_stage_reached", 0),
                _dumps(stats.get("stage_distribution", {})),
            ),
        )
        self._conn.commit()

    # ----- Query helpers for API / dashboard -----

    def get_recent_events(self, limit: int = 100) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM frame_events ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_stage_distribution(self, since: float = 0) -> dict:
        rows = self._conn.execute(
            """SELECT stage_reached, COUNT(*) as cnt
               FROM frame_events WHERE timestamp >= ?
               GROUP BY stage_reached ORDER BY stage_reached""",
            (since,),
        ).fetchall()
        return {r["stage_reached"]: r["cnt"] for r in rows}

    def get_person_count_series(self, since: float = 0, bucket_sec: int = 60) -> list[dict]:
        rows = self._conn.execute(
            """SELECT
                 CAST(timestamp / ? AS INTEGER) * ? as bucket,
                 AVG(person_count) as avg_persons,
                 MAX(person_count) as max_persons
               FROM frame_events
               WHERE timestamp >= ?
               GROUP BY bucket ORDER BY bucket""",
            (bucket_sec, bucket_sec, since),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_tracks(self, limit: int = 50) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM track_events ORDER BY last_seen DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_sessions(self, limit: int = 20) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM session_summary ORDER BY session_end DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_avg_latencies(self, since: float = 0) -> dict:
        rows = self._conn.execute(
            "SELECT latencies_json FROM frame_events WHERE timestamp >= ?", (since,)
        ).fetchall()
        if not rows:
            return {}
        totals: dict[str, list[float]] = {}
        for r in rows:
            lat = json.loads(r["latencies_json"])
            for k, v in lat.items():
                totals.setdefault(k, []).append(v)
        return {k: round(sum(v) / len(v), 2) for k, v in totals.items()}
