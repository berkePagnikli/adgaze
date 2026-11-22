"""Step 7: audience analytics (per-person table + video summary)."""

from __future__ import annotations

import pandas as pd

from .temporal import WatchTimeCounter


def person_table(counter: WatchTimeCounter, cfg) -> pd.DataFrame:
    rows = []
    for tid, t in sorted(counter.tracks.items()):
        dwell, watch = counter.dwell_s(tid), counter.watch_s(tid)
        if dwell < cfg.min_dwell_s:
            continue
        rows.append({
            "track_id": tid,
            "first_frame": t.first_frame,
            "last_frame": t.last_frame,
            "dwell_s": round(dwell, 2),        # time in front of the screen (tracked)
            "watch_s": round(watch, 2),        # time looking at the screen
            "watch_ratio": round(watch / dwell, 3) if dwell else 0.0,
            "face_ratio": round(t.face_frames / t.frames, 3),
            "viewer": int(watch >= cfg.min_watch_s),
        })
    return pd.DataFrame(rows, columns=["track_id", "first_frame", "last_frame", "dwell_s", "watch_s",
                                       "watch_ratio", "face_ratio", "viewer"])


def summarize(persons: pd.DataFrame) -> dict:
    n = len(persons)
    viewers = persons[persons["viewer"] == 1] if n else persons
    return {
        "persons": n,
        "viewers": len(viewers),
        "viewer_ratio": round(len(viewers) / n, 3) if n else 0.0,
        "total_dwell_s": round(float(persons["dwell_s"].sum()), 2) if n else 0.0,
        "total_watch_s": round(float(persons["watch_s"].sum()), 2) if n else 0.0,
        "mean_dwell_s": round(float(persons["dwell_s"].mean()), 2) if n else 0.0,
        "mean_watch_s_per_viewer": round(float(viewers["watch_s"].mean()), 2) if len(viewers) else 0.0,
    }
