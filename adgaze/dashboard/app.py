"""Streamlit dashboard for AdGaze analytics.

Run with: streamlit run adgaze/dashboard/app.py
"""

import time
from datetime import datetime

import pandas as pd
import requests
import streamlit as st

API_BASE = "http://localhost:8000/api"


def fetch(endpoint: str, params: dict | None = None):
    try:
        r = requests.get(f"{API_BASE}{endpoint}", params=params, timeout=5)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        st.error(f"API error: {e}")
        return None


def main():
    st.set_page_config(page_title="AdGaze Dashboard", layout="wide")
    st.title("AdGaze — Advertisement Engagement Dashboard")

    # Sidebar controls
    st.sidebar.header("Settings")
    auto_refresh = st.sidebar.checkbox("Auto-refresh (5s)", value=True)
    lookback_min = st.sidebar.slider("Lookback (minutes)", 1, 1440, 60)
    since = time.time() - lookback_min * 60

    # ---- Row 1: KPIs ----
    col1, col2, col3, col4 = st.columns(4)

    events = fetch("/events/recent", {"limit": 500})
    if events:
        recent_df = pd.DataFrame(events)
        if not recent_df.empty:
            col1.metric("Total Frames", len(recent_df))
            col2.metric("Avg Persons/Frame", f"{recent_df['person_count'].mean():.1f}")
            col3.metric("Faces Detected", int(recent_df["face_count"].sum()))
            col4.metric("Facing Ad", int(recent_df["any_facing_ad"].sum()))

    # ---- Row 2: Stage Distribution ----
    st.subheader("Cascade Stage Distribution")
    stage_dist = fetch("/stats/stage-distribution", {"since": since})
    if stage_dist:
        stage_names = {
            0: "Motion Only",
            1: "Person Detected",
            2: "Face Detected",
            3: "Facing Ad",
        }
        df_stage = pd.DataFrame([
            {"Stage": stage_names.get(int(k), f"Stage {k}"), "Frames": v}
            for k, v in stage_dist.items()
        ])
        if not df_stage.empty:
            st.bar_chart(df_stage.set_index("Stage"))

    # ---- Row 3: Person count over time ----
    st.subheader("Person Count Over Time")
    person_series = fetch("/stats/person-count", {"since": since, "bucket_sec": 60})
    if person_series:
        df_persons = pd.DataFrame(person_series)
        if not df_persons.empty:
            df_persons["time"] = pd.to_datetime(df_persons["bucket"], unit="s")
            st.line_chart(df_persons.set_index("time")[["avg_persons", "max_persons"]])

    # ---- Row 4: Latencies ----
    st.subheader("Average Per-Stage Latency (ms)")
    latencies = fetch("/stats/latencies", {"since": since})
    if latencies:
        df_lat = pd.DataFrame([
            {"Stage": k, "Latency (ms)": v} for k, v in latencies.items()
        ])
        if not df_lat.empty:
            st.bar_chart(df_lat.set_index("Stage"))

    # ---- Row 5: Recent tracks ----
    st.subheader("Recent Person Tracks")
    tracks = fetch("/tracks", {"limit": 20})
    if tracks:
        df_tracks = pd.DataFrame(tracks)
        if not df_tracks.empty:
            df_tracks["duration"] = df_tracks["duration_sec"].round(1)
            df_tracks["first_seen_dt"] = pd.to_datetime(df_tracks["first_seen"], unit="s")
            st.dataframe(
                df_tracks[["track_id", "first_seen_dt", "duration", "frames_seen",
                           "face_visible_frames", "facing_ad_frames"]],
                use_container_width=True,
            )

    # Auto-refresh
    if auto_refresh:
        time.sleep(5)
        st.rerun()


if __name__ == "__main__":
    main()
