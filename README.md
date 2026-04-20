# AdGaze

Cascaded micro-model pipeline for real-time advertisement engagement analytics.

## Quick Start

```bash
# 1. Install dependencies
pip install -r adgaze/requirements.txt

# 2. Download & export ONNX models
python -m adgaze.scripts.download_models

# 3. Run the pipeline (webcam)
python -m adgaze

# 4. (Optional) Start the API server
uvicorn adgaze.api.main:app --reload

# 5. (Optional) Start the dashboard
streamlit run adgaze/dashboard/app.py
```

## Pipeline Stages

| Stage | Component | Gate? | Notes |
|-------|-----------|-------|-------|
| 0 | Motion Detection | Yes | OpenCV MOG2 / frame differencing, near-zero cost |
| 1 | Person Detection | Yes | YOLOv8n (ONNX) |
| 2 | Face Detection | Yes | OpenCV YuNet (5-landmark face detector), runs on person crops |
| 3 | Head Pose Estimation | Yes | Landmark geometry (nose-to-eye-center ratio), no model needed |
| 4 | Gaze Estimation | — | *Disabled — requires custom trained model* |
| 5 | Engagement Scoring | — | *Disabled — requires data collection* |

Each stage acts as a gate: if its condition isn't met, all downstream stages are skipped.
Temporal smoothing prevents flickering at the head pose gate.

## Project Structure

```
adgaze/
├── config/pipeline.yaml       # All thresholds, model paths, camera config
├── pipeline/
│   ├── stage_motion.py        # Stage 0: Background subtraction
│   ├── stage_person.py        # Stage 1: YOLOv8n person detection
│   ├── stage_face.py          # Stage 2: YuNet face detection
│   ├── stage_headpose.py      # Stage 3: Landmark-based head pose estimation
│   ├── tracker.py             # IoU-based multi-person tracker
│   └── cascade.py             # Orchestrator with gating logic
├── api/main.py                # FastAPI analytics endpoints
├── storage/db.py              # SQLite event storage
├── dashboard/app.py           # Streamlit real-time dashboard
├── scripts/download_models.py # Model download & ONNX export
├── __main__.py                # CLI entry point
└── requirements.txt
```

## Configuration

Edit `adgaze/config/pipeline.yaml` to adjust:
- Camera source and resolution
- Per-stage confidence thresholds
- Head pose yaw/pitch limits
- Tracker parameters
- Model paths and device (CPU/CUDA)
