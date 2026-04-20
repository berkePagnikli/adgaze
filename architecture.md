# AdGaze: Cascaded Micro-Model Pipeline for Real-Time Advertisement Engagement Analytics

---

## 1. Problem Definition

### 1.1 Context

Digital and physical out-of-home (OOH) advertising is a multi-billion dollar industry, yet advertisers have almost no reliable, real-time feedback on whether their ads are actually *seen* — let alone whether viewers engage with them. Traditional metrics (impressions, foot traffic counts) are coarse proxies that cannot answer fundamental questions:

- **Was anyone present** in front of the advertisement?
- **Did they look at it**, or did they walk past without a glance?
- **How long** did they sustain attention?
- **Which part of the ad** drew the most gaze?
- **What was the emotional valence** of their reaction?

Computer vision can answer all of these, but existing solutions rely on heavyweight monolithic models that are expensive to deploy, require cloud connectivity, and raise privacy concerns by transmitting raw video to remote servers.

### 1.2 Research Gap

Current approaches to vision-based engagement tracking fall into two categories:

| Approach | Limitation |
|---|---|
| **Cloud-based monolithic models** | High latency, bandwidth cost, privacy risk, single point of failure |
| **Single large model (e.g., one large face-analysis network)** | Wastes compute when no one is present; no graceful degradation; hard to profile per-task cost |

Neither approach exploits a key observation: **most of the time, no one is looking at the ad**. A system that spends equal compute on empty frames and engaged viewers is fundamentally wasteful.

### 1.3 Thesis Contribution — Cascaded Micro-Model Pipeline

This thesis proposes **AdGaze**, a *cascaded micro-model pipeline* architecture where:

1. A sequence of **small, specialized models** are arranged in stages of increasing complexity and cost.
2. Each stage acts as a **binary gate**: if its condition is not met, all downstream (more expensive) stages are **skipped entirely**.
3. The system achieves **adaptive compute budgeting** — spending near-zero resources on empty scenes and full resources only on genuinely engaged viewers.
4. All inference runs **locally**, emitting only **anonymous engagement metrics** — never raw images or biometric data.

This is analogous to a cascade classifier (cf. Viola-Jones) but generalized across heterogeneous modern deep learning models, each optimized for a single sub-task.

### 1.4 Research Questions

- **RQ1:** How much compute (FLOPs, latency) can a gated cascade architecture save compared to an always-on monolithic pipeline, across varying traffic densities?
- **RQ2:** What is the minimum model complexity at each stage that preserves end-to-end engagement measurement accuracy within an acceptable margin?
- **RQ3:** Can per-stage confidence thresholds be learned or tuned automatically to optimize the accuracy–efficiency trade-off?

---

## 2. Tech Stack

### 2.1 Hardware & Input

| Component | Option | Notes |
|---|---|---|
| **Compute** | Personal PC (CPU or CUDA GPU) | Primary development and demo environment |
| **Camera** | Built-in PC webcam or smartphone camera | Any ≥720p source; no dedicated hardware needed |
| **Training** | Local GPU or cloud GPU (CUDA) | For custom gaze model training |

The system reads from the default webcam via OpenCV or accepts a pre-recorded video file. For phone-camera demos, the phone streams via IP webcam (e.g., DroidCam, IP Webcam app) or the video is transferred and processed offline.

### 2.2 Software & Frameworks

| Layer | Technology | Rationale |
|---|---|---|
| **Language** | Python 3.11+ | Ecosystem maturity, rapid prototyping |
| **Video I/O** | OpenCV (`cv2.VideoCapture`) | Universal camera support, efficient frame handling |
| **Inference Runtime** | ONNX Runtime (CPU/CUDA EP) | Single runtime for all models; runs on any machine with or without a GPU |
| **Person Detection** | YOLOv8n (Ultralytics, exported to ONNX) | Nano variant: 3.2M params, ~6.5 GFLOPs — sufficient for person/bounding-box detection |
| **Face Detection** | YOLOv8n-face or SCRFD-500M (ONNX) | Sub-1M param face-specific detector |
| **Face Landmarks** | MediaPipe Face Mesh (468 landmarks) or custom lightweight regressor | Head pose derivation without a separate pose model |
| **Head Pose Estimation** | PnP solve from landmarks (OpenCV `solvePnP`) | Zero additional model cost — geometric computation from landmarks |
| **Gaze Estimation** | Custom lightweight CNN (MobileNetV3-Small backbone, ~1.5M params) trained on MPIIGaze / GazeCapture | The only custom-trained model; kept small deliberately |
| **Engagement Scoring** | Rule-based temporal logic + optional lightweight LSTM (≤50K params) | Fuses time-series signals (gaze duration, revisits, head orientation stability) into a scalar score |
| **Analytics API** | FastAPI | Async, high-performance, auto-documented REST endpoints |
| **Database** | SQLite | Time-series engagement events; zero-dependency local storage |
| **Dashboard** | Grafana or Streamlit | Real-time and historical engagement visualization |
| **Containerization** | Docker (optional) | Reproducible environment for reviewers and demo |
| **Experiment Tracking** | MLflow or Weights & Biases | Track training runs, hyperparameters, model versions |

### 2.3 Model Format & Optimization

All models are exported to **ONNX** for portable, framework-independent inference. Optimization strategy:

- **FP32** as the baseline precision (default on PC).
- **FP16** when a CUDA GPU is available (automatic via ONNX Runtime CUDA EP).
- **INT8 post-training quantization** explored experimentally to measure the accuracy–latency trade-off and to demonstrate that the pipeline *could* run on constrained edge hardware in a production scenario.

---

## 3. Process Flow

### 3.1 High-Level Cascade

```
┌─────────────────────────────────────────────────────────────────┐
│                        CAMERA FEED                              │
│                     (continuous frames)                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
               ┌───────────────────────┐
               │   STAGE 0: Motion     │  ◄── Frame differencing / MOG2
               │   Detection Gate      │      (near-zero compute)
               └───────────┬───────────┘
                    no motion → LOG "idle", SKIP ──► ∅
                           │ motion detected
                           ▼
               ┌───────────────────────┐
               │   STAGE 1: Person     │  ◄── YOLOv8n (ONNX)
               │   Detection Gate      │
               └───────────┬───────────┘
                  no person → LOG "passerby: none", SKIP ──► ∅
                           │ person(s) found
                           ▼
               ┌───────────────────────┐
               │   STAGE 2: Face       │  ◄── SCRFD-500M / YOLOv8n-face
               │   Detection Gate      │
               └───────────┬───────────┘
                   no face → LOG "person present, not facing", SKIP ──► ∅
                           │ face(s) found
                           ▼
               ┌───────────────────────┐
               │   STAGE 3: Head Pose  │  ◄── Landmark PnP solve
               │   Estimation Gate     │      (pure geometry)
               └───────────┬───────────┘
              head turned away → LOG "facing away", SKIP ──► ∅
                           │ head oriented toward ad
                           ▼
               ┌───────────────────────┐
               │   STAGE 4: Gaze       │  ◄── Custom MobileNetV3-Small
               │   Estimation          │
               └───────────┬───────────┘
              gaze not on ad → LOG "glanced, not on ad" ──► ∅
                           │ gaze on ad region
                           ▼
               ┌───────────────────────┐
               │   STAGE 5: Temporal   │  ◄── Engagement Scorer
               │   Engagement Scoring  │      (rule-based + optional LSTM)
               └───────────┬───────────┘
                           │
                           ▼
               ┌───────────────────────┐
               │   EMIT: Engagement    │  ──► Database + Dashboard
               │   Event               │
               └───────────────────────┘
```

### 3.2 Stage Details

#### Stage 0 — Motion Detection (Gate)

- **Method:** Background subtraction (OpenCV MOG2) or simple frame differencing with a pixel-change threshold.
- **Cost:** < 0.5 ms per frame (pure CPU, no model).
- **Output:** Binary `motion_detected: bool`.
- **Gate logic:** If no significant motion for *N* consecutive frames, skip all downstream stages. This alone eliminates the majority of frames in low-traffic periods (e.g., nighttime).

#### Stage 1 — Person Detection (Gate)

- **Model:** YOLOv8n exported to ONNX, optionally INT8-quantized.
- **Input:** Full frame (resized to 640×640).
- **Output:** List of bounding boxes with class `person` and confidence score.
- **Gate logic:** If no `person` detection with confidence ≥ *τ₁* (tunable threshold), skip downstream. Also outputs person count for aggregate analytics.

#### Stage 2 — Face Detection (Gate)

- **Model:** SCRFD-500M or YOLOv8n-face.
- **Input:** Cropped person bounding box(es) from Stage 1 — *not* the full frame (saves compute).
- **Output:** Face bounding box(es) + 5-point facial landmarks.
- **Gate logic:** If no face detected (person's back is turned, face is occluded), skip downstream. Log the event as "person present but not facing the ad."

#### Stage 3 — Head Pose Estimation (Gate)

- **Method:** Given the 5 facial landmarks (or 468 from MediaPipe if higher fidelity is needed), use a 3D face model + OpenCV `solvePnP` to estimate yaw, pitch, roll.
- **Cost:** Pure geometry — no neural network inference.
- **Output:** Euler angles (yaw, pitch, roll).
- **Gate logic:** If yaw or pitch deviation from the ad's direction exceeds threshold *τ₃*, the person is not oriented toward the ad. Skip gaze estimation.

#### Stage 4 — Gaze Estimation

- **Model:** Custom lightweight CNN.
  - **Backbone:** MobileNetV3-Small (feature extractor, frozen or fine-tuned).
  - **Head:** Two FC layers → 2D gaze vector (yaw, pitch in radians).
  - **Training data:** MPIIGaze, GazeCapture, or a custom collected dataset.
  - **Total params:** ~1.5M.
- **Input:** Cropped eye region(s) from the face bounding box + head pose as auxiliary input.
- **Output:** 2D gaze direction vector.
- **Mapping:** Project gaze vector onto the known ad plane geometry to determine if — and *where* — the person is looking on the ad surface.

#### Stage 5 — Temporal Engagement Scoring

This stage operates on a **per-person time series**, not individual frames. It maintains a tracker (using IoU-based or feature-based tracking, e.g., ByteTrack) to associate detections across frames to the same individual.

**Tracked metrics per person:**

| Metric | Description |
|---|---|
| `presence_duration` | Total time the person was in the camera's field of view |
| `face_visible_duration` | Time the person's face was detected (facing the ad) |
| `gaze_on_ad_duration` | Time the person's gaze was on the ad surface |
| `gaze_heatmap` | Spatial distribution of gaze points on the ad |
| `attention_ratio` | `gaze_on_ad_duration / presence_duration` |
| `revisit_count` | Number of times gaze returned to the ad after looking away |
| `dwell_category` | Categorical: `glance` (<1s), `read` (1–5s), `engaged` (5–15s), `deep` (>15s) |

**Engagement score formula (rule-based baseline):**

$$E = \alpha \cdot \text{attention\_ratio} + \beta \cdot \log(1 + \text{gaze\_on\_ad\_duration}) + \gamma \cdot \text{revisit\_count}$$

Where $\alpha, \beta, \gamma$ are tunable weights. An optional LSTM-based scorer can learn these weights from labeled engagement data.

### 3.3 Data Flow & Privacy Architecture

```
┌──────────────┐     ┌──────────────────────┐     ┌────────────────┐
│   Webcam /   │────►│   Local PC           │────►│  SQLite DB     │
│   Phone Cam  │     │   (all inference     │     │  + Dashboard   │
│   (live or   │     │    runs here)        │     │  (local)       │
│    recorded) │     └──────────┬───────────┘     └───────┬────────┘
└──────────────┘                │                         │
                     Only stores:                 Visualizes:
                     • engagement scores          • engagement trends
                     • gaze heatmaps              • gaze heatmaps
                     • dwell times                • per-session reports
                     • person counts
                     • timestamps
                     (NO images, NO faces,
                      NO biometric data)
```

**Privacy guarantees (by-design, applicable if deployed in production):**
- Raw video frames are **never written to disk**.
- No face embeddings or re-identification features are stored.
- Person tracking is **intra-session only** (no cross-session re-identification).
- Architecture is designed to be **GDPR/KVKK-compliant by construction**, not just by policy.

---

## 4. Novelty & Contributions

| # | Contribution | Type |
|---|---|---|
| 1 | **Gated cascaded micro-model pipeline** — formalization of a cascade of heterogeneous DNN models with binary early-exit gates, applied to engagement analytics | Architectural |
| 2 | **Adaptive compute budgeting** — formal analysis of compute savings as a function of scene occupancy rate, with theoretical bounds and empirical validation | Theoretical + Empirical |
| 3 | **Gate threshold optimization** — learning per-stage confidence thresholds to maximize an accuracy–efficiency Pareto front under a given hardware power budget | Methodological |
| 4 | **Gaze-to-ad-surface projection** — mapping estimated gaze vectors to a known physical ad plane to produce spatial attention heatmaps | Applied |
| 5 | **Privacy-by-architecture** — demonstrating a system that emits only anonymous aggregate metrics, never raw images or biometric data | Design |
| 6 | **Comprehensive benchmark** — comparison of monolithic vs. cascade pipelines under varying traffic density scenarios, with profiling of per-stage compute cost | Empirical |

---

## 5. Evaluation Plan

### 5.1 Datasets

| Dataset | Use |
|---|---|
| **MPIIGaze / GazeCapture** | Training and evaluating the gaze estimation model |
| **WIDER FACE** | Benchmarking face detection stage |
| **Custom collected dataset** | Record volunteers viewing an ad display in controlled and in-the-wild settings; annotate engagement levels |

### 5.2 Metrics

| Category | Metrics |
|---|---|
| **Accuracy** | Per-stage precision/recall/F1; end-to-end engagement score correlation with human annotations (Pearson's r, Spearman's ρ) |
| **Efficiency** | Per-frame latency (ms), FLOPs per frame (average and worst-case), throughput (FPS) |
| **Cascade savings** | Compute reduction ratio vs. always-on full pipeline, stratified by scene occupancy rate (0%, 10%, 50%, 100% of frames contain a person) |
| **Gaze accuracy** | Angular error (degrees) on MPIIGaze test set |

### 5.3 Baselines

1. **Monolithic baseline:** Run all stages on every frame (no gating).
2. **Single large model:** An all-in-one model (e.g., fine-tuned large ResNet or ViT) that predicts engagement directly from the raw frame.
3. **Commercial SDK:** If available, compare against a commercial audience analytics SDK (e.g., Intel OpenVINO Audience Analytics).

---

## 6. Project Structure (Proposed)

```
adgaze/
├── config/
│   └── pipeline.yaml            # Per-stage thresholds, model paths, camera config
├── models/
│   ├── person_detector/         # YOLOv8n ONNX weights + export scripts
│   ├── face_detector/           # SCRFD / YOLOv8n-face ONNX weights
│   ├── gaze_estimator/          # Custom CNN training code + ONNX export
│   └── engagement_scorer/       # LSTM scorer (optional)
├── pipeline/
│   ├── stage_motion.py          # Stage 0: Motion detection
│   ├── stage_person.py          # Stage 1: Person detection
│   ├── stage_face.py            # Stage 2: Face detection
│   ├── stage_headpose.py        # Stage 3: Head pose estimation
│   ├── stage_gaze.py            # Stage 4: Gaze estimation
│   ├── stage_engagement.py      # Stage 5: Engagement scoring
│   ├── tracker.py               # Multi-person tracking (ByteTrack)
│   └── cascade.py               # Orchestrator: runs stages, handles gating
├── api/
│   └── main.py                  # FastAPI analytics endpoints
├── storage/
│   └── db.py                    # SQLite/TimescaleDB event storage
├── dashboard/
│   └── app.py                   # Streamlit or Grafana config
├── training/
│   ├── train_gaze.py            # Gaze model training loop
│   └── train_scorer.py          # Engagement scorer training
├── evaluation/
│   ├── benchmark_cascade.py     # Latency/FLOPs/accuracy benchmarks
│   └── compare_baselines.py     # Monolithic vs. cascade comparison
├── scripts/
│   ├── export_onnx.py           # Model export utilities
│   └── collect_data.py          # Data collection helper
├── tests/
├── requirements.txt
└── README.md
```

---

## 7. Timeline (Suggested Milestones)

| Phase | Tasks | Duration |
|---|---|---|
| **Phase 1 — Foundation** | Literature review; set up dev environment; implement Stage 0 + Stage 1 | 4 weeks |
| **Phase 2 — Core Pipeline** | Implement Stages 2–4; integrate person tracker; basic end-to-end pipeline running on webcam | 5 weeks |
| **Phase 3 — Gaze Model** | Curate training data; train custom gaze estimator; export to ONNX; integrate into pipeline | 4 weeks |
| **Phase 4 — Engagement Scoring** | Implement temporal engagement scorer; design and build analytics API + dashboard | 3 weeks |
| **Phase 5 — Optimization** | INT8/FP16 quantization experiments; gate threshold tuning; latency profiling and benchmarking | 3 weeks |
| **Phase 6 — Evaluation** | Collect custom dataset; run full benchmark suite; compare against baselines; statistical analysis | 4 weeks |
| **Phase 7 — Thesis Writing** | Write and revise thesis document | 5 weeks |

---

## 8. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Gaze estimation accuracy too low after quantization | Fall back to head-pose-only engagement (Stage 3 output); report as a limitation |
| Insufficient real-world data for engagement labeling | Use synthetic data augmentation; leverage existing gaze datasets; design a minimal annotation protocol |
| Person tracking drift in crowded scenes | Use ByteTrack with re-identification features from the face detector; limit study scope to low-to-moderate density |
| Webcam resolution or angle insufficient for gaze estimation | Use a higher-resolution external USB webcam; constrain demo distance to ≤1.5 m from camera |
