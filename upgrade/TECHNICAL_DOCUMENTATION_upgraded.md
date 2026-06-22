# 🔬 Upgraded Technical Documentation — Code Architecture & Breakdown

This document provides a line-by-line and architectural breakdown of the upgraded Football Highlight & Multi-Modal Analysis workspace.

---

## 🗺️ Project Architecture Overview

The system consists of a Python Flask API server backend serving a Single Page Application (SPA) frontend.

```text
+---------------------------------------------------------------------------------+
|                                 BROWSER CLIENT                                  |
|  +---------------------------------------------------------------------------+  |
|  |                SPA Frontend UI (HTML5, Vanilla CSS, JS)                   |  |
|  |                                                                           |  |
|  |  [3D Particle Scene]   [Settings Controls]       [Processing Progress Bar]|  |
|  |  (Three.js WebGL)      (Sensitivity, Duration)   (AJAX Status Polling)    |  |
|  |                                                                           |  |
|  |  [Results View]        [Scoreboard Grid]         [Audio Event Peak Plot]  |  |
|  |  (MoviePy Stitched)    (OCR Tracker Details)     (Spectral Detections)    |  |
|  |                                                                           |  |
|  |  [Semantic Clip Search][Match Frame Gallery]      [Generative Chatbot Interface]|  |
|  |  (CLIP Text Input)     (12 frame snapshots)      (RAG Context QA)         |  |
|  +---------------------------------------------------------------------------+  |
+----------------------------------------^----------------------------------------+
                                         | HTTP / JSON API
+----------------------------------------v----------------------------------------+
|                               FLASK API BACKEND                                 |
|  +-------------------------+ +-------------------------+ +--------------------+  |
|  |   Flask Server Router   | |     Pipeline Worker     | |    Model Weights   |  |
|  |      (api_server.py)    | |    (Multi-threaded)     | | (checkpoints/*.pth)|  |
|  +------------^------------+ +------------^------------+ +--------------------+  |
|               |                           |                                     |
|               | Calls                     +------------+                        |
|               v                                        v                        |
|  +--------------------------------------------------+ +----------------------+  |
|  |               Multi-Modal Engines                | |  DL Event Classifier |  |
|  |                                                  | |                      |  |
|  |  * audio_classifier.py (Spectrogram Peaks)       | |  * CNN + Bi-LSTM     |  |
|  |  * scoreboard_tracker.py (OCR Tesseract ROI)     | |    Class Detections  |  |
|  |  * clip_search.py (CLIP Zero-shot Search)        | |  * Knapsack DP Clip  |  |
|  |  * chatbot_agent.py (Document Keyword RAG)       | |    Optimization      |  |
|  |  * match_summarizer.py (LLM Narrator Summary)    | |  * MoviePy Stitcher  |  |
|  +--------------------------------------------------+ +----------------------+  |
+---------------------------------------------------------------------------------+
```

---

## 🐍 Backend Breakdown: `api_server.py`

### 1. REST API Routing (Flask)
*   `GET /`: Serves `frontend/index.html`.
*   `GET /api/status`: Returns current status of the deep learning model (loaded epoch, best validation mAP) and whether results are active.
*   `POST /api/run-pipeline`: Accepts multi-part form data uploads of the `.npy` feature file and `.mp4` video. Creates a daemon worker thread to execute the multi-modal pipeline without blocking the server.
*   `GET /api/progress/<task_id>`: Returns the step-by-step progress percentage and current executing stage.
*   `GET /api/results/<task_id>`: Returns final pipeline results (events, audio analysis, scoreboard logs, CLIP search lists, AI summaries).
*   `GET /api/frame/<task_id>?t=seconds`: Dynamically extracts and returns a JPEG frame from the source video at timestamp `t` using OpenCV (`cv2.VideoCapture`). Resizes it automatically to 640px to optimize network bandwidth.
*   `GET /api/video/<filename>`: Streams the compiled MP4 highlights.
*   `POST /api/search`: Performs a cosine similarity search on CLIP frame embeddings.
*   `POST /api/chat`: Orchestrates chatbot responses via RAG.

### 2. Pipeline Worker Thread (`_worker`)
Runs the following stages asynchronously:
1.  **Event Detections**: Loads features from `.npy`, runs sliding window inference through the `BiLSTMClassifier`, selects optimal frames using Knapsack dynamic programming, and groups consecutive event boundaries.
2.  **Audio Event Classifier**: Calls `classify_audio_events` to analyze video audio tracks for cheer volume and whistle spectrograms.
3.  **Scoreboard OCR Tracker**: Runs `track_scoreboard_in_video` to scan scoreboards at regular intervals.
4.  **CLIP Indexing**: Extracts video frame embeddings to facilitate semantic text-to-video search.
5.  **Highlights Video Stitching**: Slices and joins clips using MoviePy to create `highlights_<tid>.mp4`.
6.  **Match Summary**: Generates a text summary using the events dictionary.

---

## 🎨 Frontend Breakdown: `frontend/app.js`

### 1. Real-time 3D Scene (`initThreeScene`)
Renders an interactive WebGL scene in the page background:
*   Uses `Three.js` (loaded via index.html scripts).
*   Generates a floating particle system (`THREE.Points`) of 2,500 dots with dynamic gradient shifts.
*   Renders a spinning 3D icosahedron football with concentric rings in outer space.
*   Uses point lights (Cyan, Purple, Pink) that dynamically follow mouse coordinate trends.

### 2. Client-side SPA Router & AJAX Poll
Handles state transitions and updates:
*   Initializes default UI state, event listeners, parameters sliders, and drag-and-drop file inputs.
*   Runs an interval checking the backend model load status.
*   Upon clicking "Run Pipeline", issues a `POST` request with the uploaded files.
*   Polls `/api/progress/<task_id>` every 800ms, displaying current worker status messages and percentages.
*   Once finished, updates tabs, draws canvas timelines, renders audio charts (via Chart.js or canvas), loads key moment frame snapshots, and mounts the video player.

---

## 🧬 Core Neural Network Concepts (Academic Viva Defense)

### 1. Spatial vs. Temporal Modeling
*   **Spatial features**: Extracted by feeding raw video frames through a pre-trained **ResNet-152** network. This extracts visual semantics (shapes, players, jersey colors, ball position) into a 2048-dimensional feature vector.
*   **Dimensionality Reduction (PCA)**: Principal Component Analysis reduces the 2048-dim vectors to 512 dimensions. This matches SoccerNet's feature dimensions and removes redundant spatial information to prevent overfitting.
*   **Temporal features**: Video is sequential; a single frame cannot distinguish a goal kick from a shot. A **Bi-LSTM** processes a window of 30 frames (15 seconds) sequentially, carrying information forwards and backwards in time, allowing the classifier head to make contextual predictions.

### 2. Class Imbalance & Inverse Frequency Weights
*   **Problem**: In broadcast football videos, 95%+ of frames are "Background" (regular play, crowd cutaways). Only <2% represent Goals or Substitutions. If trained on raw cross-entropy, a model predicting "Background" for all frames would achieve 95% accuracy but detect zero events.
*   **Solution**: We calculate inverse-frequency class weights:
    $$\text{weight}_c = \frac{N_{\text{total}}}{C \cdot N_c}$$
    Where $C$ is the number of classes, $N_{\text{total}}$ is total sequences, and $N_c$ is sequences in class $c$. This scale makes errors on Goal frames significantly costlier, forcing the model to prioritize event boundaries.

### 3. Dynamic Programming Knapsack Algorithm
Instead of simply cropping every detected event, we optimize highlight selections under a strict time budget (e.g. 7 minutes maximum length).
*   **Weights**: Clip duration (seconds).
*   **Values**: Model classification confidence.
*   **Knapsack constraint**: Total selected clip length $\le$ maximum highlight duration.
*   **Dynamic Programming**: Computes an $N \times W$ lookup table in $O(N \cdot W)$ time to choose the mathematically optimal clips.
