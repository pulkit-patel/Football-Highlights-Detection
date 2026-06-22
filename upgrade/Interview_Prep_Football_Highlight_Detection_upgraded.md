# ⚽ Upgraded Interview Preparation Guide — Football Highlight & Multi-Modal Analysis Hub

This guide prepares you for any academic review or viva defense by detailing the full deep learning, multi-modal, and web architecture code.

---

## 1. Upgraded Project Architecture & Data Flow

The system has been completely upgraded from a Streamlit script to a production-grade **Flask API Server Backend + HTML5/CSS3/JavaScript SPA Frontend** featuring interactive 3D elements.

```text
[Raw Match Video MP4] ---> [Multi-Modal Pipeline Engine (api_server.py Thread)]
                                │
   ┌────────────────────────────┼───────────────────────────┬────────────────────────┐
   ▼                            ▼                           ▼                        ▼
[Spatial ResNet-152]      [Audio Librosa STFT]        [Scoreboard OCR]       [CLIP Frame Encoder]
Extract 2048D vectors      Extract sound track         Crop Scoreboard ROI    Sample at 1.0 FPS
   │                            │                           │                        │
   ▼                            ▼                           ▼                        ▼
[PCA to 512D]             Detect cheer/whistle amplitude   Tesseract text scan    Compute cosine similarity
   │                            │                           │                        │
   ▼                            ▼                           ▼                        ▼
[BiLSTM Sequential Model]  Spectral peaks timeline     Score progress log     Zero-shot Text-to-Video Match
   │                            │                           │                        │
   ▼                            ▼                           ▼                        ▼
[Knapsack DP Selection]   [Audio Event Table]         [Score Milestones]     [Semantic Search Results]
   │
   ▼
[MoviePy Video Stitcher]
   │
   ▼
[Stitched Highlight MP4] 
   │
   ▼
[3D Responsive Dashboard SPA (Three.js Background, Canvas timelines, 3-column key moment frame gallery)]
```

---

## 2. Advanced Multi-Modal Pipeline Components

### 2.1. Audio Spectrogram Peak Classification
*   **The Concept**: Sports events are loud. Decibel spikes correlate strongly with goals (crowd erupts) and fouls (referee whistle).
*   **Implementation**: Extract audio track from video $\rightarrow$ compute **Short-Time Fourier Transform (STFT)** $\rightarrow$ map frequency magnitude over time $\rightarrow$ isolate cheer energy (broadband, 500-4000Hz) vs. whistles (narrow high-frequency, 1000-2500Hz).

### 2.2. Scoreboard OCR (Optical Character Recognition)
*   **The Concept**: Video scoreboard overlays contain ground-truth state (time and scores).
*   **Implementation**: Crop Region of Interest (ROI) $\rightarrow$ apply grayscaling and adaptive thresholding $\rightarrow$ feed preprocessed image into **Tesseract OCR** to parse minute digits and score text $\rightarrow$ construct a match state timeline.

### 2.3. CLIP Semantic Text-to-Video Search
*   **The Concept**: Zero-shot retrieval allows searching for actions not explicitly labeled by our BiLSTM model (e.g. "corner kick", "headbutt", "goalkeeper save").
*   **Implementation**: Sample frames $\rightarrow$ generate visual embeddings using **CLIP (Contrastive Language-Image Pretraining)** vision encoder $\rightarrow$ compute cosine similarity against user query text encoded via CLIP text encoder.

### 2.4. Dual-Mode RAG Chatbot
*   **The Concept**: Combines match-specific analytics with project documentation Q&A.
*   **Implementation**: RAG (Retrieval-Augmented Generation) reads upgraded documents in the `upgrade/` folder, chunks them into overlap blocks, and performs keyword frequency retrieval to feed context to the generator LLM.

---

## 3. High-Yield Interview Q&A Bank

**Q1: What are the main endpoints of your Flask backend and their roles?**
*A:* 
*   `POST /api/run-pipeline`: Triggers the asynchronous worker thread to execute visual detection, audio classification, scoreboard OCR, and CLIP indexing.
*   `GET /api/progress/<tid>`: Polls status (0-100%) and current executing step.
*   `GET /api/frame/<tid>`: Dynamically extracts and crops a video frame at a given timestamp on-the-fly using OpenCV (`cv2.VideoCapture`).
*   `POST /api/search` & `POST /api/chat`: Handles semantic query retrievals and RAG chat.

**Q2: How does the front-end Three.js 3D animation work without slowing down the UI?**
*A:* Three.js utilizes **WebGL** which runs directly on the client's GPU, leaving the CPU free to handle page layout rendering and API AJAX requests. The 3D scene creates a particle system (2,500 points) and a neon football wireframe structure spinning on its axis, reacting dynamically to mouse movement offsets.

**Q3: How does the Key Match Moments gallery work under the hood?**
*A:* To keep page load times fast, the frontend does not load 12 separate video files. Instead, the backend exposes `/api/frame/<task_id>?t=seconds`. Using OpenCV, the backend opens the source video file, sets the frame pointer to the target timestamp (`CAP_PROP_POS_MSEC`), reads a single image frame, resizes it to 640px, encodes it as JPEG, and streams the raw bytes. The frontend displays these inside a responsive **3-column photo grid** with timeline tags.

**Q4: Why did you implement a local keyword-based search fallback for your RAG chatbot?**
*A:* To make the application fully self-contained and ensure it operates offline without external API dependencies. If no Hugging Face API key is provided, the local TF-IDF style keyword search retrieves the most relevant chunks from the upgraded documentation files in the `upgrade/` folder and prints them directly to the user as direct quotes.

**Q5: Explain the domain shift issue and how you resolved it in this version.**
*A:* Originally, the BiLSTM classifier was trained on ResNet-152 + PCA features from the SoccerNet dataset. Custom video inference used a lightweight ResNet-18 model, producing features in a different vector space. The model could not recognize them (domain shift). I resolved this by upgrading the custom video inference pipeline to extract features using the identical **ResNet-152 + PCA (2048 $\rightarrow$ 512)** pipeline, aligning the feature distributions.

**Q6: What is the purpose of the Knapsack dynamic programming algorithm in highlights compilation?**
*A:* A greedy algorithm simply selects the most confident events. However, it fails if a long, moderately-confident clip takes up the entire time budget, blocking several short, highly-confident clips. The 0/1 Knapsack formulation treats clip durations as weights and confidence scores as values, using dynamic programming to find the mathematically optimal combination of clips that maximizes total confidence under a strict user-defined duration cap.
