# 🎓 Upgraded Viva Defense Guide — Every Question You Could Be Asked

This document prepares you for a comprehensive defense of the deep learning, multi-modal pipelines, and full-stack web integration.

---

## 🏛️ Category 1: Architecture & Design Decisions

### Q1: Why did you choose a Flask API server + HTML/CSS/JS SPA over Streamlit/Gradio?
**Answer:** Streamlit and Gradio are excellent for simple ML prototypes, but they suffer from severe limitations:
1.  **Rendering Performance**: Streamlit re-runs the entire python script upon any state change, causing lag and reloading UI structures. Our custom SPA loads once and uses AJAX requests for seamless transitions.
2.  **Visual Customization**: Vanilla CSS and HTML5 allow complete design freedom, implementing premium aesthetics like glassmorphism card layouts, tailored HSL color indicators, and responsive grids.
3.  **Real-Time 3D Elements**: We integrated a real-time WebGL particle system using `Three.js` directly into the page backdrop, creating a premium sport-cast environment which is impossible in Streamlit.
4.  **Decoupled Control**: A Flask REST API allows client-server decoupling, enabling backend asynchronous multi-threading (running the pipeline as a daemon thread) without freezing the browser interface.

### Q2: Why ResNet-152 + PCA for spatial feature extraction?
**Answer:** The SoccerNet training features are pre-computed using a **ResNet-152 + PCA** pipeline (reducing 2048-dim vectors to 512 dimensions). For custom video inference, we use the identical extractor to prevent **Domain Shift** (which occurs when training and production features belong to different statistical distributions, causing the sequential classifier to fail). 

### Q3: Why BiLSTM over a standard LSTM or Transformer?
**Answer:** 
*   **BiLSTM vs. Standard LSTM**: Standard LSTMs only read left-to-right (past $\rightarrow$ future). Football events require future context (e.g. players celebrating in frame 25 helps identify that a goal occurred in frame 15). The BiLSTM processes sequences both forward and backward, concatenating hidden states $[h_{forward}, h_{backward}]$ to provide full temporal context.
*   **BiLSTM vs. Transformer**: Transformers lack sequential inductive bias (they assume nothing about order and must learn temporal patterns from scratch). With our dataset size of 60 matches, the Transformer underperforms because it requires substantially more data.

### Q4: Explain the 0/1 Knapsack Dynamic Programming optimizer.
**Answer:** The Knapsack DP formulation handles time-bounded highlight selection. We treat each cropped and padded match event as an item:
*   **Weight ($w_i$)**: Duration of the event clip.
*   **Value ($v_i$)**: Model prediction confidence.
*   **Constraint ($W$)**: User-defined total highlight budget (e.g. 7 minutes).
*   Unlike greedy selection (which just picks top-N by confidence and ignores length, potentially squeezing out multiple short, high-value clips in favor of one long, mediocre clip), Knapsack DP solves the recurrence:
    $$K[i][w] = \max(K[i-1][w], \, v_i + K[i-1][w-w_i])$$
    This guarantees the mathematically optimal highlight reel combination under the exact budget constraints.

---

## 🎨 Category 2: Multi-Modal Integration & Features

### Q5: How does your audio event classifier work?
**Answer:** The module `audio_classifier.py` extracts the audio track and computes the **Short-Time Fourier Transform (STFT)**. It detects peak amplitudes to identify high-energy audio anomalies. We distinguish crowd cheers (broadband, covering 500-4000Hz frequency distributions) from referee whistles (narrow high-frequency bands between 1000-2500Hz) to map high-energy match milestones.

### Q6: How does the Scoreboard OCR tracking operate?
**Answer:** The scoreboard overlay containing match minutes and score digits is cropped using a user-specified Region of Interest (ROI) (Top-Left or Top-Right). The crop is preprocessed (grayscaled, thresholded to high contrast) and sent to **Tesseract OCR**. Changes in scores indicate goals, allowing the pipeline to map exact scoring milestones.

### Q7: Explain the CLIP Text-to-Video search module.
**Answer:** Traditional search is limited to our model classes (Goal, Cards, Sub). To enable searching for arbitrary text queries (e.g., "corner kick", "headbutt", "goalkeeper save"), `clip_search.py` extracts visual embeddings from sampled video frames using the pre-trained **CLIP** vision encoder. When a user queries a text string, the CLIP text encoder encodes the query, and we calculate the **cosine similarity** between the text embedding and all frame embeddings to retrieve the top-matching moments.

### Q8: How does your local RAG chatbot function without an internet connection?
**Answer:** `chatbot_agent.py` implements a local keyword-based Retrieval-Augmented Generation (RAG) system. It reads files in the `upgrade/` directory, divides them into context chunks (1200 characters with 200 overlap), and uses keyword frequency scoring (measuring query keyword overlap) to retrieve the top-matching document blocks as context, falling back to local rule-based Q&A if external APIs are unavailable.

---

## ⚖️ Category 3: Coding & Implementation Details

### Q9: How is the Key Match Moments photo grid rendered on-the-fly?
**Answer:** The server exposes `/api/frame/<task_id>?t=seconds`. When the client requests this, OpenCV opens the video, jumps to the target timestamp, extracts a single frame image, resizes it to 640px to save network bandwidth, encodes it to JPEG, and streams the raw bytes. The frontend SPA renders these dynamically in a responsive **3-column photo grid** showing 12 evenly-spaced screenshots.

### Q10: Why do we initialize the LSTM forget gate bias to 1.0?
**Answer:** Setting the forget gate bias to 1.0 ensures the forget gate is mostly "open" ($\sigma(1.0) \approx 0.73$) at the beginning of training. This prevents the LSTM from aggressively forgetting long-term sequence history in early epochs, which stabilizes and accelerates training.

### Q11: What would you do with unlimited compute and development time?
**Answer:**
1.  **Scale Training**: Train the BiLSTM model on the full 500-match SoccerNet dataset rather than our 60-match subset.
2.  **Audio-Visual Fusion**: Integrate audio spectrogram features directly into the BiLSTM inputs alongside visual ResNet features, rather than running them as separate pipelines, allowing the neural model to learn combined audio-visual signatures.
3.  **Learned Domain Adaptation**: Implement an adversarial domain classifier layer to automatically translate ResNet-18 features into ResNet-152 feature spaces, allowing video processing on lightweight devices without domain shift failures.
