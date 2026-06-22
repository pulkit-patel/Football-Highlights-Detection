# 🚀 How to Run — Upgraded Football Highlight & Multi-Modal Hub

This guide describes how to run and test the upgraded application locally on your system using the **Flask API Backend** and the **Three.js 3D Web Dashboard**.

---

## 📋 Prerequisites & Setup

Ensure you have Python 3.8+ installed. Then install the necessary dependencies:

```bash
pip install -r requirements.txt
```

*(Optional)* If you plan to use the Generative AI Match Summary or the advanced LLM RAG chatbot features, you should get a **Hugging Face Inference API Token** (`hf_...`) and place it in the UI input key. If left blank, the system will fall back to a local rule-based match summarizer and keyword codebase search.

---

## 🏃 Starting the Server

To launch the backend API server and serve the interactive 3D frontend:

```bash
python api_server.py
```

You should see output similar to this:
```text
   Football Highlight Detection -- API Server
   Model: loaded
   -> Open  http://localhost:5000
```

Open your browser and navigate to: **[http://localhost:5000](http://localhost:5000)**

---

## 🎬 Testing the Pipeline in the 3D Dashboard

We have copied the required test files directly to the root of your project directory for quick testing:

1.  **Visual Features file**: [final_highlights_features.npy](file:///c:/Users/PULKIT/OneDrive/Desktop/Football-Highlights-Detection/final_highlights_features.npy) (Visual frame vectors from ResNet-152 + PCA)
2.  **Match Video file**: [final_highlights.mp4](file:///c:/Users/PULKIT/OneDrive/Desktop/Football-Highlights-Detection/final_highlights.mp4) (The source match MP4 video)

### Steps to Run:
1.  Open the web interface at [http://localhost:5000](http://localhost:5000).
2.  In the **Process Video Pipeline** panel, drag and drop or upload `final_highlights_features.npy` into the features dropzone.
3.  Drag and drop or upload `final_highlights.mp4` into the video dropzone.
4.  Configure side parameters if desired:
    *   **Detection Sensitivity**: Adjust detection confidence threshold.
    *   **Max Highlight Length**: Pick the total length constraint in minutes.
    *   **AI Upgrade Features**: Check/Uncheck OCR, Audio, and CLIP analysis modules.
5.  Click the **🚀 Run Multi-Modal Highlight Pipeline** button.
6.  Monitor the real-time progress bar. Once completed, the SPA will unlock interactive tabs:
    *   **🎬 Highlight Reel & Summary**: Watch the stitched MP4 highlight video, download it, see a detailed dynamic timeline, and inspect the generative AI narrative.
    *   **📸 Key Match Moments**: Review the simple 3-column photo grid showcasing 12 evenly-spaced frame screenshot snapshots of the match with timestamps.
    *   **🔤 Scoreboard Tracker (OCR)**: View team scores and game time extracted via optical character recognition.
    *   **🔊 Audio Event Analysis**: View the sound amplitude chart showing cheer and whistle events.
    *   **🔍 Semantic Video Search**: Perform natural language searches on the video (e.g. "whistle", "referee", "player celebrates").
    *   **💬 AI Chatbot Assistant**: Chat with the RAG agent about the match context or query it about codebase details.

---

## 💻 Optional: Running Jupyter Notebooks / Colab

The original training notebooks are preserved under the project root for reference:
*   [Highlight_generation_FINAL_Upgraded.ipynb](file:///c:/Users/PULKIT/OneDrive/Desktop/Football-Highlights-Detection/Highlight_generation_FINAL_Upgraded.ipynb)
*   [Highlight_generation_FINAL_Upgraded_colab.ipynb](file:///c:/Users/PULKIT/OneDrive/Desktop/Football-Highlights-Detection/Highlight_generation_FINAL_Upgraded_colab.ipynb)
*   [Highlight_generation_FINAL_colab.ipynb](file:///c:/Users/PULKIT/OneDrive/Desktop/Football-Highlights-Detection/Highlight_generation_FINAL_colab.ipynb)

These notebooks show how the CNN model (ResNet-152) and sequential classifiers (BiLSTM, Transformer, CNN Baseline) were trained and evaluated on the SoccerNet v2 Action Spotting benchmark.
