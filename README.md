# 🏟️ AI-Based Football Highlight Detection System

An automated end-to-end pipeline for detecting key match events (Goals, Cards, Substitutions) in football broadcasts and generating condensed highlight reels.

## 🌟 Overview
This project implements a deep learning approach to **Action Spotting** in soccer videos. By combining computer vision for spatial understanding and sequential modeling for temporal context, the system can identify specific events across a full 90-minute match and automatically stitch them into a high-quality highlight video.

## 🧠 Architecture
The system follows a two-stage architecture:

1.  **Spatial Feature Extraction (CNN):** 
    *   Uses a pre-trained **ResNet** backbone to extract high-dimensional spatial features from video frames at 2 FPS.
    *   Supports both ResNet18 and ResNet152 (matching the SoccerNet benchmark).
2.  **Temporal Modeling:**
    *   **Bi-Directional LSTM (Best Performer):** Captures dependencies in both forward and backward time directions.
    *   **Transformer Encoder:** Utilizes multi-head self-attention to weigh the importance of frames within a 15-second window.
3.  **Optimization:**
    *   Uses a **Knapsack DP algorithm** to select the most confident events within a user-defined time budget (e.g., "Give me a 5-minute highlight reel").
    *   **Temporal NMS:** Merges overlapping detections and replay clips for a clean output.

## 📊 Key Results
Trained on a subset of the **SoccerNet-v2** dataset:
*   **Best Model:** CNN + Bi-LSTM
*   **mAP (Mean Average Precision):** ~35.1% (Trained on 60 matches)
*   **Validation Accuracy:** ~96.3%

*Note: These results are achieved on a data-constrained subset to fit within computational limits. Standard benchmarks on 500+ matches typically reach 60-70% mAP.*

## 🚀 Getting Started

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Usage
*   **Training:** Run the `Football_Highlight_Detection.py` script or use the `Highlight_generation_FINAL.ipynb` notebook on Google Colab.
*   **Inference:**
    1.  Upload your match video (`.mp4`).
    2.  Extract features using `extract_features_from_video()`.
    3.  Generate events with `generate_highlights()`.
    4.  Stitch the video with `create_highlight_video()`.

## 📁 Repository Structure
*   `Football_Highlight_Detection.py`: Core logic for training and inference.
*   `Highlight_generation_FINAL.ipynb`: Interactive notebook for Colab.
*   `checkpoints/`: Pre-trained weights for Bi-LSTM and Transformer models.
*   `results/`: Training curves, confusion matrices, and performance metrics.
*   `requirements.txt`: Project dependencies.

## 🤝 Acknowledgments
*   **SoccerNet Team:** For the comprehensive dataset and baseline features.
*   **MoviePy:** For the automated video editing capabilities.
