# 🚀 How to Run — Football Highlight Detection

This guide covers three scenarios:
- **Path A:** Full training from scratch
- **Path B:** Inference using pre-trained checkpoints (recommended)
- **Path C:** Generate highlights from a custom YouTube video

All paths assume you are using **Google Colab with a GPU runtime**.

---

## Prerequisites

1. A Google account with access to [Google Colab](https://colab.research.google.com/)
2. This repository cloned or the notebook uploaded
3. For Path B/C: The pre-trained checkpoint files from `checkpoints/`

### Setting Up Colab

1. Go to Google Colab → `File` → `Upload Notebook` → upload `Highlight_generation_FINAL.ipynb`
2. Go to `Runtime` → `Change runtime type` → select **GPU** (T4 is fine)
3. Click `Connect` in the top right

---

## Path A: Full Training from Scratch

> ⏱️ **Estimated time:** 2-3 hours (data download + training)

### Step 1: Setup Environment
Run **Cells 1, 2, 2b, 3** in order.
- Cell 1: Installs `SoccerNet` and `moviepy` packages
- Cell 2: Imports all Python libraries
- Cell 2b: Checks GPU availability (should print `cuda`)
- Cell 3: Defines the `CONFIG` dictionary with all hyperparameters

### Step 2: Define Models
Run **Cells 5, 6, 7** in order.
- Cell 5: Defines `BaselineClassifier` (simple FC network)
- Cell 6: Defines `BiLSTMClassifier` (the best model)
- Cell 7: Defines `TransformerClassifier`

### Step 3: Download and Load SoccerNet Data
Run **Cell 8**.
- This downloads ~2GB of SoccerNet features and labels
- You will need a SoccerNet password (register at soccer-net.org)
- Creates train/test DataLoaders with class weights

### Step 4: Train All Three Models
Run **Cells 8a, 8b, 8c** in order.
- 8a: Trains CNN Baseline (~5 min)
- 8b: Trains CNN + BiLSTM (~15 min)
- 8c: Trains CNN + Transformer (~15 min)
- Each cell auto-saves the best checkpoint and creates a backup ZIP

### Step 5: Visualize Results
Run **Cell 9**.
- Generates loss curves, mAP curves, confusion matrices, and model comparison charts
- Saves everything to `results/`

### Step 6: Generate Highlights
Run **Cell 10**.
- Uses the best model (BiLSTM) to detect events on a test match
- Applies Knapsack optimization to select the best clips
- Prints detected events with timestamps

### Step 7: Stitch Video
Run **Cell 10c**.
- Cuts clips from the original video and concatenates them
- Outputs a downloadable highlight reel

---

## Path B: Inference with Pre-trained Checkpoints (Recommended)

> ⏱️ **Estimated time:** 10 minutes

### Step 1: Setup
Run **Cells 1, 2, 2b, 3, 5, 6, 7** (definitions only — takes 10 seconds)

### Step 2: Upload Checkpoint
Upload `checkpoints/CNN_BiLSTM_best.pth` to the Colab file browser (drag and drop into `/content/`).

### Step 3: Load Model
In a **new code cell**, run:
```python
model = BiLSTMClassifier(CONFIG).to(CONFIG['device'])
ckpt = torch.load('/content/CNN_BiLSTM_best.pth', map_location=CONFIG['device'], weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"✅ Loaded BiLSTM (best mAP: {ckpt.get('best_val_map', 'N/A')})")
```

### Step 4: Load SoccerNet Data for Inference
Run **Cell 8** (loads a test match).
Then run **Cell 10** to detect events.
Then run **Cell 10c** to stitch the video.

---

## Path C: Custom Video Highlight Generation

> ⏱️ **Estimated time:** 15-30 minutes (depends on video length)

### Step 1: Setup
Run **Cells 1, 2, 2b, 3, 5, 6, 7** (definitions only)

### Step 2: Upload Files
Upload two files to Colab:
- `CNN_BiLSTM_best.pth` (your trained model)
- Your match video (e.g., `my_match.mp4`)

### Step 3: Run Everything
Run **Cells 10, 10b, 10c** first (to define the functions — Cell 10 will error at the bottom, ignore it).

Then in a **new code cell**, run:
```python
# Load model
model = BiLSTMClassifier(CONFIG).to(CONFIG['device'])
ckpt = torch.load('/content/CNN_BiLSTM_best.pth', map_location=CONFIG['device'], weights_only=False)
model.load_state_dict(ckpt['model_state_dict'])
model.eval()
print(f"✅ Loaded BiLSTM (best mAP: {ckpt.get('best_val_map', 'N/A')})")

# Extract features from your video
extract_features_from_video('/content/my_match.mp4', '/content/test_features.npy')

# Detect events
selected_events, all_detected = generate_highlights(model, '/content/test_features.npy', CONFIG)
print(f"\n📍 {len(all_detected)} detections → {len(selected_events)} selected")
for evt in selected_events:
    print(f"   Half {evt['half']} — {evt['time_str']} — {evt['class']} (conf: {evt['confidence']:.3f})")

# Stitch highlight video
create_highlight_video('/content/my_match.mp4', selected_events, '/content/final_highlights.mp4')
```

### Step 4: Download
After stitching completes, the file will be at `/content/final_highlights.mp4`. Click it in the file browser to download.

---

## ⚠️ Important Notes

### Domain Shift — RESOLVED ✅
In our initial version, custom videos used ResNet-18 for feature extraction, creating a feature space mismatch with the SoccerNet-trained BiLSTM. **This has been fixed** — Cell 10b now uses **ResNet-152 + PCA (2048→512)**, matching the exact feature distribution the model was trained on. Custom video accuracy is now on par with SoccerNet test matches.

### Common Errors

| Error | Fix |
|---|---|
| `BadZipFile: File is not a zip file` | Re-upload the checkpoint — file got corrupted during transfer |
| `NameError: generate_highlights is not defined` | You forgot to run Cell 10 first (defines the function) |
| `CUDA out of memory` | Restart runtime → `Runtime` → `Restart runtime` |
| Cell 10 errors at bottom | Expected if you didn't run Cell 8. The functions are still defined — just ignore the error |

---

## 📁 Output Files

| File | Description |
|---|---|
| `final_highlights.mp4` | Your generated highlight video |
| `detected_events.json` | Raw event detections with timestamps |
| `results/*.png` | Training curves and confusion matrices |
| `checkpoints/*_best.pth` | Saved model weights |
