# 🔬 Technical Documentation — Line-by-Line Code Explanation

> **This document is for YOUR personal understanding only. It is NOT uploaded to GitHub.**

---

## Table of Contents
1. [Cell 1: Setup & Imports](#cell-1)
2. [Cell 2/2b: Dependencies & Backup Utilities](#cell-2)
3. [Cell 3: Configuration (CONFIG)](#cell-3)
4. [Cell 4: SoccerNet Download](#cell-4)
5. [Cell 5: Dataset Class](#cell-5)
6. [Cell 6: Model Definitions](#cell-6)
7. [Cell 7: Training Engine](#cell-7)
8. [Cells 8/8a/8b/8c: Data Loading & Training](#cell-8)
9. [Cell 9: Visualization](#cell-9)
10. [Cells 10/10b/10c: Highlight Generation Pipeline](#cell-10)
11. [Cell 12: Gradio UI](#cell-12)
12. [Key Concepts Explained](#concepts)

---

## Cell 1: Setup & Imports <a name="cell-1"></a>

**Lines 25-51: Standard Python imports**
```python
import torch               # PyTorch — the deep learning framework
import torch.nn as nn       # Neural network layers (Linear, LSTM, etc.)
import torch.optim as optim # Optimizers (Adam, SGD)
from torch.utils.data import Dataset, DataLoader  # Data pipeline
```
- `Dataset`: Abstract class — you define `__getitem__` and `__len__`, PyTorch handles batching
- `DataLoader`: Wraps a Dataset and provides mini-batches, shuffling, multi-worker loading

```python
from sklearn.metrics import average_precision_score  # Computes mAP
```
- **mAP (mean Average Precision)**: The primary metric for action spotting. It measures how well your model ranks true events above false ones. Higher = better. Range: 0.0 to 1.0.

**Lines 54-68: Colab vs Local detection**
```python
try:
    from google.colab import drive  # This import ONLY works on Colab
    ON_COLAB = True
except ImportError:
    ON_COLAB = False
```
- If running on Colab → mount Google Drive for persistent storage
- If running locally → use the script's directory
- **Why Drive?** Colab VMs are ephemeral. Without Drive, all files are lost when the session disconnects.

**Lines 74-78: GPU Check**
```python
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```
- `cuda` = NVIDIA GPU (fast). `cpu` = regular processor (10-50x slower)
- Colab's free T4 GPU has 15GB VRAM — more than enough for our models

---

## Cell 2/2b: Dependencies & Backup Utilities <a name="cell-2"></a>

**Cell 2 (Line 88):** Installs `SoccerNet` (dataset API) and `moviepy` (video editing)

**Cell 2b: backup_and_download() (Lines 100-150)**
- Zips all checkpoints + results + outputs into a single `.zip`
- Auto-downloads to your Mac via `google.colab.files.download()`
- **Why?** Insurance against Colab crashes. After every training cell, a backup is downloaded.

**Cell 2b: restore_from_backup() (Lines 152-197)**
- Reads a previously downloaded backup zip
- Extracts files back into the correct directories (`checkpoints/`, `results/`, `outputs/`)
- **Critical detail (Lines 1094-1100):** The training skip logic checks for BOTH `*_best.pth` AND `*_test_results.pth`. If the backup only contains `*_best.pth`, training will NOT skip — it will resume. This is the bug you hit when the test_results files were missing.

---

## Cell 3: Configuration (CONFIG) <a name="cell-3"></a>

Every hyperparameter in one dictionary. Here's what each one means and **why it has that value**:

### Data Parameters
| Parameter | Value | Why |
|---|---|---|
| `feature_dim: 512` | ResNet-152 outputs 2048-dim features, reduced to 512-dim via PCA. This matches SoccerNet's pre-extracted features exactly, eliminating domain shift. |
| `target_fps: 2` | SoccerNet features are pre-extracted at 2 frames/second. A 45-min half = 5400 frames. |
| `seq_len: 30` | 30 frames @ 2FPS = **15 seconds** of context. The model sees 15 seconds and decides "is there a goal in the middle of this window?" |
| `seq_stride: 15` | Slide the window by 15 frames = **50% overlap**. Each frame gets predictions from ~2 overlapping windows, which are averaged for smoother output. |
| `max_matches_train: 60` | Free Colab has 112GB disk. Each match is ~20MB of features. 60 matches ≈ 1.2GB. We keep it under 2GB to leave room for checkpoints. |

### Event Classes
| Parameter | Value | Why |
|---|---|---|
| `event_classes: ['Goal', 'Cards', 'Substitution', 'Background']` | 4 classes. SoccerNet has 17 event types, but we merge them into 3 meaningful categories + Background. |
| `label_map` | Maps SoccerNet labels → our classes. e.g., `'Penalty' → 'Goal'`, `'Yellow card' → 'Cards'`. |
| `event_window: 10` | ±10 frames (±5 seconds) around each labeled event. This creates the "positive" training examples. Without this window, only 1 frame per event would be labeled, making the dataset too sparse. |

### Model Parameters
| Parameter | Value | Why |
|---|---|---|
| `hidden_dim: 256` | LSTM hidden state size. 256 = good balance. Too small (64) → underfits. Too large (512) → overfits on 60 matches. |
| `num_heads: 8` | Transformer attention heads. 8 heads with 512-dim input → each head processes 64 dimensions. Standard configuration from "Attention Is All You Need" paper. |
| `num_layers: 2` | Both LSTM and Transformer use 2 layers. Deeper networks need more data; 2 layers is optimal for our 60-match dataset. |
| `dropout: 0.4` | 40% dropout — aggressive, but necessary. With only 60 matches, the model easily memorizes. Dropout forces the network to use redundant pathways, improving generalization. |

### Training Parameters
| Parameter | Value | Why |
|---|---|---|
| `lr: 5e-4` | Learning rate for Adam. 5e-4 is the "sweet spot" for Adam on sequence classification tasks. |
| `weight_decay: 1e-5` | L2 regularization. Penalizes large weights to prevent overfitting. Very small because Adam already handles this partially. |
| `batch_size: 32` | 32 sequences per gradient update. Fits comfortably in T4's 15GB VRAM. |
| `epochs: 80` | Maximum epochs. Early stopping usually triggers at epoch 15-25. |
| `patience: 15` | Stop if mAP doesn't improve for 15 consecutive epochs. Prevents wasting time on plateaus. |
| `clip_grad: 5.0` | Gradient clipping. LSTMs are prone to "exploding gradients" — clipping at 5.0 prevents NaN losses. |
| `reg_factor: 0.15` | Target: 15% of frames should be events. If the model predicts 50% events, this penalty pushes it back down. |
| `reg_lambda: 0.5` | Weight of the length regularization term. 0.5 = moderate influence. |

---

## Cell 4: SoccerNet Download <a name="cell-4"></a>

**Lines 296-363: download_soccernet_subset()**

1. Uses the official `SoccerNetDownloader` API
2. Downloads ONLY the `test` split (smallest) with 3 files per match:
   - `1_ResNET_TF2_PCA512.npy` — First half features (ResNet-152 + PCA to 512-dim)
   - `2_ResNET_TF2_PCA512.npy` — Second half features
   - `Labels-v2.json` — Event annotations with timestamps
3. Selects 100 matches (60 train + 20 val + 20 test), copies to Drive
4. Cleans up the temporary download from Colab's local disk

**Why the "test" split?** SoccerNet's `test` split is the smallest (~100 matches). The `train` split has 300+ matches and would exceed Colab's disk limit.

---

## Cell 5: Dataset Class <a name="cell-5"></a>

### SoccerNetHighlightDataset (Lines 373-555)

**_process_match() (Lines 440-509) — The most critical function for understanding the data pipeline:**

1. **Load features:** `np.load(feat_path)` → shape `(num_frames, 512)`. For a 45-min half at 2FPS, this is `(5400, 512)`.

2. **Create frame-level labels (Lines 456-488):**
   ```python
   frame_labels = np.full(num_frames, 3, dtype=np.int64)  # All Background initially
   ```
   Then for each annotation in `Labels-v2.json`:
   - Parse the timestamp: `"1 - 23:45"` → half=1, seconds=1425, frame=2850
   - Map the label: `"Yellow card"` → `"Cards"` → class index 1
   - Label a WINDOW: frames 2840 to 2861 (±10 frames = ±5 seconds) all get label 1

3. **Create sliding windows (Lines 490-509):**
   ```python
   for start_idx in range(0, num_frames - seq_len, stride):
       seq_features = features[start_idx:start_idx + 30]  # 30 frames
       center_label = seq_labels[seq_len // 2]             # Label of frame 15
   ```
   Each training sample is a 30-frame window. The label is the CENTER frame's class. This gives the model temporal context (15 frames before and after) to make its prediction.

**_compute_class_weights() (Lines 531-549):**
```python
weights[i] = total / (num_classes * label_counts[i])
```
- **Problem:** ~95% of frames are Background, ~2% Goals, ~1.5% Cards, ~1.5% Substitution
- **Solution:** Inverse frequency weighting. Background gets weight ~0.26, Goals get weight ~12.5
- This tells the loss function: "A mistake on a Goal frame is 50x worse than a mistake on a Background frame"

---

## Cell 6: Model Definitions <a name="cell-6"></a>

### Model A: BaselineCNN (Lines 593-631)
```
Input (batch, 30, 512) → Take center frame only (batch, 512) → FC(512→256) → ReLU → Dropout → LayerNorm → FC(256→128) → ReLU → Dropout → FC(128→4) → Output
```
- **No temporal modeling.** Only looks at 1 frame. This is our ablation baseline to PROVE that temporal context matters.
- **Xavier initialization:** Weights are sampled from a distribution that keeps gradients stable across layers.

### Model B: BiLSTMClassifier (Lines 636-703) — THE BEST MODEL
```
Input (batch, 30, 512) → BiLSTM(2 layers, hidden=256) → Take center output (batch, 512) → FC(512→256) → ReLU → Dropout → LayerNorm → FC(256→128) → ReLU → Dropout → FC(128→4) → Output
```

**Key architectural decisions:**

1. **Bidirectional LSTM (Line 650-657):**
   - Forward LSTM reads frames 1→30 (past→future)
   - Backward LSTM reads frames 30→1 (future→past)
   - Output is CONCATENATED: 256 forward + 256 backward = 512
   - **Why bidirectional?** A goal celebration (frame 20) helps identify the goal (frame 15) even though it comes AFTER. The backward pass catches this.

2. **Center frame output (Lines 699-701):**
   ```python
   center_idx = x.shape[1] // 2  # = 15
   center_output = lstm_out[:, center_idx, :]
   ```
   We use frame 15's output because it has seen 15 frames of past context (forward) AND 15 frames of future context (backward). It's the most informed position.

3. **Forget gate bias = 1.0 (Lines 683-685):**
   ```python
   param.data[n // 4:n // 2].fill_(1.0)
   ```
   The forget gate controls what the LSTM "remembers". Setting its bias to 1.0 means the LSTM starts by remembering everything, then learns what to forget. This is a well-known trick from the original LSTM paper that prevents the "vanishing information" problem in early training.

### Model C: TransformerClassifier (Lines 729-792)

**PositionalEncoding (Lines 708-727):**
```python
pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)
```
- Transformers have NO inherent notion of order. Without positional encoding, frames 1 and 30 are identical to the model.
- Sinusoidal encoding creates unique position "fingerprints" using sine and cosine waves at different frequencies.

**Why Transformer underperforms BiLSTM here:**
- Transformers need LOTS of data to learn attention patterns. With only 60 matches (~150K sequences), the BiLSTM's inductive bias (sequential processing) gives it an advantage.
- This is actually a great viva point: "Transformers are not universally superior — they need sufficient data to outperform recurrent architectures."

---

## Cell 7: Training Engine <a name="cell-7"></a>

### Loss Function (Lines 826-828 + 854-859)
```python
self.criterion = nn.CrossEntropyLoss(weight=class_weights)
```
Total loss = CrossEntropy + λ × LengthRegularization

1. **CrossEntropyLoss:** Standard classification loss. With class weights, it penalizes misclassifying rare events (Goals) much more than misclassifying Background.

2. **Length Regularization (Lines 854-859):**
   ```python
   event_probs = 1.0 - probs[:, -1]  # Probability of NOT being Background
   return torch.abs(event_probs.mean() - 0.15)  # Target: 15% events
   ```
   Without this, the model might predict EVERYTHING as an event (to avoid missing any). This term says: "The average event probability should be ~15%". Inspired by the CA-SUM paper.

### Optimizer & Scheduler (Lines 831-840)
- **Adam optimizer:** Adaptive learning rate per parameter. Works well out-of-the-box.
- **ReduceLROnPlateau:** If mAP doesn't improve for 5 epochs, halve the learning rate. This lets the model "fine-tune" with smaller steps when it plateaus.

### mAP Computation (Lines 925-935)
```python
for c in range(num_classes - 1):  # Exclude Background
    binary_labels = (all_labels == c).astype(int)  # One-vs-all
    ap = average_precision_score(binary_labels, all_probs[:, c])
```
- For each event class (Goal, Cards, Substitution), compute Average Precision separately
- Average Precision = area under the Precision-Recall curve
- mAP = mean of all APs
- **Background is excluded** from mAP because we only care about event detection quality

### Early Stopping (Lines 1040-1044)
```python
if self.patience_counter >= 15:
    print("Early stopping...")
    break
```
- If mAP doesn't improve for 15 epochs → stop training
- Load the BEST checkpoint (not the latest) for evaluation

---

## Cells 8/8a/8b/8c: Data Loading & Training <a name="cell-8"></a>

**Cell 8 (Lines 1067-1077):** Creates DataLoaders. If SoccerNet data doesn't exist, falls back to synthetic (random noise) data.

**train_and_evaluate_model() (Lines 1088-1146):**
- **Skip logic (Lines 1094-1111):** Checks for BOTH `*_test_results.pth` AND `*_best.pth`. If both exist → skip training entirely.
- **Training flow:** Create model → Create TrainingEngine → Train → Evaluate on test set → Save test results
- After each model, `backup_and_download()` saves everything to your Mac

---

## Cell 9: Visualization <a name="cell-9"></a>

Generates 5 plots:
1. **Training loss curves** — 3 subplots showing train vs val loss over epochs
2. **Validation mAP curves** — All 3 models on one plot, showing BiLSTM dominance
3. **Model comparison bar chart** — Side-by-side accuracy, mAP, loss
4. **Confusion matrices** — Normalized, showing what gets confused with what
5. **Classification report** — Precision, recall, F1-score per class

---

## Cells 10/10b/10c: Highlight Generation Pipeline <a name="cell-10"></a>

### generate_highlights() (Lines 1387-1498) — THE CORE INFERENCE FUNCTION

**Step 1: Sliding window inference (Lines 1422-1436)**
```python
for start in range(0, num_frames - seq_len + 1):
    seq = features[start:start + 30]  # 30-frame window
    probs = model(seq)                # → 4 class probabilities
    all_probs[center] += probs        # Accumulate at center frame
    frame_counts[center] += 1
all_probs[valid] /= frame_counts[valid]  # Average overlapping predictions
```
Each frame gets predictions from multiple overlapping windows. Averaging smooths out noise.

**Step 2: Thresholding (Line 1440)**
```python
event_mask = all_probs[:, :bg_class].max(axis=1) > 0.3
```
A frame is an "event" if ANY non-background class probability exceeds 30%. We lowered this from the original 50% to improve recall on custom videos where the model may be less confident.

**Step 3: Group consecutive detections (Lines 1443-1467)**
If frames 100-120 are all marked as events, they become ONE event spanning frames 100-120. The event's class is the average argmax, and confidence is the average probability.

**Step 4: Add padding (Lines 1456-1459)**
```python
pad = 30 * config['target_fps']  # ±30 seconds = 60 frames
events.append({
    'start': max(0, event_start - pad),
    'end': min(num_frames, i + pad),
})
```
Each detected event gets ±30 seconds of padding. A goal at 23:45 becomes a clip from 23:15 to 24:15. This ensures you see the build-up and celebration.

**Step 5: Knapsack selection (Lines 1345-1385)**
```
Items: detected events (each with a duration "weight" and confidence "value")
Constraint: total highlight duration ≤ 10 minutes (1200 frames)
Goal: maximize total confidence
Algorithm: 0/1 Knapsack using Dynamic Programming
```
This is the same algorithm from your Data Structures course! It guarantees the mathematically optimal selection of clips under the time budget.

### extract_features_from_video() (Lines 1584-1647)
- Uses **ResNet-152** (matching SoccerNet's original extractor) to extract 2048-dim features from raw MP4 frames
- Applies **PCA (2048→512)** to reduce dimensionality, aligning with SoccerNet's feature format
- Processes at 2 FPS (1 frame every 15 video frames for a 30fps video)
- **Domain shift is RESOLVED:** ResNet-152 + PCA features closely match the SoccerNet training distribution

### create_highlight_video() (Lines 1659-1708)
- Uses `moviepy.VideoFileClip` to open the source video
- `video.subclip(start_sec, end_sec)` cuts each event clip
- `concatenate_videoclips(clips)` joins them sequentially
- Outputs a final `.mp4` with `libx264` codec (H.264, universally playable)

---

## Cell 12: Gradio UI <a name="cell-12"></a>

An optional web interface for demo presentations. It:
1. Shows a dropdown of available SoccerNet test matches
2. Has a confidence threshold slider
3. Runs `generate_highlights()` and displays results as a markdown table + timeline plot
4. `share=True` creates a public URL (e.g., `https://abc123.gradio.live`) that anyone can access for 72 hours

**Limitation:** Only works with SoccerNet matches (needs `Labels-v2.json`). Does NOT support custom video upload through the UI.

---

## Key Concepts Explained <a name="concepts"></a>

### What is mAP (mean Average Precision)?
Imagine you have 100 frames and 5 of them are goals. Your model ranks all 100 frames by "goal probability". If the top 5 ranked frames are ALL actual goals → AP = 1.0 (perfect). If actual goals are scattered at ranks 1, 10, 50, 80, 99 → AP ≈ 0.3 (poor ranking). mAP is the average AP across all event classes.

### What is Domain Shift? (And How We Fixed It)
Originally, our BiLSTM learned: "When I see vector pattern X, it's a Goal." Pattern X came from ResNet-152+PCA features. When we initially used ResNet-18 features for custom videos, the vectors looked completely different — even for the same video frame. The model couldn't recognize them.

**Analogy:** You learned to recognize dogs from photographs. Now I show you X-ray images of dogs. Same dogs, completely different visual representation. You can't recognize them anymore.

**Our Solution:** We upgraded the custom video pipeline to use the same **ResNet-152 + PCA** architecture. Now the "photographs" look the same whether they come from SoccerNet or from a YouTube video. Domain shift is resolved.

### Why BiLSTM beats Transformer here?
1. **Data efficiency:** LSTMs have an inductive bias for sequential data (process left-to-right). Transformers must LEARN this from data via attention patterns.
2. **60 matches is tiny.** Transformers typically need 10,000+ examples to outperform LSTMs.
3. **Sequence length is short (30 frames).** Transformers shine on LONG sequences (1000+ tokens) where LSTMs forget. At 30 frames, LSTM memory is sufficient.

### What is the Knapsack Algorithm?
- **Problem:** You have N items, each with a weight (clip duration) and value (detection confidence). Your backpack has a weight limit (10-minute highlight budget). Pick items to maximize total value without exceeding the weight limit.
- **Solution:** Dynamic Programming table of size N × W. Time complexity: O(N × W). Guarantees the optimal solution.
- **In our case:** N ≈ 20-50 detected events, W = 1200 frames (10 min × 2 FPS × 60 sec). The table is small and solves instantly.
