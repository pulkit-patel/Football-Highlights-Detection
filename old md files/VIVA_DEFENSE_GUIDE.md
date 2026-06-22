# 🎓 Viva Defense Guide — Every Question You Could Be Asked

> **This document is for YOUR personal preparation only. NOT uploaded to GitHub.**

---

## Category 1: Architecture & Design Decisions

### Q1: Why did you choose a two-stage pipeline instead of end-to-end?
**Answer:** End-to-end training (raw pixels → event labels) would require processing 224×224×3 images through a CNN AND a temporal model simultaneously. This needs 40+ GB of GPU RAM for backpropagation. Colab's free T4 has only 15GB. The two-stage approach decouples spatial feature extraction (done once, stored as .npy files) from temporal modeling (trained iteratively), making it feasible on consumer hardware. Additionally, using pre-extracted features from SoccerNet's official pipeline ensures we're working with high-quality, standardized representations.

### Q2: Why ResNet-152 + PCA for feature extraction?
**Answer:** For the SoccerNet training data, we use pre-computed ResNet-152 + PCA features provided by the dataset creators. For custom video inference, we also use ResNet-152 + PCA to ensure there is no domain shift between training and inference features. Both pipelines output 512-dimensional vectors in the same semantic space, so the BiLSTM receives in-distribution inputs regardless of the video source. Initially we used ResNet-18 (10x faster, 11M vs 60M params) for custom videos, but we discovered the feature space mismatch and upgraded to ResNet-152 to resolve it.

### Q3: Why BiLSTM over a standard LSTM?
**Answer:** A standard LSTM only reads frames left-to-right. It can use past context but not future context. For event detection, future context is critical — a crowd celebration 3 seconds AFTER a goal helps confirm it was indeed a goal. The bidirectional LSTM reads both directions and concatenates the hidden states, giving each frame context from both past and future.

### Q4: Why did the Transformer underperform the BiLSTM?
**Answer:** Three reasons:
1. **Data scale:** We only had 60 matches (~150K sequences). Transformers require substantially more data to learn effective attention patterns. The BiLSTM has a built-in inductive bias for sequential data.
2. **Sequence length:** Our windows are 30 frames. Transformers shine on long sequences (1000+) where LSTMs struggle with vanishing gradients. At 30 frames, LSTM memory is sufficient.
3. **Positional encoding overhead:** The sinusoidal encoding adds an extra learning burden. With limited data, this becomes noise rather than signal.

### Q5: What is the Knapsack algorithm and why use it?
**Answer:** The 0/1 Knapsack is a Dynamic Programming algorithm that solves: "Given N items with weights and values, and a capacity limit, select items to maximize total value." In our case, items = detected events, weight = clip duration, value = detection confidence, capacity = 10-minute highlight budget. It guarantees the mathematically optimal selection. We chose it over greedy selection (which just picks top-N by confidence) because Knapsack considers the duration-value trade-off — a 30-second high-confidence goal is more valuable than a 2-minute medium-confidence substitution.

### Q6: Why 4 classes and not all 17 SoccerNet classes?
**Answer:** With only 60 matches, most of the 17 original SoccerNet classes have fewer than 10 examples each. Training on such sparse data would lead to severe overfitting. By merging into 4 classes (Goal, Cards, Substitution, Background), we ensure each class has hundreds of examples, making the model statistically robust.

---

## Category 2: Training & Optimization

### Q7: Explain the class imbalance problem and your solution.
**Answer:** In a 90-minute match at 2 FPS, there are ~10,800 frames. A typical match has 3-5 events, each spanning ~20 frames. So ~100 frames are events and ~10,700 are Background — a 99:1 imbalance. Without handling this, the model would predict "Background" for everything and achieve 99% accuracy while detecting zero events. Our solution: inverse frequency class weighting in CrossEntropyLoss. Background gets weight ~0.26, Goals get weight ~12.5. This makes the loss function penalize event misclassifications 50x more than background misclassifications.

### Q8: What is length regularization and why do you need it?
**Answer:** Even with class weights, the model might over-predict events to avoid missing any. Length regularization adds a penalty: `|mean(event_probability) - 0.15|`. This targets a 15% event rate. If the model predicts 50% of frames as events, this term grows large and pushes predictions back down. It's inspired by the CA-SUM (Combining Attentive Summarization) paper which uses a similar budget-aware regularization.

### Q9: Why Adam optimizer and not SGD?
**Answer:** Adam maintains per-parameter adaptive learning rates and momentum estimates. For our task with severe class imbalance and variable gradient magnitudes (rare events produce large gradients), Adam's adaptive rates prevent oscillation. SGD requires careful learning rate tuning and warm-up schedules. Adam works well out-of-the-box, which is important when training time is limited on Colab.

### Q10: Explain gradient clipping and why it's set to 5.0.
**Answer:** LSTMs are prone to exploding gradients — when the gradient through time accumulates multiplicatively and becomes extremely large, causing NaN losses and training failure. `clip_grad_norm_(params, 5.0)` rescales the entire gradient vector if its L2 norm exceeds 5.0. The value 5.0 is standard for LSTM training — small enough to prevent explosions, large enough to allow meaningful updates.

### Q11: What is ReduceLROnPlateau?
**Answer:** A learning rate scheduler that monitors validation mAP. If mAP doesn't improve for 5 consecutive epochs (patience=5), it halves the learning rate (factor=0.5). This implements a coarse-to-fine optimization strategy: large steps initially to find the right region, then small steps to fine-tune within that region. In our training, the LR typically drops from 5e-4 → 2.5e-4 → 1.25e-4 over 23 epochs.

### Q12: Why Xavier initialization for weights?
**Answer:** Xavier initialization sets weights from a distribution scaled by the number of input and output neurons: `W ~ Uniform(-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out)))`. This ensures that the variance of activations and gradients remains stable across layers, preventing vanishing or exploding gradients at initialization. For the LSTM, we use orthogonal initialization for recurrent weights, which preserves gradient norms through time steps.

---

## Category 3: Evaluation & Results

### Q13: What does 35.1% mAP actually mean?
**Answer:** mAP measures ranking quality. For each event class, we rank all frames by predicted probability and compute Average Precision (area under precision-recall curve). 35.1% mAP means our model correctly ranks about 1 in 3 event frames above most background frames. For context, SoccerNet SOTA is ~70% using 500 matches and massive compute. We achieve 50% of SOTA using 12% of the data and a single free GPU — a strong result demonstrating the approach is sound and data is the bottleneck.

### Q14: Why is test accuracy 96% but mAP only 35%?
**Answer:** Because of class imbalance. If 95% of frames are Background, predicting "Background" for everything gives 95% accuracy. High accuracy is misleading for rare-event detection. mAP directly measures event detection quality, ignoring the Background class entirely. 96% accuracy with 35% mAP means the model is excellent at Background but still misses some events — which is expected with limited training data.

### Q15: Why does Baseline have higher accuracy than Transformer but lower mAP?
**Answer:** The Baseline predicts Background more conservatively (fewer false alarms), boosting accuracy. But it also misses more events (lower recall), hurting mAP. The Transformer attempts to detect events but makes more classification errors, which lowers accuracy but keeps mAP similar. mAP is the true measure of event detection capability.

### Q16: What's in the confusion matrix?
**Answer:** Rows = true labels, columns = predicted labels. The diagonal shows correct predictions. Off-diagonal shows errors. Our BiLSTM's confusion matrix shows: Goals are sometimes confused with Cards (both involve sudden camera changes and crowd reactions). Background is predicted with 97%+ accuracy. This confusion is actually expected — the visual signatures of goals and cards share many features.

---

## Category 4: Data & Pipeline

### Q17: Why use the SoccerNet "test" split for training?
**Answer:** SoccerNet has 3 splits: train (300 matches), validation (100), test (100). The train split is too large for Colab's 112GB disk. The test split is the smallest, so we download it and re-split internally into 60/20/20 (train/val/test). This means we're not using SoccerNet's official splits, but our internal split is still valid for training and evaluation as long as there's no data leakage.

### Q18: What are the .npy files?
**Answer:** NumPy array files containing pre-extracted features. Each file stores a 2D array of shape `(num_frames, 512)` where each row is a 512-dimensional feature vector representing one video frame. These were extracted by the SoccerNet team using ResNet-152 + TF2 (TensorFlow 2) + PCA (reducing from 2048 to 512 dimensions). We use these directly instead of extracting features ourselves.

### Q19: Explain the sliding window approach.
**Answer:** We create a window of 30 consecutive frames (15 seconds). The model takes this window and predicts the class of the CENTER frame (frame 15). Then we slide the window forward by 15 frames (50% overlap) and predict again. This means each frame gets predictions from ~2 overlapping windows, which are averaged for smoother, more reliable output. This sliding window approach is standard in temporal action detection literature.

### Q20: How does the event detection work after model inference?
**Answer:** The model produces per-frame class probabilities. We threshold at 0.3 — any frame where a non-background class exceeds 30% is marked as an "event frame." We lowered this from the default 50% to improve recall on custom videos where the model may be slightly less confident. Consecutive event frames are grouped into one event. Each event gets ±30 seconds of padding (to include build-up and aftermath). Finally, the Knapsack algorithm selects the best combination under the time budget.

---

## Category 5: Domain Shift — Identified & Resolved

### Q21: Explain the domain shift problem and how you solved it.
**Answer:** Our BiLSTM was trained on feature vectors from ResNet-152 + PCA (provided by SoccerNet). In our initial version, we extracted features from custom videos using ResNet-18 (a smaller, different architecture). Although both produced 512-dimensional vectors, the feature distributions were fundamentally different — they encode visual information in incompatible ways. The BiLSTM received out-of-distribution inputs and couldn't reliably classify events. **We solved this** by upgrading the custom video pipeline to use the exact same ResNet-152 architecture + PCA reduction (2048→512), aligning the feature space with the training distribution.

### Q22: What other approaches could fix domain shift?
**Answer:** We chose the direct fix (using the same extractor), but two other approaches exist:
1. **Learned projection layer:** Train a small neural network to map ResNet-18 features into the ResNet-152+PCA feature space. Requires paired features from both extractors on the same frames.
2. **End-to-end fine-tuning:** Train CNN + LSTM jointly on raw video frames. Eliminates the feature extraction dependency entirely, but requires multi-GPU hardware.

### Q23: What would you do with unlimited compute?
**Answer:** (1) Train on all 500 SoccerNet matches. (2) Add audio features (crowd noise spikes correlate strongly with goals). (3) Implement multi-scale temporal windows (5s for cards, 30s for goal sequences). (4) Use a Transformer with pre-training on a large video corpus. (5) Add a YOLO+DeepSORT tracking pipeline for player-level statistics like speed, dribbles, and offsides.

### Q24: Why not use a pre-trained video model like I3D or SlowFast?
**Answer:** These models require RGB video frames as input and process them in 3D (spatial + temporal simultaneously). They need 32+ GB GPU RAM for training and are designed for short clips (2-10 seconds), not 45-minute halves. Our two-stage approach is more memory-efficient and compatible with the SoccerNet feature format.

---

## Category 6: Code-Specific Questions

### Q25: What does `model.eval()` do?
**Answer:** Switches the model from training mode to evaluation mode. This affects two things: (1) Dropout layers stop randomly zeroing activations — all neurons are active. (2) BatchNorm layers use running statistics instead of batch statistics. Without `model.eval()`, predictions would be non-deterministic and degraded.

### Q26: What does `@torch.no_grad()` do?
**Answer:** Disables gradient computation for the decorated function. During inference, we don't need gradients (no backpropagation). Disabling them reduces memory usage by ~50% and speeds up computation by ~20%.

### Q27: Explain the forget gate bias initialization.
**Answer:** (Lines 683-685) The LSTM forget gate decides what information to keep from the previous time step. By initializing its bias to 1.0 (sigmoid(1) ≈ 0.73), the LSTM starts by retaining most information. This prevents the "blank slate" problem where an untrained LSTM forgets everything and can't learn long-term dependencies. This technique was recommended in the original LSTM paper by Hochreiter and Schmidhuber.

### Q28: What is LayerNorm and why use it instead of BatchNorm?
**Answer:** LayerNorm normalizes across features within a single sample (independent of batch size). BatchNorm normalizes across the batch dimension. We use LayerNorm because: (1) It works the same during training and inference (no running statistics to worry about). (2) It's more stable with small batch sizes. (3) It's the standard choice for sequence models.

### Q29: Why `drop_last=True` in the training DataLoader?
**Answer:** If the last batch is smaller than batch_size (e.g., 7 samples instead of 32), BatchNorm statistics become unreliable, and the length regularization term becomes noisy. `drop_last=True` discards this incomplete batch. We only lose ~30 samples out of ~150,000 — negligible.

### Q30: What codec is used for the output video and why?
**Answer:** H.264 (`libx264`) with AAC audio. H.264 is the most universally supported video codec — playable on every device, browser, and media player. AAC is the standard audio companion. We suppress MoviePy's progress logger (`logger=None`) to keep the Colab output clean.

---

## Bonus: "Impressive" Talking Points

### If they ask "What would you do differently?"
> "Three things: First, I'd use the full 500-match SoccerNet dataset with a cloud GPU to break past the 35% mAP ceiling. Second, I'd implement a learned domain adaptation layer between the feature extractor and the temporal model, allowing the system to work with any CNN backbone without retraining. Third, I'd add an audio stream — crowd noise is one of the strongest signals for goal detection and is completely architecture-agnostic."

### If they ask "What's the most interesting thing you learned?"
> "That the Transformer is not always better than the LSTM. In the literature, Transformers dominate NLP benchmarks, but our experiments show that with limited data (60 matches), the LSTM's sequential inductive bias gives it a 4x mAP advantage. This taught me that architecture selection should be driven by data characteristics, not by popularity."

### If they ask "How is this different from commercial systems?"
> "Commercial systems like WSC Sports use proprietary datasets of 10,000+ matches, multi-modal inputs (video + audio + text overlays), and human-in-the-loop refinement. Our system demonstrates the same core algorithmic pipeline — CNN features → temporal modeling → event selection — at academic scale. The architecture is sound; the performance gap is purely a function of data and compute."
