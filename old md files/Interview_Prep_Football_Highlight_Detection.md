# ⚽ Comprehensive Deep Dive: AI-Based Football Highlight Detection System

---

## 1. Introduction and Problem Statement

The automated summarization of sports events is a highly challenging problem in computer vision and temporal sequence modeling. A standard football match lasts over 90 minutes and contains roughly 135,000 to 162,000 frames (at 25-30 FPS). Manual annotation and highlight extraction are tedious, labor-intensive tasks.

The core objective of this system is to ingest a full-length match video and output a condensed, action-packed highlight reel containing the most crucial events: **Goals, Cards (Yellow/Red), and Substitutions**, while adhering to a strict user-defined time budget (e.g., 10 minutes maximum). 

### Why is Video Understanding Difficult?
Unlike image classification, video understanding requires capturing both **spatial semantics** (what objects are in the frame—players, ball, goalposts) and **temporal dynamics** (how those objects move over time—a ball crossing the line, a referee pulling out a card, a player celebrating). Furthermore, key events in football are extremely sparse; over 95% of the video consists of "background" play, creating a severe class imbalance problem for any machine learning algorithm.

---

## 2. High-Level System Architecture

To solve this, the project employs a **Two-Stage Deep Learning Pipeline**. 

Why not an end-to-end model (feeding raw video directly to an LSTM/Transformer)? End-to-end training requires maintaining the computational graph for both the high-dimensional spatial CNN and the recurrent temporal model simultaneously. Unrolling a ResNet-152 and an LSTM for 30 frames would require upwards of 40 GB of VRAM, making it impossible to train on consumer hardware (like the 15GB Colab T4 GPU). 

Thus, the pipeline is divided into:
1. **Spatial Feature Extraction (Offline/Pre-processed):** Transforming raw pixels into compressed, meaningful semantic vectors.
2. **Temporal Modeling (Online/Trained):** Analyzing sequences of these vectors to spot patterns over time.
3. **Heuristic Selection & Video Stitching:** Post-processing probabilities and mathematically optimizing clip selection.

```text
[Raw Match Video MP4] 
        │
        ▼
[ResNet-152 CNN] ──────► Extract 2048-dim spatial features (Frame-by-frame at 2 FPS)
        │
        ▼
[PCA Reduction] ───────► Compress to 512-dim vectors 
        │
        ▼
[Sliding Window] ──────► Group into 30-frame (15-second) sequences
        │
        ▼
[Bidirectional LSTM] ──► Output event probabilities per frame
        │
        ▼
[Event Grouping] ──────► Combine consecutive predictions & add context padding
        │
        ▼
[0/1 Knapsack DP] ─────► Mathematically optimize selection under time budget
        │
        ▼
[MoviePy Stitcher] ────► Final MP4 Highlight Reel
```

---

## 3. Stage 1: Spatial Feature Extraction & Dimensionality Reduction

### 3.1. The Pre-trained ResNet-152 CNN
To understand each frame, the system passes the imagery through a **ResNet-152 (Residual Network)**. 
- **The Vanishing Gradient Problem:** Traditional deep networks suffer from vanishing gradients—as errors backpropagate through many layers, they become infinitesimally small, halting learning in early layers.
- **Residual Connections:** ResNet solves this by introducing "skip connections." Instead of a layer learning a direct mapping $H(x)$, it learns a residual function $F(x) = H(x) - x$. The output becomes $F(x) + x$. This allows gradients to bypass activation functions and flow uninhibited through the network.
- **Why ResNet-152?** It has 152 layers, enabling the extraction of highly complex, hierarchical features. Early layers detect edges and grass textures; deeper layers detect players, formations, and goalposts. The final Global Average Pooling layer outputs a dense, **2048-dimensional semantic feature vector** representing the entire frame.

### 3.2. Principal Component Analysis (PCA)
A 2048-dimensional vector per frame is still computationally expensive for the downstream sequence models. We apply **PCA** to reduce this to **512 dimensions**.
- **Mathematical Intuition:** PCA is an orthogonal linear transformation. It computes the covariance matrix of the data and calculates its eigenvalues and eigenvectors. The eigenvectors with the highest eigenvalues represent the directions of maximum variance in the data.
- By projecting the 2048D vectors onto the top 512 eigenvectors, we retain over 90% of the variance/information while discarding redundant noise. This allows our LSTM to train faster and prevents overfitting on a small dataset.

### 3.3. The Domain Shift Problem & Resolution
**Crucial Interview Concept:** Initially, the system used a lightweight ResNet-18 to process custom user videos, while the underlying model was trained on the SoccerNet dataset (which utilized ResNet-152). 
Even though both output a 512D vector (after PCA), the internal representations—the "language" the CNN speaks—were fundamentally different. The BiLSTM, expecting ResNet-152 language, failed when given ResNet-18 language. This is known as **Domain Shift**. 
**Resolution:** The custom extraction pipeline was strictly upgraded to match the ResNet-152 + PCA architecture, aligning the feature distributions and completely resolving the domain shift issue.

---

## 4. Stage 2: Temporal Modeling (The Core Brain)

The core challenge is evaluating a sequence of frames. A single frame of a ball in the net doesn't explain how it got there. We process the video at **2 FPS** and use a **Sliding Window of 30 frames** (15 seconds of context), with a stride of 15 frames (50% overlap). The model predicts the class of the center frame (frame 15).

### 4.1. Why BiLSTM? (Bidirectional Long Short-Term Memory)
LSTMs are a specialized type of Recurrent Neural Network (RNN) designed to combat the vanishing gradient problem in sequences by maintaining an internal "Cell State" ($C_t$), acting as an information highway through time.

An LSTM has three gates (using sigmoid activations to output values between 0 and 1):
1. **Forget Gate:** Decides what information from the past to throw away. 
2. **Input Gate:** Decides what new information to store in the cell state.
3. **Output Gate:** Decides what to output based on the filtered cell state.

**The Bidirectional Advantage:**
A standard LSTM reads left-to-right (past $\rightarrow$ future). However, in sports, the *future* explains the present. If the network looks at a frame and sees the ball near the goal, it might be unsure if a goal was scored. But if it reads the sequence *backwards* from frame 30 to 1, and sees the crowd erupting and the players celebrating in frame 25, it passes that information "back" to frame 15, confirming the goal.
The BiLSTM runs two independent LSTMs (forward and backward) and concatenates their hidden states ($[h_{forward}, h_{backward}]$), providing deep context from both temporal directions.

**LSTM Forget Gate Initialization Trick:**
In the codebase, the bias of the forget gate is explicitly initialized to `1.0`. 
Since $\sigma(1.0) \approx 0.73$, this ensures the forget gate is mostly "open" at the start of training. If initialized at 0 ($\sigma(0) = 0.5$), the LSTM aggressively forgets early on, struggling to learn long-term dependencies. This simple trick massively improves early training stability.

### 4.2. Transformer Encoder (Alternative Architecture)
A Transformer was also implemented and tested.
- **Self-Attention:** Instead of reading sequentially, a Transformer views all 30 frames at once. For every frame (Query), it checks every other frame (Keys) to calculate an attention score, dynamically aggregating information (Values) from the most relevant frames.
- **Positional Encoding:** Because it reads everything at once, it loses the concept of time. Sinusoidal positional encodings (using sine and cosine waves of varying frequencies) are added to the input vectors to explicitly inject time-step information.
- **Why did it fail to beat the BiLSTM?** Transformers lack **inductive bias**. LSTMs mathematically assume data is sequential, making them highly efficient on small datasets. Transformers assume nothing and must *learn* sequential patterns from scratch. With only 60 matches (150K sequences), the dataset was far too small for the Transformer to beat the LSTM's innate sequential efficiency.

### 4.3. Baseline CNN (Ablation Study)
A simple Multi-Layer Perceptron (MLP) was implemented to look at *only* the center frame, completely ignoring the other 29 frames. This baseline performed poorly, proving conclusively that temporal context is strictly necessary for video understanding.

---

## 5. Training, Optimization, and Regularization Mechanisms

### 5.1. Handling Extreme Class Imbalance
In football, events are sparse. The dataset has four classes: Goal, Cards, Substitution, and Background.
Over 95% of the frames are Background. If an algorithm simply guesses "Background" 100% of the time, it achieves 95% accuracy while doing absolutely nothing useful.

**Solution: Inverse Frequency Weighting:**
The `CrossEntropyLoss` function is modified. The weight of each class is set to `total_samples / (num_classes * class_samples)`.
Background gets a tiny weight (e.g., 0.26), while Goal gets a massive weight (e.g., 12.5). Mathematically, the loss gradient for a misclassified Goal is multiplied by 12.5, punishing the model severely for missing events and forcing it to pay attention to the minority classes.

### 5.2. Length Regularization (Budget-Aware Penalty)
Because of the heavy class weights, the model might become paranoid and start predicting "Goal" everywhere just to be safe. To prevent this, a custom **Length Regularization** penalty is added to the loss:
$$L_{reg} = \lambda \times |\text{mean}(1 - P(\text{Background})) - 0.15|$$
This forces the network to target a global average of 15% event frames across the dataset, suppressing false alarms and maintaining a realistic highlight frequency.

### 5.3. Adam Optimizer vs. Stochastic Gradient Descent (SGD)
- **SGD** updates weights using a single, fixed learning rate. On highly imbalanced datasets, gradients fluctuate wildly, causing SGD to oscillate and fail to converge.
- **Adam (Adaptive Moment Estimation)** maintains a per-parameter learning rate by tracking both the moving average of the gradients (First Moment / Momentum) and the moving average of the squared gradients (Second Moment / RMSProp). Adam smooths out erratic gradient spikes effortlessly, which is vital for our sparse event data.

### 5.4. Gradient Clipping
LSTMs suffer from **Exploding Gradients**, where recurrent multiplications over 30 time steps cause gradients to surge to infinity (`NaN`), destroying the network weights.
The codebase utilizes `torch.nn.utils.clip_grad_norm_`, which measures the L2 norm of the entire gradient vector. If it exceeds a threshold of `5.0`, the vector is mathematically scaled down to exactly 5.0, preserving the direction of the gradient while limiting its magnitude.

### 5.5. Layer Normalization vs. Batch Normalization
- **BatchNorm** standardizes features across the batch dimension. With small batch sizes (32) and temporal sequence data, the batch statistics are highly unstable and introduce massive noise.
- **LayerNorm** standardizes features across the *feature dimension* for a single sample independently. It is mathematically independent of batch size, making it the superior, stable choice for RNNs and Transformers.

### 5.6. Early Stopping & ReduceLROnPlateau
The model is trained for 80 epochs but usually peaks around epoch 20. 
- **ReduceLROnPlateau:** Monitors the validation metric (mAP). If it fails to improve for 5 epochs, the learning rate is halved. This allows the model to take large steps early on, and tiny, fine-tuning steps when it nears the local minima.
- **Early Stopping:** If validation mAP fails to improve for 15 epochs, training is abruptly halted, and the checkpoint from the absolute best epoch is restored. This rigorously prevents overfitting (memorizing the training set at the expense of generalization).

---

## 6. Post-Processing and Output Generation

### 6.1. Event Grouping and Padding
The BiLSTM outputs frame-by-frame probabilities. 
1. **Thresholding:** If the probability of any non-background class exceeds `0.3` (30%), the frame is flagged as an event.
2. **Grouping:** Consecutive event frames are merged into a single contiguous block (e.g., frames 150 to 180 become a single "Goal" event).
3. **Padding:** A highlight clip cannot start exactly when the ball crosses the line; context is required. The algorithm injects a $\pm 30$ seconds padding window around the event center, ensuring the build-up play and the resulting celebration are included in the clip.

### 6.2. The 0/1 Knapsack Algorithm (Mathematical Clip Selection)
Once all events are identified and padded, we might have 15 minutes of clips, but the user requested a strict 10-minute highlight reel. How do we choose the best clips?
A greedy approach (just picking the highest confidence clips) fails. It might select a massive 4-minute clip with 90% confidence, forcing us to drop three 1-minute clips with 89% confidence, resulting in a suboptimal reel.

**Dynamic Programming formulation:**
- **Items:** $N$ detected events.
- **Weight ($w_i$):** Duration of the clip.
- **Value ($v_i$):** Model confidence (probability) of the clip.
- **Capacity ($W$):** 10 minutes (1200 frames).

We build a 2D array $K$ where $K[i][w]$ represents the maximum value obtainable using the first $i$ items with a capacity of $w$.
$$K[i][w] = \max \left( K[i-1][w], \, v_i + K[i-1][w - w_i] \right)$$
By backtracking through this matrix, the system extracts the mathematically optimal set of highlights that maximizes total "excitement" while perfectly respecting the user's time boundary.

---

## 7. Metrics & Evaluation: Mean Average Precision (mAP)

Why is the primary metric **mAP** and not Accuracy?
As discussed, predicting "Background" for every frame yields 95% accuracy but zero utility. Accuracy measures discrete correctness, which is flawed for imbalanced datasets.

**Average Precision (AP):**
For a specific class (e.g., Goal), we rank all predictions across the entire match by their confidence score. We calculate the Precision and Recall at every possible threshold. AP is the area under the Precision-Recall (PR) curve. It rewards the model heavily for placing true Goals at the very top of the ranking list.

**Mean Average Precision (mAP):**
The mean of the APs across all non-background classes.
Achieving **35.1% mAP** on just 60 matches (12% of the dataset) using a single GPU is a massive accomplishment, proving the theoretical soundness of the BiLSTM pipeline.

---

## 8. Comprehensive Interview Q&A Bank

**Q1: Explain the end-to-end data flow of your project in under 60 seconds.**
*A:* A raw match video is processed frame-by-frame through a ResNet-152 CNN, which outputs 2048-dim spatial features. PCA compresses these to 512 dims. A sliding window groups these into 15-second blocks. These blocks are fed into a Bidirectional LSTM which analyzes the sequence forward and backward to output frame-level probabilities for Goals, Cards, or Substitutions. The output probabilities are smoothed, thresholded, padded with $\pm 30$s of context, and finally fed into a 0/1 Knapsack dynamic programming algorithm to optimally select the most confident clips that fit within a 10-minute user budget. MoviePy then stitches these clips into an MP4.

**Q2: Why did you decouple the CNN and the RNN? Why not train them jointly?**
*A:* Memory constraints. Training ResNet and LSTM jointly (unrolling 30 frames per batch) requires computing gradients for billions of intermediate activations, taking over 40GB of VRAM. By pre-extracting the spatial features, I turned the video problem into a dense-vector sequence problem, which trains incredibly fast on a standard 15GB GPU without sacrificing accuracy.

**Q3: What is "Domain Shift," and how did you encounter it?**
*A:* Domain shift happens when the data a model sees in production is statistically different from its training data. My BiLSTM was trained on SoccerNet features extracted via ResNet-152 + PCA. Initially, for custom video inference, I used a faster ResNet-18 model. Because the feature representations (the "language" of the vectors) were entirely different, the BiLSTM failed to detect events. I fixed this by upgrading the custom inference pipeline to exactly mirror the training pipeline (ResNet-152 + PCA), bringing the feature distributions back into alignment.

**Q4: Explain why BiLSTM is inherently superior to a standard LSTM for video action spotting.**
*A:* A standard LSTM only knows the past. If you see a ball flying through the air, you don't know if it's a goal or a miss. A BiLSTM reads the sequence from the future backwards as well. If the backward LSTM sees players celebrating 3 seconds later, it passes that contextual knowledge back to the frame with the flying ball, allowing the network to confidently classify it as a goal.

**Q5: Why initialize the LSTM forget gate bias to 1.0?**
*A:* If the bias is 0, the sigmoid output is 0.5. The LSTM is already forgetting 50% of its history at every step before it has even learned anything, leading to vanishing gradients early in training. Initializing to 1.0 pushes the sigmoid to ~0.73, encouraging the network to preserve gradients and remember history from epoch 1.

**Q6: Why did your Transformer model underperform the BiLSTM?**
*A:* Transformers lack inductive bias—they don't inherently assume data is sequential. They require massive datasets to learn those relationships via attention. With only 60 matches, the dataset was too small. The BiLSTM, by its very architecture, assumes data flows through time, giving it a massive head start on small sequential datasets.

**Q7: How did you solve the fact that 95% of your dataset is background noise?**
*A:* I used inverse frequency weighting in the Cross-Entropy loss. Since Background occurs 50x more often than Goals, the loss function multiplies the penalty of missing a Goal by 50x. This forces the gradients to optimize for the rare events rather than taking the lazy route of guessing "Background" everywhere.

**Q8: What is Length Regularization?**
*A:* Because of the heavy class weights, the model might over-predict goals. I added a penalty term to the loss function that calculates the absolute difference between the mean predicted event probability and 0.15. This budget-aware regularizer forces the model to target a 15% overall event rate across the match, keeping predictions conservative and realistic.

**Q9: Why use the 0/1 Knapsack instead of just picking the highest confidence clips?**
*A:* Greedy selection ignores clip duration. A 3-minute clip with 90% confidence might take up too much budget, forcing me to drop four 1-minute clips with 89% confidence. The Knapsack algorithm mathematically evaluates all combinations to find the exact subset of clips that maximizes total confidence without exceeding the exact 10-minute limit.

**Q10: Why LayerNorm instead of BatchNorm?**
*A:* BatchNorm standardizes across the batch size. In sequential models with small batch sizes, this variance is highly erratic. LayerNorm standardizes across the features of a single sample, making it completely independent of batch size and mathematically stable for RNNs.

**Q11: What would you do with a cluster of A100 GPUs and unlimited time?**
*A:* 
1. Train on the full 500-match dataset.
2. Implement an Audio CNN stream. Crowd noise decibels are heavily correlated with goals. Fusing video features with audio embeddings would drastically boost mAP.
3. Replace the BiLSTM with a large-scale Video Transformer pre-trained on Kinetics-400.
4. Implement multi-scale temporal modeling (e.g., a 5-second window for detecting rapid yellow cards, and a 45-second window for detecting long, tactical goal build-ups).
