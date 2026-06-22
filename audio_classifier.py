"""
🔊 Audio Event Classifier for Football Matches
==============================================
Extracts the audio track from football match videos, segments the audio,
computes Mel-spectrograms, and classifies events (Cheers, Whistles, Silence/Background).
Includes both a PyTorch 2D CNN model and a robust spectral-power heuristic fallback.
"""

import os
import numpy as np
import torch
import torch.nn as nn
try:
    from moviepy import VideoFileClip
except ImportError:
    try:
        from moviepy.editor import VideoFileClip
    except ImportError:
        VideoFileClip = None

# Import librosa and scipy safely
try:
    import librosa
except ImportError:
    librosa = None

try:
    import scipy.io.wavfile as wavfile
    from scipy.signal import welch
except ImportError:
    wavfile = None
    welch = None


# ============================================================
# 1. 2D CNN ARCHITECTURE DEFINITION
# ============================================================
class AudioEventCNN(nn.Module):
    """
    Lightweight 2D CNN for classifying 0.5-second Mel-Spectrograms.
    Input shape: (batch, 1, 128, 44) -> 128 mel bands, 44 time steps.
    """
    def __init__(self, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        # Downsampling: 128x44 -> pool1 -> 64x22 -> pool2 -> 32x11 -> pool3 -> 16x5
        self.fc1 = nn.Linear(64 * 16 * 5, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# ============================================================
# 2. AUDIO EXTRACTION & PREPROCESSING
# ============================================================
def extract_audio(video_path, output_wav_path):
    """Extracts mono audio WAV track from the video at 22050 Hz."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Source video not found: {video_path}")
        
    try:
        video = VideoFileClip(video_path)
        if video.audio is None:
            return False  # Video has no audio track
            
        # Extracting audio
        # Write mono, 22.05kHz audio
        video.audio.write_audiofile(
            output_wav_path,
            fps=22050,
            nbytes=2,
            buffersize=2000,
            codec="pcm_s16le",
            ffmpeg_params=["-ac", "1"],
            logger=None
        )
        video.close()
        return True
    except Exception as e:
        pass  # Failed to extract audio
        return False


def get_mel_spectrogram(y, sr, duration_sec=0.5):
    """Computes a Mel-spectrogram for a given audio signal block."""
    if librosa is None:
        raise ImportError("librosa is required to compute Mel-spectrograms.")
        
    # target frames for 0.5s with default hop_length=512 is 22050*0.5 / 512 = 22 frames
    # Let's customize n_fft and hop_length to get a clean (128, 44) shape
    # hop_length = 256 -> 22050*0.5 / 256 = 43.06 (~44 frames)
    n_mels = 128
    hop_length = 256
    n_fft = 1024
    
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
    )
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Ensure correct padding or cropping to (128, 44)
    target_width = 44
    if mel_db.shape[1] < target_width:
        pad_width = target_width - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad_width)), mode='constant', constant_values=-80.0)
    elif mel_db.shape[1] > target_width:
        mel_db = mel_db[:, :target_width]
        
    return mel_db


# ============================================================
# 3. SPECTRAL HEURISTIC FALLBACK (For immediate functionality)
# ============================================================
def heuristic_classify_audio(y, sr, chunk_size, hop_size):
    """
    Robust spectral analyzer that detects audio events by checking:
    1. Overall energy (Silence/Background vs Event).
    2. High-frequency whistle band (2000Hz - 4000Hz).
    3. Low-mid crowd cheer band (150Hz - 900Hz).
    """
    if wavfile is None or welch is None:
        # Simple amplitude threshold fallback
        pass  # Scipy not available, using simple RMS thresholding
        events = []
        for i in range(0, len(y) - chunk_size, hop_size):
            chunk = y[i:i+chunk_size]
            rms = np.sqrt(np.mean(chunk**2))
            t = (i + chunk_size/2) / sr
            if rms > 0.15:
                events.append({'time': t, 'class': 'Event', 'confidence': float(min(1.0, rms * 3))})
        return events

    events = []
    num_chunks = (len(y) - chunk_size) // hop_size + 1
    
    for idx in range(num_chunks):
        start_idx = idx * hop_size
        end_idx = start_idx + chunk_size
        chunk = y[start_idx:end_idx]
        
        # Time of the center of this chunk
        time_sec = (start_idx + chunk_size / 2) / sr
        
        # Calculate RMS amplitude
        rms = np.sqrt(np.mean(chunk**2))
        if rms < 0.005:  # Absolute silence
            continue
            
        # Frequency spectrum analysis
        freqs, psd = welch(chunk, sr, nperseg=min(256, len(chunk)))
        
        # Define bands
        whistle_band = (freqs >= 2000) & (freqs <= 4000)
        cheer_band = (freqs >= 150) & (freqs <= 900)
        all_freqs = (freqs >= 100) & (freqs <= 8000)
        
        total_power = np.sum(psd[all_freqs]) + 1e-10
        whistle_power = np.sum(psd[whistle_band])
        cheer_power = np.sum(psd[cheer_band])
        
        # Ratios
        whistle_ratio = whistle_power / total_power
        cheer_ratio = cheer_power / total_power
        
        # Event Decision Heuristic
        # Whistles have concentrated power in the 2-4kHz band
        if whistle_ratio > 0.4 and rms > 0.02:
            events.append({
                'time_seconds': time_sec,
                'class': 'Referee Whistle',
                'confidence': float(min(0.99, whistle_ratio + 0.3)),
                'rms': float(rms)
            })
        # Cheers have high energy in the lower mid-range and relatively higher RMS
        elif cheer_ratio > 0.55 and rms > 0.04:
            events.append({
                'time_seconds': time_sec,
                'class': 'Crowd Cheer',
                'confidence': float(min(0.95, cheer_ratio * 1.2)),
                'rms': float(rms)
            })
            
    return events


# ============================================================
# 4. MAIN INFERENCE FUNCTION
# ============================================================
def classify_audio_events(video_path, checkpoint_path=None, temp_dir="./data"):
    """
    Primary interface to run audio event extraction and classification.
    Automatically handles CNN model execution or falls back to Heuristic Spectral Classifier.
    """
    os.makedirs(temp_dir, exist_ok=True)
    wav_path = os.path.join(temp_dir, "temp_extracted_audio.wav")
    
    # Step 1: Extract Audio
    success = extract_audio(video_path, wav_path)
    if not success:
        return []
        
    try:
        # Load audio using librosa or fallback to wavfile
        if librosa:
            y, sr = librosa.load(wav_path, sr=22050, mono=True)
        else:
            sr, y = wavfile.read(wav_path)
            y = y.astype(np.float32) / 32768.0  # Normalize int16 to float32
            if len(y.shape) > 1:
                y = np.mean(y, axis=1)  # Convert to stereo average
                
        chunk_duration = 0.5  # 0.5s chunks
        chunk_size = int(sr * chunk_duration)
        hop_size = int(sr * 0.25)  # 50% overlap (0.25s stride)
        
        # Check if we should use CNN or Fallback
        use_cnn = False
        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = AudioEventCNN()
                model.load_state_dict(torch.load(checkpoint_path, map_location=device))
                model.to(device)
                model.eval()
                use_cnn = True
                pass  # Loaded AudioEventCNN model checkpoint
            except Exception as ex:
                pass  # Failed to load CNN checkpoint, using heuristic fallback
                
        if use_cnn and librosa:
            events = []
            num_chunks = (len(y) - chunk_size) // hop_size + 1
            classes = ['Crowd Cheer', 'Referee Whistle', 'Background']
            
            with torch.no_grad():
                for idx in range(num_chunks):
                    start_idx = idx * hop_size
                    end_idx = start_idx + chunk_size
                    chunk = y[start_idx:end_idx]
                    
                    time_sec = (start_idx + chunk_size / 2) / sr
                    
                    # Compute spectrogram
                    mel_spec = get_mel_spectrogram(chunk, sr)
                    # Normalize Mel representation
                    mel_tensor = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                    
                    logits = model(mel_tensor)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                    
                    pred_class_idx = np.argmax(probs)
                    pred_class = classes[pred_class_idx]
                    confidence = float(probs[pred_class_idx])
                    
                    if pred_class != 'Background' and confidence > 0.65:
                        events.append({
                            'time_seconds': time_sec,
                            'class': pred_class,
                            'confidence': confidence,
                            'rms': float(np.sqrt(np.mean(chunk**2)))
                        })
            return events
        else:
            pass  # Running Spectral Heuristic Audio Classifier
            return heuristic_classify_audio(y, sr, chunk_size, hop_size)
            
    except Exception as e:
        pass  # Error during audio processing
        return []
    finally:
        # Cleanup temporary audio file
        if os.path.exists(wav_path):
            try:
                os.remove(wav_path)
            except Exception:
                pass


# ============================================================
# 5. AUDIO MODEL TRAINING SCRIPT TEMPLATE
# ============================================================
def train_audio_classifier(train_loader, val_loader, epochs=10, lr=0.001, save_path="checkpoints/audio_cnn.pth"):
    """
    Template for training the AudioEventCNN model.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AudioEventCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_loss = float('inf')
    
    pass  # Training Audio CNN
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        acc = correct / total
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {acc:.2%}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            pass  # Saved best checkpoint
            
    pass  # Audio CNN model training completed
