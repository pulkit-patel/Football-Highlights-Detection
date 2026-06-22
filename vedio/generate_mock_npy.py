"""
⚽ Mock NPY Feature Generator
============================
Analyzes an MP4 video file and generates a matching mock `features.npy` file
(at 2 FPS, dimension 512) so you can test the Streamlit dashboard pipeline
instantly with any custom video!

Usage:
    python generate_mock_npy.py your_video.mp4
"""

import os
import sys
import numpy as np
import cv2

def generate_mock_features(video_path, output_npy_path="features.npy"):
    if not os.path.exists(video_path):
        print(f"❌ Error: Video file not found at '{video_path}'")
        sys.exit(1)
        
    print(f"🔍 Analyzing video: '{video_path}'...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Error: Could not open video file.")
        sys.exit(1)
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    duration_sec = total_frames / fps if fps > 0 else 0
    if duration_sec <= 0:
        print("❌ Error: Invalid video duration.")
        sys.exit(1)
        
    print(f"   - Video FPS: {fps:.2f}")
    print(f"   - Total Frames: {total_frames:,}")
    print(f"   - Duration: {duration_sec:.1f} seconds ({duration_sec/60:.2f} minutes)")
    
    # 2 FPS is the target rate of the Bi-LSTM model (1 frame every 0.5 seconds)
    target_fps = 2
    num_feature_frames = int(duration_sec * target_fps)
    
    print(f"⚡ Generating mock feature vectors of shape ({num_feature_frames}, 512)...")
    
    # Generate random features with normal distribution (512 dimensions)
    # We add some temporal structure so the Bi-LSTM is more likely to detect "mock events"
    features = np.random.normal(loc=0.0, scale=0.5, size=(num_feature_frames, 512))
    
    # Inject some mock high-energy events so detections occur
    # (e.g. simulate high confidence spikes)
    num_events = max(3, int(duration_sec / 180)) # 1 mock event every 3 minutes
    event_spacing = num_feature_frames // (num_events + 1)
    
    for i in range(1, num_events + 1):
        center = i * event_spacing
        # Create a spike of high values in a 10-frame window (5 seconds)
        start = max(0, center - 5)
        end = min(num_feature_frames, center + 5)
        # Shift distribution values to trigger non-background threshold (> 0.3)
        features[start:end, :] += np.random.uniform(0.5, 1.2, size=(end-start, 512))
        
    np.save(output_npy_path, features.astype(np.float32))
    print(f"💾 Success! Mock feature file saved as '{output_npy_path}' ({os.path.getsize(output_npy_path)/(1024*1024):.2f} MB)")
    print(f"\n🚀 Ready! Upload '{output_npy_path}' and '{video_path}' in the Streamlit UI to run the pipeline.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("❌ Error: Please specify the path to your video.")
        print("Usage: python generate_mock_npy.py <path_to_video.mp4>")
        
        # Look for the final_highlights.mp4 inside Main Files to provide as a default helper
        default_video = os.path.join("Main Files", "final_highlights.mp4")
        if os.path.exists(default_video):
            print(f"\n💡 Found a local video at '{default_video}'. Running on default...")
            generate_mock_features(default_video, "final_highlights_features.npy")
        else:
            sys.exit(1)
    else:
        generate_mock_features(sys.argv[1])
