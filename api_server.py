"""
🏟️ Flask API Backend for Football Highlight Detection
======================================================
Replaces Streamlit with clean REST endpoints.
All existing pipeline modules (audio_classifier, scoreboard_tracker,
clip_search, chatbot_agent, match_summarizer) are reused unchanged.

Usage:
    pip install flask
    python api_server.py
    → Open http://localhost:5000
"""

import os
import sys
import io
import uuid
import json
import tempfile
import threading
import numpy as np
import torch
import torch.nn as nn
import cv2
from flask import Flask, request, jsonify, send_from_directory, send_file, Response

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from audio_classifier import classify_audio_events
from scoreboard_tracker import track_scoreboard_in_video
from clip_search import index_video_frames, search_video
from chatbot_agent import FootballRAGAgent
from match_summarizer import generate_llm_summary


# ==============================================================
# 1. MODEL DEFINITION (CNN + BiLSTM — same as app.py)
# ==============================================================
class BiLSTMClassifier(nn.Module):
    """CNN features → Bidirectional LSTM → Event Classification."""
    def __init__(self, feature_dim=512, hidden_dim=256, num_layers=2,
                 num_classes=4, dropout=0.4):
        super().__init__()
        self.name = "CNN_BiLSTM"
        self.lstm = nn.LSTM(
            input_size=feature_dim, hidden_size=hidden_dim,
            num_layers=num_layers, batch_first=True, bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        lstm_output_dim = hidden_dim * 2
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.LayerNorm(256), nn.Linear(256, 128), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(128, num_classes)
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        center_idx = x.shape[1] // 2
        return self.classifier(lstm_out[:, center_idx, :])


# ==============================================================
# 2. PIPELINE UTILITIES (copied from app.py)
# ==============================================================
def knapsack_highlight_selection(events, max_duration_frames):
    if not events:
        return []
    n = len(events)
    weights = [e['end'] - e['start'] for e in events]
    values = [e['confidence'] for e in events]
    W = max_duration_frames
    K = [[0.0] * (W + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(W + 1):
            if weights[i-1] <= w:
                K[i][w] = max(values[i-1] + K[i-1][w - weights[i-1]], K[i-1][w])
            else:
                K[i][w] = K[i-1][w]
    selected = []
    w = W
    for i in range(n, 0, -1):
        if K[i][w] != K[i-1][w]:
            selected.insert(0, i - 1)
            w -= weights[i - 1]
    return [events[i] for i in selected]


@torch.no_grad()
def generate_highlights(model, features_path, threshold, max_highlight_minutes,
                        seq_len=30, num_classes=4, target_fps=2, progress_cb=None):
    model.eval()
    device = next(model.parameters()).device
    features = np.load(features_path)
    num_frames = features.shape[0]
    all_probs = np.zeros((num_frames, num_classes))
    frame_counts = np.zeros(num_frames)
    total_windows = max(1, num_frames - seq_len + 1)

    for i, start in enumerate(range(0, num_frames - seq_len + 1)):
        seq = torch.tensor(features[start:start + seq_len], dtype=torch.float32).unsqueeze(0).to(device)
        logits = model(seq)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        center = start + seq_len // 2
        all_probs[center] += probs
        frame_counts[center] += 1
        if progress_cb and i % 200 == 0:
            progress_cb(i / total_windows)

    valid = frame_counts > 0
    all_probs[valid] /= frame_counts[valid, np.newaxis]
    bg_class = num_classes - 1
    event_mask = all_probs[:, :bg_class].max(axis=1) > threshold
    events = []
    in_event = False
    event_start = 0
    classes_map = ['Goal', 'Cards', 'Substitution']

    for i in range(num_frames):
        if event_mask[i] and not in_event:
            event_start = i; in_event = True
        elif not event_mask[i] and in_event:
            confidence = float(all_probs[event_start:i, :bg_class].max(axis=1).mean())
            mean_probs = all_probs[event_start:i, :bg_class].mean(axis=0)
            class_idx = int(np.argmax(mean_probs))
            pad = 30 * target_fps
            t_sec = event_start / target_fps
            events.append({
                'start': max(0, event_start - pad), 'end': min(num_frames, i + pad),
                'event_frame': (event_start + i) // 2, 'confidence': confidence,
                'time_seconds': t_sec, 'class': classes_map[class_idx],
                'time_str': f"{int(t_sec // 60)}:{int(t_sec % 60):02d}",
                'half': 1 if t_sec < 2700 else 2
            })
            in_event = False

    if in_event:
        i = num_frames
        confidence = float(all_probs[event_start:i, :bg_class].max(axis=1).mean())
        mean_probs = all_probs[event_start:i, :bg_class].mean(axis=0)
        class_idx = int(np.argmax(mean_probs))
        pad = 30 * target_fps; t_sec = event_start / target_fps
        events.append({
            'start': max(0, event_start - pad), 'end': min(num_frames, i + pad),
            'event_frame': (event_start + i) // 2, 'confidence': confidence,
            'time_seconds': t_sec, 'class': classes_map[class_idx],
            'time_str': f"{int(t_sec // 60)}:{int(t_sec % 60):02d}",
            'half': 1 if t_sec < 2700 else 2
        })

    max_frames = max_highlight_minutes * 60 * target_fps
    selected = knapsack_highlight_selection(events, max_frames)
    selected.sort(key=lambda e: e['start'])
    return selected, events


def stitch_highlight_video(video_path, events, output_path, target_fps=2, progress_cb=None):
    try:
        from moviepy.editor import VideoFileClip, concatenate_videoclips
    except ImportError:
        from moviepy import VideoFileClip, concatenate_videoclips
    video = VideoFileClip(video_path)
    clips = []
    for i, evt in enumerate(events):
        start_sec = max(0, evt['start'] / target_fps)
        end_sec = min(video.duration, evt['end'] / target_fps)
        if end_sec <= start_sec:
            continue
        clip = video.subclip(start_sec, end_sec) if hasattr(video, 'subclip') else video.subclipped(start_sec, end_sec)
        clips.append(clip)
        if progress_cb:
            progress_cb((i + 1) / len(events))
    if not clips:
        video.close()
        return False
    final = concatenate_videoclips(clips)
    final.write_videofile(output_path, codec="libx264", audio_codec="aac", logger=None)
    video.close(); final.close()
    return True


# ==============================================================
# 3. FLASK APP
# ==============================================================
app = Flask(__name__, static_folder='frontend', static_url_path='')
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB

# CORS headers
@app.before_request
def handle_options_preflight():
    if request.method == 'OPTIONS':
        res = Response()
        res.headers['Access-Control-Allow-Origin'] = '*'
        res.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        res.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        return res

@app.after_request
def add_cors(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
    return response

# Global state
model = None
model_info = {'loaded': False, 'best_map': 'N/A', 'epoch': 'N/A'}
pipeline_results = {}   # task_id → results dict
pipeline_progress = {}  # task_id → progress dict
rag_agent = None

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def _save_results(tid, res):
    # Save NumPy embeddings separately
    emb = res.get('_clip_emb')
    if emb is not None:
        np.save(os.path.join(RESULTS_DIR, f'clip_emb_{tid}.npy'), emb)
    
    # Save the JSON-serializable keys
    serializable = {
        'selected_events': res.get('selected_events'),
        'all_events': res.get('all_events'),
        'audio_timeline': res.get('audio_timeline'),
        'ocr_timeline': res.get('ocr_timeline'),
        'clip_timestamps': res.get('clip_timestamps'),
        'has_clip': res.get('has_clip'),
        'video_duration_min': res.get('video_duration_min'),
        'summary': res.get('summary'),
        'has_video': res.get('has_video'),
        'video_filename': res.get('video_filename'),
        '_video_path': res.get('_video_path')
    }
    
    json_path = os.path.join(RESULTS_DIR, f'pipeline_results_{tid}.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2)


def _load_saved_results():
    global pipeline_results, pipeline_progress
    if not os.path.exists(RESULTS_DIR):
        return
    for fn in os.listdir(RESULTS_DIR):
        if fn.startswith('pipeline_results_') and fn.endswith('.json'):
            tid = fn[len('pipeline_results_'):-len('.json')]
            json_path = os.path.join(RESULTS_DIR, fn)
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    res = json.load(f)
                
                # Check for clip embeddings
                emb_path = os.path.join(RESULTS_DIR, f'clip_emb_{tid}.npy')
                if os.path.exists(emb_path):
                    res['_clip_emb'] = np.load(emb_path)
                else:
                    res['_clip_emb'] = None
                
                pipeline_results[tid] = res
                # Also set progress as complete for this loaded task
                pipeline_progress[tid] = {'progress': 1.0, 'step': 'Pipeline complete (restored)!', 'done': True, 'error': None}
                print(f"   [OK] Restored pipeline results for task_id: {tid}")
            except Exception as e:
                print(f"   [WARN] Failed to load cached result for {tid}: {e}")


def _load_model():
    global model, model_info
    ckpt = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints', 'CNN_BiLSTM_best.pth')
    if os.path.exists(ckpt):
        device = torch.device('cpu')
        m = BiLSTMClassifier()
        cp = torch.load(ckpt, map_location=device, weights_only=False)
        m.load_state_dict(cp['model_state_dict']); m.eval()
        model = m
        model_info = {'loaded': True, 'best_map': str(cp.get('best_val_map', 'N/A')),
                      'epoch': str(cp.get('epoch', 'N/A'))}
        print(f"   [OK] BiLSTM model loaded (epoch {model_info['epoch']})")
    else:
        print(f"   [WARN] Checkpoint not found at {ckpt}")

_load_model()
rag_agent = FootballRAGAgent()
_load_saved_results()


# ---------- ROUTES ----------

@app.route('/')
def index():
    return send_from_directory('frontend', 'index.html')

@app.route('/api/status')
def status():
    return jsonify({'model': model_info, 'has_results': bool(pipeline_results)})

@app.route('/api/run-pipeline', methods=['POST'])
def run_pipeline():
    if model is None:
        return jsonify({'error': 'Model checkpoint not loaded'}), 400
    feat = request.files.get('features')
    vid = request.files.get('video')
    if not feat or not vid:
        return jsonify({'error': 'Both .npy and .mp4 files are required'}), 400

    cfg = {
        'threshold': float(request.form.get('threshold', 0.3)),
        'max_duration': int(request.form.get('max_duration', 7)),
        'enable_audio': request.form.get('enable_audio', 'true') == 'true',
        'enable_ocr': request.form.get('enable_ocr', 'true') == 'true',
        'enable_clip': request.form.get('enable_clip', 'true') == 'true',
        'ocr_crop_pos': request.form.get('ocr_crop_pos', 'top-left'),
        'hf_token': request.form.get('hf_token', ''),
    }

    task_id = str(uuid.uuid4())[:8]
    tmp = tempfile.mkdtemp()
    fp = os.path.join(tmp, 'features.npy'); feat.save(fp)
    vp = os.path.join(tmp, 'source.mp4');   vid.save(vp)
    op = os.path.join(RESULTS_DIR, f'highlights_{task_id}.mp4')

    pipeline_progress[task_id] = {'progress': 0, 'step': 'Starting...', 'done': False, 'error': None}
    t = threading.Thread(target=_worker, args=(task_id, fp, vp, op, tmp, cfg), daemon=True)
    t.start()
    return jsonify({'task_id': task_id})


def _clean(e):
    """Make an event dict JSON-safe."""
    return {k: (float(v) if isinstance(v, (np.floating, np.integer)) else
                int(v) if isinstance(v, np.bool_) else v) for k, v in e.items()}


def _worker(tid, fp, vp, op, tmp, cfg):
    global pipeline_results
    try:
        P = pipeline_progress[tid]

        # 1 — BiLSTM
        P.update(progress=0.05, step='Running CNN-BiLSTM visual event detection…')
        n_frames = np.load(fp).shape[0]
        dur = n_frames / 2 / 60
        def inf_cb(p): P['progress'] = 0.05 + p * 0.35
        sel, all_ev = generate_highlights(model, fp, cfg['threshold'], cfg['max_duration'], progress_cb=inf_cb)
        P.update(progress=0.40, step='Visual detection complete.')

        # 2 — Audio
        audio = []
        if cfg['enable_audio']:
            P.update(progress=0.45, step='Classifying audio events…')
            try: audio = classify_audio_events(vp, temp_dir=tmp)
            except: pass
            P.update(progress=0.60, step='Audio analysis complete.')

        # 3 — OCR
        ocr = []
        if cfg['enable_ocr']:
            P.update(progress=0.65, step='Scanning scoreboard OCR…')
            try: ocr = track_scoreboard_in_video(vp, crop_position=cfg['ocr_crop_pos'], sample_interval_seconds=15.0)
            except: pass
            P.update(progress=0.80, step='Scoreboard tracking complete.')

        # 4 — CLIP
        c_ts, c_emb = [], None
        if cfg['enable_clip']:
            P.update(progress=0.82, step='Generating CLIP embeddings…')
            try:
                ix = os.path.join(tmp, 'clip.npz')
                c_ts, c_emb = index_video_frames(vp, index_path=ix, sample_fps=1.0)
                if isinstance(c_ts, np.ndarray): c_ts = c_ts.tolist()
            except: pass
            P.update(progress=0.90, step='Semantic indexing complete.')

        # 5 — Stitch
        ok = False
        if sel:
            P.update(progress=0.92, step='Stitching highlights video…')
            try: ok = stitch_highlight_video(vp, sel, op)
            except: pass

        # 6 — Summary
        P.update(progress=0.97, step='Generating match narrative…')
        summary = generate_llm_summary(sel, api_token=cfg['hf_token'] or None)

        pipeline_results[tid] = {
            'selected_events': [_clean(e) for e in sel],
            'all_events': [_clean(e) for e in all_ev],
            'audio_timeline': audio, 'ocr_timeline': ocr,
            'clip_timestamps': c_ts if isinstance(c_ts, list) else [],
            'has_clip': c_emb is not None,
            'video_duration_min': dur, 'summary': summary,
            'has_video': ok,
            'video_filename': f'highlights_{tid}.mp4' if ok else None,
            '_clip_emb': c_emb, '_video_path': vp,
        }
        try:
            _save_results(tid, pipeline_results[tid])
        except Exception as e:
            print(f"   [WARN] Failed to save results to disk for {tid}: {e}")
        P.update(progress=1.0, step='Pipeline complete!', done=True)
    except Exception as ex:
        pipeline_progress[tid] = {'progress': 0, 'step': str(ex), 'done': True, 'error': str(ex)}


@app.route('/api/progress/<tid>')
def progress(tid):
    p = pipeline_progress.get(tid)
    return jsonify(p) if p else (jsonify({'error': 'not found'}), 404)

@app.route('/api/results/<tid>')
def results(tid):
    r = pipeline_results.get(tid)
    if not r: return jsonify({'error': 'not found'}), 404
    return jsonify({k: v for k, v in r.items() if not k.startswith('_')})

@app.route('/api/frame/<tid>')
def extract_frame(tid):
    """Extract a single video frame at a given timestamp (seconds) from the source video."""
    r = pipeline_results.get(tid)
    if not r:
        return jsonify({'error': 'not found'}), 404
    vp = r.get('_video_path', '')
    if not vp or not os.path.isfile(vp):
        return jsonify({'error': 'source video not available'}), 404

    t = request.args.get('t', 0, type=float)
    w = request.args.get('w', 640, type=int)  # optional resize width

    cap = cv2.VideoCapture(vp)
    if not cap.isOpened():
        return jsonify({'error': 'cannot open video'}), 500

    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return jsonify({'error': 'frame not found'}), 404

    # Resize to target width keeping aspect ratio
    h_orig, w_orig = frame.shape[:2]
    if w < w_orig:
        scale = w / w_orig
        frame = cv2.resize(frame, (w, int(h_orig * scale)), interpolation=cv2.INTER_AREA)

    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(buf.tobytes(), mimetype='image/jpeg',
                    headers={'Cache-Control': 'public, max-age=86400'})


@app.route('/api/video/<fn>')
def video(fn):
    return send_from_directory(RESULTS_DIR, fn)

@app.route('/api/search', methods=['POST'])
def clip_search_route():
    d = request.get_json()
    r = pipeline_results.get(d.get('task_id', ''))
    if not r: return jsonify({'error': 'No results'}), 404
    emb = r.get('_clip_emb')
    if emb is None: return jsonify({'error': 'No CLIP embeddings'}), 400
    matches = search_video(query=d['query'], timestamps=r['clip_timestamps'],
                           embeddings=emb, top_k=d.get('top_k', 4),
                           event_list=r['selected_events'])
    return jsonify({'matches': matches})

@app.route('/api/chat', methods=['POST'])
def chat():
    d = request.get_json()
    r = pipeline_results.get(d.get('task_id', ''), {})
    resp = rag_agent.get_response(
        query=d['message'],
        match_events=r.get('all_events'),
        scoreboard_timeline=r.get('ocr_timeline'),
        audio_events=r.get('audio_timeline')
    )
    return jsonify({'response': resp})


if __name__ == '__main__':
    print("\n   Football Highlight Detection -- API Server")
    print(f"   Model: {'loaded' if model_info['loaded'] else 'not found'}")
    print(f"   -> Open  http://localhost:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
