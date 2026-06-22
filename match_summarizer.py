"""
⚽ Match Summarizer: CV-to-LLM Highlight Narrative Generator
===========================================================
This module bridges your Computer Vision event detection pipeline with 
Generative AI. It takes the chronologically detected events (Goals, Cards, 
Substitutions) and synthesizes them into a cohesive, sports-journalist style 
paragraph summary of the match.

Can be run locally or integrated directly into your Streamlit app.
"""

import json
import urllib.request
import urllib.error

def generate_match_storyboard(events):
    """
    Converts raw CV event predictions into a structured chronological timeline.
    """
    if not events:
        return "No major events detected during the match."
        
    # Sort events by time
    events_sorted = sorted(events, key=lambda e: (e.get('half', 1), e.get('start', 0)))
    
    storyboard = []
    for idx, evt in enumerate(events_sorted):
        half = evt.get('half', 1)
        time_str = evt.get('time_str', 'unknown time')
        event_type = evt.get('class', 'Event')
        confidence = evt.get('confidence', 1.0)
        
        storyboard.append(
            f"- Half {half} at {time_str}: {event_type} detected (Model Confidence: {confidence:.1%})"
        )
        
    return "\n".join(storyboard)


def generate_llm_summary(events, api_token=None):
    """
    Sends the event timeline to a free serverless Hugging Face LLM 
    (Mistral-7B-Instruct-v0.3 or similar) to write a sports article summary.
    If no internet or API fails, falls back to a rule-based template engine.
    """
    storyboard_text = generate_match_storyboard(events)
    if storyboard_text == "No major events detected during the match.":
        return storyboard_text

    # Prompt design for the LLM
    prompt = (
        f"You are a professional sports journalist. Summarize the following "
        f"football match highlights into an exciting, cohesive, 1-paragraph news report. "
        f"Do not output list items, and do not mention 'the model' or 'confidence'. "
        f"Focus on the drama and chronological flow of the match:\n\n"
        f"MATCH TIMELINE:\n{storyboard_text}\n\n"
        f"SPORTS REPORT SUMMARY:"
    )

    # We use Mistral-7B-Instruct via Hugging Face Serverless Inference API
    api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
    
    headers = {
        "Content-Type": "application/json",
    }
    if api_token:
        headers["Authorization"] = f"Bearer {api_token}"

    data = {
        "inputs": f"<s>[INST] {prompt} [/INST]",
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.7,
            "return_full_text": False
        }
    }
    
    req_body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(api_url, data=req_body, headers=headers)
    
    try:
        # Fetch summary from Hugging Face
        with urllib.request.urlopen(req, timeout=10) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            if isinstance(res_data, list) and len(res_data) > 0:
                summary = res_data[0].get('generated_text', '').strip()
                if summary:
                    return summary
            elif isinstance(res_data, dict) and 'generated_text' in res_data:
                return res_data['generated_text'].strip()
    except Exception as e:
        pass  # API call failed, falling back to rule-based summarizer
        
    return generate_fallback_summary(events)


def generate_fallback_summary(events):
    """
    Deterministic rule-based summary in case LLM API is unavailable.
    """
    goals = [e for e in events if e.get('class') == 'Goal']
    cards = [e for e in events if e.get('class') == 'Cards']
    subs = [e for e in events if e.get('class') == 'Substitution']
    
    summary = "The match highlights captured a series of critical inflection points. "
    
    if goals:
        goal_times = [g.get('time_str') for g in goals]
        summary += f"The game's scoring was defined by decisive goal events at {', '.join(goal_times)}. "
    else:
        summary += "Both teams struggled to break the defensive deadlock, leaving the match goal-less. "
        
    if cards:
        card_times = [c.get('time_str') for c in cards]
        summary += f"Tensions flared on the pitch, resulting in referee intervention and disciplinary cards shown at {', '.join(card_times)}. "
        
    if subs:
        sub_times = [s.get('time_str') for s in subs]
        summary += f"Tactical adjustments were made as managers introduced fresh energy from the bench at {', '.join(sub_times)}. "
        
    summary += "This highlight sequence captures the core strategy and pivotal moments that dictated the match outcome."
    return summary


# --- Demo Execution ---
if __name__ == "__main__":
    # Test payload (representing your BiLSTM detections)
    mock_events = [
        {'half': 1, 'start': 100, 'end': 200, 'class': 'Goal', 'time_str': '14:20', 'confidence': 0.89},
        {'half': 1, 'start': 500, 'end': 600, 'class': 'Cards', 'time_str': '41:05', 'confidence': 0.72},
        {'half': 2, 'start': 1000, 'end': 1100, 'class': 'Substitution', 'time_str': '68:15', 'confidence': 0.95},
        {'half': 2, 'start': 1200, 'end': 1300, 'class': 'Goal', 'time_str': '88:40', 'confidence': 0.91},
    ]
    
    print("📋 Generating Chronological Timeline:")
    print(generate_match_storyboard(mock_events))
    
    print("\n✍️ Generating AI Match Report Summary:")
    # We run without token first (which is rate-limited but usually works for basic test queries)
    summary_report = generate_llm_summary(mock_events)
    print(summary_report)
