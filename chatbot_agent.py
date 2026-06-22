"""
💬 Dual-Mode RAG Chatbot Coordinator
===================================
Orchestrates chatbot interactions between two modes:
1. Match Commentator (using video detections, OCR scoreboard, audio events).
2. Codebase Interviewer (RAG search on project files TECHNICAL_DOCUMENTATION.md, etc.).
Uses the Hugging Face Serverless Inference API with local keyword-based fallback Q&A.
"""

import os
import re
import json
import urllib.request
import urllib.error

class FootballRAGAgent:
    """
    Retrieval-Augmented Generation agent for the Football Highlight Detection workspace.
    Coordinates match context and project codebase files.
    """
    def __init__(self, workspace_path=None, api_token=None):
        self.workspace_path = workspace_path or os.getcwd()
        self.api_token = api_token
        self.doc_chunks = []
        self.load_and_chunk_docs()
        
    def load_and_chunk_docs(self):
        """Finds documentation files in the workspace and splits them into clean RAG chunks."""
        doc_files = [
            os.path.join("upgrade", "TECHNICAL_DOCUMENTATION_upgraded.md"),
            os.path.join("upgrade", "System_Architecture_and_Workflow_upgraded.md"),
            os.path.join("upgrade", "README_upgraded.md"),
            os.path.join("upgrade", "Interview_Prep_Football_Highlight_Detection_upgraded.md"),
            os.path.join("upgrade", "VIVA_DEFENSE_GUIDE_upgraded.md")
        ]
        
        self.doc_chunks = []
        for doc_name in doc_files:
            file_path = os.path.join(self.workspace_path, doc_name)
            if not os.path.exists(file_path):
                continue
                
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    
                # Split by markdown headers or substantial paragraphs
                # We split into sections of roughly 800-1500 characters
                raw_sections = re.split(r'\n##+\s+', content)
                
                # First section before any markdown header
                if raw_sections and raw_sections[0].strip():
                    self.doc_chunks.append({
                        'source': doc_name,
                        'title': "Introduction",
                        'text': raw_sections[0].strip()[:1500]
                    })
                    
                for section in raw_sections[1:]:
                    lines = section.split('\n')
                    header = lines[0].strip()
                    body = "\n".join(lines[1:]).strip()
                    
                    if body:
                        # Chunk the body if it's too long
                        chunk_size = 1200
                        for offset in range(0, len(body), chunk_size):
                            sub_text = body[offset:offset + chunk_size + 200] # overlap of 200 chars
                            self.doc_chunks.append({
                                'source': doc_name,
                                'title': header,
                                'text': f"Section: {header}\n{sub_text}"
                            })
            except Exception as e:
                pass  # Silently skip unreadable docs
                
        pass  # Index completed silently


    def retrieve_relevant_chunks(self, query, top_k=3):
        """
        Simple keyword overlap score to retrieve relevant chunks.
        Extremely fast and requires no external indexing or packages on CPU.
        """
        if not self.doc_chunks:
            return []
            
        # Extract keywords (words of length >= 3, lowercase)
        query_words = set(re.findall(r'[a-zA-Z]{3,}', query.lower()))
        
        # Stop words to remove from search matching
        stop_words = {"the", "and", "for", "that", "you", "with", "this", "what", "how", "why", "code", "model"}
        query_keywords = query_words - stop_words
        
        if not query_keywords:
            return self.doc_chunks[:top_k]
            
        scored_chunks = []
        for chunk in self.doc_chunks:
            chunk_text_lower = chunk['text'].lower()
            
            # Simple keyword matching score
            score = 0
            for keyword in query_keywords:
                # Add higher weight for exact phrase or title matches
                if keyword in chunk['title'].lower():
                    score += 5
                if keyword in chunk_text_lower:
                    # Scale score based on keyword frequency in chunk
                    score += chunk_text_lower.count(keyword)
                    
            if score > 0:
                scored_chunks.append((score, chunk))
                
        # Sort by score descending
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
        # Return top-k chunks
        results = [item[1] for item in scored_chunks[:top_k]]
        
        # Fallback to default if no keyword matches found
        if not results:
            results = self.doc_chunks[:top_k]
            
        return results


    def classify_query(self, query):
        """
        Classifies if the query is about the current match ('video')
        or the technical codebase ('codebase').
        """
        query_lower = query.lower()
        
        # Match keywords
        match_keywords = [
            "goal", "score", "whistle", "cheer", "card", "sub", "substitution", 
            "match", "game", "first half", "second half", "who won", "timeline",
            "minute", "seconds", "when did", "foul", "penalty", "kickoff", "time"
        ]
        
        for kw in match_keywords:
            # Look for exact word boundary match to prevent false matches
            if re.search(r'\b' + re.escape(kw) + r'\b', query_lower):
                return 'video'
                
        return 'codebase'


    def query_huggingface_llm(self, prompt, system_prompt=None):
        """Sends prompt to Hugging Face Inference API."""
        api_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3"
        
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_token:
            headers["Authorization"] = f"Bearer {self.api_token}"
            
        full_inputs = f"<s>[INST] {system_prompt}\n\nUSER QUERY:\n{prompt} [/INST]" if system_prompt else f"<s>[INST] {prompt} [/INST]"
        
        data = {
            "inputs": full_inputs,
            "parameters": {
                "max_new_tokens": 300,
                "temperature": 0.5,
                "return_full_text": False
            }
        }
        
        req_body = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(api_url, data=req_body, headers=headers)
        
        try:
            with urllib.request.urlopen(req, timeout=12) as response:
                res_data = json.loads(response.read().decode("utf-8"))
                if isinstance(res_data, list) and len(res_data) > 0:
                    summary = res_data[0].get('generated_text', '').strip()
                    if summary:
                        return summary
                elif isinstance(res_data, dict) and 'generated_text' in res_data:
                    return res_data['generated_text'].strip()
        except Exception as e:
            pass  # API query failed, will use fallback
            
        return None


    def generate_expert_fallback_response(self, query, query_mode, doc_context=None, match_context=None):
        """
        Determinstic expert rule-based responder that operates offline
        without any model dependencies by parsing keywords.
        """
        query_clean = query.lower()
        
        if query_mode == 'video':
            # Summarize the match events
            if not match_context or len(match_context) == 0:
                return ("I'm ready to commentate, but there is no match video processed yet. "
                        "Please upload a video and generate highlights first!")
                        
            # Custom keyword match for match commentator
            if "goal" in query_clean:
                goals = [e for e in match_context if e.get('class') == 'Goal']
                if goals:
                    times = [f"{int(g.get('time_seconds', 0)//60)}:{int(g.get('time_seconds', 0)%60):02d}" for g in goals]
                    return f"⚽ Goals were detected at the following timestamps: {', '.join(times)}. Check out these clips in the highlight panel!"
                else:
                    return "No goals were detected by the model in this match. It might have been a defensive deadlock."
                    
            if "card" in query_clean or "foul" in query_clean:
                cards = [e for e in match_context if e.get('class') == 'Cards']
                if cards:
                    times = [f"{int(c.get('time_seconds', 0)//60)}:{int(c.get('time_seconds', 0)%60):02d}" for c in cards]
                    return f"🟨/🟥 The model detected disciplinary events (cards) at: {', '.join(times)}."
                else:
                    return "No card events were detected by the model."
                    
            if "sub" in query_clean:
                subs = [e for e in match_context if e.get('class') == 'Substitution']
                if subs:
                    times = [f"{int(s.get('time_seconds', 0)//60)}:{int(s.get('time_seconds', 0)%60):02d}" for s in subs]
                    return f"🔄 Substitutions were detected by the Bi-LSTM model at: {', '.join(times)}."
                else:
                    return "No substitution events were detected in this highlight sequence."
                    
            # Default match response: print timeline summary
            timeline_summary = "🎤 Here is the match chronological timeline summary:\n"
            for idx, evt in enumerate(sorted(match_context, key=lambda x: x.get('time_seconds', 0))):
                t_sec = evt.get('time_seconds', 0)
                time_str = f"{int(t_sec // 60)}:{int(t_sec % 60):02d}"
                timeline_summary += f"- **{time_str}**: {evt.get('class')} (Confidence: {evt.get('confidence', 1.0):.1%})\n"
            return timeline_summary
            
        else:
            # Codebase / Technical doc Q&A fallback
            # Look for specific code terms in the query and answer with structural accuracy
            if "lstm" in query_clean:
                return ("💡 **Why BiLSTM?**\n"
                        "We chose a Bidirectional LSTM (`BiLSTMClassifier`) because football events have future dependencies "
                        "(e.g., a goal celebration frame at t=20 helps identify the goal event at t=15). The backward pass "
                        "captures this look-ahead context, which standard recurrent networks miss. It outperforms standard "
                        "Transformers on our small dataset (60 matches) due to a stronger sequential inductive bias.")
                        
            if "knapsack" in query_clean:
                return ("🎒 **Knapsack Optimization**\n"
                        "We use a 0/1 Knapsack Dynamic Programming algorithm (`knapsack_highlight_selection()`) to choose highlights. "
                        "Each detected event has a duration (weight) and a confidence score (value). Given a total duration limit "
                        "(e.g., 10 minutes), the Knapsack solver selects the optimal subset of non-overlapping clips that "
                        "maximizes overall prediction confidence.")
                        
            if "map" in query_clean or "metric" in query_clean:
                return ("📊 **mAP Evaluation Metric**\n"
                        "Mean Average Precision (mAP) is the standard metric in SoccerNet Action Spotting. For each event class, "
                        "we compute the Area under the Precision-Recall curve. The average across 'Goal', 'Cards', and "
                        "'Substitution' (excluding 'Background') forms our final mAP score, ensuring the model ranks "
                        "true positive events higher than false alerts.")
                        
            if "imbalance" in query_clean or "weights" in query_clean:
                return ("⚖️ **Class Imbalance Handling**\n"
                        "Since ~95% of match frames are 'Background', we apply inverse frequency weights during loss computation. "
                        "We calculate class weights as: `weights[i] = total_samples / (num_classes * count_i)`. This makes mistakes "
                        "on rare event frames (like goals) weigh roughly 50x heavier than background frame errors in the Cross Entropy loss.")
                        
            if "domain" in query_clean or "shift" in query_clean:
                return ("🔄 **Domain Shift Resolution**\n"
                        "Initially, training SoccerNet ResNet-152 features and evaluating on a local ResNet-18 pipeline led to low "
                        "accuracy because the feature vector distributions differed. We resolved this by extracting frame features "
                        "from custom videos using the exact same **ResNet-152** network, followed by a local PCA reduction to 512 dimensions.")
            
            # General documentation chunk search summary if no keyword matched
            if doc_context:
                docs_summary = "📚 **Retrieved Context from Project Documentation:**\n\n"
                for chunk in doc_context:
                    docs_summary += f"*Source: {chunk['source']} (Section: {chunk['title']})*\n> {chunk['text'][:400]}...\n\n"
                docs_summary += "\n*(Hugging Face API offline. Showing retrieved local documentation segments.)*"
                return docs_summary
                
            return ("I'm ready to explain the project codebase! Ask me about the BiLSTM architecture, "
                    "Knapsack algorithm, mAP metrics, Class Imbalance, or Domain Shift.")


    def get_response(self, query, match_events=None, scoreboard_timeline=None, audio_events=None):
        """Routes query, builds prompt context, queries LLM, and handles offline fallbacks."""
        mode = self.classify_query(query)
        
        if mode == 'video':
            # --- Mode A: Match Commentator ---
            # Compile events timeline into clean text
            events_compiled = []
            
            # Add vision events
            if match_events:
                for evt in match_events:
                    t_sec = evt.get('time_seconds', 0)
                    time_str = f"{int(t_sec // 60)}:{int(t_sec % 60):02d}"
                    events_compiled.append({
                        'time': t_sec,
                        'desc': f"Vision event: {evt.get('class')} (Confidence: {evt.get('confidence', 1.0):.1%}) at {time_str}"
                    })
            
            # Add scoreboard OCR timeline
            if scoreboard_timeline:
                for entry in scoreboard_timeline:
                    t_sec = entry.get('time_seconds', 0)
                    events_compiled.append({
                        'time': t_sec,
                        'desc': f"Scoreboard OCR status: {entry.get('score_str')} (Game time: {entry.get('game_time_str')})"
                    })
                    
            # Add audio highlights
            if audio_events:
                for entry in audio_events:
                    t_sec = entry.get('time_seconds', 0)
                    time_str = f"{int(t_sec // 60)}:{int(t_sec % 60):02d}"
                    events_compiled.append({
                        'time': t_sec,
                        'desc': f"Audio spike: {entry.get('class')} (Confidence: {entry.get('confidence', 1.0):.1%}) at {time_str}"
                    })
                    
            # Sort events chronologically
            events_compiled.sort(key=lambda x: x['time'])
            timeline_str = "\n".join([f"- {evt['desc']}" for evt in events_compiled])
            
            if not timeline_str:
                timeline_str = "No events detected in the match timeline yet."
                
            prompt = (
                f"You are a professional football match commentator. Use the following chronologically sorted match "
                f"detections to answer the viewer's question: '{query}'\n\n"
                f"MATCH EVENT LOGS:\n{timeline_str}\n\n"
                f"Answer the question concisely in a friendly, engaging commentator voice. Focus on facts from the event logs:"
            )
            
            system_prompt = (
                "You are an interactive football commentator chatbot. Use only the provided match log detections "
                "to answer user questions about what happened in the game. If the event log is empty, explain that "
                "no events have been loaded yet."
            )
            
            # Query LLM
            response = self.query_huggingface_llm(prompt, system_prompt)
            if response:
                return f"🎤 **Match Commentator**: {response}"
            else:
                # Fallback to local rule-based responder
                flat_events = []
                if match_events:
                    flat_events.extend(match_events)
                return self.generate_expert_fallback_response(query, 'video', match_context=flat_events)
                
        else:
            # --- Mode B: Codebase Interviewer ---
            # Retrieve relevant documentation chunks
            relevant_chunks = self.retrieve_relevant_chunks(query, top_k=2)
            context_str = ""
            for idx, chunk in enumerate(relevant_chunks):
                context_str += f"--- Document Section {idx+1}: {chunk['source']} ({chunk['title']}) ---\n{chunk['text']}\n\n"
                
            prompt = (
                f"You are an expert AI machine learning engineer. Answer the user's technical question about your "
                f"Football Highlight Detection system using the retrieved codebase documentation sections below:\n\n"
                f"CONTEXT DOCUMENTATION:\n{context_str}\n"
                f"USER QUESTION: '{query}'\n\n"
                f"Write a professional, detailed explanation answering their question. Be direct and clear. "
                f"Include code equations, hyperparameters, or details from the documentation sections if available:"
            )
            
            system_prompt = (
                "You are an expert ML engineering assistant explaining a Football Highlight Detection codebase. "
                "The project uses PyTorch, ResNet-152 spatial features, PCA (512-dim), a BiLSTM or Transformer for temporal "
                "context classification (Goal, Substitution, Cards, Background), and a Knapsack solver for segment compilation. "
                "Answer technical queries precisely."
            )
            
            # Query LLM
            response = self.query_huggingface_llm(prompt, system_prompt)
            if response:
                return f"🧠 **Codebase Interviewer**: {response}"
            else:
                # Fallback to local rule-based responder
                return self.generate_expert_fallback_response(query, 'codebase', doc_context=relevant_chunks)
