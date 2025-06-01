import os
import json
import logging
import re
import uuid
from datetime import datetime
from typing import List, Dict, Optional

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# -----------------------------------
# Setup logging
# -----------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app, origins="*")

# -----------------------------------
# Configuration
# -----------------------------------
OPENAI_API_KEY            = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY          = os.getenv("PINECONE_API_KEY")
PINECONE_ENV              = "us-east1-gcp"
EMBEDDING_MODEL           = "text-embedding-ada-002"
INDEX_NAME                = "triage-index"
MAX_RETRIES               = 3
MIN_SYMPTOMS_FOR_PINECONE = 2
PINECONE_SCORE_THRESHOLD  = 0.8

# -----------------------------------
# In-memory thread state
# -----------------------------------
# THREAD_CONTEXT stores just a set of all reported symptoms.
THREAD_CONTEXT: Dict[str, Dict[str, object]] = {}
# THREAD_HISTORY stores the literal chat history (user + assistant messages).
THREAD_HISTORY: Dict[str, List[Dict[str, str]]] = {}

# -----------------------------------
# Lazy-load Pinecone index (unchanged)
# -----------------------------------
_pinecone_index = None
def get_pinecone_index():
    global _pinecone_index
    if _pinecone_index is None:
        from pinecone import Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
        _pinecone_index = pc.Index(name=INDEX_NAME)
        logger.info(f"Initialized Pinecone index '{INDEX_NAME}'")
    return _pinecone_index

# ======================
# OpenAI helper functions (unchanged)
# ======================
def call_openai_chat(messages: List[Dict], model: str = "gpt-4", temperature: float = 0.3, max_tokens: int = 200) -> Optional[str]:
    try:
        import openai
        openai.api_key = OPENAI_API_KEY
        resp = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI Chat API error: {e}")
        return None

def call_openai_embeddings(texts: List[str], model: str = EMBEDDING_MODEL) -> Optional[List[List[float]]]:
    try:
        import openai
        openai.api_key = OPENAI_API_KEY
        resp = openai.Embedding.create(input=texts, model=model)
        return [item["embedding"] for item in resp["data"]]
    except Exception as e:
        logger.error(f"OpenAI Embeddings API error: {e}")
        return None

# ======================
# Utility functions (unchanged symptom‐extraction + Pinecone logic)
# ======================
def generate_thread_id() -> str:
    return f"thread_{str(uuid.uuid4()).replace('-', '')[:25]}"

def extract_symptoms_comprehensive(description: str) -> Dict:
    description_lower = description.lower()
    # (same as before: extract duration, severity, ask GPT to return JSON array of symptoms…)
    # For brevity, let’s assume we return:
    # return {"symptoms": ["chest pain", "nausea"], "duration": "yesterday", "severity": 8} 
    # (full code omitted)
    # … your existing implementation here …
    return {"symptoms": [], "duration": None, "severity": 0}

def should_query_pinecone_database(accumulated_symptoms: List[str], severity: int, full_text: str) -> bool:
    if len(accumulated_symptoms) >= MIN_SYMPTOMS_FOR_PINECONE:
        return True
    condition_keywords = ["what might be", "what could be", "what is", "infection", "condition", "diagnosis"]
    if any(kw in full_text.lower() for kw in condition_keywords):
        return True
    return False

def query_pinecone_index(query_text: str, accumulated_symptoms: List[str], top_k: int = 50) -> List[Dict]:
    # (same as before: embed → index.query → filter → dedupe → return list of matches)
    return []

def generate_condition_description(condition_name: str, user_symptoms: List[str]) -> str:
    # (same as before or fallback dictionary)
    return f"{condition_name} description…"

def rank_conditions(matches: List[Dict], accumulated_symptoms: List[str]) -> List[Dict]:
    # (same as before)
    return []

def generate_follow_up_questions(accumulated_symptoms: List[str]) -> List[str]:
    # (same as before)
    return ["Have your symptoms changed?","Are you experiencing anything else?"]

# ======================
# Main Triage Function (updated to use full history for a conversational tone)
# ======================
def triage_main(description: str, thread_id: Optional[str] = None) -> Dict:
    """
    1) Generate or reuse thread_id
    2) Append incoming user message to THREAD_HISTORY
    3) Extract new symptoms from *only* this description
    4) Accumulate them into THREAD_CONTEXT[thread_id]['symptoms']
    5) Decide if we need to query Pinecone (based on all symptoms so far)
    6) Build an assistant response using the full THREAD_HISTORY as context, so GPT can speak naturally
    """
    try:
        description = description.strip()
        logger.info(f"=== TRIAGE REQUEST === Description: '{description[:50]}...'")

        # 1) Thread ID
        if not thread_id:
            thread_id = generate_thread_id()
            logger.info(f"Generated new thread ID: {thread_id}")

        # 2) Initialize per-thread storage if needed
        if thread_id not in THREAD_CONTEXT:
            THREAD_CONTEXT[thread_id] = {"symptoms": set(), "severity": 0}
        if thread_id not in THREAD_HISTORY:
            THREAD_HISTORY[thread_id] = []  # will contain [{"role":"user","content":...}, {"role":"assistant","content":...}, ...]

        # 3) Append this new user turn to THREAD_HISTORY
        THREAD_HISTORY[thread_id].append({"role": "user", "content": description})

        # 4) Extract new symptoms + severity from this single description
        symptom_data = extract_symptoms_comprehensive(description)
        new_symptoms = symptom_data["symptoms"]          # e.g. ["chest pain","nausea"]
        severity = symptom_data["severity"]

        # Update the “max severity seen so far” (unused right now, but stored)
        THREAD_CONTEXT[thread_id]["severity"] = max(THREAD_CONTEXT[thread_id]["severity"], severity)

        # 5) Add new symptoms into the thread’s accumulated set
        for s in new_symptoms:
            THREAD_CONTEXT[thread_id]["symptoms"].add(s)

        accumulated_symptoms = sorted(THREAD_CONTEXT[thread_id]["symptoms"])
        symptoms_count = len(accumulated_symptoms)

        # 6) Decide whether to query Pinecone
        should_query = False
        possible_conditions: List[Dict] = []
        if PINECONE_API_KEY:
            should_query = should_query_pinecone_database(accumulated_symptoms, severity, description)
            if should_query:
                logger.info(f"Querying Pinecone with: {accumulated_symptoms}")
                query_text = f"Symptoms: {', '.join(accumulated_symptoms)}"
                matches = query_pinecone_index(query_text, accumulated_symptoms)
                if matches:
                    possible_conditions = rank_conditions(matches, accumulated_symptoms)
                else:
                    logger.warning("No Pinecone matches found")
        else:
            logger.warning("No PINECONE_API_KEY; skipping query")

        # 7) Build a “system” prompt and full chat history for GPT so it can be conversational
        #    We’ll stitch together: 
        #      a) A brief “system” instruction reminding GPT that it is a triage assistant
        #      b) The prior stored turns from THREAD_HISTORY[thread_id] (both user and assistant)
        #      c) A final assistant “cue”—we ask GPT to produce the next assistant message given all context so far

        system_prompt = {
            "role": "system",
            "content": (
                "You are a friendly medical triage assistant. Keep track of everything the user has said so far, "
                "and respond as if you remember their earlier symptoms. You can say things like, "
                "'Earlier, you mentioned chest pain…' so it feels conversational, not repetitive. "
                "When crafting your reply, reference their accumulated symptoms and, if needed, give follow-up questions."
            )
        }

        # 8) Build the “assistant_context” part of the history: only keep the last ~6 messages
        recent_history = THREAD_HISTORY[thread_id][-6:]  # last 6 turns (could include user+assistant)

        # Before we ask GPT, we need to insert our last assistant message into history (if any).
        # But the first time, there is no prior assistant message—so skip.
        # Actually, we build the next assistant turn only now:
        all_messages_for_gpt = [system_prompt] + recent_history

        # 9) Now ask GPT to generate a conversational response,
        #    but we also want GPT to include Pinecone results and follow-up questions if appropriate.
        #
        #    So we append one more “assistant-cue” telling GPT exactly what to include:
        assistant_cue = {
            "role": "assistant",
            "content": (
                f"Based on all prior conversation, here are the accumulated symptoms: {', '.join(accumulated_symptoms) or 'none so far'}. "
                f"{'Query Pinecone for possible conditions.' if should_query else 'No Pinecone query needed yet.'} "
                "If you did query Pinecone, list up to three possible conditions and short explanations. "
                "Otherwise, ask 2–3 follow-up questions to clarify."
            )
        }
        all_messages_for_gpt.append(assistant_cue)

        # 10) Call GPT with that entire message stack
        assistant_response = call_openai_chat(
            all_messages_for_gpt,
            model="gpt-4",
            temperature=0.7,
            max_tokens=300
        )

        if not assistant_response:
            assistant_response = "Sorry, I'm having trouble processing right now."

        # 11) Finally, we append this assistant response back into THREAD_HISTORY
        THREAD_HISTORY[thread_id].append({"role": "assistant", "content": assistant_response})

        # 12) Now build the JSON we’ll return to the client.
        # We still return the “structured fields” so the front end can see them, but
        # we let GPT produce the free‐text `text` so it’s more natural.

        # If we did query Pinecone, trust that GPT already named up to 3 conditions.
        # As a fallback, if GPT didn’t list them, we can also fill in our own `possible_conditions` field.
        if should_query and not possible_conditions:
            #    extract any bullet‐points GPT might have given? (optional; for simplicity, leave empty)
            possible_conditions = []

        # Determine follow-up questions if GPT didn’t ask them. 
        # (We expect GPT’s text to include them, but just in case:)
        fallback_follow_up = generate_follow_up_questions(accumulated_symptoms)

        return {
            "text": assistant_response,
            "possible_conditions": possible_conditions,
            "safety_measures": [
                "Monitor your symptoms closely",
                "Stay hydrated and get adequate rest",
                "Seek medical attention if symptoms worsen",
                "Contact a healthcare provider for evaluation"
            ],
            "triage": {"type": "clinic" if accumulated_symptoms else "pharmacy", "location": "Unknown"},
            "send_sos": False,
            # We assume GPT already phrased follow‐up questions in `assistant_response`. If not, return fallback.
            "follow_up_questions": fallback_follow_up,
            "thread_id": thread_id,
            "symptoms_count": symptoms_count,
            "should_query_pinecone": should_query
        }

    except Exception as e:
        logger.error(f"Error in triage_main: {e}")
        return {
            "text": "I’m sorry, I’m experiencing a technical issue. Please try again later.",
            "possible_conditions": [],
            "safety_measures": ["Stay calm", "Seek medical attention if needed"],
            "triage": {"type": "clinic", "location": "Unknown"},
            "send_sos": False,
            "follow_up_questions": [],
            "thread_id": thread_id or generate_thread_id(),
            "symptoms_count": 0,
            "should_query_pinecone": False
        }

# ======================
# Flask Routes
# ======================

@app.route('/')
def home():
    return jsonify({
        "message": "HealthMate AI Backend API – High‐Context Conversation",
        "status": "running",
        "version": "7.1.0",
        "endpoints": [
            "/triage  – POST  – Conversational medical triage with full context",
            "/health  – GET   – Health check"
        ]
    })

@app.route('/health')
def health_check():
    return {"status": "healthy", "timestamp": str(datetime.now())}

@app.route('/triage', methods=['POST'])
def triage_endpoint():
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400

        data = request.json
        if not data or 'description' not in data:
            return jsonify({'error': 'No description provided'}), 400

        description = data['description']
        thread_id   = data.get('thread_id')

        if not description.strip():
            return jsonify({'error': 'Description cannot be empty'}), 400

        result = triage_main(description, thread_id)
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in /triage endpoint: {e}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
