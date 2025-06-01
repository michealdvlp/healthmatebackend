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
# Load environment variables
# -----------------------------------
load_dotenv()

# -----------------------------------
# Initialize Flask app
# -----------------------------------
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
# In-memory thread state for triage
# -----------------------------------
# THREAD_CONTEXT stores a set of all reported symptoms per thread_id
THREAD_CONTEXT: Dict[str, Dict[str, object]] = {}
# THREAD_HISTORY stores the actual chat history (user + assistant) per thread_id
THREAD_HISTORY: Dict[str, List[Dict[str, str]]] = {}

# -----------------------------------
# Configure logging
# -----------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------
# Import extra service modules
# -----------------------------------
from awareness_service import (
    get_all_categories,
    generate_awareness_content,
    get_random_awareness_content
)

from health_analysis_service import process_user_message

from translation_service import (
    detect_language,
    translate_to_english,
    translate_to_language
)

# -----------------------------------
# Lazy-load Pinecone index
# -----------------------------------
_pinecone_index = None

def get_pinecone_index():
    global _pinecone_index
    if _pinecone_index is None:
        try:
            from pinecone import Pinecone
            pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
            _pinecone_index = pc.Index(name=INDEX_NAME)
            logger.info(f"Initialized Pinecone index '{INDEX_NAME}' (via Pinecone class)")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            raise
    return _pinecone_index

# -----------------------------------
# OpenAI helper functions
# -----------------------------------
def call_openai_chat(
    messages: List[Dict],
    model: str = "gpt-4",
    temperature: float = 0.3,
    max_tokens: int = 200
) -> Optional[str]:
    """Call the OpenAI ChatCompletion API and return the assistant’s reply or None on error."""
    try:
        import openai
        openai.api_key = OPENAI_API_KEY
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI Chat API error: {e}")
        return None

def call_openai_embeddings(
    texts: List[str],
    model: str = EMBEDDING_MODEL
) -> Optional[List[List[float]]]:
    """Call the OpenAI Embedding API and return a list of embeddings or None on error."""
    try:
        import openai
        openai.api_key = OPENAI_API_KEY
        response = openai.Embedding.create(input=texts, model=model)
        return [item["embedding"] for item in response["data"]]
    except Exception as e:
        logger.error(f"OpenAI Embeddings API error: {e}")
        return None

# -----------------------------------
# Utility functions
# -----------------------------------
def generate_thread_id() -> str:
    """Generate a pseudo-random thread ID for conversation tracking."""
    return f"thread_{str(uuid.uuid4()).replace('-', '')[:25]}"

def extract_symptoms_comprehensive(description: str) -> Dict:
    """
    1) Use regex to capture any “x/10” severity or duration phrases.
    2) Use GPT to extract a JSON array of clinically relevant symptoms.
    3) Fallback to a small regex‐based dictionary if GPT fails.
    Returns: { "symptoms": List[str], "duration": Optional[str], "severity": int }
    """
    try:
        description_lower = description.lower()

        # 1) Extract duration
        duration = None
        duration_patterns = [
            r"(since|started|began)\s*(yesterday|last night|today|[0-9]+\s*(day|hour|minute|week|month)s?\s*ago)",
            r"for\s*(about)?\s*([0-9]+\s*(minute|hour|day|week|month)s?)",
            r"(last|past)\s*([0-9]+\s*(minute|hour|day|week|month)s?)"
        ]
        for pat in duration_patterns:
            match = re.search(pat, description_lower)
            if match:
                duration = match.group(0)
                break

        # 2) Extract severity via descriptive keywords or “x/10”
        descriptive_severity = {
            "excruciating": 10, "unbearable": 10, "extremely painful": 10,
            "severe": 8, "very painful": 8, "crushing": 8, "sharp": 8,
            "painful": 6, "moderate": 6, "mild": 4
        }
        severity = 0
        for term, score in descriptive_severity.items():
            if term in description_lower:
                severity = score
                break
        match = re.search(r"pain\s+(\d+)/10", description_lower)
        if match:
            severity = int(match.group(1))

        # 3) Ask GPT to extract a JSON array of specific symptoms
        prompt = (
            "You are a medical symptom extraction specialist. Extract ONLY specific, clinically relevant symptoms "
            "from the text. Do NOT include vague terms like 'pain' without context or duration descriptions. "
            "Return a JSON array of symptom strings. Example: [\"chest pain\", \"nausea\", \"headache\"].\n\n"
            f"Text: \"{description}\""
        )
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Extract symptoms now."}
        ]
        response_text = call_openai_chat(messages, model="gpt-4", temperature=0, max_tokens=200)

        if response_text:
            try:
                gpt_symptoms = json.loads(response_text)
                if not isinstance(gpt_symptoms, list):
                    raise ValueError("Expected a JSON array of strings")
                unique_symptoms = []
                seen = set()
                for s in gpt_symptoms:
                    sc = s.lower().strip()
                    if sc and sc not in seen:
                        seen.add(sc)
                        unique_symptoms.append(sc)
            except Exception as e:
                logger.error(f"GPT parsing error in extract_symptoms_comprehensive: {e}")
                unique_symptoms = []
        else:
            # 4) Fallback: simple keyword‐based extraction
            unique_symptoms = []
            symptom_keywords = {
                'chest pain': ['chest pain', 'chest hurt', 'chest pressure'],
                'shortness of breath': ['short of breath', 'difficulty breathing', "can't breathe", 'breathless'],
                'headache': ['headache', 'head pain', 'migraine'],
                'fever': ['fever', 'temperature', 'hot', 'chills'],
                'nausea': ['nausea', 'nauseous', 'sick to stomach'],
                'abdominal pain': ['stomach pain', 'belly pain', 'abdominal pain']
            }
            for symptom, keywords in symptom_keywords.items():
                if any(keyword in description_lower for keyword in keywords):
                    unique_symptoms.append(symptom)

        logger.info(f"Extracted symptoms: {unique_symptoms}, Duration: {duration}, Severity: {severity}")
        return {"symptoms": unique_symptoms, "duration": duration, "severity": severity}

    except Exception as e:
        logger.error(f"Error in extract_symptoms_comprehensive: {e}")
        return {"symptoms": [], "duration": None, "severity": 0}

def should_query_pinecone_database(
    accumulated_symptoms: List[str],
    severity: int,
    full_text: str
) -> bool:
    """
    Decide whether to query Pinecone:
      – If we have ≥ MIN_SYMPTOMS_FOR_PINECONE accumulated symptoms → True.
      – If the user explicitly asks “what might be…,” “what is…,” etc. → True.
      – Otherwise → False.
    """
    symptom_count = len(accumulated_symptoms)
    if symptom_count >= MIN_SYMPTOMS_FOR_PINECONE:
        logger.info(f"Sufficient accumulated symptoms ({symptom_count}) → query Pinecone.")
        return True

    condition_keywords = [
        "what might be", "what could be", "what is", "infection", "condition",
        "disease", "diagnosis", "what's wrong", "what do i have"
    ]
    if any(kw in full_text.lower() for kw in condition_keywords):
        logger.info("User explicitly asked for condition identification → query Pinecone.")
        return True

    logger.info(f"Not querying Pinecone: only {symptom_count} accumulated symptoms, no explicit ask.")
    return False

def query_pinecone_index(
    query_text: str,
    accumulated_symptoms: List[str],
    top_k: int = 50
) -> List[Dict]:
    """
    1) Embed query_text via OpenAI.
    2) Query Pinecone for nearest vectors; filter by score ≥ threshold.
    3) Deduplicate by disease name.
    Returns a list of match metadata.
    """
    try:
        # 1) Get embeddings
        query_embedding = call_openai_embeddings([query_text])
        if not query_embedding:
            logger.error("Failed to generate query embedding.")
            return []

        # 2) Query Pinecone
        index = get_pinecone_index()
        response = index.query(
            vector=query_embedding[0],
            top_k=top_k,
            include_metadata=True
        )
        matches = response.get("matches", [])
        logger.info(f"Pinecone returned {len(matches)} raw matches")

        # 3) Filter + dedupe
        unique_by_disease = {}
        for m in matches:
            score = m.get("score", 0)
            if score < PINECONE_SCORE_THRESHOLD:
                continue
            disease = m["metadata"].get("disease", "unknown condition").lower()
            if disease not in unique_by_disease or score > unique_by_disease[disease]["score"]:
                unique_by_disease[disease] = {"match": m, "score": score}

        selected = [entry["match"] for entry in unique_by_disease.values()]
        logger.info(f"Selected {len(selected)} unique matches after thresholding.")
        return selected

    except Exception as e:
        logger.error(f"Error querying Pinecone index: {e}")
        return []

def generate_condition_description(
    condition_name: str,
    user_symptoms: List[str]
) -> str:
    """
    Ask GPT to produce a 2–3 sentence explanation;
    fallback to a small dictionary if GPT fails.
    """
    try:
        symptoms_text = ", ".join(user_symptoms)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a medical assistant explaining conditions to patients. "
                    "Provide a clear, simple explanation (2–3 sentences) of the medical condition. "
                    "Explain what it is, what causes it, and how it relates to their symptoms. "
                    "Use simple language that a patient can understand. "
                    "Do not provide medical advice or treatment recommendations."
                )
            },
            {
                "role": "user",
                "content": f"Explain {condition_name} to a patient experiencing: {symptoms_text}"
            }
        ]
        description = call_openai_chat(messages, model="gpt-4", temperature=0.3, max_tokens=150)
        if description and len(description) > 20:
            return description
    except Exception as e:
        logger.error(f"Error generating description for {condition_name}: {e}")

    # Fallback dictionary
    fallback = {
        "heart attack": "A heart attack occurs when blood flow to the heart muscle is blocked, often causing severe chest pain and shortness of breath. This is a medical emergency that requires immediate treatment.",
        "angina": "Angina is chest pain caused by reduced blood flow to the heart muscle, often triggered by physical activity or stress. It can cause chest tightness and breathing difficulties.",
        "pneumonia": "Pneumonia is an infection that inflames air sacs in the lungs, which may fill with fluid, causing chest pain, cough, and difficulty breathing.",
        "asthma": "Asthma is a condition where airways narrow and swell, producing extra mucus, which can cause chest tightness and difficulty breathing.",
        "gastritis": "Gastritis is inflammation of the stomach lining that can cause abdominal pain and nausea, often triggered by stress, spicy foods, or infections."
    }
    return fallback.get(
        condition_name.lower(),
        f"{condition_name} may be related to your symptoms. Please consult a healthcare professional for details."
    )

def rank_conditions(
    matches: List[Dict],
    accumulated_symptoms: List[str]
) -> List[Dict]:
    """
    For each Pinecone match, generate a description via GPT or fallback.
    Sort by score and return up to 5 conditions as dictionaries.
    """
    try:
        condition_data = []
        for match in matches:
            disease = match["metadata"].get("disease", "Unknown").title()
            score = match.get("score", 0)
            desc = generate_condition_description(disease, accumulated_symptoms)
            condition_data.append({"name": disease, "description": desc, "score": score})

        # Sort descending by score
        condition_data.sort(key=lambda x: x["score"], reverse=True)

        final = []
        for cond in condition_data[:5]:
            final.append({
                "name": cond["name"],
                "description": cond["description"],
                "file_citation": "medical_database.json"
            })
        return final

    except Exception as e:
        logger.error(f"Error ranking conditions: {e}")
        return []

def generate_follow_up_questions(accumulated_symptoms: List[str]) -> List[str]:
    """
    Ask GPT to produce 2–3 targeted follow-up questions based on ALL collected symptoms.
    Fallback to a simple default pair if GPT fails.
    """
    try:
        symptoms_text = ", ".join(accumulated_symptoms) if accumulated_symptoms else "no symptoms reported yet"
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a medical triage assistant. Generate 2–3 targeted follow-up questions to clarify symptoms "
                    "or uncover related ones. Avoid asking about symptoms already reported. "
                    "Return a JSON object with a 'questions' key containing a list of question strings."
                )
            },
            {
                "role": "user",
                "content": f"The user has reported these symptoms so far: {symptoms_text}. Generate follow-up questions."
            }
        ]
        response_text = call_openai_chat(messages, model="gpt-4", temperature=0.3, max_tokens=150)
        if response_text:
            try:
                result = json.loads(response_text)
                return result.get("questions", [])
            except (json.JSONDecodeError, KeyError):
                logger.error("JSON parsing error in follow-up questions")
    except Exception as e:
        logger.error(f"Error generating follow-up questions: {e}")

    # Simple fallback
    if not accumulated_symptoms:
        return ["Do you have any symptoms?", "When did your symptoms start?"]
    return ["Have your symptoms changed since they started?", "Are you experiencing any other symptoms?"]

# -----------------------------------
# Main Triage Function
# -----------------------------------
def triage_main(description: str, thread_id: Optional[str] = None) -> Dict:
    """
    1) Generate or reuse thread_id.
    2) Initialize THREAD_CONTEXT and THREAD_HISTORY if needed.
    3) Extract new symptoms from this description, accumulate them.
    4) Decide if Pinecone query is needed, and if so, gather possible_conditions.
    5) Build a conversational GPT prompt using THREAD_HISTORY so GPT’s reply can reference prior turns.
    6) Return a JSON with text, possible_conditions, etc.
    """
    try:
        description = description.strip()
        logger.info(f"=== TRIAGE REQUEST === Description: '{description[:50]}...'")

        # 1) Thread ID (treat empty or whitespace as “no thread”)
        if not thread_id or thread_id.strip() == "":
            thread_id = generate_thread_id()
            logger.info(f"Generated new thread ID: {thread_id}")

        # 2) Initialize per-thread storage if needed
        if thread_id not in THREAD_CONTEXT:
            THREAD_CONTEXT[thread_id] = {"symptoms": set(), "severity": 0}
        if thread_id not in THREAD_HISTORY:
            THREAD_HISTORY[thread_id] = []

        # 3) Append this new user turn to THREAD_HISTORY
        THREAD_HISTORY[thread_id].append({"role": "user", "content": description})

        # 4) Extract new symptoms + severity from this description
        symptom_data = extract_symptoms_comprehensive(description)
        logger.info(f"DEBUG → extract_symptoms_comprehensive returned: {symptom_data}")
        new_symptoms = symptom_data["symptoms"]
        severity = symptom_data["severity"]

        # Update max severity seen so far
        THREAD_CONTEXT[thread_id]["severity"] = max(THREAD_CONTEXT[thread_id]["severity"], severity)

        # 5) Accumulate new symptoms into the thread’s set
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
            logger.warning("No PINECONE_API_KEY; skipping Pinecone query")

        # 7) Build a “system” prompt and use the full recent history to craft a conversational response
        system_prompt = {
            "role": "system",
            "content": (
                "You are a friendly medical triage assistant. Keep track of everything the user has said so far, "
                "and respond as if you remember their earlier symptoms. You can say things like, "
                "'Earlier, you mentioned chest pain…' so it feels conversational. When crafting your reply, "
                "reference their accumulated symptoms and, if needed, give follow-up questions."
            )
        }

        # Keep only the last 6 turns (user+assistant) for prompt brevity
        recent_history = THREAD_HISTORY[thread_id][-6:]  # list of dicts with roles/contents

        # 8) Insert an “assistant‐cue” explaining what we want GPT to include
        assistant_cue = {
            "role": "assistant",
            "content": (
                f"Based on all prior conversation, the accumulated symptoms are: "
                f"{', '.join(accumulated_symptoms) or 'none so far'}. "
                f"{'Query Pinecone for possible conditions.' if should_query else 'No Pinecone query needed yet.'} "
                "If you did query Pinecone, list up to three possible conditions and short explanations. "
                "Otherwise, ask 2–3 follow-up questions to clarify."
            )
        }

        # Combine system prompt + recent history + assistant cue
        all_messages_for_gpt = [system_prompt] + recent_history + [assistant_cue]

        # 9) Call GPT to generate the assistant response
        assistant_response = call_openai_chat(
            all_messages_for_gpt,
            model="gpt-4",
            temperature=0.7,
            max_tokens=300
        )
        if not assistant_response:
            assistant_response = (
                "Sorry, I'm having trouble processing right now. "
                "Please try again in a moment."
            )

        # 10) Append GPT’s response into THREAD_HISTORY
        THREAD_HISTORY[thread_id].append({"role": "assistant", "content": assistant_response})

        # 11) Prepare fallback follow-up questions if GPT did not embed them
        fallback_follow_up = generate_follow_up_questions(accumulated_symptoms)

        # 12) Construct the JSON response for the client
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
            "follow_up_questions": fallback_follow_up,
            "thread_id": thread_id,
            "symptoms_count": symptoms_count,
            "should_query_pinecone": should_query
        }

    except Exception as e:
        logger.error(f"Error in triage_main: {e}")
        return {
            "text": (
                "I'm sorry, I'm experiencing a technical issue. "
                "Based on your symptoms, please seek medical attention if you feel unwell."
            ),
            "possible_conditions": [],
            "safety_measures": ["Stay calm and rest", "Seek medical attention"],
            "triage": {"type": "clinic", "location": "Unknown"},
            "send_sos": False,
            "follow_up_questions": [],
            "thread_id": thread_id or generate_thread_id(),
            "symptoms_count": 0,
            "should_query_pinecone": False
        }

# -----------------------------------
# Flask routes
# -----------------------------------

@app.route('/')
def home():
    return jsonify({
        "message": "HealthMate AI Backend API – Full Integration",
        "status": "running",
        "version": "7.2.0",
        "endpoints": [
            "/health            – GET    – Health check",
            "/triage            – POST   – Conversational medical triage with context + Pinecone",
            "/awareness         – GET    – Health awareness content",
            "/analyze           – POST   – Healthcare analysis (Azure + OpenAI)",
            "/translate         – POST   – Translation service"
        ],
        "features": {
            "triage": True,
            "awareness": True,
            "health_analysis": True,
            "translation": True
        }
    })

@app.route('/health')
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": str(datetime.now())
    }

@app.route('/triage', methods=['POST'])
def triage_endpoint():
    """
    POST /triage
    {
      "description": "I have a fever and cough",
      "thread_id": "thread_abc123"    # optional on first call
    }
    """
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

@app.route("/awareness", methods=["GET"])
def awareness_endpoint():
    """
    GET /awareness?category=Nutrition&count=3
    GET /awareness?random=5
    - If `category` is provided: returns generate_awareness_content(category, count).
    - If `random` is provided: returns get_random_awareness_content(count=random).
    - Otherwise: returns list of all categories.
    """
    try:
        category = request.args.get("category", None)
        random_count = request.args.get("random", None)
        count = int(request.args.get("count", 3))

        # 1) If random=X is set, ignore category and return X random articles
        if random_count is not None:
            try:
                rc = int(random_count)
            except ValueError:
                return jsonify({"error": "Invalid random count"}), 400

            contents = get_random_awareness_content(count=rc)
            return jsonify({"success": True, "random_count": rc, "articles": contents})

        # 2) If category is provided, generate that category
        if category:
            all_cats = get_all_categories()
            if category not in all_cats:
                return jsonify({
                    "success": False,
                    "error": f"Unknown category '{category}'. Valid options: {all_cats}"
                }), 400

            articles = generate_awareness_content(category, count=count)
            return jsonify({
                "success": True,
                "category": category,
                "count": count,
                "articles": articles
            })

        # 3) If neither, return list of all categories
        return jsonify({
            "success": True,
            "categories": get_all_categories()
        })

    except Exception as e:
        logger.error(f"Error in /awareness endpoint: {e}")
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500

@app.route("/analyze", methods=["POST"])
def analyze_endpoint():
    """
    POST /analyze
    {
      "text": "I have headache and fever for two days"
    }
    Returns a JSON object from health_analysis_service.process_user_message().
    """
    try:
        if not request.is_json:
            return jsonify({"success": False, "error": "Request must be JSON"}), 400

        data = request.json
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"success": False, "error": "Text cannot be empty"}), 400

        # Delegate to health_analysis_service
        result = process_user_message(text)
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in /analyze endpoint: {e}")
        return jsonify({"success": False, "error": "Internal server error"}), 500

@app.route("/translate", methods=["POST"])
def translate_endpoint():
    """
    POST /translate
    {
      "text": "some text",
      "to": "ig",           # target language code (optional, defaults to English)
      "from": "auto"        # source language code or "auto"
    }
    """
    try:
        if not request.is_json:
            return jsonify({"success": False, "error": "Request must be JSON"}), 400

        data = request.json
        orig_text = data.get("text", "").strip()
        if not orig_text:
            return jsonify({"success": False, "error": "Text cannot be empty"}), 400

        target_lang = data.get("to", "en").lower()
        source_lang = data.get("from", "auto").lower()

        # 1) Detect source language if "auto"
        if source_lang == "auto":
            detected = detect_language(orig_text)
            source_lang = detected
        else:
            detected = source_lang

        # 2) If source != "en", translate to English first
        if source_lang != "en":
            english_text = translate_to_english(orig_text, source_lang)
        else:
            english_text = orig_text

        # 3) If target != "en", translate English → target
        if target_lang != "en":
            back_text = translate_to_language(english_text, target_lang)
        else:
            back_text = english_text

        return jsonify({
            "success": True,
            "detected_language": source_lang,
            "original_text": orig_text,
            "translated_to_english": english_text if source_lang != "en" else None,
            "translated_to_target": back_text
        })

    except Exception as e:
        logger.error(f"Error in /translate endpoint: {e}")
        return jsonify({"success": False, "error": "Internal server error"}), 500

# -----------------------------------
# Run the Flask app
# -----------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
