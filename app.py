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
OPENAI_API_KEY           = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY         = os.getenv("PINECONE_API_KEY")
PINECONE_ENV             = "us-east1-gcp"
EMBEDDING_MODEL          = "text-embedding-ada-002"
INDEX_NAME               = "triage-index"
MAX_RETRIES              = 3
MIN_SYMPTOMS_FOR_PINECONE = 3        # lowered threshold for simpler behavior
NIGERIA_EMERGENCY_HOTLINE = "112"    # still unused, but left here in case
PINECONE_SCORE_THRESHOLD  = 0.8

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
            # Re-raise so that any code calling this knows something’s wrong
            raise
    return _pinecone_index

# ======================
# OpenAI helper functions
# ======================

def call_openai_chat(messages: List[Dict], model: str = "gpt-4", temperature: float = 0.3, max_tokens: int = 200) -> Optional[str]:
    """Call OpenAI Chat API (old, stable version). Returns the assistant’s reply or None on error."""
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

def call_openai_embeddings(texts: List[str], model: str = EMBEDDING_MODEL) -> Optional[List[List[float]]]:
    """Call OpenAI Embeddings API. Returns a list of vectors or None on error."""
    try:
        import openai
        openai.api_key = OPENAI_API_KEY
        response = openai.Embedding.create(input=texts, model=model)
        return [item["embedding"] for item in response["data"]]
    except Exception as e:
        logger.error(f"OpenAI Embeddings API error: {e}")
        return None

# ======================
# Utility functions
# ======================

def generate_thread_id() -> str:
    """Generate a pseudo-thread ID so the front end can pass it back."""
    return f"thread_{str(uuid.uuid4()).replace('-', '')[:25]}"

def extract_symptoms_comprehensive(description: str) -> Dict:
    """
    1. Use regex to capture any “<number>/10” severity or duration phrases.
    2. Call GPT to extract a JSON array of clinically relevant symptoms.
    3. Fallback to a small regex dictionary if GPT fails.
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
            m = re.search(pat, description_lower)
            if m:
                duration = m.group(0)
                break

        # 2) Extract severity via simple regex
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
        m = re.search(r"pain\s+(\d+)/10", description_lower)
        if m:
            severity = int(m.group(1))

        # 3) Ask GPT to extract the actual symptoms as a JSON array
        prompt = (
            "You are a medical symptom extraction specialist. Extract ONLY specific, clinically relevant symptoms from the text. "
            "Do NOT include vague terms like 'pain' without context or duration descriptions. "
            "Return a JSON array of symptom strings, e.g. [\"chest pain\", \"leg pain\", \"headache\"].\n\n"
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
                logger.error(f"GPT parsing error: {e}")
                unique_symptoms = []
        else:
            # 4) Fallback to a small dictionary if GPT fails
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

def should_query_pinecone_database(symptoms: List[str], severity: int = 0, full_text: str = "") -> bool:
    """
    Simplified logic:
      – If we have ≥ MIN_SYMPTOMS_FOR_PINECONE distinct symptoms, return True.
      – If the user explicitly asked for a diagnosis (“what might be…,” etc.), return True.
      – Otherwise, return False.
    """
    symptom_count = len(symptoms)

    if symptom_count >= MIN_SYMPTOMS_FOR_PINECONE:
        logger.info(f"Sufficient symptoms ({symptom_count}) → querying Pinecone.")
        return True

    condition_keywords = [
        "what might be", "what could be", "what is", "infection", "condition",
        "disease", "diagnosis", "what's wrong", "what do i have"
    ]
    if any(kw in full_text.lower() for kw in condition_keywords):
        logger.info("User explicitly asked for condition identification → querying Pinecone.")
        return True

    logger.info(f"Not querying Pinecone: only {symptom_count} symptoms, no explicit condition request.")
    return False

def query_pinecone_index(query_text: str, symptoms: List[str], top_k: int = 50) -> List[Dict]:
    """
    1) Use OpenAI embeddings to turn query_text into a vector.
    2) Query Pinecone for nearest vectors, filter by score ≥ PINECONE_SCORE_THRESHOLD.
    3) Deduplicate by disease name and return that list of matches.
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

        # 3) Filter by threshold and dedupe
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

def generate_condition_description(condition_name: str, user_symptoms: List[str]) -> str:
    """
    Ask GPT to produce a 2–3 sentence explanation. Fallback to a small canned dictionary if GPT fails.
    """
    try:
        symptoms_text = ", ".join(user_symptoms)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a medical assistant explaining conditions to patients. "
                    "Provide a clear, simple explanation (2-3 sentences) of the medical condition. "
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
    return fallback.get(condition_name.lower(),
                        f"{condition_name} may be related to your symptoms. Please consult a healthcare professional for details.")

def rank_conditions(matches: List[Dict], symptoms: List[str]) -> List[Dict]:
    """
    For each Pinecone match, generate a description via GPT or fallback, sort by score, and return top 5.
    """
    try:
        condition_data = []
        for match in matches:
            disease = match["metadata"].get("disease", "Unknown").title()
            score = match.get("score", 0)
            desc  = generate_condition_description(disease, symptoms)
            condition_data.append({"name": disease, "description": desc, "score": score})

        # Sort by descending score
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

def generate_follow_up_questions(symptoms: List[str]) -> List[str]:
    """
    Ask GPT to produce 2–3 follow-up questions. If GPT fails, fall back to a simple pair of questions.
    """
    try:
        symptoms_text = ", ".join(symptoms) if symptoms else "no symptoms reported yet"
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a medical triage assistant. Generate 2-3 targeted follow-up questions to clarify symptoms "
                    "or uncover related ones. Avoid asking about symptoms already reported. "
                    "Return a JSON object with a 'questions' key containing a list of questions."
                )
            },
            {
                "role": "user",
                "content": f"The user has reported these symptoms: {symptoms_text}. Generate follow-up questions."
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

    # Fallback questions
    if not symptoms:
        return ["Do you have any symptoms?", "When did your symptoms start?"]
    return ["Have your symptoms changed since they started?", "Are you experiencing any other symptoms?"]

# ======================
# Main Triage Function
# ======================

def triage_main(description: str, thread_id: Optional[str] = None) -> Dict:
    """
    Simplified main triage logic (no red flags):
      1) Create or reuse a thread ID
      2) Extract symptoms + severity
      3) Decide if we should query Pinecone
      4) Return JSON with:
         - text, possible_conditions, safety_measures, triage {type, location}, send_sos, 
           follow_up_questions, thread_id, symptoms_count, should_query_pinecone
    """
    try:
        description = description.strip()
        logger.info(f"=== TRIAGE REQUEST === Description: '{description[:50]}...'")

        # 1) Thread ID
        if not thread_id:
            thread_id = generate_thread_id()
            logger.info(f"Generated new thread ID: {thread_id}")

        # 2) Extract symptoms + severity
        symptom_data = extract_symptoms_comprehensive(description)
        symptoms = symptom_data["symptoms"]
        severity = symptom_data["severity"]

        # 3) Decide if we query Pinecone
        should_query = False
        possible_conditions = []
        if PINECONE_API_KEY:
            should_query = should_query_pinecone_database(symptoms, severity, description)
            if should_query:
                logger.info(f"Querying Pinecone with symptoms: {symptoms}")
                query_text = f"Symptoms: {', '.join(symptoms)}"
                matches = query_pinecone_index(query_text, symptoms)
                if matches:
                    possible_conditions = rank_conditions(matches, symptoms)
                    logger.info(f"Found {len(possible_conditions)} conditions from Pinecone")
                else:
                    logger.warning("No Pinecone matches found")
        else:
            logger.warning("PINECONE_API_KEY not set; skipping Pinecone query")

        # 4) Build response
        symptoms_text = ", ".join(symptoms) if symptoms else "your symptoms"
        # We no longer do any “emergency” branching, so send_sos will always be False.
        send_sos = False

        if possible_conditions:
            top_names = [c["name"] for c in possible_conditions[:3]]
            conditions_text = f"Possible conditions include: {', '.join(top_names)}."
        else:
            conditions_text = "This could be due to various medical conditions."

        response_text = (
            f"I understand you're experiencing {symptoms_text}. {conditions_text} "
            "It's important to seek medical attention for proper evaluation and treatment. "
            "Monitor your symptoms and seek care if they worsen."
        )

        safety_measures = [
            "Monitor your symptoms closely",
            "Stay hydrated and get adequate rest",
            "Seek medical attention if symptoms worsen",
            "Contact a healthcare provider for evaluation"
        ]

        triage_type = "clinic" if symptoms else "pharmacy"
        follow_up_questions = generate_follow_up_questions(symptoms)

        return {
            "text": response_text,
            "possible_conditions": possible_conditions,
            "safety_measures": safety_measures,
            "triage": {"type": triage_type, "location": "Unknown"},
            "send_sos": send_sos,
            "follow_up_questions": follow_up_questions,
            "thread_id": thread_id,
            "symptoms_count": len(symptoms),
            "should_query_pinecone": should_query
        }

    except Exception as e:
        logger.error(f"Error in triage_main: {e}")
        return {
            "text": (
                "I'm experiencing a technical issue. "
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

# ======================
# Flask Routes
# ======================

@app.route('/')
def home():
    return jsonify({
        "message": "HealthMate AI Backend API - Simplified (no red flags)",
        "status": "running",
        "version": "6.1.0",
        "endpoints": [
            "/triage  – POST  – Simplified medical triage (no emergency branching)",
            "/health  – GET   – Health check"
        ],
        "features": {
            "symptom_extraction": True,
            "pinecone_query": bool(PINECONE_API_KEY),
            "follow_up_questions": True
        }
    })

@app.route('/health')
def health_check():
    return {"status": "healthy", "timestamp": str(datetime.now())}

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

# For local debugging (Vercel will use WSGI entry point)
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
