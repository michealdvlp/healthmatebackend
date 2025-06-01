def is_red_flag(full_text: str, severity: int = 0, symptoms: List[str] = None) -> bool:
    """Check if text contains any red-flag keywords or emergency symptom combinations"""
    lt = full_text.lower()
    
    # Check individual red flags
    for rf in RED_FLAGS:
        if rf in lt:
            logger.info(f"Red flag detected: {rf}")
            return True
    
    # Check emergency symptom combinations
    if symptoms:
        symptoms_lower = [s.lower() for s in symptoms]
        
        # Chest pain + breathing issues = emergency
        has_chest_pain = any("chest pain" in simport os
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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") 
PINECONE_ENV = "us-east1-gcp"
EMBEDDING_MODEL = "text-embedding-ada-002"
INDEX_NAME = "triage-index"
MAX_RETRIES = 3
MIN_SYMPTOMS_FOR_PINECONE = 3
MAX_ITERATIONS = 3
NIGERIA_EMERGENCY_HOTLINE = "112"
PINECONE_SCORE_THRESHOLD = 0.8

# -----------------------------------
# Red-flag symptoms
# -----------------------------------
RED_FLAGS = [
    "bullet wound", "gunshot", "profuse bleeding", "crushing chest pain",
    "sudden shortness of breath", "loss of consciousness", "slurred speech", "seizure",
    "head trauma", "neck trauma", "high fever with stiff neck", "uncontrolled vomiting",
    "severe allergic reaction", "anaphylaxis", "difficulty breathing", "persistent cough with blood",
    "severe abdominal pain", "sudden vision loss", "chest tightness with sweating",
    "blood in urine", "inability to pass urine", "sharp abdominal pain", "intermenstrual bleeding",
    # Add more emergency combinations
    "chest pain", "shortness of breath", "severe chest pain", "chest pressure", "heart attack",
    "severe shortness of breath", "cannot breathe", "crushing pain"
]

# -----------------------------------
# Lazy-load clients
# -----------------------------------
_pinecone_index = None

def get_pinecone_index():
    """Lazy-load Pinecone index"""
    global _pinecone_index
    if _pinecone_index is None:
        try:
            # v2.x style: instantiate a Pinecone client instead of pinecone.init(...)
            from pinecone import Pinecone
            
            pc = Pinecone(
                api_key=PINECONE_API_KEY,
                environment=PINECONE_ENV
            )
            _pinecone_index = pc.Index(name=INDEX_NAME)
            logger.info(f"Initialized Pinecone index '{INDEX_NAME}' (via Pinecone class)")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            raise
    return _pinecone_index

# ======================
# OpenAI Functions (using older stable API)
# ======================

def call_openai_chat(messages: List[Dict], model: str = "gpt-4", temperature: float = 0.3, max_tokens: int = 200) -> str:
    """Call OpenAI Chat API using older stable version"""
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
    """Call OpenAI Embeddings API using older stable version"""
    try:
        import openai
        openai.api_key = OPENAI_API_KEY
        
        response = openai.Embedding.create(
            input=texts,
            model=model
        )
        return [item['embedding'] for item in response['data']]
    except Exception as e:
        logger.error(f"OpenAI Embeddings API error: {e}")
        return None

# ======================
# Utility Functions
# ======================

def generate_thread_id() -> str:
    """Generate a thread-like ID for conversation tracking"""
    return f"thread_{str(uuid.uuid4()).replace('-', '')[:25]}"

def extract_symptoms_comprehensive(description: str) -> Dict:
    """Extract symptoms using GPT-4"""
    try:
        description_lower = description.lower()

        # Extract duration and severity using regex
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

        # Use GPT-4 for symptom extraction
        prompt = (
            "You are a medical symptom extraction specialist. Extract ONLY specific, clinically relevant symptoms from the text. "
            "Do NOT include vague terms like 'pain' without context or duration descriptions. "
            "Return a JSON array of symptom strings. Be precise and map synonyms (e.g., 'leg ache' → 'leg pain'). "
            "Example output: [\"chest pain\", \"leg pain\", \"headache\"].\n\n"
            f"Text: \"{description}\""
        )

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Extract symptoms now."}
        ]

        response_text = call_openai_chat(messages, temperature=0, max_tokens=200)
        
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
            # Fallback symptom extraction
            unique_symptoms = []
            symptom_keywords = {
                'chest pain': ['chest pain', 'chest hurt', 'chest pressure'],
                'shortness of breath': ['short of breath', 'difficulty breathing', 'can\'t breathe', 'breathless'],
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
        logger.error(f"Error extracting symptoms: {e}")
        return {"symptoms": [], "duration": None, "severity": 0}

def is_red_flag(full_text: str, severity: int = 0, symptoms: List[str] = None) -> bool:
    """Check if text contains any red-flag keywords or emergency symptom combinations"""
    lt = full_text.lower()
    
    # Check individual red flags in text
    for rf in RED_FLAGS:
        if rf in lt:
            logger.info(f"Red flag detected in text: {rf}")
            return True
    
    # Check emergency symptom combinations
    if symptoms:
        symptoms_lower = [s.lower() for s in symptoms]
        
        # Chest pain + breathing issues = emergency
        has_chest_pain = any("chest pain" in s or "chest" in s for s in symptoms_lower)
        has_breathing_issue = any("shortness of breath" in s or "difficulty breathing" in s or "breathless" in s for s in symptoms_lower)
        
        if has_chest_pain and has_breathing_issue:
            logger.info(f"Emergency combination detected: chest pain + breathing issue")
            return True
        
        # Severe chest pain alone = emergency
        if any("severe chest pain" in s or "crushing chest pain" in s for s in symptoms_lower):
            logger.info(f"Emergency detected: severe chest pain")
            return True
            
        # Check for other emergency combinations
        emergency_symptoms = ["chest pain", "severe pain", "difficulty breathing", "shortness of breath", "unconscious", "seizure"]
        symptom_matches = sum(1 for symptom in symptoms_lower for emergency in emergency_symptoms if emergency in symptom)
        
        if symptom_matches >= 2:
            logger.info(f"Emergency detected: {symptom_matches} emergency symptoms")
            return True
    
    # High severity check
    if severity >= 8:
        logger.info(f"Emergency detected: high severity {severity}")
        return True
    
    return False

def should_query_pinecone_database(symptoms: List[str], severity: int = 0, full_text: str = "") -> bool:
    """Decide whether to query Pinecone"""
    symptom_count = len(symptoms)
    
    # Always query for emergencies
    if is_red_flag(full_text, severity, symptoms):
        logger.info("Emergency red flag → querying Pinecone.")
        return True

    # Query if we have enough symptoms (lowered threshold for better UX)
    if symptom_count >= 2:  # Lowered from 3 to 2
        logger.info(f"Sufficient symptoms ({symptom_count}) → querying Pinecone.")
        return True

    # Query if user explicitly asks for conditions
    condition_keywords = [
        "what might be", "what could be", "what is", "infection", "condition",
        "disease", "diagnosis", "what's wrong", "what do i have"
    ]
    if any(kw in full_text.lower() for kw in condition_keywords):
        logger.info("User explicitly asked for condition identification → querying Pinecone.")
        return True

    logger.info(f"Not querying Pinecone: only {symptom_count} symptoms")
    return False

def query_pinecone_index(query_text: str, symptoms: List[str], top_k: int = 50) -> List[Dict]:
    """Run a Pinecone vector query"""
    try:
        # Get embeddings
        query_embedding = call_openai_embeddings([query_text])
        if not query_embedding:
            logger.error("Failed to generate query embedding.")
            return []

        # Query Pinecone
        index = get_pinecone_index()
        response = index.query(
            vector=query_embedding[0],
            top_k=top_k,
            include_metadata=True
        )
        matches = response.get("matches", [])
        logger.info(f"Pinecone returned {len(matches)} raw matches")

        # Filter and deduplicate
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
    """Generate condition description using GPT-4"""
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

        description = call_openai_chat(messages, temperature=0.3, max_tokens=150)
        
        if description and len(description) > 20:
            return description
        else:
            # Use fallback descriptions
            fallback = {
                "heart attack": "A heart attack occurs when blood flow to the heart muscle is blocked, often causing severe chest pain and shortness of breath. This is a medical emergency that requires immediate treatment.",
                "angina": "Angina is chest pain caused by reduced blood flow to the heart muscle, often triggered by physical activity or stress. It can cause chest pain and breathing difficulties.",
                "pneumonia": "Pneumonia is an infection that inflames air sacs in the lungs, which may fill with fluid, causing chest pain, cough, and difficulty breathing.",
                "asthma": "Asthma is a condition where airways narrow and swell, producing extra mucus, which can cause chest tightness and difficulty breathing.",
                "gastritis": "Gastritis is inflammation of the stomach lining that can cause abdominal pain and nausea, often triggered by stress, spicy foods, or infections."
            }
            return fallback.get(condition_name.lower(), f"{condition_name} may be related to your symptoms. Please consult a healthcare professional for details.")

    except Exception as e:
        logger.error(f"Error generating description for {condition_name}: {e}")
        return f"{condition_name} may be related to your symptoms. Please consult a healthcare professional."

def rank_conditions(matches: List[Dict], symptoms: List[str]) -> List[Dict]:
    """Rank and format condition matches"""
    try:
        condition_data = []
        for match in matches:
            disease = match["metadata"].get("disease", "Unknown").title()
            score = match.get("score", 0)
            desc = generate_condition_description(disease, symptoms)
            condition_data.append({"name": disease, "description": desc, "score": score})

        # Sort by score descending
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
    """Generate follow-up questions using GPT-4"""
    try:
        symptoms_text = ", ".join(symptoms) if symptoms else "no symptoms reported yet"
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a medical triage assistant. Generate 2-3 targeted follow-up questions to clarify symptoms "
                    "or uncover related ones. Avoid asking about symptoms already reported. "
                    "Return a JSON object with a 'questions' key containing a list of question strings."
                )
            },
            {
                "role": "user",
                "content": f"The user has reported these symptoms: {symptoms_text}. Generate follow-up questions."
            }
        ]

        response_text = call_openai_chat(messages, temperature=0.3, max_tokens=150)
        
        if response_text:
            try:
                result = json.loads(response_text)
                questions = result.get("questions", [])
                return questions
            except (json.JSONDecodeError, KeyError):
                logger.error("JSON parsing error in follow-up questions")
        
        # Fallback questions
        if not symptoms:
            return ["Do you have any symptoms?", "When did your symptoms start?"]
        else:
            return ["Have your symptoms changed since they started?", "Are you experiencing any other symptoms?"]

    except Exception as e:
        logger.error(f"Error generating follow-up questions: {e}")
        return ["Do you have any other symptoms?", "When did your symptoms start?"]

# ======================
# Main Triage Function
# ======================

def triage_main(description: str, thread_id: Optional[str] = None) -> Dict:
    """
    Main triage function - simplified but effective version
    """
    try:
        description = description.strip()
        logger.info(f"=== TRIAGE REQUEST === Description: '{description[:50]}...'")

        # Generate thread ID if not provided
        if not thread_id:
            thread_id = generate_thread_id()
            logger.info(f"Generated thread ID: {thread_id}")

        # Extract symptoms
        symptom_data = extract_symptoms_comprehensive(description)
        symptoms = symptom_data["symptoms"]
        severity = symptom_data["severity"]
        
        # Check for emergency (pass symptoms for better detection)
        is_emergency = is_red_flag(description, severity, symptoms)
        
        # Determine if we should query Pinecone (pass symptoms for better detection)
        should_query = should_query_pinecone_database(symptoms, severity, description)
        
        logger.info(f"Analysis: symptoms={symptoms}, severity={severity}, is_emergency={is_emergency}, should_query={should_query}")
        
        possible_conditions = []
        if should_query and PINECONE_API_KEY:
            try:
                logger.info(f"Querying Pinecone with symptoms: {symptoms}")
                query_text = f"Symptoms: {', '.join(symptoms)}"
                matches = query_pinecone_index(query_text, symptoms)
                if matches:
                    possible_conditions = rank_conditions(matches, symptoms)
                    logger.info(f"Found {len(possible_conditions)} conditions from Pinecone")
                else:
                    logger.warning("No Pinecone matches found")
            except Exception as e:
                logger.error(f"Pinecone query failed: {e}")
        elif not PINECONE_API_KEY:
            logger.warning("Pinecone API key not available")
        else:
            logger.info("Skipping Pinecone query (should_query=False)")

        # Generate response text
        symptoms_text = ", ".join(symptoms) if symptoms else "your symptoms"
        
        if is_emergency:
            response_text = (
                f"🚨 URGENT: Your symptoms ({symptoms_text}) suggest a medical emergency. "
                f"Please call {NIGERIA_EMERGENCY_HOTLINE} immediately for emergency medical assistance. "
                "Do not delay - seek immediate medical attention."
            )
            safety_measures = [
                f"Call {NIGERIA_EMERGENCY_HOTLINE} immediately",
                "Stay calm and don't panic",
                "Have someone stay with you if possible",
                "Prepare to provide your location to emergency services"
            ]
            triage_type = "hospital"
            follow_up_questions = []
        else:
            if possible_conditions:
                conditions_list = [c['name'] for c in possible_conditions[:3]]
                conditions_text = f"Possible conditions include: {', '.join(conditions_list)}"
            else:
                conditions_text = "This could be due to various medical conditions"
                
            response_text = (
                f"I understand you're experiencing {symptoms_text}. {conditions_text}. "
                "It's important to seek medical attention for proper evaluation and treatment. "
                "Monitor your symptoms and seek immediate care if they worsen."
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
            "send_sos": is_emergency,
            "follow_up_questions": follow_up_questions,
            "thread_id": thread_id,
            "symptoms_count": len(symptoms),
            "should_query_pinecone": should_query
        }

    except Exception as e:
        logger.error(f"Error in triage function: {e}")
        return {
            "text": f"I'm experiencing a technical issue, but based on your symptoms, I recommend seeking medical attention. Call {NIGERIA_EMERGENCY_HOTLINE} if this is urgent.",
            "possible_conditions": [],
            "safety_measures": ["Stay calm and rest", "Seek medical attention"],
            "triage": {"type": "hospital", "location": "Unknown"},
            "send_sos": True,
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
        "message": "HealthMate AI Backend API - Stable Version",
        "status": "running", 
        "version": "6.0.0",
        "endpoints": [
            "/triage - POST - Advanced medical triage analysis",
            "/health - GET - Health check"
        ],
        "openai_version": "0.28.1",
        "pinecone_version": "2.2.4",
        "features": {
            "symptom_extraction": True,
            "emergency_detection": True,
            "condition_matching": bool(PINECONE_API_KEY),
            "gpt_analysis": bool(OPENAI_API_KEY)
        }
    })

@app.route('/health')
def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy", "timestamp": str(datetime.now())}

@app.route('/triage', methods=['POST'])
def triage_endpoint():
    """
    The main /triage endpoint
    """
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.json
        if not data or 'description' not in data:
            return jsonify({'error': 'No description provided'}), 400
        
        description = data['description']
        thread_id = data.get('thread_id')
        
        if not description.strip():
            return jsonify({'error': 'Description cannot be empty'}), 400
        
        # Call the main triage function
        result = triage_main(description, thread_id)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in /triage endpoint: {e}")
        return jsonify({
            "error": f"Internal server error: {str(e)}"
        }), 500

# For Vercel
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
