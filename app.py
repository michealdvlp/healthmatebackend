import os
import sys
import json
import logging
import asyncio
import re
import uuid
from datetime import datetime
from typing import List, Dict, Optional
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app, origins="*")

# Import dependencies for triage
try:
    from openai import AsyncOpenAI
    import openai
    OPENAI_AVAILABLE = True
    logger.info("OpenAI imported successfully")
except ImportError as e:
    OPENAI_AVAILABLE = False
    logger.error(f"OpenAI import failed: {e}")

try:
    from pinecone import Pinecone
    PINECONE_AVAILABLE = True
    logger.info("Pinecone imported successfully")
except ImportError as e:
    PINECONE_AVAILABLE = False
    logger.error(f"Pinecone import failed: {e}")

# Configuration from your testtriage.py
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east1-gcp")
ASSISTANT_ID = os.getenv("ASSISTANT_ID", "asst_pAhSF6XJsj60efD9GEVdEG5n")
EMBEDDING_MODEL = "text-embedding-ada-002"
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "triage-index")
MAX_RETRIES = 3
MIN_SYMPTOMS_FOR_PINECONE = 3
MAX_ITERATIONS = 3
NIGERIA_EMERGENCY_HOTLINE = "112"
PINECONE_SCORE_THRESHOLD = 0.8

# Red-flag symptoms from your testtriage.py
RED_FLAGS = [
    "bullet wound", "gunshot", "profuse bleeding", "crushing chest pain",
    "sudden shortness of breath", "loss of consciousness", "slurred speech", "seizure",
    "head trauma", "neck trauma", "high fever with stiff neck", "uncontrolled vomiting",
    "severe allergic reaction", "anaphylaxis", "difficulty breathing", "persistent cough with blood",
    "severe abdominal pain", "sudden vision loss", "chest tightness with sweating",
    "blood in urine", "inability to pass urine", "sharp abdominal pain", "intermenstrual bleeding"
]

# Initialize OpenAI client
openai_client = None
if OPENAI_AVAILABLE and OPENAI_API_KEY:
    try:
        openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        logger.info("OpenAI client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")

# Initialize Pinecone
pc = None
index = None
if PINECONE_AVAILABLE and PINECONE_API_KEY:
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(name=INDEX_NAME)
        logger.info(f"Pinecone index '{INDEX_NAME}' initialized")
    except Exception as e:
        logger.error(f"Failed to initialize Pinecone: {e}")

# Supported languages
SUPPORTED_LANGUAGES = {
    "en": "English",
    "ig": "Igbo",
    "yo": "Yoruba", 
    "ha": "Hausa",
    "pcm": "Nigerian Pidgin"
}

# Thread storage (in production, use Redis or database)
conversation_threads = {}

class TriageRequest:
    def __init__(self, description: str, thread_id: Optional[str] = None):
        self.description = description
        self.thread_id = thread_id

class TriageResponse:
    def __init__(self, **kwargs):
        self.text = kwargs.get('text', '')
        self.possible_conditions = kwargs.get('possible_conditions', [])
        self.safety_measures = kwargs.get('safety_measures', [])
        self.triage = kwargs.get('triage', {"type": "", "location": "Unknown"})
        self.send_sos = kwargs.get('send_sos', False)
        self.follow_up_questions = kwargs.get('follow_up_questions', [])
        self.thread_id = kwargs.get('thread_id', '')
        self.symptoms_count = kwargs.get('symptoms_count', 0)
        self.should_query_pinecone = kwargs.get('should_query_pinecone', False)

# Advanced triage functions from your testtriage.py
async def validate_thread(thread_id: str) -> bool:
    """Check if an OpenAI thread ID is valid"""
    if not openai_client:
        return False
    try:
        await openai_client.beta.threads.retrieve(thread_id=thread_id)
        logger.info(f"Thread {thread_id} is valid.")
        return True
    except Exception as e:
        logger.error(f"Couldn't validate thread_id {thread_id}: {e}")
        return False

async def get_thread_context(thread_id: str) -> Dict:
    """Retrieve all past user messages in this thread"""
    if not openai_client:
        return {"user_messages": [], "assistant_responses": 0, "all_symptoms": [], "max_severity": 0}
    
    try:
        messages = await openai_client.beta.threads.messages.list(thread_id=thread_id, order='asc')
        user_messages = []
        assistant_count = 0
        all_symptoms = []
        max_severity = 0

        for msg in messages.data:
            if msg.role == "user":
                content = ""
                if msg.content and hasattr(msg.content[0], "text"):
                    content = msg.content[0].text.value
                if not content:
                    logger.warning(f"Empty or malformed user message in thread {thread_id}")
                    continue
                user_messages.append(content)
                sd = await extract_symptoms_comprehensive(content)
                all_symptoms.extend(sd["symptoms"])
                max_severity = max(max_severity, sd["severity"])
            elif msg.role == "assistant":
                assistant_count += 1

        # Deduplicate symptoms
        deduped = []
        seen = set()
        for s in all_symptoms:
            sc = s.lower().strip()
            if sc not in seen:
                seen.add(sc)
                deduped.append(sc)

        return {
            "user_messages": user_messages,
            "assistant_responses": assistant_count,
            "all_symptoms": deduped,
            "max_severity": max_severity
        }

    except Exception as e:
        logger.error(f"Error getting thread context for {thread_id}: {e}")
        return {"user_messages": [], "assistant_responses": 0, "all_symptoms": [], "max_severity": 0}

async def detect_intent(description: str, thread_id: Optional[str] = None) -> Dict:
    """Use GPT-4o-mini to detect intent and extract symptom keywords"""
    if not openai_client:
        return {"intent": ["medical"], "symptoms": []}
    
    try:
        has_prior_symptoms = False
        if thread_id and await validate_thread(thread_id):
            context = await get_thread_context(thread_id)
            if context["all_symptoms"]:
                has_prior_symptoms = True
                logger.info(f"Thread {thread_id} has prior symptoms; forcing medical intent.")

        prompt = (
            "You are a medical triage assistant. Analyze the input text and classify the primary intent(s) "
            "as one or more of: 'medical' (describes symptoms), 'contextual' (provides follow-up details), "
            "or 'non_medical' (general inquiries like greetings). Also extract entities: symptoms (e.g., 'nausea', 'pain'). "
            "If prior symptoms exist in the thread context, always include 'medical' in the intent. "
            "Return a JSON object with:\n"
            "- 'intent': List of intents (e.g., ['medical', 'contextual'])\n"
            "- 'symptoms': List of symptom strings.\n\n"
            "Example:\n"
            "Input: 'I have nausea'\n"
            "Output: {'intent': ['medical'], 'symptoms': ['nausea']}"
        )

        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": description}
            ],
            temperature=0.2,
            max_tokens=200
        )

        try:
            result = json.loads(response.choices[0].message.content.strip())
            intents = result.get("intent", ["medical"])
            symptoms = result.get("symptoms", [])
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"JSON parsing error in intent detection: {e}. Falling back to keyword method.")
            intents = ["medical"]
            symptoms = []

        if has_prior_symptoms and "medical" not in intents:
            intents.append("medical")

        if not isinstance(intents, list):
            intents = [intents]
        
        valid_intents = {"medical", "contextual", "non_medical"}
        intents = [i for i in intents if i in valid_intents]
        if not intents:
            intents = ["medical"]

        return {"intent": intents, "symptoms": symptoms}

    except Exception as e:
        logger.error(f"Intent detection error: {e}. Using fallback keyword-based approach.")
        return {"intent": ["medical"], "symptoms": []}

async def extract_symptoms_comprehensive(description: str) -> Dict:
    """GPT-based comprehensive symptom extraction"""
    if not openai_client:
        return {"symptoms": [], "duration": None, "severity": 0}
    
    try:
        description_lower = description.lower()

        # Extract duration and severity
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
        
        # Check for numeric pain scale
        m = re.search(r"pain\s+(\d+)/10", description_lower)
        if m:
            severity = int(m.group(1))

        # GPT symptom extraction
        prompt = (
            "You are a medical symptom extraction specialist. Extract ONLY specific, clinically relevant symptoms from the text. "
            "Do NOT include vague terms like 'pain' without context or duration descriptions. "
            "Return a JSON array of symptom strings. Be precise and map synonyms (e.g., 'leg ache' → 'leg pain'). "
            "Example output: [\"chest pain\", \"leg pain\", \"headache\"].\n\n"
            f"Text: \"{description}\""
        )

        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Extract symptoms now."}
            ],
            temperature=0
        )

        try:
            gpt_symptoms = json.loads(response.choices[0].message.content.strip())
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
            logger.error(f"GPT returned non-JSON or invalid format: {e}. No symptoms extracted.")
            unique_symptoms = []

        logger.info(f"Extracted symptoms (GPT): {unique_symptoms}, Duration: {duration}, Severity: {severity}")
        return {"symptoms": unique_symptoms, "duration": duration, "severity": severity}

    except Exception as e:
        logger.error(f"Error extracting symptoms: {e}")
        return {"symptoms": [], "duration": None, "severity": 0}

def is_red_flag(full_text: str, severity: int = 0) -> bool:
    """Check if text contains any red-flag keywords"""
    lt = full_text.lower()
    for rf in RED_FLAGS:
        if rf in lt:
            logger.info(f"Red flag detected: {rf}")
            return True
    if severity >= 8 and any(term in lt for term in ["abdominal pain", "intermenstrual bleeding"]):
        logger.info(f"Red flag detected: high severity {severity} with critical symptom.")
        return True
    return False

async def should_query_pinecone_database(context: Dict) -> bool:
    """Decide whether to query Pinecone"""
    all_symptoms = context.get("all_symptoms", [])
    symptom_count = len(all_symptoms)
    max_severity = context.get("max_severity", 0)
    full_text = " ".join(context.get("user_messages", [])).lower()

    if is_red_flag(full_text, max_severity):
        logger.info("Emergency red flag → querying Pinecone.")
        return True

    if symptom_count >= MIN_SYMPTOMS_FOR_PINECONE:
        logger.info(f"Sufficient symptoms ({symptom_count}) → querying Pinecone.")
        return True

    condition_keywords = [
        "what might be", "what could be", "what is", "infection", "condition",
        "disease", "diagnosis", "what's wrong", "what do i have"
    ]
    if any(kw in full_text for kw in condition_keywords):
        logger.info("User explicitly asked for condition identification → querying Pinecone.")
        return True

    logger.info(f"Not querying Pinecone: only {symptom_count} symptoms, no explicit condition request.")
    return False

async def get_embeddings(texts: List[str], model: str = EMBEDDING_MODEL) -> Optional[List[List[float]]]:
    """Get embeddings from OpenAI"""
    if not openai_client:
        return None
    
    for attempt in range(MAX_RETRIES):
        try:
            response = await openai_client.embeddings.create(input=texts, model=model)
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Embedding attempt {attempt+1}/{MAX_RETRIES} failed: {e}")
            if attempt == MAX_RETRIES - 1:
                return None
    return None

async def query_index(query_text: str, symptoms: List[str], context: Dict, top_k: int = 50) -> List[Dict]:
    """Run a Pinecone vector query"""
    if not index:
        logger.warning("Pinecone index not available")
        return []
    
    query_embedding = await get_embeddings([query_text])
    if not query_embedding:
        logger.error("Failed to generate query embedding.")
        return []

    try:
        response = index.query(
            vector=query_embedding[0],
            top_k=top_k,
            include_metadata=True
        )
        matches = response.get("matches", [])
        logger.info(f"Pinecone returned {len(matches)} raw matches for '{query_text[:50]}...'")

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

async def generate_detailed_condition_description(condition_name: str, user_symptoms: List[str]) -> str:
    """Generate description for a medical condition"""
    if not openai_client:
        fallback = {
            "gastritis": "Gastritis is inflammation of the stomach lining that can cause abdominal pain and nausea.",
            "appendicitis": "Appendicitis is inflammation of the appendix, causing sharp abdominal pain.",
            "common cold": "A viral infection affecting the upper respiratory tract."
        }
        return fallback.get(condition_name.lower(), f"{condition_name} may be related to your symptoms.")
    
    try:
        symptoms_text = ", ".join(user_symptoms)
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
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
            ],
            temperature=0.3,
            max_tokens=150
        )
        desc = response.choices[0].message.content.strip()
        if desc and len(desc) > 20:
            return desc
    except Exception as e:
        logger.error(f"Error generating description for {condition_name}: {e}")
    
    return f"{condition_name} may be related to your symptoms. Please consult a healthcare professional for details."

async def rank_conditions(matches: List[Dict], symptoms: List[str], context: Dict) -> List[Dict]:
    """Rank and describe conditions from Pinecone matches"""
    try:
        condition_data = []
        for match in matches:
            disease = match["metadata"].get("disease", "Unknown").title()
            score = match.get("score", 0)
            desc = await generate_detailed_condition_description(disease, symptoms)
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

async def generate_follow_up_questions(context: Dict) -> List[str]:
    """Generate follow-up questions based on symptoms"""
    if not openai_client:
        return ["How long have you had these symptoms?", "Do you have any other symptoms?"]
    
    try:
        symptoms = context.get("all_symptoms", [])
        symptoms_text = ", ".join(symptoms) if symptoms else "no symptoms reported yet"
        
        prompt = (
            "You are a medical triage assistant. The user has reported the following symptoms: "
            f"{symptoms_text}. Generate 2-3 targeted follow-up questions to clarify these symptoms "
            "or uncover related ones. Avoid asking about symptoms already reported. "
            "Return a JSON object with a 'questions' key containing a list of question strings.\n"
            "Example: {'questions': ['Do you have any nausea or vomiting?', 'Are you experiencing any fever?']}"
        )

        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Generate follow-up questions."}
            ],
            temperature=0.3,
            max_tokens=150
        )

        try:
            result = json.loads(response.choices[0].message.content.strip())
            questions = result.get("questions", [])
        except (json.JSONDecodeError, KeyError):
            logger.error("JSON parsing error in follow-up questions. Using fallback.")
            questions = ["Do you have any other symptoms?", "When did your symptoms start?"]

        return questions

    except Exception as e:
        logger.error(f"Error generating follow-up questions: {e}")
        return ["Do you have any other symptoms?", "When did your symptoms start?"]

async def triage(request: TriageRequest) -> TriageResponse:
    """Main triage function - advanced version of your testtriage.py"""
    try:
        description = request.description.strip()
        thread_id = request.thread_id
        
        logger.info(f"=== TRIAGE REQUEST === Description: '{description[:50]}...', Thread: {thread_id}")

        # Validate or create thread
        if not thread_id or not await validate_thread(thread_id):
            if openai_client:
                new_thread = await openai_client.beta.threads.create()
                thread_id = new_thread.id
                logger.info(f"Created new thread ID: {thread_id}")
            else:
                thread_id = str(uuid.uuid4())

        # Add user message to thread
        if openai_client:
            try:
                await openai_client.beta.threads.messages.create(
                    thread_id=thread_id,
                    role="user",
                    content=description
                )
            except Exception as e:
                logger.warning(f"Could not add message to thread: {e}")

        # Get thread context
        context = await get_thread_context(thread_id)
        intent_result = await detect_intent(description, thread_id)
        intents = intent_result["intent"]

        symptom_count = len(context["all_symptoms"])
        max_severity = context["max_severity"]
        should_query = await should_query_pinecone_database(context)
        is_emergency = is_red_flag(" ".join(context["user_messages"]), max_severity)

        logger.info((
            f"Intents: {intents}, Symptoms: {context['all_symptoms']}, "
            f"Count: {symptom_count}, Max severity: {max_severity}, "
            f"Should query: {should_query}, Emergency: {is_emergency}"
        ))

        # Generate response based on analysis
        if is_emergency:
            response_text = f"🚨 URGENT: Your symptoms suggest a medical emergency. Please call {NIGERIA_EMERGENCY_HOTLINE} immediately for emergency medical assistance."
            triage_type = "hospital"
            send_sos = True
            possible_conditions = [
                {
                    "name": "Emergency Medical Condition",
                    "description": "Your symptoms require immediate medical attention and emergency care.",
                    "file_citation": "emergency_protocols.json"
                }
            ]
            safety_measures = [
                f"Call {NIGERIA_EMERGENCY_HOTLINE} immediately",
                "Stay calm and don't panic",
                "Have someone stay with you if possible",
                "Prepare to provide your location to emergency services"
            ]
            follow_up_questions = []
        
        elif should_query and symptom_count >= MIN_SYMPTOMS_FOR_PINECONE:
            # Advanced analysis with Pinecone
            logger.info(f"Querying Pinecone with {symptom_count} symptoms")
            query_text = f"Symptoms: {', '.join(context['all_symptoms'])}"
            matches = await query_index(query_text, context['all_symptoms'], context)
            
            if matches:
                possible_conditions = await rank_conditions(matches, context['all_symptoms'], context)
            else:
                possible_conditions = [
                    {
                        "name": "Possible Medical Condition",
                        "description": "Your symptoms could be due to an infection, inflammation, or other medical condition that requires professional evaluation.",
                        "file_citation": "general_medical_knowledge.json"
                    }
                ]
            
            symptoms_text = ", ".join(context['all_symptoms'])
            response_text = (
                f"I'm sorry you're dealing with {symptoms_text}. Based on my analysis:\n\n"
                f"**Symptoms Identified:** {', '.join('• ' + s for s in context['all_symptoms'])}\n\n"
                f"**Why These Symptoms Happen Together:** Your symptoms often occur together because they may affect related body systems or have similar underlying causes.\n\n"
                f"**Next Steps:** Please see a healthcare professional for proper evaluation and treatment. "
                f"This is not a medical diagnosis, but rather guidance to help you understand your symptoms.\n\n"
                f"**Important:** If your symptoms worsen or you develop new concerning symptoms, seek medical attention promptly."
            )
            
            triage_type = "clinic" if possible_conditions else "pharmacy"
            send_sos = False
            safety_measures = [
                "Monitor your symptoms closely",
                "Stay hydrated and get adequate rest",
                "Seek medical attention if symptoms worsen",
                "Keep a record of when symptoms occur"
            ]
            follow_up_questions = await generate_follow_up_questions(context)
        
        else:
            # Basic response for fewer symptoms
            symptoms_text = ", ".join(context['all_symptoms']) if context['all_symptoms'] else "your symptoms"
            response_text = (
                f"I understand you're experiencing {symptoms_text}. To provide better guidance, "
                f"I'd like to understand more about what you're experiencing."
            )
            
            possible_conditions = []
            triage_type = "clinic"
            send_sos = False
            safety_measures = [
                "Monitor your symptoms",
                "Stay hydrated",
                "Rest as needed",
                "Seek medical attention if symptoms persist or worsen"
            ]
            follow_up_questions = await generate_follow_up_questions(context)

        # Create response
        response = TriageResponse(
            text=response_text,
            possible_conditions=possible_conditions,
            safety_measures=safety_measures,
            triage={"type": triage_type, "location": "Unknown"},
            send_sos=send_sos,
            follow_up_questions=follow_up_questions,
            thread_id=thread_id,
            symptoms_count=symptom_count,
            should_query_pinecone=should_query
        )

        # Add assistant response to thread
        if openai_client:
            try:
                await openai_client.beta.threads.messages.create(
                    thread_id=thread_id,
                    role="assistant",
                    content=json.dumps(response.__dict__)
                )
            except Exception as e:
                logger.warning(f"Could not add assistant response to thread: {e}")

        return response

    except Exception as e:
        logger.error(f"Error in triage function: {e}")
        return TriageResponse(
            text=f"I'm experiencing a technical issue, but based on your symptoms, I recommend seeking medical attention. Call {NIGERIA_EMERGENCY_HOTLINE} if this is urgent.",
            possible_conditions=[],
            safety_measures=["Stay calm and rest", "Seek medical attention"],
            triage={"type": "hospital", "location": "Unknown"},
            send_sos=True,
            follow_up_questions=[],
            thread_id=thread_id or str(uuid.uuid4()),
            symptoms_count=0,
            should_query_pinecone=False
        )

# Health facts database
HEALTH_FACTS = {
    "general health": [
        {"title": "Stay Hydrated", "description": "Drink at least 8 glasses of water daily to maintain proper body functions and overall health."},
        {"title": "Get Quality Sleep", "description": "Adults need 7-9 hours of quality sleep per night for optimal health and wellbeing."},
        {"title": "Exercise Regularly", "description": "Aim for at least 150 minutes of moderate physical activity per week for cardiovascular health."},
        {"title": "Eat Balanced Meals", "description": "Include fruits, vegetables, whole grains, and lean proteins in your daily diet."},
        {"title": "Manage Stress", "description": "Practice stress-reduction techniques like meditation, deep breathing, or regular exercise."}
    ],
    "nutrition": [
        {"title": "Eat Rainbow Foods", "description": "Include colorful fruits and vegetables in your diet for diverse nutrients and vitamins."},
        {"title": "Limit Processed Foods", "description": "Reduce intake of processed and sugary foods to maintain optimal health."},
        {"title": "Portion Control", "description": "Practice mindful eating and appropriate portion sizes for better digestion."}
    ]
}

@app.route('/')
def home():
    return jsonify({
        "message": "HealthMate AI Backend API - Advanced Triage",
        "status": "running",
        "version": "2.0.0",
        "features": {
            "openai_available": OPENAI_AVAILABLE and bool(OPENAI_API_KEY),
            "pinecone_available": PINECONE_AVAILABLE and bool(PINECONE_API_KEY),
            "advanced_triage": True,
            "thread_management": True
        },
        "endpoints": [
            "/api/health/analyze - POST - Basic health analysis",
            "/api/health/languages - GET - Get supported languages", 
            "/api/health/facts - GET - Get health facts",
            "/api/health - GET - Health check",
            "/api/triage - POST - Advanced medical triage analysis"
        ]
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'HealthMate AI Backend API - Advanced',
        'environment': 'production',
        'capabilities': {
            'openai_api': OPENAI_AVAILABLE and bool(OPENAI_API_KEY),
            'pinecone_db': PINECONE_AVAILABLE and bool(PINECONE_API_KEY),
            'advanced_symptom_analysis': True,
            'emergency_detection': True,
            'thread_persistence': True
        }
    })

@app.route('/api/health/languages', methods=['GET'])
def get_supported_languages():
    return jsonify({
        'success': True,
        'supported_languages': SUPPORTED_LANGUAGES
    })

@app.route('/api/health/analyze', methods=['POST'])
def analyze_health_query():
    """Basic health analysis endpoint (fallback)"""
    try:
        if not request.is_json:
            return jsonify({'success': False, 'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'success': False, 'error': 'No message provided'}), 400
        
        user_message = data['message']
        if not user_message or not user_message.strip():
            return jsonify({'success': False, 'error': 'Message cannot be empty'}), 400
        
        # Simple symptom extraction for fallback
        symptoms = []
        text_lower = user_message.lower()
        
        symptom_keywords = {
            'headache': ['headache', 'head pain', 'head hurt', 'migraine'],
            'fever': ['fever', 'temperature', 'hot', 'chills'],
            'cough': ['cough', 'coughing'],
            'stomach pain': ['stomach pain', 'belly pain', 'abdominal pain'],
            'nausea': ['nausea', 'nauseous', 'sick'],
            'fatigue': ['tired', 'fatigue', 'exhausted', 'weak']
        }
        
        for symptom, keywords in symptom_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                symptoms.append(symptom)
        
        # Generate response
        if symptoms:
            symptom_text = ", ".join(symptoms)
            response = f"I understand you're experiencing {symptom_text}. "
        else:
            response = "I understand you have health concerns. "
            
        response += ("It's important to monitor your symptoms carefully. "
                    "If your symptoms are severe, getting worse, or you're concerned, "
                    "please consult with a healthcare professional or visit a clinic. "
                    "For emergencies, call 112 (Nigeria emergency number). "
                    "Stay hydrated, get rest, and take care of yourself.")
        
        return jsonify({
            "success": True,
            "detected_language": "English",
            "language_code": "en",
            "original_text": user_message,
            "english_translation": None,
            "health_analysis": {
                "symptoms": symptoms,
                "body_parts": [],
                "time_expressions": [],
                "medications": []
            },
            "response": response
        })
    
    except Exception as e:
        logger.error(f"Error in analyze_health_query: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': 'Please try again later'
        }), 500

@app.route('/api/triage', methods=['POST'])
def triage_endpoint():
    """
    Advanced triage analysis endpoint - Full testtriage.py implementation
    """
    try:
        if not request.is_json:
            return jsonify({'success': False, 'error': 'Request must be JSON'}), 400
        
        data = request.json
        if not data or 'description' not in data:
            return jsonify({'success': False, 'error': 'No description provided'}), 400
        
        # Check if advanced triage is available
        if not (OPENAI_AVAILABLE and OPENAI_API_KEY):
            return jsonify({
                'success': False,
                'error': 'Advanced triage service not available - OpenAI API key missing'
            }), 503
        
        # Create triage request
        triage_req = TriageRequest(
            description=data['description'],
            thread_id=data.get('thread_id')
        )
        
        logger.info(f"Processing advanced triage request: {data['description'][:50]}...")
        
        # Run the async triage function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(triage(triage_req))
        finally:
            loop.close()
        
        # Convert response to dict
        response_dict = {
            'success': True,
            'text': result.text,
            'possible_conditions': result.possible_conditions,
            'safety_measures': result.safety_measures,
            'triage': result.triage,
            'send_sos': result.send_sos,
            'follow_up_questions': result.follow_up_questions,
            'thread_id': result.thread_id,
            'symptoms_count': result.symptoms_count,
            'should_query_pinecone': result.should_query_pinecone
        }
        
        logger.info(f"Advanced triage completed successfully. Emergency: {result.send_sos}, Symptoms: {result.symptoms_count}")
        return jsonify(response_dict)
        
    except Exception as e:
        logger.error(f"Error in advanced triage endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/api/health/facts', methods=['GET'])
def get_health_facts():
    try:
        topic = request.args.get('topic', 'general health')
        count = min(int(request.args.get('count', 3)), 5)
        
        if topic.lower() in HEALTH_FACTS:
            facts = HEALTH_FACTS[topic.lower()][:count]
        else:
            facts = HEALTH_FACTS['general health'][:count]
        
        return jsonify({
            'success': True,
            'topic': topic,
            'facts': facts
        })
        
    except Exception as e:
        logger.error(f"Error in get_health_facts: {str(e)}")
        return jsonify({
            'success': True,
            'topic': 'general health',
            'facts': HEALTH_FACTS['general health'][:3]
        })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
