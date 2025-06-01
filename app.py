import os
import json
import logging
import asyncio
import re
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

# Import required libraries
try:
    from openai import AsyncOpenAI
    from pinecone import Pinecone
    
    # Initialize OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
    OPENAI_AVAILABLE = bool(openai_client)
    
    # Initialize Pinecone
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "triage-index")
    
    if PINECONE_API_KEY:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(name=INDEX_NAME)
        PINECONE_AVAILABLE = True
        logger.info(f"Initialized Pinecone index '{INDEX_NAME}'")
    else:
        PINECONE_AVAILABLE = False
        logger.warning("Pinecone not available - no API key")
        
    logger.info("Advanced AI services initialized successfully")
except ImportError as e:
    OPENAI_AVAILABLE = False
    PINECONE_AVAILABLE = False
    logger.warning(f"Advanced AI services not available: {e}")

# Configuration
EMBEDDING_MODEL = "text-embedding-ada-002"
MAX_RETRIES = 3
MIN_SYMPTOMS_FOR_PINECONE = 3
NIGERIA_EMERGENCY_HOTLINE = "112"
PINECONE_SCORE_THRESHOLD = 0.8

# Emergency red flags
RED_FLAGS = [
    "bullet wound", "gunshot", "profuse bleeding", "crushing chest pain",
    "sudden shortness of breath", "loss of consciousness", "slurred speech", "seizure",
    "head trauma", "neck trauma", "high fever with stiff neck", "uncontrolled vomiting",
    "severe allergic reaction", "anaphylaxis", "difficulty breathing", "persistent cough with blood",
    "severe abdominal pain", "sudden vision loss", "chest tightness with sweating",
    "blood in urine", "inability to pass urine", "sharp abdominal pain", "intermenstrual bleeding",
    "chest pain", "shortness of breath", "can't breathe", "severe bleeding", "unconscious"
]

# Supported languages
SUPPORTED_LANGUAGES = {
    "en": "English",
    "ig": "Igbo",
    "yo": "Yoruba", 
    "ha": "Hausa",
    "pcm": "Nigerian Pidgin"
}

# ======================
# Advanced Triage Functions (from testtriage.py)
# ======================

async def validate_thread(thread_id: str) -> bool:
    """Check if an OpenAI thread ID is valid"""
    if not OPENAI_AVAILABLE:
        return False
    try:
        await openai_client.beta.threads.retrieve(thread_id=thread_id)
        logger.info(f"Thread {thread_id} is valid.")
        return True
    except Exception as e:
        logger.error(f"Couldn't validate thread_id {thread_id}: {e}")
        return False

async def create_thread() -> str:
    """Create a new OpenAI thread"""
    if not OPENAI_AVAILABLE:
        # Generate a simple UUID as fallback
        import uuid
        return str(uuid.uuid4())
    
    try:
        new_thread = await openai_client.beta.threads.create()
        logger.info(f"Created new OpenAI thread: {new_thread.id}")
        return new_thread.id
    except Exception as e:
        logger.error(f"Error creating thread: {e}")
        import uuid
        return str(uuid.uuid4())

async def extract_symptoms_comprehensive(description: str) -> Dict:
    """Extract symptoms using GPT-4o-mini"""
    if not OPENAI_AVAILABLE:
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
            logger.error(f"GPT returned non-JSON or invalid format: {e}")
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

async def get_embeddings(texts: List[str], model: str = EMBEDDING_MODEL) -> Optional[List[List[float]]]:
    """Get embeddings from OpenAI with retries"""
    if not OPENAI_AVAILABLE:
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

async def query_pinecone_index(query_text: str, symptoms: List[str], top_k: int = 50) -> List[Dict]:
    """Query Pinecone index for condition matches"""
    if not PINECONE_AVAILABLE:
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
        logger.info(f"Pinecone returned {len(matches)} raw matches")

        # Filter by threshold and deduplicate
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
    """Generate condition description using GPT"""
    if not OPENAI_AVAILABLE:
        fallback = {
            "gastritis": "Gastritis is inflammation of the stomach lining that can cause abdominal pain and nausea.",
            "appendicitis": "Appendicitis is inflammation of the appendix, causing sharp abdominal pain that requires urgent medical attention.",
            "urinary tract infection": "A urinary tract infection (UTI) occurs when bacteria enter your urinary system, causing pain during urination."
        }
        return fallback.get(condition_name.lower(), f"{condition_name} may be related to your symptoms. Please consult a healthcare professional.")
    
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
        else:
            return f"{condition_name} may be related to your symptoms. Please consult a healthcare professional for details."
    except Exception as e:
        logger.error(f"Error generating description for {condition_name}: {e}")
        return f"{condition_name} may be related to your symptoms. Please consult a healthcare professional."

async def rank_conditions(matches: List[Dict], symptoms: List[str]) -> List[Dict]:
    """Rank and format condition matches"""
    try:
        condition_data = []
        for match in matches:
            disease = match["metadata"].get("disease", "Unknown").title()
            score = match.get("score", 0)
            desc = await generate_detailed_condition_description(disease, symptoms)
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

async def advanced_triage_analysis(description: str, thread_id: Optional[str] = None) -> Dict:
    """Perform advanced triage analysis using the testtriage.py logic"""
    try:
        # Validate or create thread
        if not thread_id or not await validate_thread(thread_id):
            thread_id = await create_thread()

        # Add user message to thread if using OpenAI threads
        if OPENAI_AVAILABLE and await validate_thread(thread_id):
            try:
                await openai_client.beta.threads.messages.create(
                    thread_id=thread_id,
                    role="user",
                    content=description
                )
            except Exception as e:
                logger.error(f"Error adding message to thread: {e}")

        # Extract symptoms
        symptom_data = await extract_symptoms_comprehensive(description)
        symptoms = symptom_data["symptoms"]
        severity = symptom_data["severity"]
        
        # Check for emergency
        is_emergency = is_red_flag(description, severity)
        
        # Determine if we should query Pinecone
        should_query = is_emergency or len(symptoms) >= MIN_SYMPTOMS_FOR_PINECONE
        
        possible_conditions = []
        if should_query and PINECONE_AVAILABLE:
            query_text = f"Symptoms: {', '.join(symptoms)}"
            matches = await query_pinecone_index(query_text, symptoms)
            if matches:
                possible_conditions = await rank_conditions(matches, symptoms)

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
        else:
            if possible_conditions:
                conditions_text = f"Possible conditions include: {', '.join([c['name'] for c in possible_conditions[:3]])}"
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

        # Format response
        result = {
            "success": True,
            "text": response_text,
            "possible_conditions": possible_conditions,
            "safety_measures": safety_measures,
            "triage": {"type": triage_type, "location": "Unknown"},
            "send_sos": is_emergency,
            "follow_up_questions": [
                "Have your symptoms changed since they started?",
                "Are you experiencing any other symptoms?"
            ] if not is_emergency else [],
            "thread_id": thread_id,
            "symptoms_count": len(symptoms),
            "should_query_pinecone": should_query
        }

        # Add assistant response to thread if using OpenAI threads
        if OPENAI_AVAILABLE and await validate_thread(thread_id):
            try:
                await openai_client.beta.threads.messages.create(
                    thread_id=thread_id,
                    role="assistant",
                    content=json.dumps(result)
                )
            except Exception as e:
                logger.error(f"Error adding assistant response to thread: {e}")

        return result

    except Exception as e:
        logger.error(f"Error in advanced triage analysis: {e}")
        return {
            "success": False,
            "text": f"I'm experiencing a technical issue, but based on your symptoms, I recommend seeking medical attention. Call {NIGERIA_EMERGENCY_HOTLINE} if this is urgent.",
            "possible_conditions": [],
            "safety_measures": ["Stay calm and rest", "Seek medical attention"],
            "triage": {"type": "hospital", "location": "Unknown"},
            "send_sos": True,
            "follow_up_questions": [],
            "thread_id": thread_id or "unknown",
            "symptoms_count": 0,
            "should_query_pinecone": False
        }

# ======================
# Flask Routes
# ======================

@app.route('/')
def home():
    return jsonify({
        "message": "HealthMate AI Backend API - Advanced Version",
        "status": "running",
        "version": "3.0.0",
        "endpoints": [
            "/api/health/analyze - POST - Basic health analysis",
            "/api/health/languages - GET - Get supported languages", 
            "/api/health/facts - GET - Get health facts",
            "/api/health - GET - Health check",
            "/api/triage - POST - Advanced triage analysis",
            "/api/config - GET - Configuration status"
        ],
        "features": {
            "openai_available": OPENAI_AVAILABLE,
            "pinecone_available": PINECONE_AVAILABLE,
            "symptom_analysis": True,
            "emergency_detection": True,
            "thread_management": OPENAI_AVAILABLE
        }
    })

@app.route('/api/config', methods=['GET'])
def get_config():
    return jsonify({
        'openai_available': OPENAI_AVAILABLE,
        'pinecone_available': PINECONE_AVAILABLE,
        'embedding_model': EMBEDDING_MODEL if OPENAI_AVAILABLE else None,
        'pinecone_index': INDEX_NAME if PINECONE_AVAILABLE else None,
        'emergency_hotline': NIGERIA_EMERGENCY_HOTLINE
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'HealthMate AI Backend API',
        'environment': 'production',
        'openai_available': OPENAI_AVAILABLE,
        'pinecone_available': PINECONE_AVAILABLE
    })

@app.route('/api/health/languages', methods=['GET'])
def get_supported_languages():
    return jsonify({
        'success': True,
        'supported_languages': SUPPORTED_LANGUAGES
    })

@app.route('/api/triage', methods=['POST'])
def triage_endpoint():
    """Advanced triage analysis endpoint"""
    try:
        if not request.is_json:
            return jsonify({'success': False, 'error': 'Request must be JSON'}), 400
        
        data = request.json
        if not data or 'description' not in data:
            return jsonify({'success': False, 'error': 'No description provided'}), 400
        
        description = data['description'].strip()
        thread_id = data.get('thread_id')
        
        if not description:
            return jsonify({'success': False, 'error': 'Description cannot be empty'}), 400
        
        logger.info(f"Processing advanced triage request: {description[:50]}...")
        
        # Run advanced analysis
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(advanced_triage_analysis(description, thread_id))
            return jsonify(result)
        finally:
            loop.close()
        
    except Exception as e:
        logger.error(f"Error in triage endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': str(e)
        }), 500

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
        
        # Simple symptom extraction for basic analysis
        symptoms = []
        symptom_keywords = {
            'headache': ['headache', 'head pain', 'migraine'],
            'fever': ['fever', 'temperature', 'hot', 'chills'],
            'cough': ['cough', 'coughing'],
            'stomach pain': ['stomach pain', 'belly pain', 'abdominal pain'],
            'chest pain': ['chest pain', 'chest hurt'],
            'shortness of breath': ['short of breath', 'difficulty breathing']
        }
        
        user_lower = user_message.lower()
        for symptom, keywords in symptom_keywords.items():
            if any(keyword in user_lower for keyword in keywords):
                symptoms.append(symptom)
        
        # Generate response
        if symptoms:
            symptom_text = ", ".join(symptoms)
            response = f"I understand you're experiencing {symptom_text}. "
        else:
            response = "I understand you have health concerns. "
            
        response += ("For a more detailed analysis, please use our advanced triage endpoint. "
                    "If your symptoms are severe, getting worse, or you're concerned, "
                    "please consult with a healthcare professional or visit a clinic. "
                    f"For emergencies, call {NIGERIA_EMERGENCY_HOTLINE}.")
        
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

@app.route('/api/health/facts', methods=['GET'])
def get_health_facts():
    """Get health facts"""
    health_facts = {
        "general health": [
            {"title": "Stay Hydrated", "description": "Drink at least 8 glasses of water daily to maintain proper body functions."},
            {"title": "Get Quality Sleep", "description": "Adults need 7-9 hours of quality sleep per night for optimal health."},
            {"title": "Exercise Regularly", "description": "Aim for at least 150 minutes of moderate physical activity per week."},
        ]
    }
    
    try:
        topic = request.args.get('topic', 'general health')
        count = min(int(request.args.get('count', 3)), 5)
        
        facts = health_facts.get(topic.lower(), health_facts['general health'])[:count]
        
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
            'facts': health_facts['general health'][:3]
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
