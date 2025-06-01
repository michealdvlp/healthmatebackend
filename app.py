import os
import json
import logging
import asyncio
import uuid
import re
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, origins="*")

# Try to import OpenAI with version compatibility
OPENAI_AVAILABLE = False
openai_client = None
try:
    import openai
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        # Use the synchronous client to avoid AsyncClient issues on Vercel
        openai.api_key = api_key
        OPENAI_AVAILABLE = True
        logger.info("✅ OpenAI configured successfully")
    else:
        logger.warning("⚠️ OpenAI API key not provided")
except ImportError as e:
    logger.warning(f"⚠️ OpenAI not available: {e}")

# Try to import Pinecone
PINECONE_AVAILABLE = False
pc = None
index = None
try:
    from pinecone import Pinecone
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "triage-index")
    if api_key:
        pc = Pinecone(api_key=api_key)
        try:
            index = pc.Index(index_name)
            PINECONE_AVAILABLE = True
            logger.info(f"✅ Pinecone configured successfully with index: {index_name}")
        except Exception as e:
            logger.warning(f"⚠️ Pinecone index '{index_name}' not accessible: {e}")
    else:
        logger.warning("⚠️ Pinecone API key not provided")
except ImportError as e:
    logger.warning(f"⚠️ Pinecone not available: {e}")

# Configuration
SUPPORTED_LANGUAGES = {
    "en": "English",
    "ig": "Igbo",
    "yo": "Yoruba", 
    "ha": "Hausa",
    "pcm": "Nigerian Pidgin"
}

# Emergency red flags
RED_FLAGS = [
    "crushing chest pain", "difficulty breathing", "shortness of breath",
    "severe bleeding", "profuse bleeding", "loss of consciousness",
    "severe headache", "stroke symptoms", "seizure", "severe allergic reaction",
    "anaphylaxis", "bullet wound", "gunshot", "head trauma", "neck trauma"
]

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

# Simple conversation storage (in production, use Redis or database)
conversation_threads = {}

def extract_symptoms(text):
    """Extract symptoms from text using keyword matching"""
    text_lower = text.lower()
    symptoms = []
    
    symptom_keywords = {
        'headache': ['headache', 'head pain', 'head hurt', 'migraine'],
        'fever': ['fever', 'temperature', 'hot', 'chills', 'burning up'],
        'cough': ['cough', 'coughing', 'hacking'],
        'stomach pain': ['stomach pain', 'belly pain', 'abdominal pain', 'tummy ache'],
        'nausea': ['nausea', 'nauseous', 'sick to stomach', 'queasy'],
        'vomiting': ['vomiting', 'throwing up', 'vomit', 'puking'],
        'diarrhea': ['diarrhea', 'loose stool', 'runny stool'],
        'fatigue': ['tired', 'fatigue', 'exhausted', 'weak', 'weakness'],
        'dizziness': ['dizzy', 'lightheaded', 'spinning'],
        'sore throat': ['sore throat', 'throat pain', 'throat hurt'],
        'chest pain': ['chest pain', 'chest hurt', 'chest pressure'],
        'shortness of breath': ['short of breath', 'difficulty breathing', 'can\'t breathe'],
        'leg pain': ['leg pain', 'leg hurt', 'leg ache'],
        'back pain': ['back pain', 'back hurt', 'back ache']
    }
    
    for symptom, keywords in symptom_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            symptoms.append(symptom)
    
    return symptoms

def detect_emergency(text):
    """Detect emergency situations"""
    text_lower = text.lower()
    for flag in RED_FLAGS:
        if flag in text_lower:
            logger.info(f"🚨 Emergency detected: {flag}")
            return True
    return False

def extract_duration(text):
    """Extract duration information from text"""
    text_lower = text.lower()
    duration_patterns = [
        r"for\s+(about\s+)?(\d+)\s+(day|hour|minute|week|month)s?",
        r"(since|started|began)\s+(yesterday|today|\d+\s+(day|hour|week|month)s?\s+ago)",
        r"(last|past)\s+(\d+)\s+(day|hour|week|month)s?"
    ]
    
    for pattern in duration_patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(0)
    return None

def assess_severity(text):
    """Assess symptom severity from descriptive words"""
    text_lower = text.lower()
    severity_scores = {
        "excruciating": 10, "unbearable": 10, "extremely painful": 10,
        "severe": 8, "very painful": 8, "crushing": 8, "sharp": 8,
        "painful": 6, "moderate": 6, "mild": 4, "slight": 2
    }
    
    max_severity = 0
    for term, score in severity_scores.items():
        if term in text_lower:
            max_severity = max(max_severity, score)
    
    # Check for numeric pain scale
    pain_match = re.search(r"pain\s+(\d+)/10", text_lower)
    if pain_match:
        max_severity = max(max_severity, int(pain_match.group(1)))
    
    return max_severity

def generate_openai_response(description, symptoms):
    """Generate response using OpenAI (synchronous to avoid Vercel issues)"""
    if not OPENAI_AVAILABLE:
        return None
    
    try:
        symptom_text = ", ".join(symptoms) if symptoms else "the reported symptoms"
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical triage assistant. Provide helpful, empathetic responses about health concerns. Always recommend consulting healthcare professionals for proper diagnosis and treatment. Be supportive but emphasize the importance of professional medical care."
                },
                {
                    "role": "user",
                    "content": f"A patient reports: {description}. The symptoms identified are: {symptom_text}. Please provide a helpful response."
                }
            ],
            temperature=0.3,
            max_tokens=300
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI request failed: {e}")
        return None

def generate_follow_up_questions(symptoms):
    """Generate relevant follow-up questions based on symptoms"""
    if not symptoms:
        return ["How long have you been experiencing these symptoms?", "Have you tried any treatments?"]
    
    questions = []
    
    if 'headache' in symptoms:
        questions.append("Is the headache throbbing or constant?")
        questions.append("Do you have sensitivity to light or sound?")
    
    if 'fever' in symptoms:
        questions.append("Have you measured your temperature?")
        questions.append("Are you experiencing chills or sweating?")
    
    if 'stomach pain' in symptoms:
        questions.append("Is the pain sharp, cramping, or dull?")
        questions.append("Have you eaten anything unusual recently?")
    
    if 'chest pain' in symptoms:
        questions.append("Does the pain worsen with breathing or movement?")
        questions.append("Do you have any shortness of breath?")
    
    # Default questions if no specific ones apply
    if not questions:
        questions = [
            "How long have you been experiencing these symptoms?",
            "Have the symptoms gotten better or worse over time?",
            "Have you tried any treatments or medications?"
        ]
    
    return questions[:3]  # Limit to 3 questions

def create_triage_response(description, symptoms, is_emergency, thread_id=None):
    """Create a comprehensive triage response"""
    if not thread_id:
        thread_id = str(uuid.uuid4())
    
    # Store conversation context
    if thread_id not in conversation_threads:
        conversation_threads[thread_id] = []
    conversation_threads[thread_id].append({
        "user": description,
        "symptoms": symptoms,
        "timestamp": datetime.now().isoformat()
    })
    
    # Generate AI response if available
    ai_response = generate_openai_response(description, symptoms)
    
    if is_emergency:
        response_text = ai_response or "🚨 URGENT: Your symptoms suggest a medical emergency. Please call 112 immediately for emergency medical assistance."
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
            "Call 112 immediately",
            "Stay calm and don't panic",
            "Have someone stay with you if possible",
            "Prepare to provide your location to emergency services"
        ]
        follow_up_questions = []
    else:
        # Generate appropriate response based on symptoms
        symptom_text = ", ".join(symptoms) if symptoms else "your symptoms"
        
        if ai_response:
            response_text = ai_response
        else:
            response_text = (
                f"I understand you're experiencing {symptom_text}. While I can provide general guidance, "
                f"it's important to consult with a healthcare professional for proper evaluation and treatment. "
                f"Monitor your symptoms and seek medical attention if they worsen or if you're concerned."
            )
        
        # Determine triage level
        if len(symptoms) >= 3 or any(s in ['chest pain', 'shortness of breath', 'severe headache'] for s in symptoms):
            triage_type = "clinic"
            possible_conditions = [
                {
                    "name": "Multiple Symptom Condition",
                    "description": f"Your combination of symptoms ({symptom_text}) suggests a condition that should be evaluated by a healthcare provider.",
                    "file_citation": "medical_knowledge_base.json"
                }
            ]
        else:
            triage_type = "pharmacy" if symptoms else "self-care"
            possible_conditions = []
        
        send_sos = False
        safety_measures = [
            "Monitor your symptoms closely",
            "Stay hydrated and get adequate rest",
            "Seek medical attention if symptoms worsen",
            "Keep a record of when symptoms occur"
        ]
        
        # Add symptom-specific safety measures
        if 'fever' in symptoms:
            safety_measures.append("Take your temperature regularly and stay cool")
        if 'stomach pain' in symptoms:
            safety_measures.append("Eat bland foods and avoid spicy or fatty foods")
        
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
        "should_query_pinecone": len(symptoms) >= 3
    }

@app.route('/')
def home():
    return jsonify({
        "message": "HealthMate AI Backend API - Stable Version",
        "status": "running",
        "version": "2.1.0",
        "features": {
            "openai_available": OPENAI_AVAILABLE,
            "pinecone_available": PINECONE_AVAILABLE,
            "emergency_detection": True,
            "symptom_analysis": True,
            "thread_management": True
        },
        "endpoints": [
            "/api/health/analyze - POST - Basic health analysis",
            "/api/health/languages - GET - Get supported languages", 
            "/api/health/facts - GET - Get health facts",
            "/api/health - GET - Health check",
            "/api/triage - POST - Medical triage analysis",
            "/api/config - GET - Configuration status"
        ]
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'HealthMate AI Backend API',
        'environment': 'production',
        'timestamp': datetime.now().isoformat(),
        'capabilities': {
            'openai_api': OPENAI_AVAILABLE,
            'pinecone_db': PINECONE_AVAILABLE,
            'emergency_detection': True,
            'symptom_analysis': True,
            'thread_persistence': True,
            'environment_variables_secured': True
        }
    })

@app.route('/api/config', methods=['GET'])
def get_config_status():
    """Check configuration status without exposing secrets"""
    return jsonify({
        'environment_variables': {
            'OPENAI_API_KEY': '✅ Set' if os.getenv('OPENAI_API_KEY') else '❌ Missing',
            'PINECONE_API_KEY': '✅ Set' if os.getenv('PINECONE_API_KEY') else '❌ Missing',
            'PINECONE_INDEX_NAME': os.getenv('PINECONE_INDEX_NAME', 'Not set'),
            'EMERGENCY_HOTLINE': os.getenv('EMERGENCY_HOTLINE', '112')
        },
        'service_status': {
            'openai_client': OPENAI_AVAILABLE,
            'pinecone_index': PINECONE_AVAILABLE,
            'basic_triage': True,
            'emergency_detection': True
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
    """Basic health analysis endpoint"""
    try:
        if not request.is_json:
            return jsonify({'success': False, 'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'success': False, 'error': 'No message provided'}), 400
        
        user_message = data['message']
        if not user_message or not user_message.strip():
            return jsonify({'success': False, 'error': 'Message cannot be empty'}), 400
        
        # Extract symptoms
        symptoms = extract_symptoms(user_message)
        
        # Generate response
        if symptoms:
            symptom_text = ", ".join(symptoms)
            response = f"I understand you're experiencing {symptom_text}. "
        else:
            response = "I understand you have health concerns. "
            
        response += ("It's important to monitor your symptoms carefully. "
                    "If your symptoms are severe, getting worse, or you're concerned, "
                    "please consult with a healthcare professional or visit a clinic. "
                    "For emergencies, call 112. Stay hydrated, get rest, and take care of yourself.")
        
        return jsonify({
            "success": True,
            "detected_language": "English",
            "language_code": "en",
            "original_text": user_message,
            "english_translation": None,
            "health_analysis": {
                "symptoms": symptoms,
                "body_parts": [],
                "time_expressions": [extract_duration(user_message)] if extract_duration(user_message) else [],
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
    Medical triage analysis endpoint with intelligent symptom assessment
    """
    try:
        if not request.is_json:
            return jsonify({'success': False, 'error': 'Request must be JSON'}), 400
        
        data = request.json
        if not data or 'description' not in data:
            return jsonify({'success': False, 'error': 'No description provided'}), 400
        
        description = data['description']
        thread_id = data.get('thread_id')
        
        logger.info(f"Processing triage request: {description[:50]}...")
        
        # Extract symptoms and assess emergency
        symptoms = extract_symptoms(description)
        is_emergency = detect_emergency(description)
        severity = assess_severity(description)
        duration = extract_duration(description)
        
        # Upgrade to emergency if high severity with concerning symptoms
        if severity >= 8 and any(s in ['chest pain', 'shortness of breath', 'headache'] for s in symptoms):
            is_emergency = True
            logger.info(f"🚨 Upgraded to emergency due to severity {severity} with concerning symptoms")
        
        # Create comprehensive response
        response = create_triage_response(description, symptoms, is_emergency, thread_id)
        
        logger.info(f"Triage completed. Emergency: {is_emergency}, Symptoms: {len(symptoms)}, Severity: {severity}")
        return jsonify({
            'success': True,
            **response
        })
        
    except Exception as e:
        logger.error(f"Error in triage endpoint: {str(e)}")
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
