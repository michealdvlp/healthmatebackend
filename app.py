import os
import json
import logging
import asyncio
from datetime import datetime
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

# Import OpenAI for triage
try:
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")
    OPENAI_AVAILABLE = True
    logger.info("OpenAI client initialized successfully")
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not available")

# Supported languages
SUPPORTED_LANGUAGES = {
    "en": "English",
    "ig": "Igbo",
    "yo": "Yoruba", 
    "ha": "Hausa",
    "pcm": "Nigerian Pidgin"
}

# Emergency red flags
RED_FLAGS = [
    "chest pain", "difficulty breathing", "shortness of breath", "can't breathe",
    "severe bleeding", "profuse bleeding", "uncontrolled bleeding",
    "loss of consciousness", "unconscious", "passed out",
    "severe headache", "worst headache ever", "sudden severe headache",
    "stroke symptoms", "face drooping", "slurred speech", "arm weakness",
    "seizure", "convulsions", "fitting",
    "severe allergic reaction", "anaphylaxis", "can't swallow",
    "compound fracture", "bone sticking out",
    "severe burns", "third degree burns",
    "poisoning", "overdose",
    "severe abdominal pain", "appendicitis symptoms",
    "heart attack", "crushing chest pain"
]

# Simple thread storage (in production, use Redis or database)
conversation_threads = {}

def generate_thread_id():
    """Generate a simple thread ID"""
    import uuid
    return str(uuid.uuid4())

def detect_emergency(text):
    """Detect emergency situations from text"""
    text_lower = text.lower()
    for flag in RED_FLAGS:
        if flag in text_lower:
            logger.info(f"Emergency detected: {flag}")
            return True
    return False

def extract_symptoms(text):
    """Extract symptoms from text"""
    text_lower = text.lower()
    symptoms = []
    
    # Common symptoms
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
        'shortness of breath': ['short of breath', 'difficulty breathing', 'can\'t breathe']
    }
    
    for symptom, keywords in symptom_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            symptoms.append(symptom)
    
    return symptoms

async def analyze_with_openai(description, thread_id=None):
    """Use OpenAI for medical triage analysis"""
    try:
        # Get conversation history if thread exists
        context = ""
        if thread_id and thread_id in conversation_threads:
            history = conversation_threads[thread_id]
            context = "\n".join([f"User: {msg['user']}\nAssistant: {msg['assistant']}" 
                               for msg in history[-3:]])  # Last 3 exchanges
        
        # Construct prompt
        system_prompt = """You are a medical triage assistant. Analyze the patient's symptoms and provide:
1. Assessment of urgency (urgent/moderate/mild)
2. Possible conditions (max 3)
3. Safety measures (3-5 actionable items)
4. Follow-up questions (2-3 specific questions)

Respond in JSON format:
{
    "urgency": "urgent|moderate|mild",
    "possible_conditions": [{"name": "condition", "description": "explanation"}],
    "safety_measures": ["measure1", "measure2"],
    "follow_up_questions": ["question1", "question2"],
    "send_sos": true/false,
    "response_text": "Comprehensive response to patient"
}

For urgent cases, set send_sos to true and recommend immediate medical attention."""

        user_prompt = f"Context: {context}\n\nCurrent symptoms: {description}"
        
        # Call OpenAI
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        
        # Parse response
        result = json.loads(response.choices[0].message.content)
        
        # Store conversation
        if not thread_id:
            thread_id = generate_thread_id()
        
        if thread_id not in conversation_threads:
            conversation_threads[thread_id] = []
        
        conversation_threads[thread_id].append({
            "user": description,
            "assistant": result.get("response_text", ""),
            "timestamp": datetime.now().isoformat()
        })
        
        return result, thread_id
        
    except Exception as e:
        logger.error(f"OpenAI analysis failed: {e}")
        return None, thread_id

def create_fallback_response(description, symptoms, is_emergency):
    """Create fallback response when OpenAI is not available"""
    if is_emergency:
        return {
            "urgency": "urgent",
            "possible_conditions": [
                {"name": "Emergency Condition", "description": "Your symptoms require immediate medical attention"}
            ],
            "safety_measures": [
                "Call 112 immediately",
                "Stay calm and don't panic",
                "Have someone stay with you if possible",
                "Prepare to provide your location to emergency services"
            ],
            "follow_up_questions": [],
            "send_sos": True,
            "response_text": "🚨 URGENT: Your symptoms suggest a medical emergency. Please call 112 immediately for emergency medical assistance."
        }
    
    symptom_text = ", ".join(symptoms) if symptoms else "your symptoms"
    
    return {
        "urgency": "moderate" if symptoms else "mild",
        "possible_conditions": [
            {"name": "Common Health Issue", "description": f"Based on {symptom_text}, this could be a common health concern that should be evaluated by a healthcare provider."}
        ],
        "safety_measures": [
            "Monitor your symptoms closely",
            "Stay hydrated and get adequate rest",
            "Seek medical attention if symptoms worsen",
            "Avoid self-medication without consulting a doctor"
        ],
        "follow_up_questions": [
            "How long have you been experiencing these symptoms?",
            "Have you tried any treatments or medications?",
            "Are there any other symptoms you haven't mentioned?"
        ],
        "send_sos": False,
        "response_text": f"I understand you're experiencing {symptom_text}. While I can provide general guidance, it's important to consult with a healthcare professional for proper evaluation and treatment. Monitor your symptoms and seek medical attention if they worsen or if you're concerned."
    }

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
        "message": "HealthMate AI Backend API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": [
            "/api/health/analyze - POST - Analyze health queries",
            "/api/health/languages - GET - Get supported languages", 
            "/api/health/facts - GET - Get health facts",
            "/api/health - GET - Health check",
            "/api/triage - POST - Advanced triage analysis"
        ]
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'HealthMate AI Backend API',
        'environment': 'production',
        'openai_available': OPENAI_AVAILABLE
    })

@app.route('/api/health/languages', methods=['GET'])
def get_supported_languages():
    return jsonify({
        'success': True,
        'supported_languages': SUPPORTED_LANGUAGES
    })

@app.route('/api/health/analyze', methods=['POST'])
def analyze_health_query():
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
        
        # Generate basic response
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
    Simplified triage analysis endpoint
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
        
        # Extract symptoms and detect emergency
        symptoms = extract_symptoms(description)
        is_emergency = detect_emergency(description)
        
        # Try OpenAI analysis first
        ai_result = None
        if OPENAI_AVAILABLE:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                ai_result, thread_id = loop.run_until_complete(
                    analyze_with_openai(description, thread_id)
                )
                loop.close()
            except Exception as e:
                logger.error(f"OpenAI analysis failed: {e}")
        
        # Use AI result or fallback
        if ai_result:
            result = ai_result
        else:
            result = create_fallback_response(description, symptoms, is_emergency)
            if not thread_id:
                thread_id = generate_thread_id()
        
        # Format response
        response = {
            "success": True,
            "text": result.get("response_text", ""),
            "possible_conditions": result.get("possible_conditions", []),
            "safety_measures": result.get("safety_measures", []),
            "triage": {
                "type": "hospital" if result.get("urgency") == "urgent" else 
                        "clinic" if result.get("urgency") == "moderate" else "pharmacy",
                "location": "Unknown"
            },
            "send_sos": result.get("send_sos", False),
            "follow_up_questions": result.get("follow_up_questions", []),
            "thread_id": thread_id,
            "symptoms_count": len(symptoms),
            "should_query_pinecone": len(symptoms) >= 2
        }
        
        logger.info(f"Triage completed. Urgency: {result.get('urgency')}, Emergency: {is_emergency}")
        return jsonify(response)
        
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
