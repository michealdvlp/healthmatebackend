import os
import sys
import json
import logging
import asyncio
import uuid
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

# Check if we can import the secure testtriage
TRIAGE_AVAILABLE = False
try:
    # Import the secure testtriage functions
    from services.secure_testtriage import (
        triage as secure_triage,
        TriageRequest,
        TriageResponse
    )
    TRIAGE_AVAILABLE = True
    logger.info("✅ Secure triage service imported successfully")
except ImportError as e:
    logger.warning(f"⚠️  Could not import secure triage service: {e}")
    logger.info("Will use basic triage fallback")

# Alternative: Try to import OpenAI directly for basic functionality
OPENAI_AVAILABLE = False
try:
    import openai
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = bool(os.getenv("OPENAI_API_KEY"))
    if OPENAI_AVAILABLE:
        openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        logger.info("✅ OpenAI client available")
except ImportError:
    logger.warning("⚠️  OpenAI not available")

# Supported languages
SUPPORTED_LANGUAGES = {
    "en": "English",
    "ig": "Igbo",
    "yo": "Yoruba", 
    "ha": "Hausa",
    "pcm": "Nigerian Pidgin"
}

# Emergency red flags for basic fallback
RED_FLAGS = [
    "crushing chest pain", "difficulty breathing", "shortness of breath",
    "severe bleeding", "loss of consciousness", "severe headache",
    "stroke symptoms", "seizure", "severe allergic reaction"
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

def extract_basic_symptoms(text):
    """Basic symptom extraction for fallback"""
    text_lower = text.lower()
    symptoms = []
    
    symptom_keywords = {
        'headache': ['headache', 'head pain', 'head hurt', 'migraine'],
        'fever': ['fever', 'temperature', 'hot', 'chills'],
        'cough': ['cough', 'coughing'],
        'stomach pain': ['stomach pain', 'belly pain', 'abdominal pain'],
        'nausea': ['nausea', 'nauseous', 'sick'],
        'fatigue': ['tired', 'fatigue', 'exhausted', 'weak'],
        'chest pain': ['chest pain', 'chest hurt', 'chest pressure'],
        'shortness of breath': ['short of breath', 'difficulty breathing', 'can\'t breathe']
    }
    
    for symptom, keywords in symptom_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            symptoms.append(symptom)
    
    return symptoms

def detect_emergency_basic(text):
    """Basic emergency detection"""
    text_lower = text.lower()
    for flag in RED_FLAGS:
        if flag in text_lower:
            return True
    return False

async def basic_openai_analysis(description):
    """Basic OpenAI analysis when full triage isn't available"""
    if not OPENAI_AVAILABLE:
        return None
    
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a medical triage assistant. Analyze the symptoms and provide a helpful response. Be empathetic and recommend appropriate medical care."
                },
                {
                    "role": "user",
                    "content": f"Patient describes: {description}"
                }
            ],
            temperature=0.3,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"OpenAI analysis failed: {e}")
        return None

def create_basic_response(description, symptoms, is_emergency):
    """Create a basic triage response"""
    if is_emergency:
        return {
            "text": "🚨 URGENT: Your symptoms suggest a medical emergency. Please call 112 immediately for emergency medical assistance.",
            "possible_conditions": [
                {"name": "Emergency Medical Condition", "description": "Your symptoms require immediate medical attention."}
            ],
            "safety_measures": [
                "Call 112 immediately",
                "Stay calm and don't panic",
                "Have someone stay with you if possible"
            ],
            "triage": {"type": "hospital", "location": "Unknown"},
            "send_sos": True,
            "follow_up_questions": [],
            "symptoms_count": len(symptoms),
            "should_query_pinecone": False
        }
    
    symptom_text = ", ".join(symptoms) if symptoms else "your symptoms"
    return {
        "text": f"I understand you're experiencing {symptom_text}. It's important to monitor your symptoms and seek medical attention if they worsen or if you're concerned. For non-emergency medical concerns, consider visiting a clinic.",
        "possible_conditions": [
            {"name": "Common Health Issue", "description": f"Based on {symptom_text}, this could be a health concern that should be evaluated by a healthcare provider."}
        ],
        "safety_measures": [
            "Monitor your symptoms closely",
            "Stay hydrated and get adequate rest",
            "Seek medical attention if symptoms worsen"
        ],
        "triage": {"type": "clinic" if symptoms else "pharmacy", "location": "Unknown"},
        "send_sos": False,
        "follow_up_questions": [
            "How long have you been experiencing these symptoms?",
            "Have you tried any treatments?"
        ],
        "symptoms_count": len(symptoms),
        "should_query_pinecone": len(symptoms) >= 2
    }

@app.route('/')
def home():
    return jsonify({
        "message": "HealthMate AI Backend API - Secure Version",
        "status": "running",
        "version": "2.0.0",
        "security": "Environment variables secured",
        "features": {
            "advanced_triage": TRIAGE_AVAILABLE,
            "openai_available": OPENAI_AVAILABLE,
            "basic_fallback": True
        },
        "endpoints": [
            "/api/health/analyze - POST - Basic health analysis",
            "/api/health/languages - GET - Get supported languages", 
            "/api/health/facts - GET - Get health facts",
            "/api/health - GET - Health check",
            "/api/triage - POST - Advanced medical triage analysis",
            "/api/config - GET - Configuration status"
        ]
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'HealthMate AI Backend API - Secure',
        'environment': 'production',
        'timestamp': datetime.now().isoformat(),
        'capabilities': {
            'advanced_triage': TRIAGE_AVAILABLE,
            'openai_api': OPENAI_AVAILABLE,
            'basic_triage': True,
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
            'advanced_triage': TRIAGE_AVAILABLE,
            'openai_client': OPENAI_AVAILABLE,
            'fallback_available': True
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
        
        # Extract symptoms and detect emergency
        symptoms = extract_basic_symptoms(user_message)
        
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
    Advanced triage analysis endpoint - Uses secure testtriage if available
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
        
        # Try advanced triage first
        if TRIAGE_AVAILABLE:
            try:
                # Create triage request
                triage_req = TriageRequest(
                    description=description,
                    thread_id=thread_id
                )
                
                # Run the async secure triage function
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(secure_triage(triage_req))
                finally:
                    loop.close()
                
                # Convert response to dict
                if hasattr(result, '__dict__'):
                    response_dict = result.__dict__
                else:
                    response_dict = {
                        'text': result.get('text', ''),
                        'possible_conditions': result.get('possible_conditions', []),
                        'safety_measures': result.get('safety_measures', []),
                        'triage': result.get('triage', {}),
                        'send_sos': result.get('send_sos', False),
                        'follow_up_questions': result.get('follow_up_questions', []),
                        'thread_id': result.get('thread_id', ''),
                        'symptoms_count': result.get('symptoms_count', 0),
                        'should_query_pinecone': result.get('should_query_pinecone', False)
                    }
                
                logger.info(f"Advanced triage completed successfully")
                return jsonify({
                    'success': True,
                    **response_dict
                })
                
            except Exception as e:
                logger.error(f"Advanced triage failed: {e}")
                logger.info("Falling back to basic triage")
        
        # Fallback to basic triage
        symptoms = extract_basic_symptoms(description)
        is_emergency = detect_emergency_basic(description)
        
        # Try OpenAI enhancement if available
        ai_response = None
        if OPENAI_AVAILABLE:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    ai_response = loop.run_until_complete(basic_openai_analysis(description))
                finally:
                    loop.close()
            except Exception as e:
                logger.error(f"OpenAI enhancement failed: {e}")
        
        # Create basic response
        basic_resp = create_basic_response(description, symptoms, is_emergency)
        
        # Enhance with AI response if available
        if ai_response:
            basic_resp['text'] = ai_response
        
        # Generate thread ID if not provided
        if not thread_id:
            thread_id = str(uuid.uuid4())
        basic_resp['thread_id'] = thread_id
        
        logger.info(f"Basic triage completed. Emergency: {is_emergency}, Symptoms: {len(symptoms)}")
        return jsonify({
            'success': True,
            **basic_resp
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
