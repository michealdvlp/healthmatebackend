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

# Conversation storage - In production, use Redis or database
conversation_threads = {}

class ConversationThread:
    def __init__(self, thread_id):
        self.thread_id = thread_id
        self.messages = []
        self.all_symptoms = []
        self.conversation_count = 0
        self.created_at = datetime.now()
        
    def add_message(self, user_message, assistant_response, extracted_symptoms):
        self.messages.append({
            "user": user_message,
            "assistant": assistant_response,
            "symptoms": extracted_symptoms,
            "timestamp": datetime.now().isoformat()
        })
        self.conversation_count += 1
        
        # Add new symptoms to accumulated list
        for symptom in extracted_symptoms:
            if symptom.lower() not in [s.lower() for s in self.all_symptoms]:
                self.all_symptoms.append(symptom.lower())
    
    def get_context_summary(self):
        return {
            "thread_id": self.thread_id,
            "conversation_count": self.conversation_count,
            "all_symptoms": self.all_symptoms,
            "symptoms_count": len(self.all_symptoms),
            "last_message": self.messages[-1] if self.messages else None,
            "created_at": self.created_at.isoformat()
        }

def generate_thread_id():
    """Generate a thread ID"""
    return f"thread_{str(uuid.uuid4()).replace('-', '')[:24]}"

def detect_emergency(text, symptoms_list=None):
    """Detect emergency situations from text and symptoms"""
    text_lower = text.lower()
    
    # Check for red flag keywords
    for flag in RED_FLAGS:
        if flag in text_lower:
            logger.info(f"Emergency detected: {flag}")
            return True
    
    # Check for emergency symptom combinations
    if symptoms_list:
        symptoms_lower = [s.lower() for s in symptoms_list]
        
        # Cardiac emergency pattern
        cardiac_symptoms = {"chest pain", "shortness of breath", "sweating", "nausea"}
        if len(set(symptoms_lower) & cardiac_symptoms) >= 3:
            logger.info("Emergency: Possible cardiac event detected")
            return True
            
        # Severe infection pattern
        infection_symptoms = {"fever", "severe pain", "difficulty breathing"}
        if len(set(symptoms_lower) & infection_symptoms) >= 2:
            severity_indicators = ["severe", "extreme", "unbearable", "can't"]
            if any(indicator in text_lower for indicator in severity_indicators):
                logger.info("Emergency: Severe infection pattern detected")
                return True
    
    return False

async def extract_symptoms_comprehensive(text):
    """Extract symptoms from text using GPT-4o-mini for dynamic symptom detection"""
    try:
        # Try GPT-4o-mini first for dynamic symptom extraction
        if OPENAI_AVAILABLE:
            prompt = f"""
            You are a medical symptom extraction specialist. Extract ALL specific symptoms and health complaints from this text.
            
            Rules:
            - Extract specific symptoms like "chest pain", "nausea", "headache", etc.
            - Include body parts when relevant (e.g., "stomach pain" not just "pain")
            - Normalize similar terms (e.g., "tummy ache" → "stomach pain")
            - Include severity descriptors if mentioned (e.g., "severe headache")
            - DO NOT include vague terms like "feeling unwell" or "not good"
            - DO NOT include duration information as symptoms
            
            Text: "{text}"
            
            Return ONLY a JSON array of symptom strings:
            ["symptom1", "symptom2", "symptom3"]
            """
            
            try:
                client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a medical symptom extraction AI. Return only valid JSON arrays of symptoms."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=150
                )
                
                content = response.choices[0].message.content.strip()
                
                # Clean up the response to extract JSON
                if content.startswith('```json'):
                    content = content.replace('```json', '').replace('```', '')
                elif content.startswith('```'):
                    content = content.replace('```', '')
                
                content = content.strip()
                
                # Parse JSON response
                gpt_symptoms = json.loads(content)
                
                if isinstance(gpt_symptoms, list):
                    # Normalize and clean symptoms
                    normalized_symptoms = []
                    for symptom in gpt_symptoms:
                        if isinstance(symptom, str) and len(symptom.strip()) > 2:
                            normalized_symptoms.append(symptom.lower().strip())
                    
                    logger.info(f"DEBUG → GPT-4o-mini extracted symptoms: {normalized_symptoms}")
                    
                    # Extract severity and duration using regex
                    severity_score = extract_severity(text)
                    duration = extract_duration(text)
                    
                    return {
                        'symptoms': normalized_symptoms,
                        'severity': severity_score,
                        'duration': duration
                    }
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing error from GPT-4o-mini: {e}. Content: {content}")
            except Exception as e:
                logger.error(f"GPT-4o-mini symptom extraction failed: {e}")
        
        # Fallback to regex-based extraction if GPT fails
        logger.warning("Using fallback regex-based symptom extraction")
        return extract_symptoms_fallback(text)
        
    except Exception as e:
        logger.error(f"Error in symptom extraction: {e}")
        return extract_symptoms_fallback(text)

def extract_severity(text):
    """Extract severity score from text"""
    text_lower = text.lower()
    severity_score = 0
    
    # Severity indicators
    severity_words = {
        'mild': 3, 'slight': 3, 'minor': 3, 'little': 3,
        'moderate': 5, 'noticeable': 5, 'some': 4,
        'severe': 8, 'bad': 7, 'terrible': 8, 'awful': 8, 'intense': 7,
        'extreme': 9, 'excruciating': 10, 'unbearable': 10, 'worst': 10
    }
    
    for word, score in severity_words.items():
        if word in text_lower:
            severity_score = max(severity_score, score)
    
    # Numeric pain scale (1-10)
    pain_scale_match = re.search(r'(\d+)\s*(?:/10|out of 10|scale)', text_lower)
    if pain_scale_match:
        try:
            score = int(pain_scale_match.group(1))
            if 1 <= score <= 10:
                severity_score = score
        except ValueError:
            pass
    
    return severity_score

def extract_duration(text):
    """Extract duration information from text"""
    text_lower = text.lower()
    
    # Duration patterns
    duration_patterns = [
        r'for\s+(\d+)\s+(minute|hour|day|week|month)s?',
        r'(\d+)\s+(minute|hour|day|week|month)s?\s+ago',
        r'since\s+(yesterday|last\s+night|this\s+morning|last\s+week)',
        r'started\s+(\d+)\s+(minute|hour|day|week)s?\s+ago',
        r'about\s+(\d+)\s+(minute|hour|day|week)s?',
        r'for\s+the\s+past\s+(\d+)\s+(minute|hour|day|week)s?'
    ]
    
    for pattern in duration_patterns:
        match = re.search(pattern, text_lower)
        if match:
            return match.group(0)
    
    return None

def extract_symptoms_fallback(text):
    """Fallback regex-based symptom extraction for when GPT fails"""
    text_lower = text.lower()
    symptoms = []
    
    # Basic symptom patterns - fewer but more reliable
    basic_patterns = {
        'chest pain': [r'chest\s+pain', r'pain\s+in.*chest', r'chest.*hurt'],
        'shortness of breath': [r'short.*breath', r'difficulty.*breath', r'can\'?t.*breath'],
        'headache': [r'head.*ache', r'head.*pain', r'migraine'],
        'nausea': [r'nausea', r'nauseous', r'feel.*sick', r'queasy'],
        'fever': [r'fever', r'temperature', r'feel.*hot', r'chills'],
        'sweating': [r'sweat', r'sweaty', r'perspir'],
        'dizziness': [r'dizz', r'lightheaded', r'spinning'],
        'stomach pain': [r'stomach.*pain', r'belly.*pain', r'abdominal.*pain'],
        'cough': [r'cough', r'hacking'],
        'fatigue': [r'tired', r'fatigue', r'exhausted', r'weak'],
        'vomiting': [r'vomit', r'throwing.*up', r'puking'],
        'diarrhea': [r'diarr', r'loose.*stool'],
    }
    
    # Extract symptoms using basic patterns
    for symptom, patterns in basic_patterns.items():
        for pattern in patterns:
            if re.search(pattern, text_lower):
                if symptom not in symptoms:
                    symptoms.append(symptom)
                break
    
    logger.info(f"DEBUG → Fallback extracted symptoms: {symptoms}")
    
    return {
        'symptoms': symptoms,
        'severity': extract_severity(text),
        'duration': extract_duration(text)
    }

def detect_intent(text, thread_context=None):
    """Detect user intent from text"""
    text_lower = text.lower()
    
    # Non-medical greetings and general questions
    non_medical_patterns = [
        r'\b(hi|hello|hey|good\s+(morning|afternoon|evening))\b',
        r'what\s+(can\s+you|do\s+you)\s+do',
        r'how\s+(are\s+you|does\s+this\s+work)',
        r'tell\s+me\s+about',
        r'^(thanks?|thank\s+you|ok|okay)$'
    ]
    
    for pattern in non_medical_patterns:
        if re.search(pattern, text_lower) and not thread_context:
            return 'non_medical'
    
    # Medical symptom indicators
    symptom_indicators = [
        r'\b(pain|hurt|ache|feel|symptom|sick|ill|unwell)\b',
        r'\b(fever|temperature|hot|cold|chills)\b',
        r'\b(nausea|dizzy|tired|weak|cough|headache)\b',
        r'\b(bleed|sweat|vomit|diarrhea)\b'
    ]
    
    has_symptoms = any(re.search(pattern, text_lower) for pattern in symptom_indicators)
    
    if thread_context and thread_context.all_symptoms:
        return 'contextual' if not has_symptoms else 'medical'
    
    return 'medical' if has_symptoms else 'contextual'

async def generate_openai_response(description, thread_context, intent):
    """Generate response using OpenAI with proper context"""
    try:
        # Build context string
        context_str = ""
        if thread_context and thread_context.conversation_count > 0:
            context_str = f"Previous symptoms mentioned: {', '.join(thread_context.all_symptoms)}\n"
            context_str += f"Conversation turn: {thread_context.conversation_count + 1}\n"
            if thread_context.messages:
                last_exchange = thread_context.messages[-1]
                context_str += f"Last user message: {last_exchange['user']}\n"
        
        # Create appropriate prompt based on intent and context
        if intent == 'non_medical':
            system_prompt = """You are a medical triage assistant for Nigeria. Introduce yourself and explain how you can help with health concerns. Be warm and professional."""
            user_prompt = description
        
        elif thread_context and len(thread_context.all_symptoms) < 3 and thread_context.conversation_count < 3:
            # Phase 1: Gathering symptoms
            system_prompt = """You are a medical triage assistant. The user is describing symptoms. Your job is to:
1. Acknowledge their symptoms with empathy
2. Ask 2-3 specific follow-up questions to gather more symptom details
3. Provide basic comfort measures
4. DO NOT diagnose or suggest specific conditions yet

Respond in JSON format:
{
    "response_text": "Empathetic response with follow-up questions",
    "urgency": "mild",
    "safety_measures": ["basic comfort measures"],
    "follow_up_questions": ["specific question 1", "specific question 2"],
    "send_sos": false
}"""
            user_prompt = f"{context_str}Current message: {description}"
        
        else:
            # Phase 2: Analysis and recommendations
            system_prompt = """You are a medical triage assistant. The user has provided sufficient symptom information. Provide:
1. Comprehensive analysis of their symptoms
2. 2-3 possible conditions with explanations
3. Specific safety measures and self-care
4. Clear next steps for medical care
5. Emergency guidance if needed

Respond in JSON format:
{
    "response_text": "Comprehensive response explaining symptoms and recommendations",
    "urgency": "urgent|moderate|mild",
    "possible_conditions": [{"name": "condition", "description": "explanation"}],
    "safety_measures": ["specific measure 1", "specific measure 2"],
    "follow_up_questions": ["question 1", "question 2"],
    "send_sos": true/false
}"""
            user_prompt = f"{context_str}Current message: {description}"
        
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
        try:
            result = json.loads(response.choices[0].message.content)
            return result
        except json.JSONDecodeError:
            # Fallback if JSON parsing fails
            return {
                "response_text": response.choices[0].message.content,
                "urgency": "moderate",
                "possible_conditions": [],
                "safety_measures": ["Consult a healthcare provider"],
                "follow_up_questions": ["How are you feeling now?"],
                "send_sos": False
            }
        
    except Exception as e:
        logger.error(f"OpenAI analysis failed: {e}")
        return None

def create_fallback_response(description, symptoms, is_emergency, intent, thread_context):
    """Create fallback response when OpenAI is not available"""
    if intent == 'non_medical':
        return {
            "response_text": "Hi! I'm a medical triage assistant. I can help you assess symptoms and suggest possible health conditions. Please describe any symptoms you're experiencing.",
            "urgency": "mild",
            "possible_conditions": [],
            "safety_measures": [],
            "follow_up_questions": ["Are you experiencing any symptoms?"],
            "send_sos": False
        }
    
    if is_emergency:
        return {
            "response_text": "🚨 URGENT: Your symptoms suggest a medical emergency. Please call 112 immediately for emergency medical assistance.",
            "urgency": "urgent",
            "possible_conditions": [
                {"name": "Emergency Condition", "description": "Your symptoms require immediate medical attention"}
            ],
            "safety_measures": [
                "Call 112 immediately",
                "Stay calm and don't panic",
                "Have someone stay with you if possible"
            ],
            "follow_up_questions": [],
            "send_sos": True
        }
    
    symptom_text = ", ".join(symptoms) if symptoms else "your symptoms"
    
    if thread_context and len(thread_context.all_symptoms) < 3:
        # Phase 1: Gathering more symptoms
        return {
            "response_text": f"I'm sorry you're experiencing {symptom_text}. To help me understand better, please answer these questions:",
            "urgency": "mild",
            "possible_conditions": [],
            "safety_measures": ["Stay hydrated", "Rest as needed"],
            "follow_up_questions": [
                "How long have you had these symptoms?",
                "Do you have any other symptoms?",
                "On a scale of 1-10, how severe is your discomfort?"
            ],
            "send_sos": False
        }
    
    # Phase 2: Analysis
    return {
        "response_text": f"Based on your symptoms ({symptom_text}), I recommend seeking medical attention for proper evaluation and treatment.",
        "urgency": "moderate",
        "possible_conditions": [
            {"name": "Common Health Issue", "description": f"Your symptoms may indicate a condition that requires medical evaluation."}
        ],
        "safety_measures": [
            "Monitor your symptoms closely",
            "Stay hydrated and get adequate rest",
            "Seek medical attention if symptoms worsen"
        ],
        "follow_up_questions": [
            "Have your symptoms changed?",
            "Are you experiencing anything else?"
        ],
        "send_sos": False
    }

# Health facts database (keeping existing implementation)
HEALTH_FACTS = {
    "general health": [
        {"title": "Stay Hydrated", "description": "Drink at least 8 glasses of water daily to maintain proper body functions and overall health."},
        {"title": "Get Quality Sleep", "description": "Adults need 7-9 hours of quality sleep per night for optimal health and wellbeing."},
        {"title": "Exercise Regularly", "description": "Aim for at least 150 minutes of moderate physical activity per week for cardiovascular health."},
        {"title": "Eat Balanced Meals", "description": "Include fruits, vegetables, whole grains, and lean proteins in your daily diet."},
        {"title": "Manage Stress", "description": "Practice stress-reduction techniques like meditation, deep breathing, or regular exercise."}
    ]
}

@app.route('/')
def home():
    return jsonify({
        "message": "HealthMate AI Backend API",
        "status": "running",
        "version": "2.0.0",
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
        'openai_available': OPENAI_AVAILABLE,
        'active_threads': len(conversation_threads)
    })

@app.route('/api/health/languages', methods=['GET'])
def get_supported_languages():
    return jsonify({
        'success': True,
        'supported_languages': SUPPORTED_LANGUAGES
    })

@app.route('/api/triage', methods=['POST'])
def triage_endpoint():
    """
    Enhanced triage analysis endpoint with proper context management
    """
    try:
        if not request.is_json:
            return jsonify({'success': False, 'error': 'Request must be JSON'}), 400
        
        data = request.json
        if not data or 'description' not in data:
            return jsonify({'success': False, 'error': 'No description provided'}), 400
        
        description = data['description'].strip()
        thread_id = data.get('thread_id', '').strip()
        
        logger.info(f"Processing triage request: {description[:50]}...")
        
        # Get or create thread
        if thread_id and thread_id.strip() and thread_id in conversation_threads:
            thread_context = conversation_threads[thread_id]
            logger.info(f"Using existing thread: {thread_id}")
        else:
            thread_id = generate_thread_id()
            thread_context = ConversationThread(thread_id)
            conversation_threads[thread_id] = thread_context
            logger.info(f"Created new thread: {thread_id}")
        
        # Extract symptoms from current message
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        symptom_analysis = loop.run_until_complete(extract_symptoms_comprehensive(description))
        loop.close()
        
        current_symptoms = symptom_analysis['symptoms']
        
        logger.info(f"DEBUG → Current symptoms extracted: {current_symptoms}")
        logger.info(f"DEBUG → Previous symptoms in thread: {thread_context.all_symptoms}")
        
        # Detect intent
        intent = detect_intent(description, thread_context)
        
        # Check for emergency
        all_symptoms_combined = thread_context.all_symptoms + current_symptoms
        is_emergency = detect_emergency(description, all_symptoms_combined)
        
        # Determine if should query Pinecone
        should_query_pinecone = (
            len(set(thread_context.all_symptoms + current_symptoms)) >= 3 and 
            thread_context.conversation_count >= 1
        ) or is_emergency
        
        logger.info(f"Intent: {intent}, Symptoms: {current_symptoms}, "
                   f"Total symptoms: {len(set(thread_context.all_symptoms + current_symptoms))}, "
                   f"Conversation count: {thread_context.conversation_count}, "
                   f"Should query Pinecone: {should_query_pinecone}, Emergency: {is_emergency}")
        
        # Generate response
        ai_result = None
        if OPENAI_AVAILABLE:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                ai_result = loop.run_until_complete(
                    generate_openai_response(description, thread_context, intent)
                )
                loop.close()
            except Exception as e:
                logger.error(f"OpenAI analysis failed: {e}")
        
        # Use AI result or fallback
        if ai_result:
            result = ai_result
        else:
            result = create_fallback_response(
                description, current_symptoms, is_emergency, intent, thread_context
            )
        
        # Store the conversation (this is where symptoms get accumulated!)
        thread_context.add_message(description, result.get("response_text", ""), current_symptoms)
        
        logger.info(f"DEBUG → After adding message, total symptoms: {thread_context.all_symptoms}")
        logger.info(f"DEBUG → Symptoms count: {len(thread_context.all_symptoms)}")
        
        # Format final response
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
            "symptoms_count": len(thread_context.all_symptoms),  # This should now be correct!
            "should_query_pinecone": should_query_pinecone,
            "conversation_count": thread_context.conversation_count,
            "intent": intent
        }
        
        logger.info(f"Triage completed. Thread: {thread_id}, "
                   f"Symptoms count: {len(thread_context.all_symptoms)}, "
                   f"Conversation: {thread_context.conversation_count}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in triage endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/api/health/analyze', methods=['POST'])
def analyze_health_query():
    """Legacy endpoint for backward compatibility"""
    try:
        if not request.is_json:
            return jsonify({'success': False, 'error': 'Request must be JSON'}), 400
        
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'success': False, 'error': 'No message provided'}), 400
        
        user_message = data['message']
        symptom_analysis = extract_symptoms_comprehensive(user_message)
        symptoms = symptom_analysis['symptoms']
        
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
            "health_analysis": {
                "symptoms": symptoms,
                "severity": symptom_analysis['severity'],
                "duration": symptom_analysis['duration']
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

# Debug endpoint to view thread context
@app.route('/api/debug/thread/<thread_id>', methods=['GET'])
def debug_thread(thread_id):
    if thread_id in conversation_threads:
        thread = conversation_threads[thread_id]
        return jsonify(thread.get_context_summary())
    return jsonify({"error": "Thread not found"}), 404

# Test endpoint to verify symptom extraction
@app.route('/api/test/symptoms', methods=['POST'])
def test_symptom_extraction():
    try:
        data = request.json
        text = data.get('text', '')
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Run async function in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(extract_symptoms_comprehensive(text))
        loop.close()
        
        return jsonify({
            'input_text': text,
            'extracted_symptoms': result['symptoms'],
            'severity': result['severity'],
            'duration': result['duration']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

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
