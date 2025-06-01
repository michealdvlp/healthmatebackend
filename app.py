import os
import json
import logging
import asyncio
import uuid
import re
import requests
from datetime import datetime, timedelta
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

# Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
AZURE_TRANSLATOR_KEY = os.getenv("AZURE_TRANSLATOR_KEY")
AZURE_TRANSLATOR_ENDPOINT = os.getenv("AZURE_TRANSLATOR_ENDPOINT", "https://api.cognitive.microsofttranslator.com")
AZURE_TRANSLATOR_REGION = os.getenv("AZURE_TRANSLATOR_REGION", "eastus")

PINECONE_ENV = "us-east1-gcp"
EMBEDDING_MODEL = "text-embedding-ada-002"
INDEX_NAME = "triage-index"
MIN_SYMPTOMS_FOR_PINECONE = 3
PINECONE_SCORE_THRESHOLD = 0.8

# Check for required API keys
OPENAI_AVAILABLE = bool(OPENAI_API_KEY)
PINECONE_AVAILABLE = bool(PINECONE_API_KEY)
TRANSLATOR_AVAILABLE = bool(AZURE_TRANSLATOR_KEY)

if OPENAI_AVAILABLE:
    try:
        import openai
        openai.api_key = OPENAI_API_KEY
        logger.info("OpenAI client initialized successfully")
    except ImportError:
        OPENAI_AVAILABLE = False
        logger.warning("OpenAI not available - install openai package")

# Supported languages
SUPPORTED_LANGUAGES = {
    "en": "English",
    "ig": "Igbo", 
    "yo": "Yoruba",
    "ha": "Hausa",
    "pcm": "Nigerian Pidgin"
}

# Health categories for awareness content
HEALTH_CATEGORIES = [
    "Nutrition", "Exercise", "Mental Health", "Sleep", "Preventive Care",
    "Common Illnesses", "First Aid", "Chronic Conditions", "Women's Health",
    "Men's Health", "Children's Health", "Elderly Care"
]

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

# In-memory storage for conversations and cache
conversation_threads = {}
_content_cache = {}
_cache_expiry = {}
_pinecone_index = None
CACHE_DURATION = 24 * 60 * 60  # 24 hours

# ======================
# Core Classes
# ======================

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

# ======================
# Utility Functions
# ======================

def generate_thread_id():
    """Generate a thread ID"""
    return f"thread_{str(uuid.uuid4()).replace('-', '')[:24]}"

def get_pinecone_index():
    """Lazy-load Pinecone index"""
    global _pinecone_index
    if _pinecone_index is None and PINECONE_AVAILABLE:
        try:
            from pinecone import Pinecone
            pc = Pinecone(api_key=PINECONE_API_KEY)
            _pinecone_index = pc.Index(name=INDEX_NAME)
            logger.info(f"Initialized Pinecone index '{INDEX_NAME}'")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
    return _pinecone_index

# ======================
# OpenAI Functions
# ======================

def call_openai_chat(messages: List[Dict], model: str = "gpt-4", temperature: float = 0.3, max_tokens: int = 200) -> Optional[str]:
    """Call OpenAI Chat API"""
    if not OPENAI_AVAILABLE:
        return None
    
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
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
    """Call OpenAI Embeddings API"""
    if not OPENAI_AVAILABLE:
        return None
    
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.embeddings.create(input=texts, model=model)
        return [item.embedding for item in response.data]
    except Exception as e:
        logger.error(f"OpenAI Embeddings API error: {e}")
        return None

# ======================
# Translation Functions
# ======================

def detect_language(text):
    """Detect language using Azure Translator"""
    if not TRANSLATOR_AVAILABLE:
        logger.warning("Azure Translator not available, defaulting to English")
        return "en"
    
    url = f"{AZURE_TRANSLATOR_ENDPOINT}/detect"
    params = {'api-version': '3.0'}
    headers = {
        'Ocp-Apim-Subscription-Key': AZURE_TRANSLATOR_KEY,
        'Ocp-Apim-Subscription-Region': AZURE_TRANSLATOR_REGION,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }
    body = [{'text': text}]
    
    try:
        response = requests.post(url, params=params, headers=headers, json=body)
        response.raise_for_status()
        detected_language = response.json()[0]["language"]
        return detected_language
    except Exception as e:
        logger.error(f"Error detecting language: {e}")
        return "en"

def translate_to_english(text, source_language):
    """Translate text to English"""
    if not TRANSLATOR_AVAILABLE or source_language == "en":
        return text
    
    url = f"{AZURE_TRANSLATOR_ENDPOINT}/translate"
    params = {
        'api-version': '3.0',
        'from': source_language,
        'to': ['en']
    }
    headers = {
        'Ocp-Apim-Subscription-Key': AZURE_TRANSLATOR_KEY,
        'Ocp-Apim-Subscription-Region': AZURE_TRANSLATOR_REGION,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }
    body = [{'text': text}]
    
    try:
        response = requests.post(url, params=params, headers=headers, json=body)
        response.raise_for_status()
        translated_text = response.json()[0]["translations"][0]["text"]
        return translated_text
    except Exception as e:
        logger.error(f"Error translating text: {e}")
        return text

def translate_to_language(text, target_language):
    """Translate text to target language"""
    if not TRANSLATOR_AVAILABLE or target_language == "en":
        return text
        
    url = f"{AZURE_TRANSLATOR_ENDPOINT}/translate"
    params = {
        'api-version': '3.0',
        'from': 'en',
        'to': [target_language]
    }
    headers = {
        'Ocp-Apim-Subscription-Key': AZURE_TRANSLATOR_KEY,
        'Ocp-Apim-Subscription-Region': AZURE_TRANSLATOR_REGION,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }
    body = [{'text': text}]
    
    try:
        response = requests.post(url, params=params, headers=headers, json=body)
        response.raise_for_status()
        translated_text = response.json()[0]["translations"][0]["text"]
        return translated_text
    except Exception as e:
        logger.error(f"Error translating back to {target_language}: {e}")
        return text

# ======================
# Symptom Extraction
# ======================

async def extract_symptoms_comprehensive(text):
    """Extract symptoms using GPT-4o-mini dynamically"""
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
                client = openai.OpenAI(api_key=OPENAI_API_KEY)
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
    """Fallback regex-based symptom extraction"""
    text_lower = text.lower()
    symptoms = []
    
    # Basic symptom patterns
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

# ======================
# Emergency Detection
# ======================

def detect_emergency(text, symptoms_list=None):
    """Detect emergency situations"""
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

# ======================
# Pinecone Functions
# ======================

def should_query_pinecone_database(accumulated_symptoms: List[str], severity: int, full_text: str) -> bool:
    """Decide whether to query Pinecone"""
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

def query_pinecone_index(query_text: str, accumulated_symptoms: List[str], top_k: int = 50) -> List[Dict]:
    """Query Pinecone for similar conditions"""
    if not PINECONE_AVAILABLE:
        logger.warning("Pinecone not available")
        return []
    
    try:
        # Get embeddings
        query_embedding = call_openai_embeddings([query_text])
        if not query_embedding:
            logger.error("Failed to generate query embedding.")
            return []

        # Query Pinecone
        index = get_pinecone_index()
        if not index:
            return []
            
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
    """Generate condition description using GPT or fallback"""
    if not OPENAI_AVAILABLE:
        return f"{condition_name} may be related to your symptoms. Please consult a healthcare professional for details."
    
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

def rank_conditions(matches: List[Dict], accumulated_symptoms: List[str]) -> List[Dict]:
    """Rank and format conditions from Pinecone matches"""
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

# ======================
# Health Awareness Functions
# ======================

def get_color_for_category(category):
    """Return a consistent color for each category"""
    colors = {
        "Nutrition": "#4CAF50",
        "Exercise": "#2196F3",
        "Mental Health": "#9C27B0",
        "Sleep": "#673AB7",
        "Preventive Care": "#00BCD4",
        "Common Illnesses": "#FF5722",
        "First Aid": "#F44336",
        "Chronic Conditions": "#795548",
        "Women's Health": "#E91E63",
        "Men's Health": "#3F51B5",
        "Children's Health": "#FFEB3B",
        "Elderly Care": "#607D8B"
    }
    return colors.get(category, "#FF9800")

def generate_awareness_content(category, count=3):
    """Generate health awareness content using OpenAI"""
    cache_key = f"{category}_{count}"
    
    # Check cache first
    if cache_key in _content_cache and datetime.now() < _cache_expiry.get(cache_key, datetime.now()):
        logger.info(f"Returning cached content for {category}")
        return _content_cache[cache_key]
    
    if not OPENAI_AVAILABLE:
        return get_fallback_content(category, count)
    
    try:
        prompt = f"""
        Generate {count} informative and evidence-based health awareness articles about {category}.
        For each article, provide:
        1. An attention-grabbing title (5-8 words)
        2. Informative content (100-150 words) with practical advice
        3. Make the content culturally sensitive and appropriate for diverse populations
        4. Ensure medical accuracy and avoid overly technical language
        
        Return as JSON array: [{"title": "...", "content": "..."}, ...]
        """
        
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a health educator providing accurate, helpful health information. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1500
        )
        
        # Parse the response
        response_content = json.loads(response.choices[0].message.content)
        
        # Ensure we have the expected structure
        if isinstance(response_content, list):
            articles = response_content
        elif "articles" in response_content:
            articles = response_content["articles"]
        else:
            # Try to find any list in the response
            for key, value in response_content.items():
                if isinstance(value, list) and len(value) > 0:
                    articles = value
                    break
            else:
                raise ValueError("Unexpected response structure from OpenAI")
        
        # Format the articles with category and color
        formatted_articles = []
        for article in articles:
            formatted_articles.append({
                "title": article.get("title", "Health Tips"),
                "content": article.get("content", "Information not available"),
                "category": category,
                "color": get_color_for_category(category)
            })
        
        # Cache the results
        _content_cache[cache_key] = formatted_articles
        _cache_expiry[cache_key] = datetime.now() + timedelta(seconds=CACHE_DURATION)
        
        return formatted_articles
        
    except Exception as e:
        logger.error(f"Error generating awareness content for {category}: {str(e)}")
        return get_fallback_content(category, count)

def get_fallback_content(category, count=3):
    """Get fallback content when OpenAI is not available"""
    fallback_content = {
        "Nutrition": [
            {
                "title": "The Importance of Balanced Diet",
                "content": "A balanced diet provides essential nutrients needed for good health. Include fruits, vegetables, whole grains, lean proteins, and healthy fats in your daily meals. Aim for variety and moderation to ensure you get all necessary vitamins and minerals.",
                "category": "Nutrition",
                "color": "#4CAF50"
            }
        ],
        "Exercise": [
            {
                "title": "Benefits of Regular Physical Activity",
                "content": "Regular exercise improves cardiovascular health, strengthens muscles, and enhances mood. Aim for at least 150 minutes of moderate activity weekly. Even short walks provide benefits.",
                "category": "Exercise",
                "color": "#2196F3"
            }
        ]
    }
    
    default_content = [{
        "title": f"Important {category} Tips",
        "content": f"Taking care of your {category.lower()} is essential for overall wellbeing. Regular attention to this aspect of health can prevent problems and improve quality of life.",
        "category": category,
        "color": get_color_for_category(category)
    }]
    
    content = fallback_content.get(category, default_content)
    return content[:count]

def get_random_awareness_content(count=5):
    """Get random awareness content across categories"""
    import random
    
    result = []
    categories = random.sample(HEALTH_CATEGORIES, min(count, len(HEALTH_CATEGORIES)))
    
    for category in categories:
        try:
            content = generate_awareness_content(category, count=1)
            if content:
                result.extend(content)
        except Exception as e:
            logger.error(f"Error getting random content for {category}: {str(e)}")
            result.extend(get_fallback_content(category, 1))
    
    return result

# ======================
# Response Generation Functions
# ======================

async def generate_openai_response(description, thread_context, intent):
    """Generate response using OpenAI with proper context"""
    if not OPENAI_AVAILABLE:
        return None
    
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
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
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

def generate_follow_up_questions(accumulated_symptoms: List[str]) -> List[str]:
    """Generate follow-up questions based on symptoms"""
    if not OPENAI_AVAILABLE:
        if not accumulated_symptoms:
            return ["Do you have any symptoms?", "When did your symptoms start?"]
        return ["Have your symptoms changed since they started?", "Are you experiencing any other symptoms?"]
    
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

# ======================
# Main Triage Function
# ======================

async def triage_main(description: str, thread_id: Optional[str] = None) -> Dict:
    """Main triage processing function"""
    try:
        description = description.strip()
        logger.info(f"=== TRIAGE REQUEST === Description: '{description[:50]}...'")

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
        symptom_analysis = await extract_symptoms_comprehensive(description)
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
            len(set(thread_context.all_symptoms + current_symptoms)) >= MIN_SYMPTOMS_FOR_PINECONE and 
            thread_context.conversation_count >= 1
        ) or is_emergency
        
        logger.info(f"Intent: {intent}, Symptoms: {current_symptoms}, "
                   f"Total symptoms: {len(set(thread_context.all_symptoms + current_symptoms))}, "
                   f"Conversation count: {thread_context.conversation_count}, "
                   f"Should query Pinecone: {should_query_pinecone}, Emergency: {is_emergency}")
        
        # Generate response
        ai_result = await generate_openai_response(description, thread_context, intent)
        
        # Use AI result or fallback
        if ai_result:
            result = ai_result
        else:
            result = create_fallback_response(
                description, current_symptoms, is_emergency, intent, thread_context
            )
        
        # Query Pinecone if needed
        possible_conditions = []
        if should_query_pinecone and PINECONE_AVAILABLE:
            try:
                all_symptoms = list(set(thread_context.all_symptoms + current_symptoms))
                query_text = f"Symptoms: {', '.join(all_symptoms)}"
                matches = query_pinecone_index(query_text, all_symptoms)
                if matches:
                    possible_conditions = rank_conditions(matches, all_symptoms)
                logger.info(f"Pinecone query completed. Found {len(possible_conditions)} conditions.")
            except Exception as e:
                logger.error(f"Pinecone query failed: {e}")
        
        # Override possible_conditions if we got them from Pinecone
        if possible_conditions:
            result["possible_conditions"] = possible_conditions
        
        # Store the conversation (this is where symptoms get accumulated!)
        thread_context.add_message(description, result.get("response_text", ""), current_symptoms)
        
        logger.info(f"DEBUG → After adding message, total symptoms: {thread_context.all_symptoms}")
        logger.info(f"DEBUG → Symptoms count: {len(thread_context.all_symptoms)}")
        
        # Format final response
        response = {
            "success": True,
            "text": result.get("response_text", ""),
            "possible_conditions": result.get("possible_conditions", []),
            "safety_measures": result.get("safety_measures", [
                "Monitor your symptoms closely",
                "Stay hydrated and get adequate rest",
                "Seek medical attention if symptoms worsen",
                "Contact a healthcare provider for evaluation"
            ]),
            "triage": {
                "type": "hospital" if result.get("urgency") == "urgent" else 
                        "clinic" if result.get("urgency") == "moderate" else "pharmacy",
                "location": "Unknown"
            },
            "send_sos": result.get("send_sos", False),
            "follow_up_questions": result.get("follow_up_questions", generate_follow_up_questions(thread_context.all_symptoms)),
            "thread_id": thread_id,
            "symptoms_count": len(thread_context.all_symptoms),  # This should now be correct!
            "should_query_pinecone": should_query_pinecone,
            "conversation_count": thread_context.conversation_count,
            "intent": intent
        }
        
        logger.info(f"Triage completed. Thread: {thread_id}, "
                   f"Symptoms count: {len(thread_context.all_symptoms)}, "
                   f"Conversation: {thread_context.conversation_count}")
        return response
        
    except Exception as e:
        logger.error(f"Error in triage_main: {str(e)}")
        return {
            "success": False,
            "text": "I'm sorry, I'm experiencing a technical issue. Please try again later.",
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
# Health Analysis Function
# ======================

def process_user_message(user_input):
    """Process user message for health analysis"""
    try:
        # Detect language
        detected_language_code = detect_language(user_input)
        detected_language = SUPPORTED_LANGUAGES.get(detected_language_code, "English")
        
        # Translate to English if needed
        if detected_language_code != "en":
            english_translation = translate_to_english(user_input, detected_language_code)
        else:
            english_translation = None
        
        # Extract symptoms (use sync version for this function)
        text_to_analyze = english_translation if english_translation else user_input
        
        # Use fallback extraction for sync context
        symptom_data = extract_symptoms_fallback(text_to_analyze)
        symptoms = symptom_data['symptoms']
        
        # Generate response based on symptoms
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
        
        # Translate response back if needed
        if detected_language_code != "en":
            response = translate_to_language(response, detected_language_code)
        
        return {
            "success": True,
            "detected_language": detected_language,
            "language_code": detected_language_code,
            "original_text": user_input,
            "english_translation": english_translation,
            "health_analysis": {
                "symptoms": symptoms,
                "body_parts": [],
                "time_expressions": [symptom_data['duration']] if symptom_data['duration'] else [],
                "medications": []
            },
            "response": response
        }
        
    except Exception as e:
        logger.error(f"Error in process_user_message: {e}")
        return {
            "success": False,
            "message": "Sorry, I encountered an error processing your request. Please try again."
        }

# ======================
# Flask Routes
# ======================

@app.route('/')
def home():
    return jsonify({
        "message": "HealthMate AI Backend API - Complete Integration",
        "status": "running",
        "version": "2.1.0",
        "endpoints": [
            "/api/health - GET - Health check",
            "/api/health/languages - GET - Get supported languages", 
            "/api/health/analyze - POST - Legacy health analysis",
            "/api/health/facts - GET - Get health facts",
            "/api/triage - POST - Advanced triage analysis with context",
            "/api/awareness - GET - Health awareness content",
            "/api/translate - POST - Translation service",
            "/api/test/symptoms - POST - Test symptom extraction",
            "/api/debug/thread/<id> - GET - Debug thread context"
        ],
        "features": {
            "openai_available": OPENAI_AVAILABLE,
            "pinecone_available": PINECONE_AVAILABLE,
            "translator_available": TRANSLATOR_AVAILABLE,
            "active_threads": len(conversation_threads)
        }
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'service': 'HealthMate AI Backend API',
        'environment': 'production',
        'openai_available': OPENAI_AVAILABLE,
        'pinecone_available': PINECONE_AVAILABLE,
        'translator_available': TRANSLATOR_AVAILABLE,
        'active_threads': len(conversation_threads),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/health/languages', methods=['GET'])
def get_supported_languages():
    return jsonify({
        'success': True,
        'supported_languages': SUPPORTED_LANGUAGES
    })

@app.route('/api/triage', methods=['POST'])
def triage_endpoint():
    """Enhanced triage analysis endpoint with proper context management"""
    try:
        if not request.is_json:
            return jsonify({'success': False, 'error': 'Request must be JSON'}), 400
        
        data = request.json
        if not data or 'description' not in data:
            return jsonify({'success': False, 'error': 'No description provided'}), 400
        
        description = data['description'].strip()
        thread_id = data.get('thread_id', '').strip()
        
        if not description:
            return jsonify({'success': False, 'error': 'Description cannot be empty'}), 400
        
        logger.info(f"Processing triage request: {description[:50]}...")
        
        # Run async function in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(triage_main(description, thread_id))
        loop.close()
        
        return jsonify(result)
        
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
        if not user_message or not user_message.strip():
            return jsonify({'success': False, 'error': 'Message cannot be empty'}), 400
        
        result = process_user_message(user_message)
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error in analyze_health_query: {str(e)}")
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'message': 'Please try again later'
        }), 500

@app.route('/api/awareness', methods=['GET'])
def awareness_endpoint():
    """Health awareness content endpoint"""
    try:
        category = request.args.get("category", None)
        random_count = request.args.get("random", None)
        count = int(request.args.get("count", 3))

        # Random articles
        if random_count is not None:
            try:
                rc = int(random_count)
            except ValueError:
                return jsonify({"error": "Invalid random count"}), 400

            contents = get_random_awareness_content(count=rc)
            return jsonify({"success": True, "random_count": rc, "articles": contents})

        # Specific category
        if category:
            if category not in HEALTH_CATEGORIES:
                return jsonify({
                    "success": False,
                    "error": f"Unknown category '{category}'. Valid options: {HEALTH_CATEGORIES}"
                }), 400

            articles = generate_awareness_content(category, count=count)
            return jsonify({
                "success": True,
                "category": category,
                "count": count,
                "articles": articles
            })

        # List all categories
        return jsonify({
            "success": True,
            "categories": HEALTH_CATEGORIES
        })

    except Exception as e:
        logger.error(f"Error in /awareness endpoint: {e}")
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500

@app.route('/api/translate', methods=['POST'])
def translate_endpoint():
    """Translation service endpoint"""
    try:
        if not request.is_json:
            return jsonify({"success": False, "error": "Request must be JSON"}), 400

        data = request.json
        orig_text = data.get("text", "").strip()
        if not orig_text:
            return jsonify({"success": False, "error": "Text cannot be empty"}), 400

        target_lang = data.get("to", "en").lower()
        source_lang = data.get("from", "auto").lower()

        # Detect source language if "auto"
        if source_lang == "auto":
            detected = detect_language(orig_text)
            source_lang = detected
        else:
            detected = source_lang

        # Translate to English if needed
        if source_lang != "en":
            english_text = translate_to_english(orig_text, source_lang)
        else:
            english_text = orig_text

        # Translate to target language if needed
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

@app.route('/api/health/facts', methods=['GET'])
def get_health_facts():
    """Get health facts"""
    try:
        topic = request.args.get('topic', 'general health')
        count = min(int(request.args.get('count', 3)), 5)
        
        # Generate facts using awareness content system
        if topic in HEALTH_CATEGORIES:
            facts_content = generate_awareness_content(topic, count)
            facts = [{"title": item["title"], "description": item["content"]} for item in facts_content]
        else:
            # Fallback facts
            facts = [
                {"title": "Stay Hydrated", "description": "Drink at least 8 glasses of water daily to maintain proper body functions."},
                {"title": "Get Quality Sleep", "description": "Adults need 7-9 hours of quality sleep per night for optimal health and wellbeing."},
                {"title": "Exercise Regularly", "description": "Aim for at least 150 minutes of moderate physical activity per week."}
            ][:count]
        
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
            'facts': [{"title": "Stay Healthy", "description": "Maintain a balanced lifestyle with good nutrition, exercise, and adequate rest."}]
        })

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

# Debug endpoint to view thread context
@app.route('/api/debug/thread/<thread_id>', methods=['GET'])
def debug_thread(thread_id):
    if thread_id in conversation_threads:
        thread = conversation_threads[thread_id]
        return jsonify(thread.get_context_summary())
    return jsonify({"error": "Thread not found"}), 404

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
