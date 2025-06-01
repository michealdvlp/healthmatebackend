import os
import json
import logging
import re
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
# Configuration (same as testtriage.py)
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
# Red-flag symptoms (exact same as testtriage.py)
# -----------------------------------
RED_FLAGS = [
    "bullet wound", "gunshot", "profuse bleeding", "crushing chest pain",
    "sudden shortness of breath", "loss of consciousness", "slurred speech", "seizure",
    "head trauma", "neck trauma", "high fever with stiff neck", "uncontrolled vomiting",
    "severe allergic reaction", "anaphylaxis", "difficulty breathing", "persistent cough with blood",
    "severe abdominal pain", "sudden vision loss", "chest tightness with sweating",
    "blood in urine", "inability to pass urine", "sharp abdominal pain", "intermenstrual bleeding"
]

# -----------------------------------
# Lazy-load clients to avoid initialization errors
# -----------------------------------
_openai_client = None
_pinecone_index = None

def get_openai_client():
    """Lazy-load OpenAI client using older compatible version"""
    global _openai_client
    if _openai_client is None:
        import openai
        openai.api_key = OPENAI_API_KEY
        _openai_client = openai
    return _openai_client

def get_pinecone_index():
    """Lazy-load Pinecone index"""
    global _pinecone_index
    if _pinecone_index is None:
        try:
            import pinecone
            pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
            _pinecone_index = pinecone.Index(INDEX_NAME)
            logger.info(f"Initialized Pinecone index '{INDEX_NAME}'")
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            raise
    return _pinecone_index

# ======================
# Utility Functions (adapted for older OpenAI API)
# ======================

def validate_thread(thread_id: str) -> bool:
    """Check if an OpenAI thread ID is valid using older API"""
    try:
        openai = get_openai_client()
        openai.beta.threads.retrieve(thread_id)
        logger.info(f"Thread {thread_id} is valid.")
        return True
    except Exception as e:
        logger.error(f"Couldn't validate thread_id {thread_id}: {e}")
        return False

def get_thread_context(thread_id: str) -> Dict:
    """Retrieve all past user messages in this thread using older API"""
    try:
        openai = get_openai_client()
        messages = openai.beta.threads.messages.list(thread_id=thread_id, order='asc')
        user_messages = []
        assistant_count = 0
        all_symptoms = []
        max_severity = 0

        for msg in messages.data:
            if msg.role == "user":
                content = ""
                if msg.content and len(msg.content) > 0:
                    content = msg.content[0].text.value
                if not content:
                    logger.warning(f"Empty or malformed user message in thread {thread_id}")
                    continue
                user_messages.append(content)
                sd = extract_symptoms_comprehensive(content)
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

def detect_intent(description: str, thread_id: Optional[str] = None) -> Dict:
    """Use GPT-4 to detect intent using older API"""
    try:
        has_prior_symptoms = False
        if thread_id and validate_thread(thread_id):
            context = get_thread_context(thread_id)
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

        openai = get_openai_client()
        response = openai.ChatCompletion.create(
            model="gpt-4",
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
            logger.error(f"JSON parsing error in intent detection: {e}")
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
        logger.error(f"Intent detection error: {e}")
        return {"intent": ["medical"], "symptoms": []}

def extract_symptoms_comprehensive(description: str) -> Dict:
    """Extract symptoms using GPT-4 with older API"""
    try:
        description_lower = description.lower()

        # Extract duration and severity (same regex logic)
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

        # GPT symptom extraction using older API
        prompt = (
            "You are a medical symptom extraction specialist. Extract ONLY specific, clinically relevant symptoms from the text. "
            "Do NOT include vague terms like 'pain' without context or duration descriptions. "
            "Return a JSON array of symptom strings. Be precise and map synonyms (e.g., 'leg ache' → 'leg pain'). "
            "Example output: [\"chest pain\", \"leg pain\", \"headache\"].\n\n"
            f"Text: \"{description}\""
        )

        openai = get_openai_client()
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Extract symptoms now."}
            ],
            temperature=0,
            max_tokens=200
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
            logger.error(f"GPT parsing error: {e}")
            unique_symptoms = []

        logger.info(f"Extracted symptoms: {unique_symptoms}, Duration: {duration}, Severity: {severity}")
        return {"symptoms": unique_symptoms, "duration": duration, "severity": severity}

    except Exception as e:
        logger.error(f"Error extracting symptoms: {e}")
        return {"symptoms": [], "duration": None, "severity": 0}

def generate_follow_up_questions(context: Dict) -> List[str]:
    """Generate follow-up questions using older OpenAI API"""
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

        openai = get_openai_client()
        response = openai.ChatCompletion.create(
            model="gpt-4",
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
            logger.error("JSON parsing error in follow-up questions")
            questions = ["Do you have any other symptoms?", "When did your symptoms start?"]

        return questions

    except Exception as e:
        logger.error(f"Error generating follow-up questions: {e}")
        return ["Do you have any other symptoms?", "When did your symptoms start?"]

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

def should_query_pinecone_database(context: Dict) -> bool:
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

    logger.info(f"Not querying Pinecone: only {symptom_count} symptoms")
    return False

def get_embeddings(texts: List[str], model: str = EMBEDDING_MODEL) -> Optional[List[List[float]]]:
    """Get embeddings using older OpenAI API"""
    for attempt in range(MAX_RETRIES):
        try:
            openai = get_openai_client()
            response = openai.Embedding.create(input=texts, model=model)
            return [item['embedding'] for item in response['data']]
        except Exception as e:
            logger.error(f"Embedding attempt {attempt+1}/{MAX_RETRIES} failed: {e}")
            if attempt == MAX_RETRIES - 1:
                return None
    return None

def query_index(query_text: str, symptoms: List[str], context: Dict, top_k: int = 50) -> List[Dict]:
    """Run a Pinecone vector query using older API"""
    query_embedding = get_embeddings([query_text])
    if not query_embedding:
        logger.error("Failed to generate query embedding.")
        return []

    try:
        index = get_pinecone_index()
        response = index.query(
            vector=query_embedding[0],
            top_k=top_k,
            include_metadata=True
        )
        matches = response.get("matches", [])
        logger.info(f"Pinecone returned {len(matches)} raw matches")

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

def generate_detailed_condition_description(condition_name: str, user_symptoms: List[str]) -> str:
    """Generate condition description using older OpenAI API"""
    try:
        symptoms_text = ", ".join(user_symptoms)
        for attempt in range(MAX_RETRIES):
            try:
                openai = get_openai_client()
                response = openai.ChatCompletion.create(
                    model="gpt-4",
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
                    logger.info(f"Generated description for {condition_name}")
                    return desc
            except Exception as e:
                logger.error(f"Error generating description attempt {attempt+1}: {e}")

        # Fallback descriptions
        fallback = {
            "gastritis": "Gastritis is inflammation of the stomach lining that can cause abdominal pain and nausea, often triggered by stress, spicy foods, or infections like H. pylori.",
            "appendicitis": "Appendicitis is inflammation of the appendix, causing sharp abdominal pain that may start near the navel and shift to the lower right side, often requiring urgent medical attention.",
            "irritable bowel syndrome": "Irritable bowel syndrome (IBS) is a condition affecting the digestive system, causing abdominal pain, bloating, and changes in bowel habits, often triggered by stress or certain foods.",
            "urinary tract infection": "A urinary tract infection (UTI) occurs when bacteria enter and multiply in your urinary system, causing pain during urination, frequent urination, and sometimes pelvic pain.",
            "pelvic inflammatory disease": "Pelvic inflammatory disease (PID) is an infection of the female reproductive organs that can cause pelvic pain, painful urination, and pain during intercourse."
        }
        return fallback.get(condition_name.lower(), f"{condition_name} may be related to your symptoms. Please consult a healthcare professional for details.")

    except Exception as e:
        logger.error(f"Fatal error generating description for {condition_name}: {e}")
        return f"{condition_name} may be related to your symptoms. Please consult a healthcare professional."

def rank_conditions(matches: List[Dict], symptoms: List[str], context: Dict) -> List[Dict]:
    """Rank and format condition matches"""
    try:
        condition_data = []
        for match in matches:
            disease = match["metadata"].get("disease", "Unknown").title()
            score = match.get("score", 0)
            desc = generate_detailed_condition_description(disease, symptoms)
            condition_data.append({"name": disease, "description": desc, "score": score})

        # Sort descending by score
        condition_data.sort(key=lambda x: x["score"], reverse=True)

        final = []
        for cond in condition_data[:5]:
            final.append({
                "name": cond["name"],
                "description": cond["description"],
                "file_citation": "use.json"
            })
        return final

    except Exception as e:
        logger.error(f"Error ranking conditions: {e}")
        return []

def generate_non_medical_response(thread_id: str) -> Dict:
    """Generate non-medical greeting response"""
    response = {
        "text": (
            "Hi! I'm a medical triage assistant. I can help you assess symptoms and suggest possible health conditions. "
            "For example, you can say, 'I have a cough and fever,' and I'll guide you on what to do next.\n\nPlease let me know:\n"
            "- Are you experiencing any symptoms?"
        ),
        "possible_conditions": [],
        "safety_measures": [],
        "triage": {"type": "", "location": "Unknown"},
        "send_sos": False,
        "follow_up_questions": ["Are you experiencing any symptoms?"],
        "symptoms_count": 0,
        "should_query_pinecone": False
    }
    try:
        openai = get_openai_client()
        openai.beta.threads.messages.create(
            thread_id=thread_id,
            role="assistant",
            content=json.dumps(response)
        )
    except Exception as e:
        logger.error(f"Error adding non-medical response to thread {thread_id}: {e}")
    return response

def generate_phase1_response(description: str, context: Dict, thread_id: str) -> Dict:
    """Phase 1: Collect more symptoms"""
    try:
        symptoms = context["all_symptoms"]
        symptoms_text = ", ".join(symptoms) if symptoms else "no symptoms reported yet"
        follow_up_questions = generate_follow_up_questions(context)
        text = f"I'm sorry you're dealing with {symptoms_text}. To help me understand better, please answer:\n" + "\n".join(
            f"- {q}" for q in follow_up_questions
        )
        response = {
            "text": text,
            "possible_conditions": [],
            "safety_measures": ["Stay hydrated", "Rest as needed"],
            "triage": {"type": "clinic", "location": "Unknown"},
            "send_sos": False,
            "follow_up_questions": follow_up_questions,
            "symptoms_count": len(symptoms),
            "should_query_pinecone": False
        }
        try:
            openai = get_openai_client()
            openai.beta.threads.messages.create(
                thread_id=thread_id,
                role="assistant",
                content=json.dumps(response)
            )
        except Exception as e:
            logger.error(f"Error adding phase1 response to thread {thread_id}: {e}")
        return response

    except Exception as e:
        logger.error(f"Error generating Phase 1 response: {e}")
        return {
            "text": "I need more information. Please describe your symptoms.",
            "possible_conditions": [],
            "safety_measures": ["Stay calm and rest"],
            "triage": {"type": "clinic", "location": "Unknown"},
            "send_sos": False,
            "follow_up_questions": ["Do you have any symptoms?"],
            "symptoms_count": 0,
            "should_query_pinecone": False
        }

def generate_phase2_response(context: Dict, is_emergency: bool, thread_id: Optional[str] = None) -> Dict:
    """Phase 2: Provide detailed analysis with conditions"""
    try:
        symptoms = context["all_symptoms"]
        symptoms_text = ", ".join(symptoms) if symptoms else "your symptoms"
        symptom_count = len(symptoms)
        should_query = should_query_pinecone_database(context)

        possible_conditions = []
        if should_query:
            logger.info(f"Querying Pinecone with {symptom_count} symptoms: {symptoms_text}")
            query_text = f"Symptoms: {symptoms_text}"
            matches = query_index(query_text, symptoms, context)
            if not matches:
                logger.warning("No Pinecone matches found")
                possible_conditions = [
                    {
                        "name": "Possible condition",
                        "description": "Your symptoms could be due to an infection, inflammation, or other medical condition that requires professional evaluation.",
                        "file_citation": "use.json"
                    }
                ]
            else:
                possible_conditions = rank_conditions(matches, symptoms, context)

        why_together = f"Your symptoms ({symptoms_text}) often happen together because they affect the same area of your body or are caused by similar issues."
        causes = []
        seen_conditions = set()
        if possible_conditions:
            for cond in possible_conditions[:3]:
                cname = cond["name"]
                if cname.lower() not in seen_conditions:
                    causes.append(f"- **{cname}**: {cond['description']}")
                    seen_conditions.add(cname.lower())
        else:
            causes.append(
                "- Based on your symptoms, this could be due to an infection, inflammation, or other medical condition that requires professional evaluation."
            )

        red_flags_text = "\n".join([
            "Seek urgent care right away if you have:",
            "- High fever or chills",
            "- Severe pain that's getting worse",
            "- Bleeding that won't stop",
            "- Difficulty breathing",
            "- Chest pain or pressure",
            "- Signs of severe dehydration",
            "- Any symptoms that are rapidly worsening"
        ])
        
        self_care = [
            "Stay hydrated by drinking plenty of water",
            "Get adequate rest to help your body heal",
            "Avoid alcohol, caffeine, and spicy foods until you feel better"
        ]
        if any("pain" in s.lower() for s in symptoms):
            self_care.append("Apply a warm compress to painful areas for 10-15 minutes")
        if any("bleeding" in s.lower() for s in symptoms):
            self_care.append("Monitor the bleeding and seek immediate care if it becomes heavy")

        action_plan = "Book a visit to a doctor or clinic soon. A proper examination and tests can determine what's causing your symptoms."
        if is_emergency:
            action_plan = f"This appears serious! Call {NIGERIA_EMERGENCY_HOTLINE} now or visit a hospital immediately."

        follow_up = ["Have your symptoms changed or are there any new ones?"]
        text = (
            f"I'm sorry you're dealing with {symptoms_text}. Here's what might be going on:\n\n"
            f"**Symptoms Identified:**\n"
            + (f"{', '.join(f'• {s}' for s in symptoms)}\n\n" if symptoms else "• No symptoms reported\n\n")
            + f"**Why These Symptoms Happen Together:**\n{why_together}\n\n"
            f"**Possible Causes:**\n" + "\n".join(causes) + "\n\n"
            f"**Signs You Need Urgent Care Right Away:**\n{red_flags_text}\n\n"
            f"**What You Can Do Now:**\n" +
            "\n".join(f"- {item}" for item in self_care) + "\n\n"
            f"**Next Steps:**\n- {action_plan}\n"
            "- This is not a medical diagnosis. Please see a healthcare professional as soon as possible.\n"
        )
        if follow_up:
            text += "\n**Please Also Answer:**\n" + "\n".join(f"- {q}" for q in follow_up)

        response = {
            "text": text,
            "possible_conditions": possible_conditions,
            "safety_measures": self_care,
            "triage": {
                "type": "hospital" if is_emergency else ("clinic" if possible_conditions else "pharmacy"),
                "location": "Unknown"
            },
            "send_sos": is_emergency,
            "follow_up_questions": follow_up,
            "symptoms_count": symptom_count,
            "should_query_pinecone": should_query
        }
        
        if thread_id:
            try:
                openai = get_openai_client()
                openai.beta.threads.messages.create(
                    thread_id=thread_id,
                    role="assistant",
                    content=json.dumps(response)
                )
            except Exception as e:
                logger.error(f"Error adding phase2 response to thread {thread_id}: {e}")

        return response

    except Exception as e:
        logger.error(f"Error in generate_phase2_response: {e}")
        return {
            "text": f"I'm experiencing a technical issue, but based on your symptoms, I recommend seeking medical attention. Call {NIGERIA_EMERGENCY_HOTLINE} if this is urgent.",
            "possible_conditions": [],
            "safety_measures": ["Stay calm and rest", "Seek medical attention"],
            "triage": {"type": "hospital", "location": "Unknown"},
            "send_sos": True,
            "follow_up_questions": [],
            "symptoms_count": len(context.get("all_symptoms", [])),
            "should_query_pinecone": False
        }

def generate_contextual_response(description: str, thread_id: str, context: Dict) -> Dict:
    """Handle contextual follow-up responses"""
    try:
        symptoms = context.get("all_symptoms", [])
        symptoms_text = ", ".join(symptoms) if symptoms else "your previously mentioned symptoms"
        symptom_count = len(symptoms)
        max_severity = context.get("max_severity", 0)
        is_emergency = is_red_flag(" ".join(context.get("user_messages", [])), max_severity)

        text = f"Thank you for providing more details. Based on your symptoms ({symptoms_text}), I recommend seeking medical attention.\n\n"
        action_plan = "Book a visit to a doctor or clinic soon. A proper examination and tests can determine what's causing your symptoms."
        if is_emergency:
            action_plan = f"This appears serious! Call {NIGERIA_EMERGENCY_HOTLINE} now or visit a hospital immediately."
        text += f"**Next Steps:**\n- {action_plan}\n- This is not a medical diagnosis. Please see a healthcare professional as soon as possible.\n\n"

        follow_up = generate_follow_up_questions(context)
        if follow_up:
            text += "**Please Also Answer:**\n" + "\n".join(f"- {q}" for q in follow_up) + "\n"

        should_query = should_query_pinecone_database(context)
        possible_conditions = []
        if should_query:
            query_text = f"Symptoms: {symptoms_text}"
            matches = query_index(query_text, symptoms, context)
            possible_conditions = rank_conditions(matches, symptoms, context)

        response = {
            "text": text,
            "possible_conditions": possible_conditions,
            "safety_measures": [
                "Stay hydrated by drinking plenty of water",
                "Get adequate rest to help your body heal",
                "Avoid alcohol, caffeine, and spicy foods until you feel better"
            ],
            "triage": {"type": "hospital" if is_emergency else "clinic", "location": "Unknown"},
            "send_sos": is_emergency,
            "follow_up_questions": follow_up,
            "symptoms_count": symptom_count,
            "should_query_pinecone": should_query
        }

        try:
            openai = get_openai_client()
            openai.beta.threads.messages.create(
                thread_id=thread_id,
                role="assistant",
                content=json.dumps(response)
            )
        except Exception as e:
            logger.error(f"Error adding contextual response to thread {thread_id}: {e}")

        return response

    except Exception as e:
        logger.error(f"Error in generate_contextual_response: {e}")
        return {
            "text": "Thank you for your response. Please provide more details about your symptoms so I can assist you further.",
            "possible_conditions": [],
            "safety_measures": ["Stay calm and rest"],
            "triage": {"type": "clinic", "location": "Unknown"},
            "send_sos": False,
            "follow_up_questions": ["Do you have any other symptoms?"],
            "symptoms_count": 0,
            "should_query_pinecone": False
        }

def should_continue_conversation(thread_id: str, description: str) -> bool:
    """Decide if we should continue an ongoing conversation"""
    try:
        context = get_thread_context(thread_id)
        if context["all_symptoms"] or context["assistant_responses"] > 0:
            logger.info(f"Continuing conversation in thread {thread_id}")
            return True

        intent_result = detect_intent(description, thread_id)
        intents = intent_result["intent"]
        logger.info(f"Intent for thread: {intents}")
        return any(i in ["medical", "contextual"] for i in intents)

    except Exception as e:
        logger.error(f"Error checking conversation continuity: {e}")
        return False

# ======================
# Main Triage Function (EXACT same logic as testtriage.py)
# ======================

def triage_main(description: str, thread_id: Optional[str] = None) -> Dict:
    """
    The main triage function - exact same logic as testtriage.py
    """
    try:
        description = description.strip()
        logger.info(f"=== TRIAGE REQUEST === Description: '{description[:50]}...', Thread: {thread_id}")

        # 1) Validate or create a thread
        if not thread_id or not validate_thread(thread_id):
            openai = get_openai_client()
            new_thread = openai.beta.threads.create()
            thread_id = new_thread.id
            logger.info(f"Created new thread ID: {thread_id}")

        # 2) Append user message
        openai = get_openai_client()
        openai.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=description
        )

        # 3) Recompute thread context
        context = get_thread_context(thread_id)
        intent_result = detect_intent(description, thread_id)
        intents = intent_result["intent"]

        symptom_count = len(context["all_symptoms"])
        max_severity = context["max_severity"]
        is_continuing = should_continue_conversation(thread_id, description)
        should_query = should_query_pinecone_database(context)
        is_emergency = is_red_flag(" ".join(context["user_messages"]), max_severity)

        logger.info((
            f"Intents: {intents}, Continuing: {is_continuing}, "
            f"Symptoms: {context['all_symptoms']}, Count: {symptom_count}, "
            f"Max severity: {max_severity}, Should query: {should_query}, Emergency: {is_emergency}"
        ))

        # 4) Decide which "phase" to run
        if "non_medical" in intents and not is_continuing and not context["all_symptoms"]:
            response = generate_non_medical_response(thread_id)
        elif "contextual" in intents and context["all_symptoms"]:
            response = generate_contextual_response(description, thread_id, context)
        elif is_emergency or should_query:
            logger.info(f"Phase 2 (emergency={is_emergency}, should_query={should_query})")
            response = generate_phase2_response(context, is_emergency, thread_id)
        else:
            logger.info(f"Phase 1 (assistant_responses={context['assistant_responses']+1}/{MAX_ITERATIONS})")
            response = generate_phase1_response(description, context, thread_id)

        response["thread_id"] = thread_id
        return response

    except Exception as e:
        logger.error(f"Error in triage function: {e}")
        return {
            "text": f"Internal server error: {str(e)}",
            "possible_conditions": [],
            "safety_measures": [],
            "triage": {"type": "hospital", "location": "Unknown"},
            "send_sos": True,
            "follow_up_questions": [],
            "thread_id": thread_id or "error",
            "symptoms_count": 0,
            "should_query_pinecone": False
        }

# ======================
# Flask Routes
# ======================

@app.route('/')
def home():
    return jsonify({
        "message": "HealthMate AI Backend API - testtriage.py Compatible",
        "status": "running", 
        "version": "5.0.0",
        "endpoints": [
            "/triage - POST - Advanced medical triage analysis (same as testtriage.py)",
            "/health - GET - Health check"
        ],
        "openai_version": "0.28.1",
        "pinecone_version": "2.2.4"
    })

@app.route('/health')
def health_check():
    """Simple health check endpoint"""
    return {"status": "healthy", "timestamp": str(datetime.now())}

@app.route('/triage', methods=['POST'])
def triage_endpoint():
    """
    The main /triage endpoint - EXACT same as testtriage.py but as Flask route
    """
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400
        
        data = request.json
        if not data or 'description' not in data:
            return jsonify({'error': 'No description provided'}), 400
        
        description = data['description']
        thread_id = data.get('thread_id')
        
        # Call the main triage function (same logic as testtriage.py)
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
