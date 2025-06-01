import os
import signal
import json
import logging
import asyncio
import re
from datetime import datetime
from typing import List, Dict, Optional

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# -----------------------------------
# Pinecone imports (new style)
# -----------------------------------
from pinecone import Pinecone

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# -----------------------------------
# OpenAI import (no openai.error)
# -----------------------------------
from openai import AsyncOpenAI

# -----------------------------------
# Setup logging
# -----------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------
# Configuration from Environment Variables
# -----------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV", "us-east1-gcp")
ASSISTANT_ID = os.getenv("ASSISTANT_ID", "asst_pAhSF6XJsj60efD9GEVdEG5n")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "triage-index")
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
MIN_SYMPTOMS_FOR_PINECONE = int(os.getenv("MIN_SYMPTOMS_FOR_PINECONE", "3"))
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "3"))
NIGERIA_EMERGENCY_HOTLINE = os.getenv("EMERGENCY_HOTLINE", "112")
PINECONE_SCORE_THRESHOLD = float(os.getenv("PINECONE_SCORE_THRESHOLD", "0.8"))
PORT = int(os.getenv("PORT", "8000"))

# Validate required environment variables
if not OPENAI_API_KEY:
    logger.error("OPENAI_API_KEY environment variable is required")
    raise ValueError("Missing required environment variable: OPENAI_API_KEY")

if not PINECONE_API_KEY:
    logger.error("PINECONE_API_KEY environment variable is required")
    raise ValueError("Missing required environment variable: PINECONE_API_KEY")

logger.info("✅ All required environment variables loaded successfully")
logger.info(f"Using OpenAI API Key: {OPENAI_API_KEY[:10]}...")
logger.info(f"Using Pinecone API Key: {PINECONE_API_KEY[:10]}...")
logger.info(f"Pinecone Environment: {PINECONE_ENV}")
logger.info(f"Index Name: {INDEX_NAME}")

# -----------------------------------
# Red-flag symptoms (still used for urgency detection)
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
# Initialize OpenAI Async Client
# -----------------------------------
try:
    openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    logger.info("✅ OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"❌ Failed to initialize OpenAI client: {e}")
    raise

# -----------------------------------
# Initialize Pinecone (new style)
# -----------------------------------
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(name=INDEX_NAME)
    logger.info(f"✅ Initialized Pinecone index '{INDEX_NAME}' in environment '{PINECONE_ENV}'")
    
    # Test the connection
    stats = index.describe_index_stats()
    logger.info(f"📊 Index stats: {stats.total_vector_count} vectors, {stats.dimension} dimensions")
    
except Exception as e:
    logger.error(f"❌ Failed to initialize Pinecone index: {e}")
    
    # Try to list available indexes for debugging
    try:
        available_indexes = pc.list_indexes().names()
        logger.error(f"Available indexes: {available_indexes}")
        logger.error(f"Requested index '{INDEX_NAME}' not found. Please check the index name.")
    except Exception as list_error:
        logger.error(f"Could not list available indexes: {list_error}")
    
    raise

# -----------------------------------
# FastAPI app & Pydantic models
# -----------------------------------
app = FastAPI(
    title="HealthMate AI Triage Service",
    description="Advanced medical triage system using OpenAI and Pinecone",
    version="2.0.0"
)

class TriageRequest(BaseModel):
    description: str
    thread_id: Optional[str] = None

class TriageResponse(BaseModel):
    text: str
    possible_conditions: List[Dict] = []
    safety_measures: List[str] = []
    triage: Dict = {"type": "", "location": "Unknown"}
    send_sos: bool = False
    follow_up_questions: List[str] = []
    thread_id: str
    symptoms_count: int = 0
    should_query_pinecone: bool = False

# ======================
# Utility Functions
# ======================

async def validate_thread(thread_id: str) -> bool:
    """
    Check if an OpenAI thread ID is valid. If the call errors, return False.
    """
    try:
        await openai_client.beta.threads.retrieve(thread_id=thread_id)
        logger.info(f"Thread {thread_id} is valid.")
        return True
    except Exception as e:
        logger.error(f"Couldn't validate thread_id {thread_id}: {e}")
        return False

async def get_thread_context(thread_id: str) -> Dict:
    """
    Retrieve all past user messages in this thread, re-extract symptoms & severity,
    and return a context dictionary.
    """
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
    """
    Use GPT-4o-mini to detect intent and extract any symptom keywords.
    If GPT parsing fails, fallback to simple keyword matching.
    """
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
            symptom_keywords = [
                "pain", "bleeding", "fever", "cough", "discomfort", "hurt", "ache", "nausea",
                "vomiting", "dizziness", "swelling", "shortness of breath", "palpitations"
            ]
            desc_lower = description.lower()
            for kw in symptom_keywords:
                if kw in desc_lower:
                    symptoms.append(kw)

        if has_prior_symptoms and "medical" not in intents:
            intents.append("medical")

        if not isinstance(intents, list):
            intents = [intents]
        valid_intents = {"medical", "contextual", "non_medical"}
        intents = [i for i in intents if i in valid_intents]
        if not intents:
            intents = ["medical"]

        if not symptoms and "non_medical" not in intents:
            non_medical_keywords = ["what can you do", "hi", "hello", "how are you"]
            if any(kw in description.lower() for kw in non_medical_keywords):
                logger.info("Fallback: Non-medical due to greeting keywords.")
                intents = ["non_medical"]

        return {"intent": intents, "symptoms": symptoms}

    except Exception as e:
        logger.error(f"Intent detection error: {e}. Using fallback keyword-based approach.")
        symptom_keywords = [
            "pain", "bleeding", "fever", "cough", "discomfort", "hurt", "ache", "nausea",
            "vomiting", "dizziness", "swelling", "shortness of breath", "palpitations"
        ]
        desc_lower = description.lower()
        intents = []
        symptoms = []
        non_medical_keywords = ["what can you do", "hi", "hello", "how are you"]

        if any(kw in desc_lower for kw in non_medical_keywords):
            intents.append("non_medical")
        if any(kw in desc_lower for kw in symptom_keywords):
            intents.append("medical")
            symptoms = [kw for kw in symptom_keywords if kw in desc_lower]

        if thread_id and await validate_thread(thread_id):
            ctx = await get_thread_context(thread_id)
            if ctx["all_symptoms"] and "medical" not in intents:
                intents.append("medical")

        if not intents:
            intents = ["medical"]

        return {"intent": intents, "symptoms": symptoms}

async def extract_symptoms_comprehensive(description: str) -> Dict:
    """
    **Now: fully GPT-based extraction** (no regex dictionary). Send the text to GPT-4o-mini
    and ask for a JSON array of all clinically relevant symptoms. Return a dict with 
    'symptoms', 'duration', and 'severity'. Fallback to no symptoms if something breaks.
    """
    try:
        description_lower = description.lower()

        # 1) Extract duration phrases and severity scores via regex (optional, but keep)
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

        # 2) Now call GPT to extract symptoms exclusively
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
            # Normalize to lowercase, strip whitespace
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

async def generate_follow_up_questions(context: Dict) -> List[str]:
    """
    Given context with 'all_symptoms', ask GPT to propose 2–3 follow-up questions.
    """
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
            logger.error("JSON parsing error in follow-up questions. Using fallback questions.")
            questions = ["Do you have any other symptoms?", "When did your symptoms start?"]

        return questions

    except Exception as e:
        logger.error(f"Error generating follow-up questions: {e}")
        return ["Do you have any other symptoms?", "When did your symptoms start?"]

def is_red_flag(full_text: str, severity: int = 0) -> bool:
    """
    Check if text contains any red-flag keywords, or if severity ≥ 8 plus
    mention of a critical symptom.
    """
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
    """
    Decide whether to query Pinecone. We do so if:
    - A red flag is present.
    - We have ≥ MIN_SYMPTOMS_FOR_PINECONE distinct symptoms.
    - Or the user explicitly asked for a condition/diagnosis.
    """
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
    """
    Wrapper around openai_client.embeddings.create(...) with retries.
    """
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
    """
    Run a Pinecone vector query and return matches with score ≥ threshold,
    deduplicated by disease name.
    """
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
    """
    Ask GPT-4o-mini to give a simple 2-3 sentence description of condition_name
    and how it relates to the user's symptoms. Fall back to a canned dictionary if GPT fails.
    """
    try:
        symptoms_text = ", ".join(user_symptoms)
        for attempt in range(MAX_RETRIES):
            try:
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
                    logger.info(f"Generated description for {condition_name}: {desc}")
                    return desc
                logger.warning(f"Description too short for {condition_name}. Retrying...")
            except Exception as e:
                logger.error(f"Error generating description for {condition_name} attempt {attempt+1}: {e}")

        # Fallback dictionary
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

async def rank_conditions(matches: List[Dict], symptoms: List[str], context: Dict) -> List[Dict]:
    """
    Given Pinecone matches, ask GPT for a short description of each, sort by score descending,
    and return up to 5 conditions with name, description, and citation.
    """
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
                "file_citation": "medical_knowledge_base.json"
            })
        return final

    except Exception as e:
        logger.error(f"Error ranking conditions: {e}")
        return [
            {
                "name": m["metadata"].get("disease", "Unknown").title(),
                "description": await generate_detailed_condition_description(m["metadata"].get("disease", "Unknown"), symptoms),
                "file_citation": "medical_knowledge_base.json"
            }
            for m in matches[:3] if m.get("score", 0) >= PINECONE_SCORE_THRESHOLD
        ]

async def generate_non_medical_response(thread_id: str) -> Dict:
    """
    A canned "hello, I'm a medical triage assistant" message for non-medical greetings.
    """
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
        await openai_client.beta.threads.messages.create(
            thread_id=thread_id,
            role="assistant",
            content=json.dumps(response)
        )
    except Exception as e:
        logger.error(f"Error adding non-medical response to thread {thread_id}: {e}")
    return response

async def generate_phase1_response(description: str, context: Dict, thread_id: str) -> Dict:
    """
    Collect more symptoms: apologize and ask 2-3 follow-up questions.
    """
    try:
        symptoms = context["all_symptoms"]
        symptoms_text = ", ".join(symptoms) if symptoms else "no symptoms reported yet"
        follow_up_questions = await generate_follow_up_questions(context)
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
            await openai_client.beta.threads.messages.create(
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
            "text": "I'm sorry, I need more information to assist you. Please let me know:\n- Do you have any symptoms?",
            "possible_conditions": [],
            "safety_measures": ["Stay calm and rest"],
            "triage": {"type": "clinic", "location": "Unknown"},
            "send_sos": False,
            "follow_up_questions": ["Do you have any symptoms?"],
            "symptoms_count": 0,
            "should_query_pinecone": False
        }

async def generate_phase2_response(context: Dict, is_emergency: bool, thread_id: Optional[str] = None) -> Dict:
    """
    We have enough symptoms (or a red flag). Query Pinecone if needed, rank conditions, and respond.
    """
    try:
        symptoms = context["all_symptoms"]
        symptoms_text = ", ".join(symptoms) if symptoms else "your symptoms"
        symptom_count = len(symptoms)
        should_query = await should_query_pinecone_database(context)

        possible_conditions = []
        if should_query:
            logger.info(f"Querying Pinecone with {symptom_count} symptoms: {symptoms_text}")
            query_text = f"Symptoms: {symptoms_text}"
            matches = await query_index(query_text, symptoms, context)
            if not matches:
                logger.warning("No Pinecone matches found, using fallback condition.")
                possible_conditions = [
                    {
                        "name": "Possible condition",
                        "description": "Your symptoms could be due to an infection, inflammation, or other medical condition that requires professional evaluation.",
                        "file_citation": "medical_knowledge_base.json"
                    }
                ]
            else:
                possible_conditions = await rank_conditions(matches, symptoms, context)

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
                await openai_client.beta.threads.messages.create(
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

async def generate_contextual_response(description: str, thread_id: str, context: Dict) -> Dict:
    """
    If the user is providing new context after we've already identified some symptoms (contextual intent),
    then we thank them, give next steps, potentially query Pinecone again, and ask follow-ups.
    """
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

        follow_up = await generate_follow_up_questions(context)
        if follow_up:
            text += "**Please Also Answer:**\n" + "\n".join(f"- {q}" for q in follow_up) + "\n"

        should_query = await should_query_pinecone_database(context)
        possible_conditions = []
        if should_query:
            query_text = f"Symptoms: {symptoms_text}"
            matches = await query_index(query_text, symptoms, context)
            possible_conditions = await rank_conditions(matches, symptoms, context)

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
            await openai_client.beta.threads.messages.create(
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

async def should_continue_conversation(thread_id: str, description: str) -> bool:
    """
    Decide if we should continue an ongoing conversation instead of resetting it.
    We continue if we have prior symptoms or if GPT says the intent is 'medical' or 'contextual'.
    """
    try:
        context = await get_thread_context(thread_id)
        if context["all_symptoms"] or context["assistant_responses"] > 0:
            logger.info(f"Continuing conversation in thread {thread_id} (prior symptoms/responses).")
            return True

        intent_result = await detect_intent(description, thread_id)
        intents = intent_result["intent"]
        logger.info(f"Intent for thread: {intents}")
        return any(i in ["medical", "contextual"] for i in intents)

    except Exception as e:
        logger.error(f"Error checking conversation continuity: {e}")
        return False

# ======================
# Main Triage Endpoint
# ======================

@app.post("/triage", response_model=TriageResponse)
async def triage(request: TriageRequest):
    """
    The main /triage endpoint. Steps:
      1. Validate or create a thread.
      2. Append the user's message to that thread.
      3. Recompute thread context (all past user symptoms, severity).
      4. Run intent detection on the new message.
      5. Decide if this is non-medical, contextual, or first-phase or second-phase.
      6. Generate appropriate response JSON, append it to the thread, and return it.
    """
    try:
        description = request.description.strip()
        thread_id = request.thread_id
        logger.info(f"=== TRIAGE REQUEST === Description: '{description[:50]}...', Thread: {thread_id}")

        # 1) Validate or create a thread
        if not thread_id or not await validate_thread(thread_id):
            new_thread = await openai_client.beta.threads.create()
            thread_id = new_thread.id
            logger.info(f"Created new thread ID: {thread_id}")

        # 2) Append user message
        await openai_client.beta.threads.messages.create(
            thread_id=thread_id,
            role="user",
            content=description
        )

        # 3) Recompute thread context
        context = await get_thread_context(thread_id)
        intent_result = await detect_intent(description, thread_id)
        intents = intent_result["intent"]

        symptom_count = len(context["all_symptoms"])
        max_severity = context["max_severity"]
        is_continuing = await should_continue_conversation(thread_id, description)
        should_query = await should_query_pinecone_database(context)
        is_emergency = is_red_flag(" ".join(context["user_messages"]), max_severity)

        logger.info((
            f"Intents: {intents}, Continuing: {is_continuing}, "
            f"Symptoms: {context['all_symptoms']}, Count: {symptom_count}, "
            f"Max severity: {max_severity}, Should query: {should_query}, Emergency: {is_emergency}"
        ))

        # 4) Decide which "phase" to run
        if "non_medical" in intents and not is_continuing and not context["all_symptoms"]:
            response = await generate_non_medical_response(thread_id)
        elif "contextual" in intents and context["all_symptoms"]:
            response = await generate_contextual_response(description, thread_id, context)
        elif is_emergency or should_query:
            logger.info(f"Phase 2 (emergency={is_emergency}, should_query={should_query})")
            response = await generate_phase2_response(context, is_emergency, thread_id)
        else:
            logger.info(f"Phase 1 (assistant_responses={context['assistant_responses']+1}/{MAX_ITERATIONS})")
            response = await generate_phase1_response(description, context, thread_id)

        response["thread_id"] = thread_id
        return TriageResponse(**response)

    except Exception as e:
        logger.error(f"Error in /triage endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Enhanced health check endpoint with configuration status.
    """
    return {
        "status": "healthy",
        "timestamp": str(datetime.now()),
        "service": "HealthMate AI Triage Service",
        "version": "2.0.0",
        "configuration": {
            "openai_configured": bool(OPENAI_API_KEY),
            "pinecone_configured": bool(PINECONE_API_KEY),
            "index_name": INDEX_NAME,
            "emergency_hotline": NIGERIA_EMERGENCY_HOTLINE,
            "min_symptoms_for_pinecone": MIN_SYMPTOMS_FOR_PINECONE,
            "pinecone_score_threshold": PINECONE_SCORE_THRESHOLD
        }
    }

@app.get("/config")
async def get_configuration():
    """
    Endpoint to check current configuration (without exposing API keys).
    """
    return {
        "environment_variables": {
            "OPENAI_API_KEY": "✅ Set" if OPENAI_API_KEY else "❌ Missing",
            "PINECONE_API_KEY": "✅ Set" if PINECONE_API_KEY else "❌ Missing",
            "PINECONE_ENV": PINECONE_ENV,
            "ASSISTANT_ID": ASSISTANT_ID,
            "INDEX_NAME": INDEX_NAME,
            "EMERGENCY_HOTLINE": NIGERIA_EMERGENCY_HOTLINE
        },
        "configuration": {
            "MAX_RETRIES": MAX_RETRIES,
            "MIN_SYMPTOMS_FOR_PINECONE": MIN_SYMPTOMS_FOR_PINECONE,
            "MAX_ITERATIONS": MAX_ITERATIONS,
            "PINECONE_SCORE_THRESHOLD": PINECONE_SCORE_THRESHOLD,
            "PORT": PORT
        }
    }

# -----------------------------------
# Startup validation
# -----------------------------------
@app.on_event("startup")
async def startup_event():
    """
    Validate configuration and connections on startup.
    """
    logger.info("🚀 Starting HealthMate AI Triage Service...")
    
    # Validate Pinecone index exists
    try:
        if pc:
            existing_indexes = pc.list_indexes().names()
            if INDEX_NAME not in existing_indexes:
                logger.error(f"❌ Index '{INDEX_NAME}' not found. Available: {existing_indexes}")
                raise RuntimeError(f"Missing Pinecone index '{INDEX_NAME}'")
            logger.info(f"✅ Verified Pinecone index '{INDEX_NAME}' exists.")
        else:
            logger.warning("⚠️  Pinecone client not initialized")
    except Exception as e:
        logger.error(f"❌ Startup validation error: {e}")
        raise

    logger.info("✅ HealthMate AI Triage Service started successfully!")

# -----------------------------------
# Run with Uvicorn
# -----------------------------------
if __name__ == "__main__":
    def handle_shutdown():
        logger.info("Shutting down server...")
        exit(0)

    signal.signal(signal.SIGINT, lambda s, f: handle_shutdown())

    logger.info(f"🌟 Starting HealthMate AI Triage server on port {PORT}...")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
