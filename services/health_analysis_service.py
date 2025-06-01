import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Supported languages
SUPPORTED_LANGUAGES = {
    "en": "English",
    "ig": "Igbo",
    "yo": "Yoruba", 
    "ha": "Hausa",
    "pcm": "Nigerian Pidgin"
}

def process_user_message(user_input):
    """
    Simplified processing for serverless environment
    """
    try:
        # For now, assume English and provide basic response
        detected_language = "en"
        
        # Simple keyword-based analysis
        symptoms = []
        if any(word in user_input.lower() for word in ['headache', 'head', 'pain in head']):
            symptoms.append('headache')
        if any(word in user_input.lower() for word in ['fever', 'temperature', 'hot']):
            symptoms.append('fever')
        if any(word in user_input.lower() for word in ['cough', 'coughing']):
            symptoms.append('cough')
        if any(word in user_input.lower() for word in ['stomach', 'belly', 'abdomen']):
            symptoms.append('stomach pain')
        
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
        
        return {
            "success": True,
            "detected_language": SUPPORTED_LANGUAGES[detected_language],
            "language_code": detected_language,
            "original_text": user_input,
            "english_translation": None,
            "health_analysis": {
                "symptoms": symptoms,
                "body_parts": [],
                "time_expressions": [],
                "medications": []
            },
            "response": response
        }
        
    except Exception as e:
        print(f"Error in process_user_message: {e}")
        return {
            "success": False,
            "message": "Sorry, I encountered an error processing your request. Please try again."
        }

def generate_health_facts(topic="general health", count=3):
    """
    Generate basic health facts
    """
    try:
        facts_database = {
            "nutrition": [
                {
                    "title": "Eat a Balanced Diet",
                    "description": "Include fruits, vegetables, whole grains, and lean proteins in your daily meals for optimal nutrition."
                },
                {
                    "title": "Stay Hydrated",
                    "description": "Drink at least 8 glasses of water daily to maintain proper body functions."
                },
                {
                    "title": "Limit Processed Foods",
                    "description": "Reduce intake of processed and sugary foods to maintain good health."
                }
            ],
            "exercise": [
                {
                    "title": "Regular Physical Activity",
                    "description": "Aim for at least 150 minutes of moderate exercise per week for cardiovascular health."
                },
                {
                    "title": "Strength Training",
                    "description": "Include muscle-strengthening activities at least 2 days per week."
                },
                {
                    "title": "Stay Active Daily",
                    "description": "Even simple activities like walking can significantly improve your health."
                }
            ],
            "general health": [
                {
                    "title": "Get Quality Sleep",
                    "description": "Adults need 7-9 hours of quality sleep per night for optimal health and wellbeing."
                },
                {
                    "title": "Regular Health Checkups",
                    "description": "Schedule regular visits with healthcare providers to monitor your health."
                },
                {
                    "title": "Manage Stress",
                    "description": "Practice stress-reduction techniques like meditation, deep breathing, or regular exercise."
                },
                {
                    "title": "Wash Hands Frequently",
                    "description": "Regular handwashing helps prevent the spread of infections and diseases."
                },
                {
                    "title": "Stay Hydrated",
                    "description": "Drinking adequate water helps maintain body temperature and supports organ function."
                }
            ]
        }
        
        # Get facts for the topic
        topic_lower = topic.lower()
        if topic_lower in facts_database:
            available_facts = facts_database[topic_lower]
        else:
            available_facts = facts_database["general health"]
        
        # Return requested number of facts
        return available_facts[:count]
        
    except Exception as e:
        print(f"Error generating health facts: {e}")
        return [
            {
                "title": "Stay Healthy",
                "description": "Maintain a balanced lifestyle with good nutrition, exercise, and adequate rest."
            }
        ]
