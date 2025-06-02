import os
import sys
import importlib.util
import logging  # Add this import
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from dotenv import load_dotenv

# Load health analysis service module
file_path = os.path.join(os.path.dirname(__file__), 'services', 'health_analysis_service.py')
spec = importlib.util.spec_from_file_location("health_analysis_service", file_path)
health_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(health_module)

# Get the functions and variables from the loaded module
process_user_message = health_module.process_user_message
SUPPORTED_LANGUAGES = health_module.SUPPORTED_LANGUAGES
generate_health_facts = health_module.generate_health_facts

# Load translation service module directly
translation_file_path = os.path.join(os.path.dirname(__file__), 'services', 'translation_service.py')
translation_spec = importlib.util.spec_from_file_location("translation_service", translation_file_path)
translation_module = importlib.util.module_from_spec(translation_spec)
translation_spec.loader.exec_module(translation_module)

# Get translation functions
translate_to_language = translation_module.translate_to_language
translate_to_english = translation_module.translate_to_english

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("healthmate.log"),
        logging.StreamHandler()
    ]
)

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/api/health/analyze', methods=['POST'])
def analyze_health_query():
    """
    Endpoint to analyze health queries in multiple languages
    
    Expects JSON with: {
        "message": "User's health concern in any supported language"
    }
    """
    if not request.is_json:
        return jsonify({
            'success': False,
            'error': 'Request must be JSON'
        }), 400
    
    data = request.json
    
    if not data or 'message' not in data:
        return jsonify({
            'success': False,
            'error': 'No message provided'
        }), 400
    
    user_message = data['message']
    result = process_user_message(user_message)
    
    return jsonify(result)

@app.route('/api/health/languages', methods=['GET'])
def get_supported_languages():
    """Endpoint to retrieve all supported languages"""
    return jsonify({
        'success': True,
        'supported_languages': SUPPORTED_LANGUAGES
    })

@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint to verify API is running"""
    return jsonify({
        'status': 'healthy',
        'service': 'HealthMate AI API'
    })

@app.route('/api/health/facts', methods=['GET'])
def get_health_facts():
    """Endpoint to generate random health facts using OpenAI"""
    # Get query parameters
    topic = request.args.get('topic', 'general health')
    count = min(int(request.args.get('count', 3)), 5)  # Reduce default to 3, max to 5
    
    try:
        # Get the health facts with a timeout
        facts = generate_health_facts(topic, count)
        
        return jsonify({
            'success': True,
            'topic': topic,
            'facts': facts
        })
        
    except Exception as e:
        # Return some default facts if API fails
        default_facts = [
            {
                "title": "Stay Hydrated",
                "description": "Drinking adequate water is essential for overall health, helping maintain body temperature and remove waste."
            },
            {
                "title": "Sleep Matters",
                "description": "Adults need 7-9 hours of quality sleep per night for optimal cognitive function and physical health."
            },
            {
                "title": "Regular Exercise",
                "description": "Just 30 minutes of moderate activity daily can significantly improve heart health and mental wellbeing."
            }
        ]
        
        return jsonify({
            'success': True,
            'topic': topic,
            'facts': default_facts,
            'note': 'Using default facts due to API timeout'
        })

# Load awareness service module directly from file
awareness_file_path = os.path.join(os.path.dirname(__file__), 'services', 'awareness_service.py')
awareness_spec = importlib.util.spec_from_file_location("awareness_service", awareness_file_path)
awareness_module = importlib.util.module_from_spec(awareness_spec)
awareness_spec.loader.exec_module(awareness_module)

# Get functions from the awareness module
generate_awareness_content = awareness_module.generate_awareness_content
get_all_categories = awareness_module.get_all_categories
get_random_awareness_content = awareness_module.get_random_awareness_content

@app.route('/api/health/awareness/categories', methods=['GET'])
def get_awareness_categories():
    """Endpoint to get all health awareness categories"""
    try:
        categories = get_all_categories()
        return jsonify({
            'success': True,
            'categories': categories
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health/awareness/content', methods=['GET'])
def get_category_content():
    """Endpoint to get awareness content for a specific category"""
    try:
        category = request.args.get('category', 'Nutrition')
        count = min(int(request.args.get('count', 3)), 5)  # Limit to 5 max
        
        content = generate_awareness_content(category, count)
        
        return jsonify({
            'success': True,
            'category': category,
            'content': content
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/health/awareness/random', methods=['GET'])
def get_random_content():
    """Endpoint to get random awareness content across categories"""
    try:
        count = min(int(request.args.get('count', 5)), 10)  # Limit to 10 max
        
        content = get_random_awareness_content(count)
        
        return jsonify({
            'success': True,
            'content': content
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# Add this new endpoint for translating awareness content
@app.route('/api/translate/awareness', methods=['POST'])
def translate_awareness_content():
    """Endpoint to translate awareness content to a specific language"""
    try:
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Request must be JSON'
            }), 400
        
        data = request.json
        
        if not data or 'content' not in data or 'target_language' not in data:
            return jsonify({
                'success': False,
                'error': 'Missing required fields (content, target_language)'
            }), 400
        
        content = data['content']
        target_language = data['target_language']
        
        # If already English and requesting English, just return the content
        if target_language == 'en':
            return jsonify({
                'success': True,
                'translated_content': content
            })
        
        # Initialize translated content with same structure as original
        translated_content = []
        
        # Translate each content item
        for item in content:
            try:
                # Translate title, content AND category
                translated_title = translate_to_language(item['title'], target_language)
                translated_body = translate_to_language(item['content'], target_language)
                translated_category = translate_to_language(item['category'], target_language)
                
                # Create translated version of the item
                translated_item = item.copy()
                translated_item['title'] = translated_title
                translated_item['content'] = translated_body
                translated_item['category'] = translated_category  # Add this line
                translated_item['original_language'] = 'en'
                translated_item['translated'] = True
                
                translated_content.append(translated_item)
            except Exception as e:
                # If translation fails for an item, use the original
                item['translated'] = False
                translated_content.append(item)
                logging.error(f"Error translating content: {str(e)}")
        
        return jsonify({
            'success': True,
            'translated_content': translated_content
        })
        
    except Exception as e:
        logging.error(f"Error in translation endpoint: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    # Use debug=True during development, set to False for production
    app.run(host='0.0.0.0', port=port, debug=True)