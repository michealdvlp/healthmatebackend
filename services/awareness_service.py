import os
import json
import time
import logging
from datetime import datetime, timedelta
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging for serverless environment (no file logging)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()  # Only console output for Vercel
    ]
)
logger = logging.getLogger("healthmate_awareness")

# OpenAI configuration
openai.api_key = os.getenv("OPENAI_API_KEY")

# Simple in-memory cache
_content_cache = {}
_cache_expiry = {}
CACHE_DURATION = 24 * 60 * 60  # 24 hours in seconds

# Health categories
HEALTH_CATEGORIES = [
    "Nutrition", "Exercise", "Mental Health", "Sleep", "Preventive Care",
    "Common Illnesses", "First Aid", "Chronic Conditions", "Women's Health",
    "Men's Health", "Children's Health", "Elderly Care"
]

# Fallback content for when API fails
FALLBACK_CONTENT = {
    "Nutrition": [
        {
            "title": "The Importance of Balanced Diet",
            "content": "A balanced diet provides essential nutrients needed for good health. Include fruits, vegetables, whole grains, lean proteins, and healthy fats in your daily meals. Aim for variety and moderation to ensure you get all necessary vitamins and minerals. Proper nutrition supports immune function, energy levels, and overall wellbeing.",
            "category": "Nutrition",
            "color": "#4CAF50"
        },
        {
            "title": "Hydration and Health",
            "content": "Drinking enough water is crucial for bodily functions. Aim for 8 glasses daily, more during hot weather or physical activity. Water helps regulate body temperature, lubricate joints, and transport nutrients. Signs of dehydration include headaches, fatigue, and dark urine. Make water your primary beverage for optimal health.",
            "category": "Nutrition",
            "color": "#4CAF50"
        }
    ],
    "Exercise": [
        {
            "title": "Benefits of Regular Physical Activity",
            "content": "Regular exercise improves cardiovascular health, strengthens muscles, and enhances mood. Aim for at least 150 minutes of moderate activity weekly. Even short walks provide benefits. Find activities you enjoy to make exercise sustainable. Remember to include both cardio and strength training for balanced fitness.",
            "category": "Exercise",
            "color": "#2196F3"
        }
    ],
    "Mental Health": [
        {
            "title": "Managing Stress Effectively",
            "content": "Chronic stress impacts both mental and physical health. Practice stress management through deep breathing, meditation, or gentle movement. Regular exercise and adequate sleep also help reduce stress levels. Don't hesitate to seek professional support when needed. Remember that managing stress is an essential part of overall health.",
            "category": "Mental Health",
            "color": "#9C27B0"
        }
    ],
    "Sleep": [
        {
            "title": "Importance of Quality Sleep",
            "content": "Quality sleep is essential for cognitive function, immune health, and emotional wellbeing. Adults should aim for 7-9 hours nightly. Establish a regular sleep schedule and create a restful environment. Limit screen time before bed and avoid caffeine in the afternoon. Consistent sleep habits contribute significantly to overall health.",
            "category": "Sleep",
            "color": "#673AB7"
        }
    ],
    "Preventive Care": [
        {
            "title": "Regular Health Screenings",
            "content": "Preventive screenings help detect health issues early when they're most treatable. Schedule regular check-ups with your healthcare provider. Common screenings include blood pressure, cholesterol, and cancer screenings appropriate for your age and risk factors. Staying current with vaccinations is also an important aspect of preventive healthcare.",
            "category": "Preventive Care",
            "color": "#00BCD4"
        }
    ]
}

# Initialize fallback content for remaining categories
for category in HEALTH_CATEGORIES:
    if category not in FALLBACK_CONTENT:
        FALLBACK_CONTENT[category] = [{
            "title": f"Important {category} Tips",
            "content": f"Taking care of your {category.lower()} is essential for overall wellbeing. Regular attention to this aspect of health can prevent problems and improve quality of life. Consult healthcare professionals for personalized advice regarding {category.lower()}.",
            "category": category,
            "color": "#FF9800"
        }]

def get_color_for_category(category):
    """Return a consistent color for each category"""
    colors = {
        "Nutrition": "#4CAF50",      # Green
        "Exercise": "#2196F3",       # Blue
        "Mental Health": "#9C27B0",  # Purple
        "Sleep": "#673AB7",          # Deep Purple
        "Preventive Care": "#00BCD4", # Cyan
        "Common Illnesses": "#FF5722", # Deep Orange
        "First Aid": "#F44336",      # Red
        "Chronic Conditions": "#795548", # Brown
        "Women's Health": "#E91E63", # Pink
        "Men's Health": "#3F51B5",   # Indigo
        "Children's Health": "#FFEB3B", # Yellow
        "Elderly Care": "#607D8B"    # Blue Grey
    }
    return colors.get(category, "#FF9800")  # Default to amber if category not found

def generate_awareness_content(category, count=3):
    """Generate health awareness content using OpenAI"""
    cache_key = f"{category}_{count}"
    
    # Check cache first
    if cache_key in _content_cache and datetime.now() < _cache_expiry.get(cache_key, datetime.now()):
        logger.info(f"Returning cached content for {category}")
        return _content_cache[cache_key]
    
    try:
        # Define the prompt for GPT-4
        prompt = f"""
        Generate {count} informative and evidence-based health awareness articles about {category}.
        For each article, provide:
        1. An attention-grabbing title (5-8 words)
        2. Informative content (100-150 words) with practical advice
        3. Make the content culturally sensitive and appropriate for diverse populations
        4. Ensure medical accuracy and avoid overly technical language
        
        Format as a JSON array with objects containing 'title' and 'content' fields.
        """
        
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Call the OpenAI API with timeout handling
        start_time = time.time()
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a health educator providing accurate, helpful health information."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            max_tokens=1500,
            timeout=10  # 10 second timeout
        )
        elapsed_time = time.time() - start_time
        logger.info(f"OpenAI API call for {category} took {elapsed_time:.2f} seconds")
        
        # Parse the response
        response_content = json.loads(response.choices[0].message.content)
        
        # Ensure we have the expected structure
        if "articles" not in response_content and isinstance(response_content, list):
            # If it's a list, assume it's the articles directly
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
                # Fallback if structure is unexpected
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
        
        # Return fallback content for the category
        return FALLBACK_CONTENT.get(category, FALLBACK_CONTENT["Preventive Care"])

def get_all_categories():
    """Return list of all health categories"""
    return HEALTH_CATEGORIES

def get_random_awareness_content(count=5):
    """Get random awareness content across categories"""
    import random
    
    result = []
    categories = random.sample(HEALTH_CATEGORIES, min(count, len(HEALTH_CATEGORIES)))
    
    for category in categories:
        try:
            # Get just one article per category for variety
            content = generate_awareness_content(category, count=1)
            if content:
                result.extend(content)
        except Exception as e:
            logger.error(f"Error getting random content for {category}: {str(e)}")
            # Add fallback content
            fallback = FALLBACK_CONTENT.get(category, [])[0:1]
            result.extend(fallback)
    
    return result

# For testing purposes
if __name__ == "__main__":
    print("Testing awareness content generation...")
    for category in ["Nutrition", "Exercise", "Mental Health"]:
        content = generate_awareness_content(category, 2)
        print(f"\n=== {category} ===")
        for article in content:
            print(f"Title: {article['title']}")
            print(f"Content: {article['content'][:50]}...")
