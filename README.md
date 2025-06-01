# HealthMate AI Backend

This directory contains the backend codebase for the HealthMate AI project, a multilingual health assistant designed to provide reliable health information in multiple Nigerian languages.

## 📋 Overview

The HealthMate AI backend is built with Flask and provides API endpoints for:

- Health query analysis
- Language detection and translation
- Health facts generation
- Health awareness content

## 🛠️ Technology Stack

- Python 3.10+
- Flask (Web framework)
- OpenAI API (AI processing)
- Azure Cognitive Services (Language detection, translation, health entity recognition)

## 🚀 Getting Started

### Prerequisites
- Python 3.10 or higher
- API keys for OpenAI and Azure services

### Installation

1. Create and activate a virtual environment:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r ../requirements.txt
```

3. Create a `.env` file in the project root (one level up) based on `.env.example` and add your API keys.

4. Start the backend server:
```bash
python app.py
```

5. The server will start on http://localhost:5000

## 📂 Project Structure

- `app.py` - Main Flask application and API routes
- `services/` - Core functionality modules
  - `health_analysis_service.py` - Health query processing
  - `translation_service.py` - Language detection and translation
  - `awareness_service.py` - Health awareness content generation
- `test_api.py` - API testing

## 🔌 API Reference

For detailed API documentation, see [API_DOCS.md](../API_DOCS.md) in the project root.

## 🧪 Testing

Run the tests with:
```bash
pytest
```

## 📝 Code Style

This project follows PEP 8 style guidelines. You can check your code style with:
```bash
flake8 .
```

## 🔒 Security Considerations

- API keys are stored in environment variables, not in the codebase
- CORS is enabled to allow frontend requests
- Input validation is performed on all endpoints
- Rate limiting is applied to prevent abuse

## 🚧 Development Guidelines

1. Create a new branch for your feature or bugfix
2. Write tests for new functionality
3. Document new API endpoints in API_DOCS.md
4. Follow the existing code style and patterns
5. Submit a pull request with a clear description of your changes

## 📄 License

This project is licensed under the MIT License - see the LICENSE file in the project root for details.
