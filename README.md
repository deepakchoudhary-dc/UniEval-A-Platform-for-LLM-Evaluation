# AI Chatbot with Memory and Explainability

An advanced AI chatbot featuring human-like memory, transparent decision-making, and comprehensive search capabilities.

## Features

- **Memory System**: Persistent conversation memory with metadata tracking
- **Search Functionality**: Query past conversations and stored information
- **Explainability**: LIME/SHAP integration for transparent AI decisions
- **Fairness**: Bias detection and mitigation tools
- **Audit Logging**: Complete decision-making process tracking
- **Model Cards**: Automated documentation of AI capabilities
- This project also includes the security trails and observability cards for the LLMs.

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. Run the chatbot:
```bash
python main.py
```

## Project Structure

```
chatbot/
├── src/
│   ├── core/
│   │   ├── chatbot.py          # Main chatbot logic
│   │   ├── memory.py           # Memory management
│   │   └── search.py           # Search functionality
│   ├── explainability/
│   │   ├── lime_explainer.py   # LIME explanations
│   │   ├── shap_explainer.py   # SHAP explanations
│   │   └── model_card.py       # Model documentation
│   ├── fairness/
│   │   └── bias_detector.py    # Bias detection and mitigation
│   ├── data/
│   │   └── database.py         # Database operations
│   └── api/
│       └── routes.py           # API endpoints
├── tests/
├── config/
│   └── settings.py             # Configuration
└── main.py                     # Entry point
```

## Configuration

Edit `config/settings.py` to customize:
- Model parameters
- Memory retention policies
- Explainability settings
- Fairness thresholds

## Usage Examples

### Basic Chat
```python
from src.core.chatbot import TransparentChatbot

chatbot = TransparentChatbot()
response = chatbot.chat("What's the weather like today?")
print(response.answer)
print(response.explanation)
```

### Memory Search
```python
results = chatbot.search_memory("weather conversation")
for result in results:
    print(f"Found: {result.snippet}")
    print(f"From: {result.timestamp}")
```

### Get Explanation
```python
explanation = chatbot.explain_last_response()
print(f"Source: {explanation.data_source}")
print(f"Confidence: {explanation.confidence}")
```

## Development

Run tests:
```bash
pytest tests/
```

Format code:
```bash
black src/
flake8 src/
```

## License

MIT License - see LICENSE file for details.
