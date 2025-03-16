# Conversation Data Annotation Tool

This tool provides a web-based interface for running and annotating conversational workflows. It helps visualize and track various components of conversational AI systems including Named Entity Recognition (NER), search functionality, and response critiques.

## Features

- **Workflow Visualization**: Step-by-step workflow tracking
- **Real-time Data Display**: View NER results, search calls, and search results as they happen
- **Conversation History**: Track and manage the full conversation history
- **Response Evaluation**: View critiques of AI responses
- **Asynchronous Processing**: Background processing via Celery for long-running tasks
- **Real-time Updates**: WebSocket-based updates for immediate UI feedback
- **Conversation Management**: Save, clear, and restart conversations

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- Redis server (for Celery)
- API keys for the following services:
  - OpenAI
  - Together AI
  - Anthropic Claude
  - Groq

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/conversation-annotation-tool.git
   cd conversation-annotation-tool
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create API key files in the project root:
   - `openai.key`: Your OpenAI API key
   - `together.key`: Your Together API key
   - `claude.key`: Your Anthropic Claude API key
   - `groq.key`: Your Groq API key

5. Start Redis server:
   ```
   redis-server
   ```

6. Start Celery worker:
   ```
   celery -A app.celery worker --loglevel=info
   ```

7. Start the Flask application:
   ```
   python app.py
   ```

8. Visit `http://localhost:5000` in your browser

## Project Structure

```
conversation-annotation-tool/
├── app.py                 # Main Flask application
├── templates/
│   └── index.html         # Main UI template
├── static/
│   └── css/
│       └── style.css      # CSS styles
├── saved_conversations/   # Directory for saved conversations
├── logs/                  # Application logs
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── *.key                  # API key files (not included in repo)
```

## Usage Guide

1. **Start Workflow**: Click "Start Workflow" to initialize a new conversation
2. **View Data**: Use the tabs to switch between different data views
3. **Next Step**: Click "Next" to proceed to the next step in the workflow
4. **Regenerate**: Use "Regenerate Assistant Response" to get a new response
5. **Save**: Save the conversation for future reference
6. **Clear**: Clear the current conversation while keeping persona and requirements
7. **Reset**: Reset everything and start from scratch

## Requirements

- Flask
- Flask-SocketIO
- Celery
- Redis
- OpenAI
- Together
- Anthropic
- Groq

## License

MIT License