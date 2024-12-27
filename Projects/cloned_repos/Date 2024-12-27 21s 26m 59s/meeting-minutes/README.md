# Meeting Minutes - AI-Powered Meeting Assistant

An AI-powered meeting assistant that captures live meeting audio, transcribes it in real-time, and generates summaries while ensuring user privacy. Perfect for teams who want to focus on discussions while automatically capturing and organizing meeting content.

## Why?

While there are many meeting transcription tools available, this solution stands out by offering:
- **Privacy First**: All processing happens locally on your device
- **Cost Effective**: Uses open-source AI models instead of expensive APIs
- **Flexible**: Works offline, supports multiple meeting platforms
- **Customizable**: Self-host and modify for your specific needs
- **Intelligent**: Built-in knowledge graph for semantic search across meetings

## Features

✅ Modern, responsive UI with real-time updates

✅ Export to Markdown/PDF

🚧 Real-time audio capture using SoundDevice

🚧 Real-time audio visualization

🚧 Automatic meeting detection (Zoom, Google Meet, Teams)

🚧 Live audio transcription using OpenAI's Whisper

🚧 Real-time display of transcription

🚧 Post-meeting summarization

🚧 Local processing for privacy

## System Architecture

The application is built with a modern stack focusing on performance and user privacy. For detailed architecture documentation and diagrams, see [Architecture Documentation](docs/architecture.md).

![High Level Architecture](docs/Diagram-High%20level%20architecture%20diagram.jpg)

Key Components:

- **Frontend** (Electron JS + Next JS)
  - User interface and real-time updates
  - Cross-platform desktop application
  - WebSocket communication

- **Backend** (FastAPI)
  - Audio processing pipeline
  - AI integration and coordination
  - Database operations
  
- **AI Engine** (Whisper + Qwen/Llama 3.2)
  - Real-time transcription
  - Meeting summarization
  - Natural language processing

- **Storage**
  - Local SQLite database for secure data storage
  - Knowledge Graph/VectorDB for semantic search

- **Integration**
  - Virtual Audio Driver for system-level audio capture
  - Ollama with Agentic Tools for extended AI capabilities

## Prerequisites

- Node.js >= 18
- Python >= 3.9
- Virtual audio driver:
  - macOS: BlackHole (recommended)
  - Windows: Virtual Audio Cable
  - Linux: PulseAudio

## Setup Instructions

### 1. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server (Terminal 1)
npm run dev

# Start Electron app (Terminal 2)
npm start
```

### 2. Backend Setup

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Windows: .\venv\Scripts\activate

# Navigate to backend directory
cd backend

# Install dependencies
pip install -r requirements.txt

# Start backend server
cd app
uvicorn main:app --reload
```

### 3. Audio Setup

1. Install the virtual audio driver for your OS
2. Configure system audio to route through the virtual device
3. Verify audio routing in the application settings

## Development Guidelines

- Follow the established project structure
- Write tests for new features
- Document API changes
- Use type hints in Python code
- Follow ESLint configuration for JavaScript/TypeScript

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT License - Feel free to use this project for your own purposes.

Last updated: December 26, 2024
