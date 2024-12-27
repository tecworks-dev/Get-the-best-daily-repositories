from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import whisper
import asyncio
import json
from typing import Optional
import sounddevice as sd
import numpy as np

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Whisper model
model = whisper.load_model("base")

class AudioTranscriber:
    def __init__(self):
        self.stream = None
        self.is_recording = False
        self.buffer = []
        
    async def start_recording(self, sample_rate=16000):
        self.is_recording = True
        self.stream = sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype=np.float32,
            callback=self.audio_callback
        )
        self.stream.start()
        
    def audio_callback(self, indata, frames, time, status):
        if self.is_recording:
            self.buffer.append(indata.copy())
            
    def stop_recording(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
        self.is_recording = False
        
    def get_audio_data(self):
        if not self.buffer:
            return None
        audio_data = np.concatenate(self.buffer)
        self.buffer = []
        return audio_data

transcriber = AudioTranscriber()

@app.websocket("/ws/transcribe")
async def transcribe_audio(websocket: WebSocket):
    await websocket.accept()
    
    try:
        await transcriber.start_recording()
        
        while True:
            # Process audio in chunks
            await asyncio.sleep(2.0)  # Process every 2 seconds
            audio_data = transcriber.get_audio_data()
            
            if audio_data is not None:
                # Convert audio data to format expected by Whisper
                result = model.transcribe(audio_data)
                
                if result["text"]:
                    await websocket.send_json({
                        "type": "transcription",
                        "text": result["text"]
                    })
                    
    except Exception as e:
        print(f"Error in WebSocket connection: {e}")
    finally:
        transcriber.stop_recording()

@app.post("/summarize")
async def summarize_transcript(transcript: str):
    # TODO: Implement summarization using Qwen/GPT
    # For now, return a mock summary
    return {
        "overview": "Meeting summary will be generated here",
        "key_points": ["Point 1", "Point 2"],
        "action_items": ["Action 1", "Action 2"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
