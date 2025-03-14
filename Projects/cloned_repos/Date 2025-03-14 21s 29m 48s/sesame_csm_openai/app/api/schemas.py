# Update app/api/schemas.py
from enum import Enum
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field


class Voice(str, Enum):
    """Voice options for CSM model - these are just speaker IDs"""
    alloy = "0"  # Speaker 0
    echo = "1"   # Speaker 1
    fable = "2"  # Speaker 2
    onyx = "3"   # Speaker 3
    nova = "4"   # Speaker 4
    shimmer = "5" # Speaker 5


class ResponseFormat(str, Enum):
    mp3 = "mp3"
    opus = "opus"
    aac = "aac"
    flac = "flac"
    wav = "wav"


class TTSRequest(BaseModel):
    model: str = Field("csm-1b", description="The TTS model to use")
    input: str = Field(..., description="The text to generate audio for")
    voice: Voice = Field(..., description="The voice to use for generation")
    response_format: Optional[ResponseFormat] = Field(ResponseFormat.mp3, description="The format of the audio response")
    speed: Optional[float] = Field(1.0, description="The speed of the audio", ge=0.25, le=4.0)
    
    # These are CSM-specific parameters that aren't in standard OpenAI API
    max_audio_length_ms: Optional[float] = Field(90000, description="Maximum audio length in milliseconds")
    temperature: Optional[float] = Field(0.9, description="Sampling temperature", ge=0.0, le=2.0)
    topk: Optional[int] = Field(50, description="Top-k for sampling", ge=1, le=100)

    # Make all optional fields truly optional in JSON
    class Config:
        populate_by_name = True
        extra = "ignore"  # Allow extra fields without error


class TTSResponse(BaseModel):
    """Only used for API documentation"""
    pass