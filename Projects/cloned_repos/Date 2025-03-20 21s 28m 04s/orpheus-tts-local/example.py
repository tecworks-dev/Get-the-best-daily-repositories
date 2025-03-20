#!/usr/bin/env python3
"""
Simple example of using Orpheus TTS as a library.
This script demonstrates how to generate speech and save it to a file.
"""

from gguf_orpheus import generate_speech_from_api, AVAILABLE_VOICES

def text_to_speech(text, voice="tara", output_file=None):
    """
    Convert text to speech using Orpheus TTS.
    
    Args:
        text (str): The text to convert to speech
        voice (str): The voice to use (default: tara)
        output_file (str): Path to save the audio file (default: None)
    
    Returns:
        list: Audio segments
    """
    print(f"Converting: '{text}' with voice '{voice}'")
    
    # Generate speech
    audio_segments = generate_speech_from_api(
        prompt=text,
        voice=voice,
        output_file=output_file
    )
    
    return audio_segments

def main():
    # Example 1: Generate speech with Tara voice
    text_to_speech(
        "Hello, I'm Tara. This is an example of using Orpheus TTS as a library.",
        voice="tara",
        output_file="example_tara.wav"
    )
    
    # Example 2: Generate speech with a different voice
    text_to_speech(
        "Hi there, I'm Leo. I have a different voice than Tara.",
        voice="leo",
        output_file="example_leo.wav"
    )
    
    print("All available voices:")
    for voice in AVAILABLE_VOICES:
        print(f"- {voice}")

if __name__ == "__main__":
    main() 