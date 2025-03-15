"""Audio processing utilities for CSM-1B TTS API."""
import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)

def remove_long_silences(
    audio: torch.Tensor, 
    sample_rate: int,
    min_speech_energy: float = 0.015,
    max_silence_sec: float = 0.4,
    keep_silence_sec: float = 0.1,
) -> torch.Tensor:
    """
    Remove uncomfortably long silences from audio while preserving natural pauses.
    
    Args:
        audio: Audio tensor
        sample_rate: Sample rate in Hz
        min_speech_energy: Minimum RMS energy to consider as speech
        max_silence_sec: Maximum silence duration to keep in seconds
        keep_silence_sec: Amount of silence to keep at speech boundaries
        
    Returns:
        Audio with long silences removed
    """
    # Convert to numpy for processing
    audio_np = audio.cpu().numpy()
    
    # Calculate frame size and hop length
    frame_size = int(0.02 * sample_rate)  # 20ms frames
    hop_length = int(0.01 * sample_rate)  # 10ms hop
    
    # Compute frame energy
    frames = []
    for i in range(0, len(audio_np) - frame_size + 1, hop_length):
        frames.append(audio_np[i:i+frame_size])
    
    if len(frames) < 2:  # If audio is too short for analysis
        return audio
        
    frames = np.array(frames)
    # Root mean square energy
    frame_energy = np.sqrt(np.mean(frames**2, axis=1))
    
    # Adaptive threshold based on audio content
    # Uses a percentile to adapt to different audio characteristics
    energy_threshold = max(
        min_speech_energy,  # Minimum threshold
        np.percentile(frame_energy, 10)  # Adapt to audio
    )
    
    # Identify speech frames
    is_speech = frame_energy > energy_threshold
    
    # Convert frame indices to sample indices considering overlapping frames
    speech_segments = []
    in_speech = False
    speech_start = 0
    
    for i in range(len(is_speech)):
        if is_speech[i] and not in_speech:
            # Start of speech
            in_speech = True
            # Calculate start sample including keep_silence
            speech_start = max(0, i * hop_length - int(keep_silence_sec * sample_rate))
            
        elif not is_speech[i] and in_speech:
            # Potential end of speech, look ahead to check if silence continues
            silence_length = 0
            for j in range(i, min(len(is_speech), i + int(max_silence_sec * sample_rate / hop_length))):
                if not is_speech[j]:
                    silence_length += 1
                else:
                    break
                    
            if silence_length * hop_length >= max_silence_sec * sample_rate:
                # End of speech, long enough silence detected
                in_speech = False
                # Calculate end sample including keep_silence
                speech_end = min(len(audio_np), i * hop_length + int(keep_silence_sec * sample_rate))
                speech_segments.append((speech_start, speech_end))
    
    # Handle the case where audio ends during speech
    if in_speech:
        speech_segments.append((speech_start, len(audio_np)))
    
    if not speech_segments:
        logger.warning("No speech segments detected, returning original audio")
        return audio
    
    # Combine speech segments with controlled silence durations
    result = []
    
    # Add initial silence if the first segment doesn't start at the beginning
    if speech_segments[0][0] > 0:
        # Add a short leading silence (100ms)
        silence_samples = min(int(0.1 * sample_rate), speech_segments[0][0])
        if silence_samples > 0:
            result.append(audio_np[speech_segments[0][0] - silence_samples:speech_segments[0][0]])
    
    # Process each speech segment
    for i, (start, end) in enumerate(speech_segments):
        # Add this speech segment
        result.append(audio_np[start:end])
        
        # Add a controlled silence between segments
        if i < len(speech_segments) - 1:
            next_start = speech_segments[i+1][0]
            # Calculate available silence duration
            available_silence = next_start - end
            
            if available_silence > 0:
                # Use either the actual silence (if shorter than max) or the max allowed
                silence_duration = min(available_silence, int(max_silence_sec * sample_rate))
                # Take the first portion of the silence - usually cleaner
                result.append(audio_np[end:end + silence_duration])
    
    # Combine all parts
    processed_audio = np.concatenate(result)
    
    # Log the results
    original_duration = len(audio_np) / sample_rate
    processed_duration = len(processed_audio) / sample_rate
    logger.info(f"Silence removal: {original_duration:.2f}s -> {processed_duration:.2f}s ({processed_duration/original_duration*100:.1f}%)")
    
    # Return as tensor with original device and dtype
    return torch.tensor(processed_audio, device=audio.device, dtype=audio.dtype)

def enhance_audio_quality(audio: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """
    Enhance audio quality by applying various processing techniques.
    
    Args:
        audio: Audio tensor
        sample_rate: Sample rate in Hz
        
    Returns:
        Enhanced audio tensor
    """
    try:
        audio_np = audio.cpu().numpy()
        
        # Remove DC offset
        audio_np = audio_np - np.mean(audio_np)
        
        # Apply light compression to improve perceived loudness
        from scipy import signal
        
        # Compress by reducing peaks and increasing quieter parts slightly
        threshold = 0.5
        ratio = 1.5
        attack = 0.01
        release = 0.1
        
        # Simple compression algorithm
        gain = np.ones_like(audio_np)
        for i in range(1, len(audio_np)):
            level = abs(audio_np[i])
            if level > threshold:
                gain[i] = threshold + (level - threshold) / ratio
                gain[i] = gain[i] / level if level > 0 else 1.0
            else:
                gain[i] = 1.0
            
            # Smooth gain changes
            gain[i] = gain[i-1] + (gain[i] - gain[i-1]) * (attack if gain[i] < gain[i-1] else release)
        
        audio_np = audio_np * gain
        
        # Optional: Apply subtle EQ to enhance speech clarity
        # Boost mid-high frequencies slightly
        sos = signal.butter(2, 4000, 'highshelf', fs=sample_rate, output='sos')
        audio_np = signal.sosfilt(sos, audio_np)
        
        # Normalize to prevent clipping
        max_val = np.max(np.abs(audio_np))
        if max_val > 0:
            audio_np = audio_np * 0.95 / max_val
        
        return torch.tensor(audio_np, device=audio.device, dtype=audio.dtype)
        
    except Exception as e:
        logger.warning(f"Audio quality enhancement failed: {e}")
        return audio