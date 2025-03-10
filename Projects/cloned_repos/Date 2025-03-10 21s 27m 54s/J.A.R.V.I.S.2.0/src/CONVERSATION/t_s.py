import threading
import time
import speech_recognition as sr

recognizer = sr.Recognizer()

def recognize_speech():
    """Recognize speech from the microphone."""
    recognizer.dynamic_energy_threshold = True
    recognizer.energy_threshold = 3000
    recognizer.dynamic_energy_adjustment_damping = 0.010  # More sensitive with lower values
    recognizer.dynamic_energy_ratio = 1.0
    recognizer.pause_threshold = 0.8
    recognizer.operation_timeout = None  # No timeout
    recognizer.non_speaking_duration = 0.5

    try:
        available = sr.Microphone.list_microphone_names()
        if len(available) <= 1:
            print("No microphones available.")
            return None 
        
        with sr.Microphone() as source:
            print("[=] Adjusting for ambient noise... Please wait.")
            recognizer.adjust_for_ambient_noise(source)  # Adjust for noise
            print("[+] Listening...")
            audio = recognizer.listen(source)  # Start listening

            print("Recognizing...")
            text = recognizer.recognize_google(audio)  # Recognize speech
            print(f"[Recognized] {text}")
            return text
        
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
        
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    return None 

def listen_in_background():
    """Runs speech recognition in the background."""
    while True:
        spoken_text = recognize_speech()
        if spoken_text:
            print(f"Detected speech: {spoken_text}")
        time.sleep(1)  # Adjust as needed

# Run speech recognition in a separate thread
listener_thread = threading.Thread(target=listen_in_background)
listener_thread.daemon = True
listener_thread.start()

# Main program keeps running while background thread listens
while True:
    time.sleep(5)  # Main thread sleeps, letting the listener thread do the work
