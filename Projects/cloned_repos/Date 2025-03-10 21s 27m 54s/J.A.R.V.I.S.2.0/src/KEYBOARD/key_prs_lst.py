import pynput.keyboard as keyboard
import speech_recognition as sr
import threading

# Global flags and variables
is_listening = False
recognized_text = ""  # Variable to store recognized text
recognizer = sr.Recognizer()

def recognize_speech():
    """Recognize speech from the microphone continuously."""
    global recognized_text
    with sr.Microphone() as source:
        print("Listening for speech...")
        recognizer.adjust_for_ambient_noise(source)  # To adjust for ambient noise

        while is_listening:  # Continuously listen while 'up' key is pressed
            try:
                audio = recognizer.listen(source, timeout=10)  # Timeout to prevent endless listening
                print("Recognizing...")
                recognized_text = recognizer.recognize_google(audio)  # Store recognized text globally
                print(f"Recognized: {recognized_text}")
            
            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand audio")
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")
            except sr.WaitTimeoutError:
                print("Listening timed out.")

def on_press(key):
    """Handles key press events."""
    global is_listening
    try:
        # Only start recording if the 'up' key is pressed and we're not already listening
        if key == keyboard.Key.up and not is_listening:  
            is_listening = True
            print("Recording started... Listening for speech.")
            threading.Thread(target=recognize_speech, daemon=True).start()  # Start speech recognition in a new thread
    except AttributeError:
        pass

def on_release(key):
    """Handles key release events."""
    global is_listening
    if key == keyboard.Key.up and is_listening:  # Release 'up' to stop recording
        is_listening = False
        print("Recording stopped.")
        print(f"Final recorded text: {recognized_text}")  # Output the final recognized text when stopped

    if key == keyboard.Key.esc:  # Press 'esc' to exit the program
        return False

if __name__ == "__main__":
    print("Press and hold the 'up' key to start recording. Release it to stop recording.")
    try:
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()
    except KeyboardInterrupt:
        print("\nProgram interrupted. Exiting...")
