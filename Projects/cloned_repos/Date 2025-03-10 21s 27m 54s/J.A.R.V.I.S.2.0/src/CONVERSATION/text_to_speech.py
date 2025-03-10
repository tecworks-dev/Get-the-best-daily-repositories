
import pyttsx3
from random import randint
def speak(text:str) -> None:
    try:
        # Initialize the TTS engine
        engine = pyttsx3.init()
        # Set the speaking rate
        try:
            rate = engine.getProperty('rate')
            engine.setProperty('rate', 154)  # Setting up a new speaking rate
        except Exception as e:
            print(f"Error setting rate: {e}")

        # Set the volume
        try:
            volume = engine.getProperty('volume')
            engine.setProperty('volume', 1.0)  # Setting volume level between 0 and 1
        except Exception as e:
            print(f"Error setting volume: {e}")

        # Set the voice
        try:
            voices = engine.getProperty('voices')
            choice = randint(0,1)
            if len(voices) > 15: 
                if choice == 1:
                    engine.setProperty('voice', voices[93].id)  # Set the voice by index
                else:
                    engine.setProperty('voice', voices[15].id)
            else:
                engine.setProperty('voice', voices[0].id)
        except Exception as e:
            print(f"Error setting voice: {e}")

        # Speak the text
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"Error speaking text: {e}")
        finally:
            engine.stop()

    except Exception as e:
        print(f"Error initializing TTS engine: {e}")

if __name__ == "__main__":
    speak("Good evening, sir. It’s good to see you again. Shall I begin your usual routine? ")
    speak("Welcome home, sir. Shall I initiate the house protocol and prepare everything for your arrival?")
    speak("hello how are you sir , good morning do you need any asistance.")