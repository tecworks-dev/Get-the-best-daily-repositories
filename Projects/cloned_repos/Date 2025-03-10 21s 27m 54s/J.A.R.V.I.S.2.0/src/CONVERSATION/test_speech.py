import speech_recognition as sr

def recognize_speech():
    """Recognize speech from the microphone."""
    recognizer = sr.Recognizer()
    #recognizer.dynamic_energy_threshold = True
    # recognizer.energy_threshold = 3000
    # recognizer.dynamic_energy_adjustment_damping =  0.07  # less more active
    # recognizer.dynamic_energy_ratio = 1.5
    # recognizer.pause_threshold = 0.6
    # recognizer.operation_timeout = None
    # recognizer.non_speaking_duration = 0.5
    
    
    recognizer.energy_threshold = 10000  # Higher to filter out background noise
    recognizer.dynamic_energy_adjustment_damping = 0.03  # Fast adjustments to changing noise
    recognizer.dynamic_energy_ratio = 2.0  # Speech must be significantly louder than noise
    recognizer.pause_threshold = 0.8  # Shorter pauses allowed
    recognizer.operation_timeout = 10  # Stops if no speech for 10s
    recognizer.non_speaking_duration = 0.4  # Ends sooner if background noise is stable

    with sr.Microphone() as source:
        print("[=] Adjusting for ambient noise... Please wait.")
        recognizer.adjust_for_ambient_noise(source)
        print("[+] Listening...")
        audio = recognizer.listen(source)

        try:
            print("Recognizing...")
            text = recognizer.recognize_google(audio)
            return text
        
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
            return None
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
            return None

# if __name__ == "__main__":
#     print("Say 'start listening' to activate, or 'exit' to quit.")
#     listening_mode = False
#     full_text = ""  # To store the complete speech input

#     while True:
#         spoken_text = recognize_speech()
#         if spoken_text:
#             print(f"You said: {spoken_text}")

#             if "hey jarvis" in spoken_text.lower():
#                 print("Listening mode activated.")
#                 listening_mode = True
#                 continue
                
#             elif "stop" in spoken_text.lower():
#                 listening_mode = False
#                 print("Listening mode deactivated...")  # Will still allow the program to continue running
#                 continue 
            
            
#             if listening_mode:
#                 full_text += spoken_text + " "  # Append the recognized speech
#                 print(f"Current text: {full_text}")
                
#                 if spoken_text.lower() == "exit":
#                     print("Exiting...")
#                     break  # Option to exit the program when 'exit' is said

