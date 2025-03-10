from pynput.keyboard import Listener

# Flag to track recording status
is_recording = False

def on_press(key):
    global is_recording
    try:
        if key == key.up:
            if not is_recording:
                is_recording = True
                print("Recording started...")
    except AttributeError:
        pass

def on_release(key):
    global is_recording
    try:
        if key == key.up:
            if is_recording:
                is_recording = False
                print("Recording stopped.")
                return False  # Stop listener after key release
    except AttributeError:
        pass

if __name__ == "__main__":
    print("Press and hold the 'up' key to start recording. Release it to stop recording.")
    
    # Start listening for key events
    with Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
