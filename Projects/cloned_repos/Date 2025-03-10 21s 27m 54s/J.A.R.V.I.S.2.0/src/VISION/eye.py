# Adapted from OpenAI's Vision example 
import base64
import cv2 
from PIL import Image 
from src.FUNCTION.get_env import load_variable
from google import genai




def resize_image(image_path:str , require_width=336 , require_height=336) -> bool:
    with Image.open(image_path) as img:
        width, height = img.size
        if height <= require_height and width <= require_width:
            return True 
        try:
            img = img.resize((require_width, require_height), Image.ANTIALIAS)
            img.save(image_path)
            print(f"Image saved to {image_path}, size: {require_width}x{require_height}")
        except Exception as e:
            print(e)
            return False 
        return True 
    
def capture_image_and_save(image_path="captured_image.png") -> None:
    # Initialize the camera
    cap = cv2.VideoCapture(0)  # 0 is the default camera

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return None 

    try:
        # Capture a single frame
        ret, frame = cap.read()

        if ret:
            # Save the image in PNG format
            cv2.imwrite(image_path, frame)
            print(f"Image captured and saved as {image_path}")
            return image_path 
        else:
            print("Error: Could not capture image.")
            return None 
    finally:
        # Release the camera
        cap.release()
        cv2.destroyAllWindows()


def detect_image(image_path:str) -> str | None:
    image = Image.open(image_path)
    genai_key = load_variable("genai-key")
    client = genai.Client(api_key=genai_key)
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=["What is this image?", image])
        return response.text 
    except Exception as e:
        print(f"Error: {e}")
        return None 


# if __name__ == "__main__":
#     image = capture_image_and_save()
    
#image = capture_image_and_save()