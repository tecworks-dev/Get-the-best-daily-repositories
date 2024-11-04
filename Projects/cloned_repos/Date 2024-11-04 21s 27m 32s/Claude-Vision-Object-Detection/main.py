import os
import base64
import json
from PIL import Image, ImageDraw, ImageFont
import anthropic
from typing import List, Dict, Union
import glob
import sys
from dotenv import load_dotenv
import random
import colorsys

# Load environment variables from .env file
load_dotenv()

class ClaudeVisionProcessor:
    def __init__(self):
        """Initialize the processor with your Anthropic API key from .env."""
        self.client = anthropic.Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        # Create output directory in current path
        self.output_dir = os.path.join(os.getcwd(), 'output')
        os.makedirs(self.output_dir, exist_ok=True)

    def encode_image(self, image_path: str) -> tuple[str, str]:
        """Encode an image file to base64 and determine its media type."""
        with open(image_path, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

        ext = image_path.lower().split('.')[-1]
        media_types = {
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'png': 'image/png',
            'gif': 'image/gif',
            'webp': 'image/webp'
        }
        media_type = media_types.get(ext, 'image/jpeg')

        return encoded_string, media_type

    def process_images(self, image_paths: Union[str, List[str]]) -> Dict:
        """Process images to detect objects and their bounding boxes."""
        if isinstance(image_paths, str):
            if os.path.isdir(image_paths):
                image_paths = []
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.webp']:
                    image_paths.extend(glob.glob(os.path.join(image_paths, ext)))
                if not image_paths:
                    raise ValueError(f"No supported images found in directory {image_paths}")
            else:
                if not os.path.exists(image_paths):
                    raise ValueError(f"Image path does not exist: {image_paths}")
                image_paths = [image_paths]

        print(f"Processing {len(image_paths)} images...")

        content = []
        for idx, img_path in enumerate(image_paths, 1):
            print(f"Processing image {idx}: {img_path}")
            content.append({
                "type": "text",
                "text": f"Image {idx}:"
            })

            try:
                encoded_image, media_type = self.encode_image(img_path)
                content.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": encoded_image
                    }
                })
            except Exception as e:
                print(f"Error encoding image {img_path}: {str(e)}")
                continue

        # Specific system prompt for precise bounding box detection
        system_prompt = """
        You are an expert computer vision system. First describe the image in accurate details, then analyze the provided images and return ONLY a JSON object containing bounding boxes. Be super precise and try to detect as many objects as possible.
        Be accurate and try to detect as many objects as possible. Really open your eyes and see the world.

        Follow these strict rules:
        1. Output MUST be valid JSON with no additional text
        2. Each detected object must have:
           - 'element': descriptive name of the object
           - 'bbox': [x1, y1, x2, y2] coordinates (normalized 0-1)
           - 'confidence': confidence score (0-1)
        3. Use this exact format:
           {
             "image_number": [
               {
                 "element": "object_name",
                 "bbox": [x1, y1, x2, y2],
                 "confidence": 0.95
               }
             ]
           }
        4. Coordinates must be precise and properly normalized
        5. DO NOT include any explanation or additional text
        """

        try:
            print("Analyzing images with Claude...")
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=8000,
                system=system_prompt,
                messages=[{
                    "role": "user",
                    "content": content
                }]
            )

            response_text = response.content[0].text
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1

            if json_start >= 0 and json_end > json_start:
                bboxes = json.loads(response_text[json_start:json_end])
                print("Successfully extracted bounding boxes")
                return bboxes
            else:
                raise ValueError("No valid JSON found in response")

        except Exception as e:
            print(f"Error processing images: {str(e)}")
            return None

    def get_random_color(self):
        """Generate a random vibrant color using HSV color space."""
        # Use golden ratio to generate well-distributed hues
        golden_ratio = 0.618033988749895
        hue = random.random()
        hue = (hue + golden_ratio) % 1.0
        
        # Use high saturation and value for vibrant colors
        saturation = 0.85 + random.random() * 0.15  # 0.85-1.00
        value = 0.85 + random.random() * 0.15       # 0.85-1.00
        
        # Convert HSV to RGB
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        
        # Convert to hex color string
        return '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        )

    def draw_bounding_boxes(self, image_path: str, bboxes: List[Dict]):
        """Draw bounding boxes on an image and save the result."""
        try:
            image = Image.open(image_path)  # Remove the .convert('RGBA')
            draw = ImageDraw.Draw(image)
            width, height = image.size

            # Try to load a larger font, fall back to default if not available
            try:
                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)  # Larger font size
            except:
                font = ImageFont.load_default()

            for bbox in bboxes:
                # Get a random bright color for this box
                color = self.get_random_color()

                x1, y1, x2, y2 = bbox['bbox']
                x1, x2 = x1 * width, x2 * width
                y1, y2 = y1 * height, y2 * height

                # Draw rectangle with thicker outline
                draw.rectangle([x1, y1, x2, y2], outline=color.strip(), width=4)

                # Draw label with confidence
                label = f"{bbox['element']} ({bbox['confidence']:.2f})"

                # Get text size for background
                text_bbox = draw.textbbox((x1, y1-40), label, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]

                # Draw black background for text for better visibility
                draw.rectangle([x1, y1-40, x1+text_width, y1-40+text_height],
                             fill=(0, 0, 0))

                # Draw text
                draw.text((x1, y1-40), label, fill=color.strip(), font=font)

            # Save to output directory
            basename = os.path.basename(image_path)
            output_path = os.path.join(self.output_dir, f'detected_{basename}')
            image.save(output_path)
            print(f"Saved annotated image to: {output_path}")

        except Exception as e:
            print(f"Error drawing bounding boxes for {image_path}: {str(e)}")

def main():
    """Process images with Claude Vision API"""
    print("\n=== Claude Vision Object Detection ===")
    print("Enter the path to an image file or a directory containing images.")
    input_path = input("\nPath: ").strip()

    if not os.path.exists(input_path):
        print(f"Error: Path '{input_path}' does not exist!")
        return

    try:
        processor = ClaudeVisionProcessor()
        result = processor.process_images(input_path)

        if result:
            for image_num, bboxes in result.items():
                if os.path.isdir(input_path):
                    # Find matching image in directory
                    images = glob.glob(os.path.join(input_path, f'*{image_num}.*'))
                    if images:
                        image_path = images[0]
                else:
                    image_path = input_path

                processor.draw_bounding_boxes(image_path, bboxes)

    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
