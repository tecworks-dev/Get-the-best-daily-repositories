# Claude Vision Object Detection

A powerful Python tool that leverages Claude 3.5 Sonnet Vision API to detect and visualize objects in images. The script automatically draws bounding boxes around detected objects, labels them, and displays confidence scores.

![Example Output](output/detected_test.webp)

## Features

- 🖼️ Process single images or entire directories
- 📦 Automatic object detection with bounding boxes
- 🎯 High-precision confidence scores
- 🎨 Vibrant, distinct colors for each detected object
- 💾 Saves annotated images with detection results

## Requirements

- Python 3.7+
- Anthropic API key
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/doriandarko/claude-vision-object-detection.git
cd claude-vision-detection
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root and add your Anthropic API key:
```
ANTHROPIC_API_KEY=your_api_key_here
```

## Usage

1. Run the script:
```bash
python main.py
```

2. When prompted, enter either:
   - Path to a single image file
   - Path to a directory containing multiple images

3. The script will:
   - Process each image using Claude Vision API
   - Draw bounding boxes around detected objects
   - Add labels with confidence scores
   - Save annotated images in an `output` directory

## Supported Image Formats

- JPEG (.jpg, .jpeg)
- PNG (.png)
- GIF (.gif)
- WebP (.webp)

## Output

The script creates an `output` directory in the current working directory. Processed images are saved with the prefix `detected_` followed by the original filename.

### Error Handling

The script includes comprehensive error handling for:
- Invalid image paths
- Unsupported file formats
- API communication issues
- Image processing errors

## Contributing

Contributions are welcome! Please feel free to submit pull requests or create issues for bugs and feature requests.

## License

Copyright (c) 2024 Pietro Schirano

This project is licensed under a modified MIT License with attribution requirements. See the [LICENSE](LICENSE) file for details.

### Attribution Requirements

When using this software or its derivatives, you must include:
- The name of the original project (Claude Vision Object Detection)
- A link to the original repository
- The name of the original author

## Acknowledgments

- Built using the Claude 3.5 Sonnet Vision API by Anthropic
- Uses PIL (Python Imaging Library) for image processing
- Implements the golden ratio for color generation

## Support

For issues, questions, or contributions, please:
1. Check existing GitHub issues
2. Create a new issue with a detailed description
3. Include sample images if relevant (without sensitive data)

---

*Note: This tool relies on the Claude Vision API, which requires an API key from Anthropic. Make sure you have appropriate access and credits before using.*
