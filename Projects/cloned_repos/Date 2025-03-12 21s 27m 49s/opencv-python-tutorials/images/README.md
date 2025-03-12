# Sample Images for OpenCV Tutorials

This directory contains sample images used throughout the OpenCV tutorials. Using these shared images ensures consistency across examples and makes it easier for users to follow along.

## Adding Your Own Images

To use these tutorials with your own images:

1. Download sample images from public domain sources like [Pexels](https://www.pexels.com/), [Unsplash](https://unsplash.com/), or [Pixabay](https://pixabay.com/)
2. Add them to this directory
3. Use the same filenames referenced in the tutorials, or update the scripts to use your filenames

## Recommended Images to Download

For the best experience with these tutorials, download the following types of images:

1. `chessboard.jpg` - A chessboard pattern (for camera calibration)
2. `faces.jpg` - An image with one or more faces (for face detection)
3. `shapes.jpg` - An image with various geometric shapes (for contour detection)
4. `landscape.jpg` - A natural scene (for general processing)
5. `text.jpg` - An image containing text (for OCR examples)
6. `noisy.jpg` - An image with noise (for filtering examples)
7. `low_contrast.jpg` - A low contrast image (for histogram equalization)
8. `objects.jpg` - An image with multiple distinct objects (for object detection)
9. `panorama_1.jpg` and `panorama_2.jpg` - Overlapping images (for stitching examples)
10. `stereo_left.jpg` and `stereo_right.jpg` - Stereo pair images (for depth estimation)

## Usage in Scripts

When using these images in the tutorial scripts, use relative paths like:

```python
import cv2
import os

# Get the path to the images directory relative to the script
script_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.join(os.path.dirname(script_dir), 'images')

# Load an image
img = cv2.imread(os.path.join(images_dir, 'landscape.jpg'))
```

This ensures that the scripts can find the images regardless of which directory they're run from.

## License Information

Please ensure any images you use comply with appropriate licensing. The tutorials are designed to work with any suitable images, so feel free to use your own copyright-compliant images.