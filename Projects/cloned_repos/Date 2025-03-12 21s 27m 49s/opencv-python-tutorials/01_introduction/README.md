# Introduction to OpenCV

## What is OpenCV?

OpenCV (Open Source Computer Vision Library) is an open-source library focused on computer vision, machine learning, and image processing. It was designed to provide a common infrastructure for computer vision applications and to accelerate the use of machine perception in commercial products.

The library has more than 2500 optimized algorithms, which includes a comprehensive set of both classic and state-of-the-art computer vision and machine learning algorithms. These algorithms can be used to:

- Detect and recognize faces
- Identify objects
- Classify human actions in videos
- Track camera movements
- Track moving objects
- Extract 3D models of objects
- Produce 3D point clouds from stereo cameras
- Stitch images together to produce a high-resolution image of an entire scene
- Find similar images from an image database
- Remove red eyes from images taken using flash
- Follow eye movements
- Recognize scenery and establish markers to overlay it with augmented reality
- And much more!

## History of OpenCV

- OpenCV was initially developed by Intel in 1999
- The first release came in 2000
- In 2008, Willow Garage took over support
- In 2012, a non-profit foundation OpenCV.org now maintains it
- Currently, OpenCV is available under a BSD license and hence it's free for both academic and commercial use

## OpenCV-Python

OpenCV-Python is the Python API for OpenCV. It combines the best qualities of OpenCV C++ API and Python language:

- Python is easy to learn and implement
- OpenCV-Python is a Python wrapper for the original C++ implementation
- OpenCV-Python uses Numpy, which is a highly optimized library for numerical operations
- All the OpenCV array structures are converted to and from Numpy arrays

## Installation

### Method 1: Using pip (Recommended)

```bash
# For main modules
pip install opencv-python

# For main modules and extra modules
pip install opencv-contrib-python
```

### Method 2: From Source

Building from source gives you more control but is more complex. See the [official documentation](https://docs.opencv.org/master/d2/de6/tutorial_py_setup_in_ubuntu.html) for details.

## Verifying Installation

Let's create a simple script to verify that OpenCV is installed correctly:

```python
import cv2
import numpy as np

# Print OpenCV version
print(f"OpenCV Version: {cv2.__version__}")

# Create a simple image
img = np.zeros((300, 300, 3), dtype=np.uint8)
img[:] = (255, 0, 0)  # Blue color in BGR

# Display text
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'OpenCV', (75, 150), font, 2, (0, 255, 255), 5)

# Show image
cv2.imshow('OpenCV Test', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## Basic Concepts

### Images in OpenCV

In OpenCV, images are represented as multi-dimensional arrays (numpy arrays). A color image is a 3D array with dimensions:

- Height (number of rows)
- Width (number of columns)
- Channels (typically 3 for BGR images)

A grayscale image is a 2D array with only height and width dimensions.

### Color Spaces

OpenCV uses BGR (Blue-Green-Red) color space by default, not RGB. This is important to remember when working with colors.

### Coordinate System

In OpenCV:
- The origin (0,0) is at the top-left corner of the image
- The x-coordinate increases moving right
- The y-coordinate increases moving down

## Next Steps

Now that you understand the basics of OpenCV, you're ready to move on to the next tutorial where we'll explore how to read, display, and manipulate images.

Check out the accompanying Python script (`intro.py`) to see these concepts in action.