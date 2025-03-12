# Image Thresholding with OpenCV

Thresholding is a simple yet effective technique for image segmentation. It separates pixels in the image that are of interest from the background based on their intensity values.

## Basic Concept

In its simplest form, thresholding replaces each pixel in an image with a black pixel if the image intensity is less than a fixed constant (the threshold value), or a white pixel if the intensity is greater than that constant.

## Simple Thresholding

OpenCV provides the `cv2.threshold()` function for basic thresholding:

```python
import cv2
import numpy as np

# Read the image in grayscale
img = cv2.imread('image.jpg', cv2.IMREAD_GRAYSCALE)

# Apply simple thresholding
# Parameters:
# - source image (must be grayscale)
# - threshold value
# - maximum value to use with the THRESH_BINARY and THRESH_BINARY_INV thresholding types
# - thresholding type
ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
```

### Thresholding Types

1. **THRESH_BINARY**: If pixel intensity is greater than the threshold, set it to the maximum value, otherwise set it to 0.
   ```
   dst(x,y) = maxVal if src(x,y) > thresh else 0
   ```

2. **THRESH_BINARY_INV**: Inverse of THRESH_BINARY.
   ```
   dst(x,y) = 0 if src(x,y) > thresh else maxVal
   ```

3. **THRESH_TRUNC**: If pixel intensity is greater than the threshold, set it to the threshold, otherwise keep it as is.
   ```
   dst(x,y) = thresh if src(x,y) > thresh else src(x,y)
   ```

4. **THRESH_TOZERO**: If pixel intensity is greater than the threshold, keep it as is, otherwise set it to 0.
   ```
   dst(x,y) = src(x,y) if src(x,y) > thresh else 0
   ```

5. **THRESH_TOZERO_INV**: Inverse of THRESH_TOZERO.
   ```
   dst(x,y) = 0 if src(x,y) > thresh else src(x,y)
   ```

## Adaptive Thresholding

Simple thresholding uses a global value for all pixels, which may not be ideal when an image has different lighting conditions in different areas. Adaptive thresholding calculates the threshold for small regions of the image, giving us different thresholds for different regions.

```python
# Apply adaptive thresholding
# Parameters:
# - source image (must be grayscale)
# - maximum value to use with the THRESH_BINARY and THRESH_BINARY_INV thresholding types
# - adaptive method (ADAPTIVE_THRESH_MEAN_C or ADAPTIVE_THRESH_GAUSSIAN_C)
# - thresholding type (usually THRESH_BINARY or THRESH_BINARY_INV)
# - block size (size of the pixel neighborhood used to calculate the threshold)
# - constant subtracted from the mean or weighted mean
adaptive_thresh1 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)
adaptive_thresh2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
```

### Adaptive Methods

1. **ADAPTIVE_THRESH_MEAN_C**: The threshold value is the mean of the neighborhood area minus the constant.
2. **ADAPTIVE_THRESH_GAUSSIAN_C**: The threshold value is a weighted sum (Gaussian window) of the neighborhood values minus the constant.

## Otsu's Thresholding

Otsu's method determines an optimal global threshold value from the image histogram. It works best for bimodal images (images with two distinct peaks in their histogram).

```python
# Apply Otsu's thresholding
# Combine simple thresholding with Otsu's method using the additional flag THRESH_OTSU
ret, otsu_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# The threshold value is returned as 'ret'
print(f"Otsu's threshold value: {ret}")
```

## Multi-level Thresholding

For more complex images, you might need multiple threshold levels. This can be achieved by applying thresholding multiple times with different values or using more advanced techniques.

```python
# Example of multi-level thresholding
ret1, thresh1 = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
ret2, thresh2 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
ret3, thresh3 = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
ret4, thresh4 = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

# Combine the results
multi_level = np.zeros_like(img)
multi_level = np.where(thresh1 == 255, 64, multi_level)
multi_level = np.where(thresh2 == 255, 128, multi_level)
multi_level = np.where(thresh3 == 255, 192, multi_level)
multi_level = np.where(thresh4 == 255, 255, multi_level)
```

## Practical Applications

Thresholding is used in many applications:

1. **Document Scanning**: Converting scanned documents to binary images for OCR
2. **Medical Imaging**: Segmenting regions of interest in medical scans
3. **Object Detection**: Separating objects from backgrounds
4. **Quality Control**: Detecting defects in manufacturing
5. **License Plate Recognition**: Isolating characters on license plates

## Tips for Effective Thresholding

1. **Pre-processing**: Apply Gaussian blur to reduce noise before thresholding
2. **Histogram Analysis**: Analyze the image histogram to determine appropriate threshold values
3. **Adaptive vs. Global**: Use adaptive thresholding for images with varying illumination
4. **Otsu's Method**: Use Otsu's method when the image histogram has two distinct peaks
5. **Post-processing**: Apply morphological operations to clean up the thresholded image

## Practical Example

Check out the accompanying Python script (`thresholding.py`) for a complete example demonstrating these concepts.

## Next Steps

Now that you understand image thresholding, you're ready to move on to more advanced topics like image gradients and edge detection in the next tutorial.