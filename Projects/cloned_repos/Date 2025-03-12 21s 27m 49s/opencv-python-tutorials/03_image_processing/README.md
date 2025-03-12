# Image Processing with OpenCV

This tutorial covers essential image processing techniques in OpenCV, including color space conversions, image transformations, and filtering operations.

## Color Space Conversions

OpenCV supports more than 150 color space conversions. The most commonly used ones are BGR ↔ Gray, BGR ↔ HSV, etc.

```python
import cv2
import numpy as np

# Read an image (in BGR format by default)
img = cv2.imread('image.jpg')

# Convert BGR to Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Convert BGR to HSV (Hue, Saturation, Value)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Convert BGR to RGB
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Convert BGR to LAB
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
```

### Why HSV?

HSV is often preferred for color-based segmentation because it separates the image intensity (Value) from the color information (Hue and Saturation). This makes it easier to filter colors regardless of lighting conditions.

## Geometric Transformations

### Translation

Translation is the shifting of an object's location:

```python
# Create a translation matrix
# tx = shift in x direction, ty = shift in y direction
tx, ty = 100, 50
translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])

# Apply the translation
# Parameters: image, translation matrix, output image dimensions
height, width = img.shape[:2]
translated = cv2.warpAffine(img, translation_matrix, (width, height))
```

### Rotation

Rotation around a specified point:

```python
# Define the rotation center, angle, and scale
center = (width // 2, height // 2)
angle = 45
scale = 1.0

# Calculate the rotation matrix
rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

# Apply the rotation
rotated = cv2.warpAffine(img, rotation_matrix, (width, height))
```

### Affine Transformation

Affine transformation preserves lines and parallelism but not angles or distances:

```python
# Define three points from the input image
src_points = np.float32([[0, 0], [width - 1, 0], [0, height - 1]])

# Define where those points will be in the output image
dst_points = np.float32([[0, 0], [width - 1, 0], [width // 3, height - 1]])

# Calculate the affine transformation matrix
affine_matrix = cv2.getAffineTransform(src_points, dst_points)

# Apply the affine transformation
affine = cv2.warpAffine(img, affine_matrix, (width, height))
```

### Perspective Transformation

Perspective transformation changes the perspective of an image:

```python
# Define four points in the input image
src_points = np.float32([[56, 65], [368, 52], [28, 387], [389, 390]])

# Define where those points will be in the output image
dst_points = np.float32([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]])

# Calculate the perspective transformation matrix
perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

# Apply the perspective transformation
perspective = cv2.warpPerspective(img, perspective_matrix, (width, height))
```

## Image Filtering

### Blurring/Smoothing

#### 1. Average Blurring

Simple averaging of pixels within a kernel area:

```python
# Apply average blur with a 5x5 kernel
blur = cv2.blur(img, (5, 5))
```

#### 2. Gaussian Blur

Weighted average where center pixels have higher weight:

```python
# Apply Gaussian blur with a 5x5 kernel and standard deviation of 0
gaussian = cv2.GaussianBlur(img, (5, 5), 0)
```

#### 3. Median Blur

Replaces each pixel with the median of neighboring pixels:

```python
# Apply median blur with a kernel size of 5
median = cv2.medianBlur(img, 5)
```

#### 4. Bilateral Filter

Preserves edges while reducing noise:

```python
# Apply bilateral filter
# Parameters: image, diameter of pixel neighborhood, sigma color, sigma space
bilateral = cv2.bilateralFilter(img, 9, 75, 75)
```

### Morphological Operations

#### 1. Erosion

Erodes away the boundaries of foreground objects:

```python
# Define a kernel
kernel = np.ones((5, 5), np.uint8)

# Apply erosion
erosion = cv2.erode(img, kernel, iterations=1)
```

#### 2. Dilation

Increases the white region in the image:

```python
# Apply dilation
dilation = cv2.dilate(img, kernel, iterations=1)
```

#### 3. Opening

Erosion followed by dilation, useful for removing noise:

```python
# Apply opening
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
```

#### 4. Closing

Dilation followed by erosion, useful for closing small holes:

```python
# Apply closing
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
```

#### 5. Morphological Gradient

Difference between dilation and erosion:

```python
# Apply morphological gradient
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
```

## Image Gradients

Gradients are used to detect edges in images:

### Sobel Derivatives

```python
# Apply Sobel in x direction
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)

# Apply Sobel in y direction
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

# Calculate the magnitude
magnitude = cv2.magnitude(sobelx, sobely)
```

### Laplacian Derivatives

```python
# Apply Laplacian
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
```

## Image Pyramids

Image pyramids are used for multi-scale processing:

### Gaussian Pyramid

```python
# Create a Gaussian pyramid (downsampling)
lower_res = cv2.pyrDown(img)  # Reduces to half the size
even_lower_res = cv2.pyrDown(lower_res)  # Reduces to quarter the size
```

### Laplacian Pyramid

```python
# Create a Laplacian pyramid
higher_res = cv2.pyrUp(lower_res)  # Increases size but loses detail
laplacian = cv2.subtract(img, higher_res)  # Contains the lost detail
```

## Practical Example

Check out the accompanying Python script (`image_processing.py`) for a complete example demonstrating these concepts.

## Next Steps

Now that you understand various image processing techniques in OpenCV, you're ready to move on to more advanced topics like drawing and writing on images in the next tutorial.