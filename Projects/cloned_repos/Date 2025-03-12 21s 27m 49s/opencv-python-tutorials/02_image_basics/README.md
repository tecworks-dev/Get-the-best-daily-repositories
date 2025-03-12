# Image Basics with OpenCV

This tutorial covers the fundamental operations for working with images in OpenCV: reading, displaying, manipulating, and saving images.

## Reading Images

OpenCV provides the `cv2.imread()` function to read an image from a file:

```python
import cv2

# Read an image
# Parameters:
# - Path to the image file
# - Flag specifying how to read the image:
#   cv2.IMREAD_COLOR (1): Loads a color image (default)
#   cv2.IMREAD_GRAYSCALE (0): Loads image in grayscale mode
#   cv2.IMREAD_UNCHANGED (-1): Loads image as is including alpha channel
img = cv2.imread('image.jpg', cv2.IMREAD_COLOR)

# Check if image was loaded successfully
if img is None:
    print("Error: Could not read image")
else:
    print("Image loaded successfully")
```

## Displaying Images

To display an image in a window, use `cv2.imshow()`:

```python
# Create a window and display the image
# Parameters:
# - Window name (string)
# - Image to be shown
cv2.imshow('Image Window', img)

# Wait for a key press (0 means wait indefinitely)
cv2.waitKey(0)

# Close all OpenCV windows
cv2.destroyAllWindows()
```

## Image Properties

You can access various properties of an image:

```python
# Image dimensions (height, width, channels)
height, width, channels = img.shape
print(f"Image Dimensions: {width}x{height}")
print(f"Number of Channels: {channels}")

# Image data type
print(f"Image Data Type: {img.dtype}")

# Total number of pixels
print(f"Total Pixels: {img.size}")
```

For grayscale images, `img.shape` returns only height and width.

## Accessing and Modifying Pixels

Images in OpenCV are NumPy arrays, so you can access and modify pixels using array indexing:

```python
# Access a pixel value at row=100, col=50
# Returns [B, G, R] for color images
pixel = img[100, 50]
print(f"Pixel at (50, 100): {pixel}")

# Modify a pixel value
img[100, 50] = [255, 0, 0]  # Set to blue in BGR

# Access only blue channel of a pixel
blue = img[100, 50, 0]

# Modify only the green channel
img[100, 50, 1] = 255
```

## Region of Interest (ROI)

You can extract and process specific regions of an image:

```python
# Extract a region (rectangle from x=100, y=50 with width=200, height=150)
roi = img[50:200, 100:300]

# Display the ROI
cv2.imshow('Region of Interest', roi)
cv2.waitKey(0)

# Copy one region to another
img[300:450, 200:400] = roi
```

## Splitting and Merging Channels

For color images, you can work with individual color channels:

```python
# Split the BGR image into separate channels
b, g, r = cv2.split(img)

# Display individual channels
cv2.imshow('Blue Channel', b)
cv2.imshow('Green Channel', g)
cv2.imshow('Red Channel', r)
cv2.waitKey(0)

# Merge channels back together
merged_img = cv2.merge((b, g, r))
```

The `cv2.split()` function is computationally expensive. For simple channel access, NumPy indexing is more efficient:

```python
# More efficient way to access channels
b = img[:, :, 0]
g = img[:, :, 1]
r = img[:, :, 2]
```

## Resizing Images

Resize images using `cv2.resize()`:

```python
# Resize to specific dimensions
resized_img = cv2.resize(img, (800, 600))

# Resize by scaling factor
# fx and fy are scaling factors for width and height
half_size = cv2.resize(img, None, fx=0.5, fy=0.5)
double_size = cv2.resize(img, None, fx=2, fy=2)

# Specify interpolation method
# cv2.INTER_AREA - good for shrinking
# cv2.INTER_CUBIC, cv2.INTER_LINEAR - good for enlarging
resized_img = cv2.resize(img, (800, 600), interpolation=cv2.INTER_CUBIC)
```

## Rotating Images

Rotate images using transformation matrices:

```python
# Get the image dimensions
height, width = img.shape[:2]

# Define the rotation matrix
# Parameters: center point, angle (degrees), scale
rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), 45, 1)

# Apply the rotation
rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height))
```

## Flipping Images

Flip images horizontally, vertically, or both:

```python
# Flip horizontally (1)
horizontal_flip = cv2.flip(img, 1)

# Flip vertically (0)
vertical_flip = cv2.flip(img, 0)

# Flip both horizontally & vertically (-1)
both_flip = cv2.flip(img, -1)
```

## Saving Images

Save processed images using `cv2.imwrite()`:

```python
# Save an image to a file
# Parameters:
# - Output file name
# - Image to be saved
result = cv2.imwrite('output.jpg', img)

if result:
    print("Image saved successfully")
else:
    print("Error: Could not save image")
```

## Practical Example

Check out the accompanying Python script (`image_basics.py`) for a complete example demonstrating these concepts.

## Next Steps

Now that you understand the basics of working with images in OpenCV, you're ready to move on to more advanced image processing techniques in the next tutorial.