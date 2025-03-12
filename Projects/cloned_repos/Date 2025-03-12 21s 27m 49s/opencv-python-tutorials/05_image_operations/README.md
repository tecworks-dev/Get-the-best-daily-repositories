# Image Arithmetic and Bitwise Operations with OpenCV

This tutorial covers image arithmetic operations (addition, subtraction, etc.) and bitwise operations (AND, OR, XOR, NOT) in OpenCV. These operations are fundamental for many image processing tasks such as image blending, masking, and extracting regions of interest.

## Image Arithmetic Operations

### Addition

Adding two images combines their pixel values. OpenCV provides the `cv2.add()` function which performs a saturated operation (values are clipped to 255 if they exceed it).

```python
import cv2
import numpy as np

# Load two images
img1 = cv2.imread('image1.jpg')
img2 = cv2.imread('image2.jpg')

# Resize the second image to match the first if needed
img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

# Add the images
added_img = cv2.add(img1, img2)

# Display the result
cv2.imshow('Added Image', added_img)
cv2.waitKey(0)
```

### Weighted Addition (Blending)

Weighted addition allows you to specify the contribution of each image to the final result. This is useful for creating transparent overlays or smooth transitions.

```python
# Blend the images with weights
# Parameters:
# - First image
# - Weight of the first image
# - Second image
# - Weight of the second image
# - Scalar added to each sum
alpha = 0.7  # Weight of the first image
beta = 0.3   # Weight of the second image
blended_img = cv2.addWeighted(img1, alpha, img2, beta, 0)

# Display the result
cv2.imshow('Blended Image', blended_img)
cv2.waitKey(0)
```

### Subtraction

Subtracting one image from another can be used to find differences between images or remove backgrounds.

```python
# Subtract img2 from img1
subtracted_img = cv2.subtract(img1, img2)

# Display the result
cv2.imshow('Subtracted Image', subtracted_img)
cv2.waitKey(0)
```

### Multiplication

Multiplying images is useful for masking operations.

```python
# Multiply the images
multiplied_img = cv2.multiply(img1, img2)

# Display the result
cv2.imshow('Multiplied Image', multiplied_img)
cv2.waitKey(0)
```

### Division

Division can be used for normalization or to remove lighting effects.

```python
# Divide img1 by img2
divided_img = cv2.divide(img1, img2)

# Display the result
cv2.imshow('Divided Image', divided_img)
cv2.waitKey(0)
```

## Bitwise Operations

Bitwise operations work on binary images (or treat images as binary patterns) and are particularly useful for creating masks and extracting regions.

### Creating a Mask

First, let's create a simple mask to demonstrate bitwise operations:

```python
# Create a black image
height, width = img1.shape[:2]
mask = np.zeros((height, width), dtype=np.uint8)

# Draw a white circle in the middle
center = (width // 2, height // 2)
radius = 100
cv2.circle(mask, center, radius, 255, -1)

# Display the mask
cv2.imshow('Mask', mask)
cv2.waitKey(0)
```

### Bitwise AND

The bitwise AND operation extracts the region of the image that corresponds to the white region in the mask.

```python
# Apply the mask using bitwise AND
# Parameters:
# - First image
# - Second image
masked_img = cv2.bitwise_and(img1, img1, mask=mask)

# Display the result
cv2.imshow('Masked Image (AND)', masked_img)
cv2.waitKey(0)
```

### Bitwise OR

The bitwise OR operation combines the pixels of two images.

```python
# Create another image with a different shape
img3 = np.zeros((height, width, 3), dtype=np.uint8)
cv2.rectangle(img3, (width//4, height//4), (3*width//4, 3*height//4), (0, 0, 255), -1)

# Apply bitwise OR
or_img = cv2.bitwise_or(img1, img3)

# Display the result
cv2.imshow('OR Operation', or_img)
cv2.waitKey(0)
```

### Bitwise XOR

The bitwise XOR operation returns 1 only if the corresponding bits are different.

```python
# Apply bitwise XOR
xor_img = cv2.bitwise_xor(img1, img3)

# Display the result
cv2.imshow('XOR Operation', xor_img)
cv2.waitKey(0)
```

### Bitwise NOT

The bitwise NOT operation inverts the image, changing black to white and vice versa.

```python
# Apply bitwise NOT
not_img = cv2.bitwise_not(img1)

# Display the result
cv2.imshow('NOT Operation', not_img)
cv2.waitKey(0)
```

## Practical Applications

### Image Blending with Masks

You can use masks to blend specific regions of images:

```python
# Create a mask with a gradient
gradient_mask = np.zeros((height, width), dtype=np.uint8)
for i in range(width):
    gradient_mask[:, i] = i * 255 // width

# Blend images using the gradient mask
img1_masked = cv2.bitwise_and(img1, img1, mask=gradient_mask)
img2_masked = cv2.bitwise_and(img2, img2, mask=cv2.bitwise_not(gradient_mask))
blended_with_mask = cv2.add(img1_masked, img2_masked)

# Display the result
cv2.imshow('Blended with Mask', blended_with_mask)
cv2.waitKey(0)
```

### Background Removal

You can use bitwise operations to remove backgrounds:

```python
# Assume we have a foreground mask (e.g., from segmentation)
foreground_mask = np.zeros((height, width), dtype=np.uint8)
cv2.rectangle(foreground_mask, (width//4, height//4), (3*width//4, 3*height//4), 255, -1)

# Extract the foreground
foreground = cv2.bitwise_and(img1, img1, mask=foreground_mask)

# Create a colored background
background = np.ones((height, width, 3), dtype=np.uint8) * [0, 255, 0]  # Green background
background_mask = cv2.bitwise_not(foreground_mask)
background = cv2.bitwise_and(background, background, mask=background_mask)

# Combine foreground and new background
result = cv2.add(foreground, background)

# Display the result
cv2.imshow('Background Removal', result)
cv2.waitKey(0)
```

### Logo Watermarking

You can add a logo or watermark to an image:

```python
# Load or create a logo
logo = np.zeros((100, 100, 3), dtype=np.uint8)
cv2.circle(logo, (50, 50), 40, (0, 0, 255), -1)
cv2.putText(logo, 'CV', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Resize the logo if needed
logo_resized = cv2.resize(logo, (width//4, height//4))
logo_height, logo_width = logo_resized.shape[:2]

# Create a region of interest (ROI) in the top-right corner
roi = img1[0:logo_height, width-logo_width:width]

# Create a mask of the logo and its inverse
logo_gray = cv2.cvtColor(logo_resized, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(logo_gray, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

# Black-out the area of the logo in ROI
roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

# Take only the logo region from the logo image
logo_fg = cv2.bitwise_and(logo_resized, logo_resized, mask=mask)

# Put the logo in ROI and modify the original image
dst = cv2.add(roi_bg, logo_fg)
img1[0:logo_height, width-logo_width:width] = dst

# Display the result
cv2.imshow('Watermarked Image', img1)
cv2.waitKey(0)
```

## Tips and Best Practices

1. **Image Dimensions**: Ensure that images have the same dimensions before performing arithmetic operations
2. **Data Types**: Be aware of the data types of your images to avoid overflow or underflow
3. **Normalization**: Consider normalizing images before operations for consistent results
4. **Masking**: Use masks to restrict operations to specific regions of interest
5. **Saturation**: Remember that OpenCV's arithmetic operations are saturated (clipped to valid ranges)

## Practical Example

Check out the accompanying Python script (`image_operations.py`) for a complete example demonstrating these concepts.

## Next Steps

Now that you understand image arithmetic and bitwise operations, you're ready to move on to more advanced topics like image thresholding in the next tutorial.