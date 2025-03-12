# Image Transformations with OpenCV

This tutorial covers various image transformation techniques using OpenCV, including geometric transformations, perspective transformations, and image warping.

## Table of Contents
1. [Geometric Transformations](#geometric-transformations)
2. [Affine Transformations](#affine-transformations)
3. [Perspective Transformations](#perspective-transformations)
4. [Image Warping](#image-warping)
5. [Interpolation Methods](#interpolation-methods)

## Geometric Transformations

### Scaling (Resizing)

```python
import cv2
import numpy as np

def resize_image(image, scale_factor=None, dimensions=None):
    """
    Resize image using different methods
    """
    if scale_factor is not None:
        # Resize by scale factor
        resized = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)
    elif dimensions is not None:
        # Resize to specific dimensions
        resized = cv2.resize(image, dimensions)
    else:
        raise ValueError("Either scale_factor or dimensions must be provided")
    
    return resized

# Example usage
image = cv2.imread('input.jpg')
# Resize to 50% of original size
resized_scale = resize_image(image, scale_factor=0.5)
# Resize to specific dimensions
resized_dim = resize_image(image, dimensions=(800, 600))
```

### Rotation

```python
def rotate_image(image, angle, center=None, scale=1.0):
    """
    Rotate image by given angle
    """
    height, width = image.shape[:2]
    
    if center is None:
        center = (width // 2, height // 2)
    
    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    
    # Perform rotation
    rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
    
    return rotated

# Example usage
rotated_image = rotate_image(image, angle=45)
```

### Translation

```python
def translate_image(image, x, y):
    """
    Translate image by (x,y)
    """
    height, width = image.shape[:2]
    
    # Create translation matrix
    translation_matrix = np.float32([[1, 0, x],
                                   [0, 1, y]])
    
    # Perform translation
    translated = cv2.warpAffine(image, translation_matrix, (width, height))
    
    return translated

# Example usage
translated_image = translate_image(image, 100, 50)  # Move 100px right, 50px down
```

## Affine Transformations

```python
def affine_transform(image, src_points, dst_points):
    """
    Apply affine transformation
    """
    # Get affine transform matrix
    affine_matrix = cv2.getAffineTransform(src_points, dst_points)
    
    # Apply transformation
    height, width = image.shape[:2]
    transformed = cv2.warpAffine(image, affine_matrix, (width, height))
    
    return transformed

# Example usage
src_pts = np.float32([[0,0], [width-1,0], [0,height-1]])
dst_pts = np.float32([[width*0.2,height*0.1], [width*0.9,height*0.2], 
                      [width*0.1,height*0.9]])
affine_image = affine_transform(image, src_pts, dst_pts)
```

## Perspective Transformations

```python
def perspective_transform(image, src_points, dst_points):
    """
    Apply perspective transformation
    """
    # Get perspective transform matrix
    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Apply transformation
    height, width = image.shape[:2]
    transformed = cv2.warpPerspective(image, perspective_matrix, (width, height))
    
    return transformed

# Example usage
src_pts = np.float32([[0,0], [width-1,0], [width-1,height-1], [0,height-1]])
dst_pts = np.float32([[width*0.1,height*0.1], [width*0.9,height*0.1],
                      [width*0.9,height*0.9], [width*0.1,height*0.9]])
perspective_image = perspective_transform(image, src_pts, dst_pts)
```

## Image Warping

### Polar Transformation

```python
def polar_warp(image, center=None, maxRadius=None):
    """
    Convert image to polar coordinates
    """
    if center is None:
        center = (image.shape[1]//2, image.shape[0]//2)
    if maxRadius is None:
        maxRadius = min(center[0], center[1])
    
    # Linear Polar
    polar = cv2.linearPolar(image, center, maxRadius, cv2.WARP_FILL_OUTLIERS)
    
    # Log Polar
    log_polar = cv2.logPolar(image, center, maxRadius, cv2.WARP_FILL_OUTLIERS)
    
    return polar, log_polar
```

### Remapping

```python
def remap_image(image):
    """
    Demonstrate image remapping
    """
    height, width = image.shape[:2]
    
    # Create maps
    map_x = np.zeros((height, width), np.float32)
    map_y = np.zeros((height, width), np.float32)
    
    # Populate maps
    for i in range(height):
        for j in range(width):
            map_x[i,j] = j
            map_y[i,j] = height - i - 1
    
    # Apply remapping
    remapped = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
    
    return remapped
```

## Interpolation Methods

```python
def compare_interpolation_methods(image, new_size):
    """
    Compare different interpolation methods
    """
    methods = {
        'INTER_NEAREST': cv2.INTER_NEAREST,
        'INTER_LINEAR': cv2.INTER_LINEAR,
        'INTER_CUBIC': cv2.INTER_CUBIC,
        'INTER_LANCZOS4': cv2.INTER_LANCZOS4,
        'INTER_AREA': cv2.INTER_AREA
    }
    
    results = {}
    for name, method in methods.items():
        results[name] = cv2.resize(image, new_size, interpolation=method)
    
    return results
```

## Best Practices and Tips

1. **Choosing Interpolation Methods**
   - Use INTER_AREA for shrinking
   - Use INTER_CUBIC or INTER_LINEAR for enlarging
   - Use INTER_NEAREST for binary images

2. **Handling Borders**
   - Consider border effects in transformations
   - Use appropriate border modes
   - Add padding when necessary

3. **Performance Optimization**
   - Cache transformation matrices for repeated operations
   - Use fixed-point arithmetic when possible
   - Consider using GPU acceleration for large images

4. **Common Issues and Solutions**
   - Handle image distortion
   - Maintain aspect ratio
   - Deal with information loss

## Applications

1. **Image Registration**
   - Medical imaging
   - Satellite imagery
   - Panorama stitching

2. **Camera Calibration**
   - Lens distortion correction
   - Perspective correction
   - 3D reconstruction

3. **Document Processing**
   - Document scanning
   - Text deskewing
   - OCR preprocessing

4. **Visual Effects**
   - Image warping
   - Special effects
   - Artistic transformations

## Further Reading

1. [OpenCV Documentation](https://docs.opencv.org/)
2. Computer Vision textbooks
3. Research papers on image transformation
4. Advanced topics in geometric computer vision