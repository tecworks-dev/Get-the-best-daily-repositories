# Contours in OpenCV

Contours are one of the most powerful features in OpenCV for shape analysis and object detection. They represent the boundaries of objects in an image and are essential for many computer vision applications.

## What are Contours?

Contours can be explained simply as a curve joining all the continuous points (along the boundary) having the same color or intensity. They are useful for shape analysis, object detection and recognition.

## Finding Contours

Before finding contours, we need to prepare our image:
1. Convert the image to grayscale
2. Apply thresholding or edge detection to get a binary image
3. Find contours using `cv2.findContours()`

```python
import cv2
import numpy as np

# Read image
img = cv2.imread('shapes.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply threshold
ret, thresh = cv2.threshold(gray, 127, 255, 0)

# Find contours
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours
cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
```

### Contour Retrieval Modes

OpenCV provides different modes for contour retrieval:

- `cv2.RETR_EXTERNAL`: Retrieves only the extreme outer contours
- `cv2.RETR_LIST`: Retrieves all contours without establishing any hierarchical relationships
- `cv2.RETR_CCOMP`: Retrieves all contours and organizes them into a two-level hierarchy
- `cv2.RETR_TREE`: Retrieves all contours and reconstructs a full hierarchy of nested contours

### Contour Approximation Methods

- `cv2.CHAIN_APPROX_NONE`: Stores all the contour points
- `cv2.CHAIN_APPROX_SIMPLE`: Compresses horizontal, vertical, and diagonal segments and leaves only their end points

## Contour Properties

Once we have the contours, we can extract various properties:

### 1. Contour Area

```python
area = cv2.contourArea(contour)
```

### 2. Contour Perimeter (Arc Length)

```python
perimeter = cv2.arcLength(contour, True)  # True indicates closed contour
```

### 3. Contour Approximation

We can approximate a contour shape to another shape with fewer vertices:

```python
epsilon = 0.02 * cv2.arcLength(contour, True)
approx = cv2.approxPolyDP(contour, epsilon, True)
```

### 4. Convex Hull

The convex hull is the smallest convex set that contains the contour:

```python
hull = cv2.convexHull(contour)
```

### 5. Checking Convexity

```python
is_convex = cv2.isContourConvex(contour)
```

### 6. Bounding Rectangle

There are two types of bounding rectangles:

**Straight Bounding Rectangle:**
```python
x, y, w, h = cv2.boundingRect(contour)
cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
```

**Rotated Rectangle:**
```python
rect = cv2.minAreaRect(contour)
box = cv2.boxPoints(rect)
box = np.int0(box)
cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
```

### 7. Minimum Enclosing Circle

```python
(x, y), radius = cv2.minEnclosingCircle(contour)
center = (int(x), int(y))
radius = int(radius)
cv2.circle(img, center, radius, (0, 255, 0), 2)
```

### 8. Fitting an Ellipse

```python
ellipse = cv2.fitEllipse(contour)
cv2.ellipse(img, ellipse, (0, 255, 0), 2)
```

### 9. Fitting a Line

```python
[vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
rows, cols = img.shape[:2]
lefty = int((-x * vy / vx) + y)
righty = int(((cols - x) * vy / vx) + y)
cv2.line(img, (cols-1, righty), (0, lefty), (0, 255, 0), 2)
```

## Contour Moments

Moments help us to calculate the center of mass, area, and other properties of a contour:

```python
M = cv2.moments(contour)
if M['m00'] != 0:
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    # Mark the center
    cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
```

## Shape Identification

We can identify basic shapes using contour approximation:

```python
def identify_shape(contour):
    shape = "unknown"
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
    
    # Triangle
    if len(approx) == 3:
        shape = "triangle"
    
    # Square or rectangle
    elif len(approx) == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)
        shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
    
    # Pentagon
    elif len(approx) == 5:
        shape = "pentagon"
    
    # Circle
    elif len(approx) > 8:
        shape = "circle"
    
    return shape
```

## Contour Hierarchy

Contours can have hierarchical relationships (parent-child). For example, if there's a contour inside another contour, the inner one is a child of the outer one.

The hierarchy array contains information about the relationship between contours:
- First value: Index of the next contour at the same level
- Second value: Index of the previous contour at the same level
- Third value: Index of the first child contour
- Fourth value: Index of the parent contour

```python
# Access hierarchy information
for i, contour in enumerate(contours):
    # Get hierarchy for this contour
    if hierarchy is not None:
        parent = hierarchy[0][i][3]
        if parent != -1:
            print(f"Contour {i} has parent contour {parent}")
```

## Practical Applications

### 1. Object Counting

```python
def count_objects(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY)[1]
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter small contours
    filtered_contours = [c for c in contours if cv2.contourArea(c) > 100]
    
    # Draw contours
    for contour in filtered_contours:
        cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
    
    return len(filtered_contours), image
```

### 2. Shape Detection

```python
def detect_shapes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blur, 60, 255, cv2.THRESH_BINARY)[1]
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Process each contour
    for contour in contours:
        if cv2.contourArea(contour) < 100:
            continue
            
        # Get shape name
        shape = identify_shape(contour)
        
        # Get the center of the contour
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            # Draw the contour and shape name
            cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
            cv2.putText(image, shape, (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 2)
    
    return image
```

### 3. Document Scanner

A simple document scanner can be built using contour detection:

```python
def scan_document(image):
    # Convert to grayscale and apply edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 75, 200)
    
    # Find contours
    contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    
    # Find the document contour
    document_contour = None
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        
        # If our approximated contour has four points, we can assume it's the document
        if len(approx) == 4:
            document_contour = approx
            break
    
    if document_contour is None:
        return None
    
    # Apply perspective transform
    return four_point_transform(image, document_contour.reshape(4, 2))
```

## Conclusion

Contours are a powerful tool in OpenCV for shape analysis and object detection. By understanding how to find, analyze, and manipulate contours, you can build a wide range of computer vision applications, from simple shape detection to complex document scanners and object tracking systems.

In the next tutorial, we'll explore histograms, which are another important tool for image analysis in OpenCV.