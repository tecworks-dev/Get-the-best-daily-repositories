# Drawing and Writing on Images with OpenCV

This tutorial covers how to draw various shapes and write text on images using OpenCV. These functions are essential for annotating images, creating visualizations, and highlighting regions of interest.

## Creating a Blank Image

Before we start drawing, let's create a blank image to work with:

```python
import cv2
import numpy as np

# Create a blank black image
# Parameters: height, width, channels (3 for BGR)
height, width = 500, 800
img = np.zeros((height, width, 3), dtype=np.uint8)

# Create a white image instead
white_img = np.ones((height, width, 3), dtype=np.uint8) * 255
```

## Drawing Lines

Draw a line using the `cv2.line()` function:

```python
# Draw a line
# Parameters:
# - image
# - start point (x1, y1)
# - end point (x2, y2)
# - color in BGR
# - thickness in pixels
cv2.line(img, (0, 0), (width, height), (0, 255, 0), 3)

# Draw a dashed line
def draw_dashed_line(img, pt1, pt2, color, thickness, dash_length):
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
    dashes = int(dist / dash_length)
    for i in range(dashes):
        start = (int(pt1[0] + (pt2[0] - pt1[0]) * i / dashes),
                 int(pt1[1] + (pt2[1] - pt1[1]) * i / dashes))
        end = (int(pt1[0] + (pt2[0] - pt1[0]) * (i + 0.5) / dashes),
               int(pt1[1] + (pt2[1] - pt1[1]) * (i + 0.5) / dashes))
        cv2.line(img, start, end, color, thickness)

draw_dashed_line(img, (0, height//2), (width, height//2), (255, 0, 0), 3, 20)
```

## Drawing Rectangles

Draw rectangles using the `cv2.rectangle()` function:

```python
# Draw a rectangle
# Parameters:
# - image
# - top-left corner (x1, y1)
# - bottom-right corner (x2, y2)
# - color in BGR
# - thickness (negative thickness fills the rectangle)
cv2.rectangle(img, (100, 100), (300, 300), (0, 0, 255), 2)

# Draw a filled rectangle
cv2.rectangle(img, (400, 100), (600, 300), (255, 255, 0), -1)
```

## Drawing Circles

Draw circles using the `cv2.circle()` function:

```python
# Draw a circle
# Parameters:
# - image
# - center (x, y)
# - radius
# - color in BGR
# - thickness (negative thickness fills the circle)
cv2.circle(img, (width//2, height//2), 100, (255, 0, 255), 3)

# Draw a filled circle
cv2.circle(img, (200, 400), 50, (0, 255, 255), -1)
```

## Drawing Ellipses

Draw ellipses using the `cv2.ellipse()` function:

```python
# Draw an ellipse
# Parameters:
# - image
# - center (x, y)
# - axes lengths (major, minor)
# - angle (rotation of the ellipse)
# - start angle
# - end angle
# - color in BGR
# - thickness (negative thickness fills the ellipse)
cv2.ellipse(img, (width//2, height//2), (100, 50), 0, 0, 360, (255, 255, 255), 2)

# Draw a rotated ellipse
cv2.ellipse(img, (width//2, height//2), (100, 50), 45, 0, 360, (0, 255, 0), 2)

# Draw a partial ellipse (arc)
cv2.ellipse(img, (width//2, height//2), (100, 50), 135, 0, 180, (0, 0, 255), 2)
```

## Drawing Polygons

Draw polygons using the `cv2.polylines()` function:

```python
# Define vertices of a polygon
pts = np.array([[100, 50], [200, 300], [400, 200], [300, 100]], np.int32)
# Reshape to the required format
pts = pts.reshape((-1, 1, 2))

# Draw a polygon
# Parameters:
# - image
# - array of polygons (each an array of points)
# - whether to close the polygon
# - color in BGR
# - thickness
cv2.polylines(img, [pts], True, (0, 255, 255), 3)

# Draw a filled polygon using fillPoly
cv2.fillPoly(img, [pts], (255, 0, 0))
```

## Adding Text

Add text to images using the `cv2.putText()` function:

```python
# Add text to an image
# Parameters:
# - image
# - text string
# - bottom-left corner of the text
# - font type
# - font scale
# - color in BGR
# - thickness
# - line type
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'OpenCV Drawing', (50, 50), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

# Available fonts:
# cv2.FONT_HERSHEY_SIMPLEX
# cv2.FONT_HERSHEY_PLAIN
# cv2.FONT_HERSHEY_DUPLEX
# cv2.FONT_HERSHEY_COMPLEX
# cv2.FONT_HERSHEY_TRIPLEX
# cv2.FONT_HERSHEY_COMPLEX_SMALL
# cv2.FONT_HERSHEY_SCRIPT_SIMPLEX
# cv2.FONT_HERSHEY_SCRIPT_COMPLEX
```

## Text Properties and Positioning

To position text properly, you can get the text size:

```python
# Get the size of the text
text = "Hello OpenCV"
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
thickness = 2
(text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

# Calculate position to center the text
textX = (img.shape[1] - text_width) // 2
textY = (img.shape[0] + text_height) // 2

# Draw the text
cv2.putText(img, text, (textX, textY), font, font_scale, (255, 255, 255), thickness)
```

## Drawing Arrows

Draw arrows using the `cv2.arrowedLine()` function:

```python
# Draw an arrow
# Parameters:
# - image
# - start point
# - end point
# - color in BGR
# - thickness
# - line type
# - shift
# - tipLength (length of the arrow tip as a fraction of the line length)
cv2.arrowedLine(img, (50, 400), (200, 400), (255, 255, 0), 2, tipLength=0.3)
```

## Drawing Markers

Draw markers (crosshairs) using the `cv2.drawMarker()` function:

```python
# Draw a marker
# Parameters:
# - image
# - position
# - marker type
# - marker size
# - color in BGR
# - thickness
# - line type
cv2.drawMarker(img, (400, 400), (0, 255, 0), cv2.MARKER_CROSS, 20, 2)

# Available marker types:
# cv2.MARKER_CROSS
# cv2.MARKER_TILTED_CROSS
# cv2.MARKER_STAR
# cv2.MARKER_DIAMOND
# cv2.MARKER_SQUARE
# cv2.MARKER_TRIANGLE_UP
# cv2.MARKER_TRIANGLE_DOWN
```

## Practical Applications

Drawing functions are useful for:

1. **Annotation**: Labeling objects or regions in images
2. **Visualization**: Highlighting features or results
3. **User Interface**: Creating interactive elements
4. **Data Presentation**: Plotting graphs or charts
5. **Object Tracking**: Marking tracked objects with bounding boxes

## Practical Example

Check out the accompanying Python script (`drawing.py`) for a complete example demonstrating these concepts.

## Next Steps

Now that you understand how to draw shapes and write text on images, you're ready to move on to more advanced topics like image arithmetic and bitwise operations in the next tutorial.