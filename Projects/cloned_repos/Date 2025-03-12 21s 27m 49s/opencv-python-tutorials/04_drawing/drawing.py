#!/usr/bin/env python3
"""
Drawing and Writing on Images with OpenCV
This script demonstrates how to draw various shapes and write text on images
"""

import cv2
import numpy as np
import os

def main():
    # Create a blank black image
    height, width = 600, 800
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 1. Drawing Lines
    print("\n1. DRAWING LINES")
    
    # Draw a green line
    cv2.line(img, (0, 0), (width-1, height-1), (0, 255, 0), 3)
    
    # Draw a blue dashed line
    draw_dashed_line(img, (0, height//2), (width-1, height//2), (255, 0, 0), 2, 20)
    
    # Draw a red line with anti-aliasing
    cv2.line(img, (width-1, 0), (0, height-1), (0, 0, 255), 3, cv2.LINE_AA)
    
    # Display the result
    cv2.imshow('Lines', img.copy())
    print("Press any key to continue...")
    cv2.waitKey(0)
    
    # 2. Drawing Rectangles
    print("\n2. DRAWING RECTANGLES")
    
    # Draw a red rectangle
    cv2.rectangle(img, (100, 100), (300, 300), (0, 0, 255), 2)
    
    # Draw a filled yellow rectangle
    cv2.rectangle(img, (400, 100), (600, 300), (0, 255, 255), -1)
    
    # Display the result
    cv2.imshow('Rectangles', img.copy())
    print("Press any key to continue...")
    cv2.waitKey(0)
    
    # 3. Drawing Circles
    print("\n3. DRAWING CIRCLES")
    
    # Draw a magenta circle
    cv2.circle(img, (width//2, height//2), 100, (255, 0, 255), 3)
    
    # Draw a filled cyan circle
    cv2.circle(img, (200, 400), 50, (255, 255, 0), -1)
    
    # Display the result
    cv2.imshow('Circles', img.copy())
    print("Press any key to continue...")
    cv2.waitKey(0)
    
    # 4. Drawing Ellipses
    print("\n4. DRAWING ELLIPSES")
    
    # Draw a white ellipse
    cv2.ellipse(img, (width//2, height//2), (100, 50), 0, 0, 360, (255, 255, 255), 2)
    
    # Draw a rotated green ellipse
    cv2.ellipse(img, (width//2, height//2), (100, 50), 45, 0, 360, (0, 255, 0), 2)
    
    # Draw a partial red ellipse (arc)
    cv2.ellipse(img, (width//2, height//2), (100, 50), 135, 0, 180, (0, 0, 255), 2)
    
    # Display the result
    cv2.imshow('Ellipses', img.copy())
    print("Press any key to continue...")
    cv2.waitKey(0)
    
    # 5. Drawing Polygons
    print("\n5. DRAWING POLYGONS")
    
    # Define vertices of a polygon
    pts = np.array([[100, 50], [200, 300], [400, 200], [300, 100]], np.int32)
    # Reshape to the required format
    pts = pts.reshape((-1, 1, 2))
    
    # Draw a cyan polygon outline
    polygon_img = img.copy()
    cv2.polylines(polygon_img, [pts], True, (255, 255, 0), 3)
    
    # Display the result
    cv2.imshow('Polygon Outline', polygon_img)
    print("Press any key to continue...")
    cv2.waitKey(0)
    
    # Draw a filled blue polygon
    filled_polygon_img = img.copy()
    cv2.fillPoly(filled_polygon_img, [pts], (255, 0, 0))
    
    # Display the result
    cv2.imshow('Filled Polygon', filled_polygon_img)
    print("Press any key to continue...")
    cv2.waitKey(0)
    
    # 6. Adding Text
    print("\n6. ADDING TEXT")
    
    # Create a new image for text examples
    text_img = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background
    
    # Add text with different fonts
    fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_DUPLEX, 
             cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_COMPLEX_SMALL,
             cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, cv2.FONT_HERSHEY_SCRIPT_COMPLEX]
    
    font_names = ["SIMPLEX", "PLAIN", "DUPLEX", "COMPLEX", "TRIPLEX", 
                  "COMPLEX_SMALL", "SCRIPT_SIMPLEX", "SCRIPT_COMPLEX"]
    
    for i, (font, name) in enumerate(zip(fonts, font_names)):
        y_pos = 50 + i * 60
        cv2.putText(text_img, f"OpenCV Font: {name}", (50, y_pos), font, 1, (0, 0, 0), 2, cv2.LINE_AA)
    
    # Display the result
    cv2.imshow('Text with Different Fonts', text_img)
    print("Press any key to continue...")
    cv2.waitKey(0)
    
    # 7. Centered Text
    print("\n7. CENTERED TEXT")
    
    # Create a new image for centered text example
    centered_img = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background
    
    # Get the size of the text
    text = "Hello OpenCV"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    thickness = 3
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Calculate position to center the text
    textX = (centered_img.shape[1] - text_width) // 2
    textY = (centered_img.shape[0] + text_height) // 2
    
    # Draw the text
    cv2.putText(centered_img, text, (textX, textY), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
    
    # Draw the bounding box (for demonstration)
    cv2.rectangle(centered_img, (textX, textY - text_height), (textX + text_width, textY + baseline), (0, 255, 0), 1)
    
    # Display the result
    cv2.imshow('Centered Text', centered_img)
    print("Press any key to continue...")
    cv2.waitKey(0)
    
    # 8. Drawing Arrows and Markers
    print("\n8. DRAWING ARROWS AND MARKERS")
    
    # Create a new image
    arrow_marker_img = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background
    
    # Draw arrows
    cv2.arrowedLine(arrow_marker_img, (50, 100), (200, 100), (255, 0, 0), 2, tipLength=0.3)
    cv2.arrowedLine(arrow_marker_img, (50, 200), (200, 200), (0, 255, 0), 2, tipLength=0.1)
    cv2.arrowedLine(arrow_marker_img, (50, 300), (200, 300), (0, 0, 255), 2, tipLength=0.5)
    
    # Draw markers
    marker_types = [cv2.MARKER_CROSS, cv2.MARKER_TILTED_CROSS, cv2.MARKER_STAR, 
                   cv2.MARKER_DIAMOND, cv2.MARKER_SQUARE, cv2.MARKER_TRIANGLE_UP, 
                   cv2.MARKER_TRIANGLE_DOWN]
    
    for i, marker in enumerate(marker_types):
        x_pos = 300 + i * 70
        cv2.drawMarker(arrow_marker_img, (x_pos, 200), (0, 0, 0), marker, 30, 2)
    
    # Display the result
    cv2.imshow('Arrows and Markers', arrow_marker_img)
    print("Press any key to continue...")
    cv2.waitKey(0)
    
    # 9. Practical Application: Annotating an Image
    print("\n9. PRACTICAL APPLICATION: ANNOTATING AN IMAGE")
    
    # Create a sample image or use a blank one
    annotated_img = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background
    
    # Draw a face (circle for head, lines for features)
    cv2.circle(annotated_img, (width//2, height//2), 100, (0, 0, 0), 2)  # Head
    cv2.circle(annotated_img, (width//2 - 30, height//2 - 30), 10, (0, 0, 0), -1)  # Left eye
    cv2.circle(annotated_img, (width//2 + 30, height//2 - 30), 10, (0, 0, 0), -1)  # Right eye
    cv2.ellipse(annotated_img, (width//2, height//2 + 20), (30, 10), 0, 0, 180, (0, 0, 0), 2)  # Smile
    
    # Add annotations
    cv2.putText(annotated_img, "Face", (width//2 - 30, height//2 - 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.arrowedLine(annotated_img, (width//2 - 30, height//2 - 130), 
                   (width//2, height//2 - 100), (0, 0, 255), 2)
    
    cv2.putText(annotated_img, "Eyes", (width//2 + 70, height//2 - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.arrowedLine(annotated_img, (width//2 + 60, height//2 - 30), 
                   (width//2 + 40, height//2 - 30), (255, 0, 0), 2)
    
    cv2.putText(annotated_img, "Smile", (width//2 + 70, height//2 + 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.arrowedLine(annotated_img, (width//2 + 60, height//2 + 30), 
                   (width//2 + 40, height//2 + 20), (0, 255, 0), 2)
    
    # Display the result
    cv2.imshow('Annotated Image', annotated_img)
    print("Press any key to exit...")
    cv2.waitKey(0)
    
    # Clean up
    cv2.destroyAllWindows()
    print("\nAll operations completed successfully!")

def draw_dashed_line(img, pt1, pt2, color, thickness, dash_length):
    """
    Draw a dashed line from pt1 to pt2 with specified color and thickness.
    dash_length is the length of each dash.
    """
    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
    dashes = int(dist / dash_length)
    for i in range(dashes):
        start = (int(pt1[0] + (pt2[0] - pt1[0]) * i / dashes),
                 int(pt1[1] + (pt2[1] - pt1[1]) * i / dashes))
        end = (int(pt1[0] + (pt2[0] - pt1[0]) * (i + 0.5) / dashes),
               int(pt1[1] + (pt2[1] - pt1[1]) * (i + 0.5) / dashes))
        cv2.line(img, start, end, color, thickness)

if __name__ == "__main__":
    main()