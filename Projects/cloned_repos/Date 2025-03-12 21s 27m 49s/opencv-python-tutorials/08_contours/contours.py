#!/usr/bin/env python3
"""
Contours Tutorial Script
This script demonstrates various contour operations and applications using OpenCV
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def create_sample_image():
    """Create a sample image with basic shapes for contour detection"""
    # Create a black image
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    
    # Draw some shapes
    # Rectangle
    cv2.rectangle(img, (50, 50), (150, 150), (255, 255, 255), -1)
    
    # Circle
    cv2.circle(img, (300, 100), 50, (255, 255, 255), -1)
    
    # Triangle
    pts = np.array([[250, 300], [150, 400], [350, 400]], np.int32)
    cv2.fillPoly(img, [pts], (255, 255, 255))
    
    # Star shape
    center = (400, 300)
    pts = []
    for i in range(5):
        # Outer points of the star
        x = center[0] + 50 * np.cos(2 * np.pi * i / 5 - np.pi / 2)
        y = center[1] + 50 * np.sin(2 * np.pi * i / 5 - np.pi / 2)
        pts.append([int(x), int(y)])
        
        # Inner points of the star
        x = center[0] + 25 * np.cos(2 * np.pi * i / 5 + np.pi / 5 - np.pi / 2)
        y = center[1] + 25 * np.sin(2 * np.pi * i / 5 + np.pi / 5 - np.pi / 2)
        pts.append([int(x), int(y)])
    
    pts = np.array(pts, np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(img, [pts], (255, 255, 255))
    
    return img

def find_and_draw_contours(image, mode=cv2.RETR_EXTERNAL):
    """Find and draw contours on an image"""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply threshold
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, hierarchy = cv2.findContours(thresh, mode, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a copy for drawing
    img_contours = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    # Draw all contours
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2)
    
    return img_contours, contours, hierarchy

def analyze_contour_properties(image, contours):
    """Analyze and display various properties of contours"""
    img_properties = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    for i, contour in enumerate(contours):
        # Calculate basic properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Calculate centroid
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            
            # Draw centroid
            cv2.circle(img_properties, (cx, cy), 5, (0, 0, 255), -1)
            
            # Add text with properties
            cv2.putText(img_properties, f"Area: {area:.0f}", (cx - 40, cy + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(img_properties, f"Perimeter: {perimeter:.0f}", (cx - 40, cy + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    return img_properties

def draw_shape_approximations(image, contours):
    """Draw various shape approximations for contours"""
    img_approx = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    for contour in contours:
        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img_approx, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Minimum area rectangle
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img_approx, [box], 0, (0, 255, 0), 2)
        
        # Minimum enclosing circle
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)
        cv2.circle(img_approx, center, radius, (0, 0, 255), 2)
        
        # Fit ellipse if possible (needs at least 5 points)
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            cv2.ellipse(img_approx, ellipse, (255, 255, 0), 2)
    
    return img_approx

def identify_shapes(image, contours):
    """Identify and label basic shapes in the image"""
    img_shapes = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    for contour in contours:
        # Skip small contours
        if cv2.contourArea(contour) < 100:
            continue
        
        # Calculate shape properties
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        
        # Get the center of the contour
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            # Identify shape based on vertices
            if len(approx) == 3:
                shape = "Triangle"
            elif len(approx) == 4:
                # Check if it's a square or rectangle
                x, y, w, h = cv2.boundingRect(approx)
                ar = w / float(h)
                shape = "Square" if 0.95 <= ar <= 1.05 else "Rectangle"
            elif len(approx) == 5:
                shape = "Pentagon"
            elif len(approx) == 10:
                shape = "Star"
            elif len(approx) > 10:
                shape = "Circle"
            else:
                shape = "Unknown"
            
            # Draw the shape name
            cv2.putText(img_shapes, shape, (cx - 40, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    return img_shapes

def convex_hull_demo(image, contours):
    """Demonstrate convex hull calculation"""
    img_hull = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    for contour in contours:
        # Calculate convex hull
        hull = cv2.convexHull(contour)
        
        # Draw original contour in green
        cv2.drawContours(img_hull, [contour], -1, (0, 255, 0), 2)
        
        # Draw convex hull in blue
        cv2.drawContours(img_hull, [hull], -1, (255, 0, 0), 2)
        
        # Calculate and display convexity defects if possible
        if len(contour) >= 5:
            hull_indices = cv2.convexHull(contour, returnPoints=False)
            try:
                defects = cv2.convexityDefects(contour, hull_indices)
                if defects is not None:
                    for i in range(defects.shape[0]):
                        s, e, f, d = defects[i, 0]
                        far = tuple(contour[f][0])
                        cv2.circle(img_hull, far, 5, (0, 0, 255), -1)
            except:
                pass
    
    return img_hull

def main():
    # Create or load an image
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), 'images')
    
    # Try to load an image from the images directory, or create a sample one
    try:
        img_path = os.path.join(images_dir, 'shapes.jpg')
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
        else:
            img = create_sample_image()
            # Save the sample image
            if not os.path.exists(images_dir):
                os.makedirs(images_dir)
            cv2.imwrite(os.path.join(images_dir, 'shapes.jpg'), img)
            print("Created and saved sample shapes image")
    except:
        img = create_sample_image()
    
    # 1. Basic Contour Detection
    print("\n1. Basic Contour Detection")
    img_contours, contours, hierarchy = find_and_draw_contours(img)
    plt.figure(figsize=(10, 5))
    plt.subplot(121), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image'), plt.axis('off')
    plt.subplot(122), plt.imshow(cv2.cvtColor(img_contours, cv2.COLOR_BGR2RGB))
    plt.title('Contours Detected'), plt.axis('off')
    plt.show()
    
    # 2. Contour Properties
    print("\n2. Analyzing Contour Properties")
    img_properties = analyze_contour_properties(img, contours)
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(img_properties, cv2.COLOR_BGR2RGB))
    plt.title('Contour Properties'), plt.axis('off')
    plt.show()
    
    # 3. Shape Approximations
    print("\n3. Shape Approximations")
    img_approx = draw_shape_approximations(img, contours)
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(img_approx, cv2.COLOR_BGR2RGB))
    plt.title('Shape Approximations'), plt.axis('off')
    plt.show()
    
    # 4. Shape Identification
    print("\n4. Shape Identification")
    img_shapes = identify_shapes(img, contours)
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(img_shapes, cv2.COLOR_BGR2RGB))
    plt.title('Shape Identification'), plt.axis('off')
    plt.show()
    
    # 5. Convex Hull
    print("\n5. Convex Hull Demonstration")
    img_hull = convex_hull_demo(img, contours)
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(img_hull, cv2.COLOR_BGR2RGB))
    plt.title('Convex Hull'), plt.axis('off')
    plt.show()
    
    # 6. Hierarchy Demonstration
    print("\n6. Contour Hierarchy Demonstration")
    # Create an image with nested contours
    nested_img = np.zeros((500, 500, 3), dtype=np.uint8)
    cv2.rectangle(nested_img, (100, 100), (400, 400), (255, 255, 255), -1)
    cv2.rectangle(nested_img, (200, 200), (300, 300), (0, 0, 0), -1)
    cv2.circle(nested_img, (250, 250), 30, (255, 255, 255), -1)
    
    # Find contours with hierarchy
    img_hierarchy, contours, hierarchy = find_and_draw_contours(nested_img, cv2.RETR_TREE)
    
    # Print hierarchy information
    print("\nHierarchy Information:")
    for i, h in enumerate(hierarchy[0]):
        print(f"Contour {i}: Next: {h[0]}, Previous: {h[1]}, Child: {h[2]}, Parent: {h[3]}")
    
    plt.figure(figsize=(10, 5))
    plt.subplot(121), plt.imshow(cv2.cvtColor(nested_img, cv2.COLOR_BGR2RGB))
    plt.title('Nested Contours'), plt.axis('off')
    plt.subplot(122), plt.imshow(cv2.cvtColor(img_hierarchy, cv2.COLOR_BGR2RGB))
    plt.title('Hierarchy Visualization'), plt.axis('off')
    plt.show()
    
    print("\nAll contour operations completed successfully!")

if __name__ == "__main__":
    main()