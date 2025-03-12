#!/usr/bin/env python3
"""
Introduction to OpenCV - Basic Script
This script demonstrates basic OpenCV functionality to verify installation
"""

import cv2
import numpy as np

def main():
    # Print OpenCV version
    print(f"OpenCV Version: {cv2.__version__}")
    
    # Create a simple black image (zeros = black)
    height, width = 500, 700
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Draw a blue rectangle
    # Parameters: image, top-left corner, bottom-right corner, color (BGR), thickness
    cv2.rectangle(img, (50, 50), (300, 200), (255, 0, 0), thickness=2)
    
    # Draw a green circle
    # Parameters: image, center, radius, color (BGR), thickness (-1 means filled)
    cv2.circle(img, (400, 100), 75, (0, 255, 0), thickness=-1)
    
    # Draw a red line
    # Parameters: image, start point, end point, color (BGR), thickness
    cv2.line(img, (100, 300), (600, 300), (0, 0, 255), thickness=5)
    
    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'OpenCV Introduction', (150, 400), font, 1.5, (255, 255, 255), 2)
    
    # Display the image
    cv2.imshow('OpenCV Basics', img)
    
    # Wait for a key press and then close all windows
    print("Press any key to close the window...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("OpenCV is working correctly!")

if __name__ == "__main__":
    main()