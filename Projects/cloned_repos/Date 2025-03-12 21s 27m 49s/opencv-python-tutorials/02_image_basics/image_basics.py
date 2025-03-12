#!/usr/bin/env python3
"""
Image Basics with OpenCV
This script demonstrates fundamental operations for working with images in OpenCV
"""

import cv2
import numpy as np
import os
import sys

def main():
    # Create a sample image if no image is provided
    sample_img = create_sample_image()
    
    # Save the sample image
    cv2.imwrite('sample_image.jpg', sample_img)
    print("Created and saved sample_image.jpg")
    
    # 1. Reading an image
    print("\n1. READING IMAGES")
    img = cv2.imread('sample_image.jpg')
    
    if img is None:
        print("Error: Could not read image")
        sys.exit()
    else:
        print("Image loaded successfully")
    
    # 2. Image properties
    print("\n2. IMAGE PROPERTIES")
    height, width, channels = img.shape
    print(f"Image Dimensions: {width}x{height}")
    print(f"Number of Channels: {channels}")
    print(f"Image Data Type: {img.dtype}")
    print(f"Total Pixels: {img.size}")
    
    # 3. Displaying the image
    print("\n3. DISPLAYING IMAGES")
    print("Press any key to continue to the next step...")
    cv2.imshow('Original Image', img)
    cv2.waitKey(0)
    
    # 4. Accessing and modifying pixels
    print("\n4. ACCESSING AND MODIFYING PIXELS")
    # Create a copy to work with
    pixel_img = img.copy()
    
    # Access a pixel
    y, x = 100, 150  # row=100, col=150
    pixel = pixel_img[y, x]
    print(f"Pixel at ({x}, {y}): {pixel} (BGR)")
    
    # Modify a pixel
    pixel_img[y, x] = [0, 0, 255]  # Set to red in BGR
    
    # Draw a circle to highlight the modified pixel
    cv2.circle(pixel_img, (x, y), 10, (0, 255, 255), 2)
    
    # Display the result
    cv2.imshow('Modified Pixel', pixel_img)
    cv2.waitKey(0)
    
    # 5. Region of Interest (ROI)
    print("\n5. REGION OF INTEREST (ROI)")
    roi_img = img.copy()
    
    # Extract a region
    roi = roi_img[50:200, 100:300]
    
    # Highlight the ROI in the original image
    cv2.rectangle(roi_img, (100, 50), (300, 200), (0, 255, 0), 2)
    
    # Display the original with highlighted ROI
    cv2.imshow('Image with ROI', roi_img)
    cv2.waitKey(0)
    
    # Display the extracted ROI
    cv2.imshow('Extracted ROI', roi)
    cv2.waitKey(0)
    
    # 6. Splitting and merging channels
    print("\n6. SPLITTING AND MERGING CHANNELS")
    # Split the BGR image into separate channels
    b, g, r = cv2.split(img)
    
    # Create images to visualize each channel (grayscale)
    cv2.imshow('Blue Channel', b)
    cv2.imshow('Green Channel', g)
    cv2.imshow('Red Channel', r)
    cv2.waitKey(0)
    
    # Create colored single-channel images for better visualization
    zeros = np.zeros_like(b)
    blue_img = cv2.merge([b, zeros, zeros])  # Only blue channel
    green_img = cv2.merge([zeros, g, zeros])  # Only green channel
    red_img = cv2.merge([zeros, zeros, r])    # Only red channel
    
    # Display colored single-channel images
    cv2.imshow('Blue Channel (Colored)', blue_img)
    cv2.imshow('Green Channel (Colored)', green_img)
    cv2.imshow('Red Channel (Colored)', red_img)
    cv2.waitKey(0)
    
    # 7. Resizing images
    print("\n7. RESIZING IMAGES")
    # Resize to specific dimensions
    resized_img = cv2.resize(img, (400, 300))
    cv2.imshow('Resized to 400x300', resized_img)
    cv2.waitKey(0)
    
    # Resize by scaling factor
    half_size = cv2.resize(img, None, fx=0.5, fy=0.5)
    double_size = cv2.resize(img, None, fx=2, fy=2)
    
    cv2.imshow('Half Size', half_size)
    cv2.waitKey(0)
    cv2.imshow('Double Size', double_size)
    cv2.waitKey(0)
    
    # 8. Rotating images
    print("\n8. ROTATING IMAGES")
    # Get the image dimensions
    height, width = img.shape[:2]
    
    # Define the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), 45, 1)
    
    # Apply the rotation
    rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height))
    
    cv2.imshow('Rotated Image (45 degrees)', rotated_img)
    cv2.waitKey(0)
    
    # 9. Flipping images
    print("\n9. FLIPPING IMAGES")
    # Flip horizontally
    horizontal_flip = cv2.flip(img, 1)
    cv2.imshow('Horizontal Flip', horizontal_flip)
    cv2.waitKey(0)
    
    # Flip vertically
    vertical_flip = cv2.flip(img, 0)
    cv2.imshow('Vertical Flip', vertical_flip)
    cv2.waitKey(0)
    
    # Flip both horizontally & vertically
    both_flip = cv2.flip(img, -1)
    cv2.imshow('Both Flip', both_flip)
    cv2.waitKey(0)
    
    # 10. Saving images
    print("\n10. SAVING IMAGES")
    # Save the rotated image
    result = cv2.imwrite('rotated_image.jpg', rotated_img)
    
    if result:
        print("Rotated image saved successfully as 'rotated_image.jpg'")
    else:
        print("Error: Could not save rotated image")
    
    # Clean up
    cv2.destroyAllWindows()
    print("\nAll operations completed successfully!")

def create_sample_image():
    """Create a sample image with shapes for demonstration"""
    # Create a blank image with white background
    img = np.ones((400, 600, 3), dtype=np.uint8) * 255
    
    # Draw a blue rectangle
    cv2.rectangle(img, (50, 50), (200, 150), (255, 0, 0), -1)
    
    # Draw a green circle
    cv2.circle(img, (400, 100), 75, (0, 255, 0), -1)
    
    # Draw a red triangle
    pts = np.array([[300, 300], [200, 200], [400, 200]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(img, [pts], (0, 0, 255))
    
    # Add some text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'OpenCV', (250, 350), font, 2, (0, 0, 0), 3)
    
    return img

if __name__ == "__main__":
    main()