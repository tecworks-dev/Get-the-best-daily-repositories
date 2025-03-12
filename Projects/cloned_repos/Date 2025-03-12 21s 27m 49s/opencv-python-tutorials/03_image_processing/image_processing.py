#!/usr/bin/env python3
"""
Image Processing with OpenCV
This script demonstrates various image processing techniques in OpenCV
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
    
    # Read the image
    img = cv2.imread('sample_image.jpg')
    if img is None:
        print("Error: Could not read image")
        sys.exit()
    
    # Get image dimensions
    height, width = img.shape[:2]
    
    # 1. Color Space Conversions
    print("\n1. COLOR SPACE CONVERSIONS")
    
    # Convert BGR to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Convert BGR to RGB
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Convert BGR to LAB
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    # Display the conversions
    cv2.imshow('Original (BGR)', img)
    cv2.imshow('Grayscale', gray)
    cv2.imshow('HSV', hsv)
    # Note: OpenCV imshow expects BGR, so RGB will look color-inverted
    cv2.imshow('RGB (looks wrong in imshow)', rgb)
    cv2.imshow('LAB', lab)
    
    print("Press any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 2. Geometric Transformations
    print("\n2. GEOMETRIC TRANSFORMATIONS")
    
    # Translation
    tx, ty = 100, 50
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    translated = cv2.warpAffine(img, translation_matrix, (width, height))
    
    # Rotation
    center = (width // 2, height // 2)
    angle = 45
    scale = 1.0
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(img, rotation_matrix, (width, height))
    
    # Affine Transformation
    src_points = np.float32([[0, 0], [width - 1, 0], [0, height - 1]])
    dst_points = np.float32([[0, 0], [width - 1, 0], [width // 3, height - 1]])
    affine_matrix = cv2.getAffineTransform(src_points, dst_points)
    affine = cv2.warpAffine(img, affine_matrix, (width, height))
    
    # Perspective Transformation
    # Define four points in the input image (these should be meaningful points in your image)
    src_points = np.float32([[50, 50], [width-50, 50], [50, height-50], [width-50, height-50]])
    # Define where those points will be in the output image
    dst_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    perspective = cv2.warpPerspective(img, perspective_matrix, (width, height))
    
    # Display the transformations
    cv2.imshow('Original', img)
    cv2.imshow('Translated', translated)
    cv2.imshow('Rotated', rotated)
    cv2.imshow('Affine Transformation', affine)
    cv2.imshow('Perspective Transformation', perspective)
    
    print("Press any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 3. Image Filtering
    print("\n3. IMAGE FILTERING")
    
    # Average Blur
    blur = cv2.blur(img, (5, 5))
    
    # Gaussian Blur
    gaussian = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Median Blur
    median = cv2.medianBlur(img, 5)
    
    # Bilateral Filter
    bilateral = cv2.bilateralFilter(img, 9, 75, 75)
    
    # Display the filters
    cv2.imshow('Original', img)
    cv2.imshow('Average Blur', blur)
    cv2.imshow('Gaussian Blur', gaussian)
    cv2.imshow('Median Blur', median)
    cv2.imshow('Bilateral Filter', bilateral)
    
    print("Press any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 4. Morphological Operations
    print("\n4. MORPHOLOGICAL OPERATIONS")
    
    # Create a binary image for morphological operations
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Define a kernel
    kernel = np.ones((5, 5), np.uint8)
    
    # Erosion
    erosion = cv2.erode(binary, kernel, iterations=1)
    
    # Dilation
    dilation = cv2.dilate(binary, kernel, iterations=1)
    
    # Opening
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    
    # Closing
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Morphological Gradient
    gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
    
    # Display the morphological operations
    cv2.imshow('Binary Image', binary)
    cv2.imshow('Erosion', erosion)
    cv2.imshow('Dilation', dilation)
    cv2.imshow('Opening', opening)
    cv2.imshow('Closing', closing)
    cv2.imshow('Morphological Gradient', gradient)
    
    print("Press any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 5. Image Gradients
    print("\n5. IMAGE GRADIENTS")
    
    # Convert to float64 for gradient operations
    gray_float = np.float32(gray)
    
    # Sobel Derivatives
    sobelx = cv2.Sobel(gray_float, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_float, cv2.CV_64F, 0, 1, ksize=3)
    
    # Convert back to uint8 for display
    sobelx_abs = cv2.convertScaleAbs(sobelx)
    sobely_abs = cv2.convertScaleAbs(sobely)
    
    # Combine the two Sobel images
    sobel_combined = cv2.addWeighted(sobelx_abs, 0.5, sobely_abs, 0.5, 0)
    
    # Laplacian Derivative
    laplacian = cv2.Laplacian(gray_float, cv2.CV_64F)
    laplacian_abs = cv2.convertScaleAbs(laplacian)
    
    # Display the gradients
    cv2.imshow('Original Grayscale', gray)
    cv2.imshow('Sobel X', sobelx_abs)
    cv2.imshow('Sobel Y', sobely_abs)
    cv2.imshow('Sobel Combined', sobel_combined)
    cv2.imshow('Laplacian', laplacian_abs)
    
    print("Press any key to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 6. Image Pyramids
    print("\n6. IMAGE PYRAMIDS")
    
    # Gaussian Pyramid (downsampling)
    lower_res = cv2.pyrDown(img)
    even_lower_res = cv2.pyrDown(lower_res)
    
    # Laplacian Pyramid
    higher_res = cv2.pyrUp(lower_res)
    # Resize higher_res to match original image size
    higher_res_resized = cv2.resize(higher_res, (width, height))
    # Calculate the difference (Laplacian)
    laplacian_diff = cv2.subtract(img, higher_res_resized)
    
    # Display the pyramids
    cv2.imshow('Original', img)
    cv2.imshow('Lower Resolution (pyrDown)', lower_res)
    cv2.imshow('Even Lower Resolution (pyrDown twice)', even_lower_res)
    cv2.imshow('Higher Resolution (pyrUp)', higher_res)
    cv2.imshow('Laplacian Difference', laplacian_diff)
    
    print("Press any key to exit...")
    cv2.waitKey(0)
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
    
    # Add some noise for filtering demonstrations
    noise = np.zeros(img.shape, np.uint8)
    cv2.randu(noise, 0, 50)
    img = cv2.add(img, noise)
    
    return img

if __name__ == "__main__":
    main()