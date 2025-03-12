#!/usr/bin/env python3
"""
Image Gradients and Edge Detection with OpenCV
This script demonstrates various gradient calculation and edge detection techniques in OpenCV
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def main():
    # Get the path to the images directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), 'images')
    
    # Try to load an image from the images directory
    # If no suitable image is found, create a sample image
    try:
        # Try to load a grayscale image if available
        img_path = os.path.join(images_dir, 'shapes.jpg')
        if os.path.exists(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        else:
            # Try other potential images
            for img_name in ['landscape.jpg', 'objects.jpg', 'text.jpg']:
                img_path = os.path.join(images_dir, img_name)
                if os.path.exists(img_path):
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    break
            else:
                # If no image is found, create a sample image
                img = create_sample_image()
    except:
        # Fallback to creating a sample image
        img = create_sample_image()
    
    # Ensure we have a valid image
    if img is None or img.size == 0:
        img = create_sample_image()
        cv2.imwrite('sample_image.jpg', img)
        print("Created and saved sample_image.jpg")
    
    # 1. Sobel Derivatives
    print("\n1. SOBEL DERIVATIVES")
    
    # Apply Sobel operator in x direction
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    
    # Apply Sobel operator in y direction
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    # Convert to absolute values
    abs_sobelx = cv2.convertScaleAbs(sobelx)
    abs_sobely = cv2.convertScaleAbs(sobely)
    
    # Combine the two gradients
    sobel_combined = cv2.addWeighted(abs_sobelx, 0.5, abs_sobely, 0.5, 0)
    
    # Display the results
    display_images([img, abs_sobelx, abs_sobely, sobel_combined], 
                  ['Original Image', 'Sobel X', 'Sobel Y', 'Sobel Combined'])
    print("Press any key to continue...")
    cv2.waitKey(0)
    
    # 2. Scharr Derivatives
    print("\n2. SCHARR DERIVATIVES")
    
    # Apply Scharr operator in x direction
    scharrx = cv2.Scharr(img, cv2.CV_64F, 1, 0)
    
    # Apply Scharr operator in y direction
    scharry = cv2.Scharr(img, cv2.CV_64F, 0, 1)
    
    # Convert to absolute values
    abs_scharrx = cv2.convertScaleAbs(scharrx)
    abs_scharry = cv2.convertScaleAbs(scharry)
    
    # Combine the two gradients
    scharr_combined = cv2.addWeighted(abs_scharrx, 0.5, abs_scharry, 0.5, 0)
    
    # Display the results
    display_images([img, abs_scharrx, abs_scharry, scharr_combined], 
                  ['Original Image', 'Scharr X', 'Scharr Y', 'Scharr Combined'])
    print("Press any key to continue...")
    cv2.waitKey(0)
    
    # 3. Laplacian Derivatives
    print("\n3. LAPLACIAN DERIVATIVES")
    
    # Apply Laplacian operator
    laplacian = cv2.Laplacian(img, cv2.CV_64F)
    
    # Convert to absolute values
    abs_laplacian = cv2.convertScaleAbs(laplacian)
    
    # Display the results
    display_images([img, abs_laplacian], ['Original Image', 'Laplacian'])
    print("Press any key to continue...")
    cv2.waitKey(0)
    
    # 4. Canny Edge Detector
    print("\n4. CANNY EDGE DETECTOR")
    
    # Apply Canny edge detector with different threshold values
    edges1 = cv2.Canny(img, 50, 150)
    edges2 = cv2.Canny(img, 100, 200)
    edges3 = cv2.Canny(img, 150, 250)
    
    # Display the results
    display_images([img, edges1, edges2, edges3], 
                  ['Original Image', 'Canny (50, 150)', 'Canny (100, 200)', 'Canny (150, 250)'])
    print("Press any key to continue...")
    cv2.waitKey(0)
    
    # 5. Automatic Threshold Selection for Canny
    print("\n5. AUTOMATIC THRESHOLD SELECTION FOR CANNY")
    
    # Calculate median of the image
    median = np.median(img)
    
    # Set lower and upper thresholds based on median
    lower = int(max(0, (1.0 - 0.33) * median))
    upper = int(min(255, (1.0 + 0.33) * median))
    
    print(f"Median: {median}, Lower Threshold: {lower}, Upper Threshold: {upper}")
    
    # Apply Canny edge detector with automatic thresholds
    edges_auto = cv2.Canny(img, lower, upper)
    
    # Display the results
    display_images([img, edges_auto], ['Original Image', f'Canny (Auto: {lower}, {upper})'])
    print("Press any key to continue...")
    cv2.waitKey(0)
    
    # 6. Pre-processing for Better Edge Detection
    print("\n6. PRE-PROCESSING FOR BETTER EDGE DETECTION")
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Apply Canny edge detector on original and blurred images
    edges_original = cv2.Canny(img, 100, 200)
    edges_blurred = cv2.Canny(blurred, 100, 200)
    
    # Display the results
    display_images([img, blurred, edges_original, edges_blurred], 
                  ['Original Image', 'Gaussian Blur', 'Canny on Original', 'Canny on Blurred'])
    print("Press any key to continue...")
    cv2.waitKey(0)
    
    # 7. Morphological Operations on Edges
    print("\n7. MORPHOLOGICAL OPERATIONS ON EDGES")
    
    # Get edges using Canny
    edges = cv2.Canny(blurred, 100, 200)
    
    # Apply dilation to thicken edges
    kernel = np.ones((3, 3), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Apply erosion to thin edges
    eroded_edges = cv2.erode(edges, kernel, iterations=1)
    
    # Apply closing to close small gaps
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Display the results
    display_images([edges, dilated_edges, eroded_edges, closed_edges], 
                  ['Original Edges', 'Dilated Edges', 'Eroded Edges', 'Closed Edges'])
    print("Press any key to continue...")
    cv2.waitKey(0)
    
    # 8. Custom Edge Detection with Sobel
    print("\n8. CUSTOM EDGE DETECTION WITH SOBEL")
    
    # Calculate gradient magnitude
    gradient_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate gradient magnitude and direction
    magnitude = cv2.magnitude(gradient_x, gradient_y)
    direction = cv2.phase(gradient_x, gradient_y, angleInDegrees=True)
    
    # Normalize magnitude to 0-255 range
    magnitude_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    # Apply threshold to get edges
    _, edges_sobel = cv2.threshold(magnitude_norm, 50, 255, cv2.THRESH_BINARY)
    
    # Display the results
    display_images([img, magnitude_norm, edges_sobel], 
                  ['Original Image', 'Gradient Magnitude', 'Thresholded Edges'])
    print("Press any key to continue...")
    cv2.waitKey(0)
    
    # 9. Finding Contours from Edges
    print("\n9. FINDING CONTOURS FROM EDGES")
    
    # Find contours from edges
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on original image
    contour_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
    
    # Display the results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(edges, cmap='gray')
    plt.title('Edges')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
    plt.title('Contours')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    print("Press any key to continue...")
    cv2.waitKey(0)
    
    # 10. Line Detection with Hough Transform
    print("\n10. LINE DETECTION WITH HOUGH TRANSFORM")
    
    # Create a copy of the original image for line detection
    line_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Apply Hough Line Transform
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
    
    # Draw lines on original image
    if lines is not None:
        for i in range(min(10, len(lines))):  # Draw at most 10 lines
            rho, theta = lines[i][0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # Apply Probabilistic Hough Line Transform
    prob_line_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    lines_p = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
    
    # Draw lines on original image
    if lines_p is not None:
        for line in lines_p:
            x1, y1, x2, y2 = line[0]
            cv2.line(prob_line_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Display the results
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(edges, cmap='gray')
    plt.title('Edges')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB))
    plt.title('Standard Hough Lines')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(prob_line_img, cv2.COLOR_BGR2RGB))
    plt.title('Probabilistic Hough Lines')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    print("Press any key to continue...")
    cv2.waitKey(0)
    
    # 11. Circle Detection with Hough Circle Transform
    print("\n11. CIRCLE DETECTION WITH HOUGH CIRCLE TRANSFORM")
    
    # Create a copy of the original image for circle detection
    circle_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    
    # Apply Hough Circle Transform
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 20,
                              param1=50, param2=30, minRadius=0, maxRadius=0)
    
    # Draw circles on original image
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Draw outer circle
            cv2.circle(circle_img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Draw center of the circle
            cv2.circle(circle_img, (i[0], i[1]), 2, (0, 0, 255), 3)
    
    # Display the results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(circle_img, cv2.COLOR_BGR2RGB))
    plt.title('Detected Circles')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    print("Press any key to continue...")
    cv2.waitKey(0)
    
    # 12. Comparing Edge Detection Methods
    print("\n12. COMPARING EDGE DETECTION METHODS")
    
    # Apply different edge detection methods
    sobel_edges = sobel_combined
    laplacian_edges = abs_laplacian
    canny_edges = cv2.Canny(img, 100, 200)
    
    # Display the results
    display_images([img, sobel_edges, laplacian_edges, canny_edges], 
                  ['Original Image', 'Sobel', 'Laplacian', 'Canny'])
    print("Press any key to exit...")
    cv2.waitKey(0)
    
    # Clean up
    cv2.destroyAllWindows()
    print("\nAll operations completed successfully!")

def create_sample_image():
    """Create a sample image with shapes for edge detection demonstration"""
    # Create a blank image
    width, height = 500, 500
    img = np.ones((height, width), dtype=np.uint8) * 255  # White background
    
    # Add some shapes
    # Rectangle
    cv2.rectangle(img, (100, 100), (300, 300), 0, -1)
    
    # Circle
    cv2.circle(img, (400, 400), 80, 0, -1)
    
    # Triangle
    pts = np.array([[250, 400], [100, 450], [400, 450]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(img, [pts], 0)
    
    # Line
    cv2.line(img, (50, 50), (450, 50), 0, 5)
    
    # Add some noise
    noise = np.zeros((height, width), dtype=np.uint8)
    cv2.randu(noise, 0, 25)
    img = cv2.add(img, noise)
    
    # Add some blur to make it more realistic
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    return img

def display_images(images, titles):
    """Display multiple images in a grid using matplotlib"""
    n_images = len(images)
    n_cols = 2
    n_rows = (n_images + n_cols - 1) // n_cols  # Ceiling division
    
    plt.figure(figsize=(12, 5 * n_rows))
    
    for i in range(n_images):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()