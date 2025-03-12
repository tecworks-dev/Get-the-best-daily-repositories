#!/usr/bin/env python3
"""
Histograms Tutorial Script
This script demonstrates various histogram operations and applications using OpenCV
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def create_sample_image():
    """Create a sample image with varying intensities and patterns"""
    # Create a gradient image
    gradient = np.zeros((256, 256), dtype=np.uint8)
    for i in range(256):
        gradient[:, i] = i
    
    # Create a pattern image
    pattern = np.zeros((256, 256), dtype=np.uint8)
    cv2.circle(pattern, (128, 128), 50, 255, -1)
    cv2.rectangle(pattern, (20, 20), (80, 80), 127, -1)
    cv2.line(pattern, (200, 200), (250, 250), 200, 3)
    
    # Combine images horizontally
    combined = np.hstack((gradient, pattern))
    
    # Create color version
    color = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
    color[:, 256:, 0] = pattern  # Blue channel
    color[:, 256:, 1] = cv2.flip(pattern, 0)  # Green channel
    color[:, 256:, 2] = cv2.flip(pattern, 1)  # Red channel
    
    return color

def plot_histogram(image, title='Histogram', color=True):
    """Plot histogram of an image"""
    plt.figure(figsize=(10, 4))
    
    if color:
        colors = ('b', 'g', 'r')
        for i, col in enumerate(colors):
            hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            plt.plot(hist, color=col)
    else:
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        plt.plot(hist)
    
    plt.title(title)
    plt.xlim([0, 256])
    plt.grid(True)
    plt.show()

def plot_2d_histogram(image):
    """Plot 2D histogram of an image"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(122)
    plt.imshow(hist, interpolation='nearest')
    plt.title('2D Histogram (Hue vs Saturation)')
    plt.colorbar()
    plt.show()

def equalize_histogram(image):
    """Apply histogram equalization to an image"""
    if len(image.shape) == 3:
        # For color images, equalize each channel separately
        b, g, r = cv2.split(image)
        b_eq = cv2.equalizeHist(b)
        g_eq = cv2.equalizeHist(g)
        r_eq = cv2.equalizeHist(r)
        return cv2.merge([b_eq, g_eq, r_eq])
    else:
        return cv2.equalizeHist(image)

def apply_clahe(image, clip_limit=2.0, grid_size=(8, 8)):
    """Apply CLAHE to an image"""
    if len(image.shape) == 3:
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        l_eq = clahe.apply(l)
        
        # Merge channels and convert back to BGR
        lab_eq = cv2.merge([l_eq, a, b])
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        return clahe.apply(image)

def compare_histograms(image1, image2):
    """Compare histograms of two images using different methods"""
    # Convert images to HSV
    hsv1 = cv2.cvtColor(image1, cv2.COLOR_BGR2HSV)
    hsv2 = cv2.cvtColor(image2, cv2.COLOR_BGR2HSV)
    
    # Calculate histograms
    hist1 = cv2.calcHist([hsv1], [0, 1], None, [180, 256], [0, 180, 0, 256])
    hist2 = cv2.calcHist([hsv2], [0, 1], None, [180, 256], [0, 180, 0, 256])
    
    # Normalize histograms
    cv2.normalize(hist1, hist1, 0, 1, cv2.NORM_MINMAX)
    cv2.normalize(hist2, hist2, 0, 1, cv2.NORM_MINMAX)
    
    # Compare using different methods
    methods = [
        ('Correlation', cv2.HISTCMP_CORREL),
        ('Chi-Square', cv2.HISTCMP_CHISQR),
        ('Intersection', cv2.HISTCMP_INTERSECT),
        ('Bhattacharyya', cv2.HISTCMP_BHATTACHARYYA)
    ]
    
    results = {}
    for method_name, method in methods:
        result = cv2.compareHist(hist1, hist2, method)
        results[method_name] = result
    
    return results

def histogram_backprojection(image, target):
    """Perform histogram backprojection"""
    # Convert images to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    target_hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
    
    # Calculate target histogram
    target_hist = cv2.calcHist([target_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    
    # Normalize histogram
    cv2.normalize(target_hist, target_hist, 0, 255, cv2.NORM_MINMAX)
    
    # Calculate back projection
    back_proj = cv2.calcBackProject([hsv], [0, 1], target_hist, [0, 180, 0, 256], 1)
    
    return back_proj

def main():
    # Create or load sample images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), 'images')
    
    # Try to load an image from the images directory, or create a sample one
    try:
        img_path = os.path.join(images_dir, 'sample.jpg')
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
        else:
            img = create_sample_image()
            # Save the sample image
            if not os.path.exists(images_dir):
                os.makedirs(images_dir)
            cv2.imwrite(img_path, img)
            print("Created and saved sample image")
    except:
        img = create_sample_image()
    
    # 1. Basic Histogram Calculation and Visualization
    print("\n1. Basic Histogram Visualization")
    plot_histogram(img, 'Color Histogram of Original Image')
    
    # Convert to grayscale and show histogram
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plot_histogram(gray, 'Grayscale Histogram', color=False)
    
    # 2. 2D Histogram
    print("\n2. 2D Histogram Visualization")
    plot_2d_histogram(img)
    
    # 3. Histogram Equalization
    print("\n3. Histogram Equalization")
    equalized = equalize_histogram(img)
    
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(cv2.cvtColor(equalized, cv2.COLOR_BGR2RGB))
    plt.title('Histogram Equalized')
    plt.axis('off')
    
    plt.subplot(133)
    plot_histogram(equalized, 'Equalized Histogram')
    plt.show()
    
    # 4. CLAHE
    print("\n4. CLAHE Enhancement")
    clahe_img = apply_clahe(img)
    
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(cv2.cvtColor(clahe_img, cv2.COLOR_BGR2RGB))
    plt.title('CLAHE Enhanced')
    plt.axis('off')
    
    plt.subplot(133)
    plot_histogram(clahe_img, 'CLAHE Histogram')
    plt.show()
    
    # 5. Histogram Comparison
    print("\n5. Histogram Comparison")
    # Create a modified version of the original image
    modified = img.copy()
    modified = cv2.addWeighted(modified, 1.2, np.zeros(modified.shape, modified.dtype), 0, 30)
    
    # Compare histograms
    comparison = compare_histograms(img, modified)
    print("\nHistogram Comparison Results:")
    for method, value in comparison.items():
        print(f"{method}: {value:.4f}")
    
    # 6. Histogram Backprojection
    print("\n6. Histogram Backprojection")
    # Create a target image (use a section of the original image)
    target = img[100:150, 100:150].copy()
    
    # Calculate backprojection
    back_proj = histogram_backprojection(img, target)
    
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(cv2.cvtColor(target, cv2.COLOR_BGR2RGB))
    plt.title('Target')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(back_proj, cmap='gray')
    plt.title('Backprojection')
    plt.axis('off')
    plt.show()
    
    # 7. Practical Application: Image Enhancement
    print("\n7. Practical Application: Image Enhancement")
    # Create a low contrast image
    low_contrast = cv2.convertScaleAbs(img, alpha=0.5, beta=30)
    
    # Apply different enhancement methods
    hist_eq = equalize_histogram(low_contrast)
    clahe_enhanced = apply_clahe(low_contrast, clip_limit=3.0)
    
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(low_contrast, cv2.COLOR_BGR2RGB))
    plt.title('Low Contrast Image')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(cv2.cvtColor(hist_eq, cv2.COLOR_BGR2RGB))
    plt.title('Histogram Equalization')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(cv2.cvtColor(clahe_enhanced, cv2.COLOR_BGR2RGB))
    plt.title('CLAHE Enhancement')
    plt.axis('off')
    plt.show()
    
    print("\nAll histogram operations completed successfully!")

if __name__ == "__main__":
    main()