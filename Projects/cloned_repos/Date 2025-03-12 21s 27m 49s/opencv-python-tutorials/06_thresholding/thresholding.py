#!/usr/bin/env python3
"""
Image Thresholding with OpenCV
This script demonstrates various thresholding techniques in OpenCV
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
        img_path = os.path.join(images_dir, 'low_contrast.jpg')
        if os.path.exists(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        else:
            # Try other potential images
            for img_name in ['landscape.jpg', 'shapes.jpg', 'text.jpg']:
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
    
    # Save the sample image if it was created
    if isinstance(img, tuple) and len(img) == 2:
        img, is_sample = img
        if is_sample:
            cv2.imwrite('sample_image.jpg', img)
            print("Created and saved sample_image.jpg")
    
    # 1. Simple Thresholding
    print("\n1. SIMPLE THRESHOLDING")
    
    # Apply different thresholding techniques
    ret, thresh_binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    ret, thresh_binary_inv = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    ret, thresh_trunc = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
    ret, thresh_tozero = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
    ret, thresh_tozero_inv = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)
    
    # Display the results
    titles = ['Original Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
    images = [img, thresh_binary, thresh_binary_inv, thresh_trunc, thresh_tozero, thresh_tozero_inv]
    
    display_multiple_images(images, titles)
    print("Press any key to continue...")
    cv2.waitKey(0)
    
    # 2. Adaptive Thresholding
    print("\n2. ADAPTIVE THRESHOLDING")
    
    # Apply adaptive thresholding
    adaptive_mean = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
    adaptive_gaussian = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 11, 2)
    
    # Display the results
    titles = ['Original Image', 'Adaptive Mean', 'Adaptive Gaussian']
    images = [img, adaptive_mean, adaptive_gaussian]
    
    display_multiple_images(images, titles)
    print("Press any key to continue...")
    cv2.waitKey(0)
    
    # 3. Otsu's Thresholding
    print("\n3. OTSU'S THRESHOLDING")
    
    # Apply global thresholding
    ret1, global_thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    
    # Apply Otsu's thresholding
    ret2, otsu_thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Apply Gaussian blur + Otsu's thresholding for better results
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ret3, otsu_blur_thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    print(f"Global Threshold Value: {ret1}")
    print(f"Otsu's Threshold Value: {ret2}")
    print(f"Otsu's Threshold Value after Gaussian Blur: {ret3}")
    
    # Display the results
    titles = ['Original Image', 'Global Thresholding (v=127)',
              f"Otsu's Thresholding (v={ret2})", 'Gaussian + Otsu']
    images = [img, global_thresh, otsu_thresh, otsu_blur_thresh]
    
    display_multiple_images(images, titles)
    print("Press any key to continue...")
    cv2.waitKey(0)
    
    # 4. Multi-level Thresholding
    print("\n4. MULTI-LEVEL THRESHOLDING")
    
    # Apply multi-level thresholding
    ret1, thresh1 = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    ret2, thresh2 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    ret3, thresh3 = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
    ret4, thresh4 = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
    
    # Combine the results to create a multi-level thresholded image
    multi_level = np.zeros_like(img)
    multi_level = np.where(thresh1 == 255, 64, multi_level)
    multi_level = np.where(thresh2 == 255, 128, multi_level)
    multi_level = np.where(thresh3 == 255, 192, multi_level)
    multi_level = np.where(thresh4 == 255, 255, multi_level)
    
    # Display the results
    titles = ['Original Image', 'Threshold (v=50)', 'Threshold (v=100)', 
              'Threshold (v=150)', 'Threshold (v=200)', 'Multi-level']
    images = [img, thresh1, thresh2, thresh3, thresh4, multi_level]
    
    display_multiple_images(images, titles)
    print("Press any key to continue...")
    cv2.waitKey(0)
    
    # 5. Thresholding with Pre-processing
    print("\n5. THRESHOLDING WITH PRE-PROCESSING")
    
    # Apply histogram equalization to enhance contrast
    equalized = cv2.equalizeHist(img)
    
    # Apply thresholding on the equalized image
    ret, thresh_eq = cv2.threshold(equalized, 127, 255, cv2.THRESH_BINARY)
    
    # Apply adaptive thresholding on the equalized image
    adaptive_eq = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
    
    # Display the results
    titles = ['Original Image', 'Histogram Equalized', 
              'Thresholding after Equalization', 'Adaptive after Equalization']
    images = [img, equalized, thresh_eq, adaptive_eq]
    
    display_multiple_images(images, titles)
    print("Press any key to exit...")
    cv2.waitKey(0)
    
    # Clean up
    cv2.destroyAllWindows()
    print("\nAll operations completed successfully!")

def create_sample_image():
    """Create a sample image with varying intensities for thresholding demonstration"""
    # Create a gradient image
    width, height = 400, 400
    img = np.zeros((height, width), dtype=np.uint8)
    
    # Create horizontal gradient
    for i in range(width):
        img[:, i] = i * 255 // width
    
    # Add some shapes with different intensities
    # Circle in the top-left
    cv2.circle(img, (100, 100), 50, 200, -1)
    
    # Rectangle in the bottom-right
    cv2.rectangle(img, (250, 250), (350, 350), 50, -1)
    
    # Add some noise
    noise = np.zeros((height, width), dtype=np.uint8)
    cv2.randu(noise, 0, 50)
    img = cv2.add(img, noise)
    
    # Add some text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'Threshold', (130, 200), font, 1, 150, 2)
    
    return (img, True)  # Return the image and a flag indicating it's a sample

def display_multiple_images(images, titles):
    """Display multiple images in a grid using matplotlib"""
    n_images = len(images)
    n_cols = 3
    n_rows = (n_images + n_cols - 1) // n_cols  # Ceiling division
    
    plt.figure(figsize=(15, 5 * n_rows))
    
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