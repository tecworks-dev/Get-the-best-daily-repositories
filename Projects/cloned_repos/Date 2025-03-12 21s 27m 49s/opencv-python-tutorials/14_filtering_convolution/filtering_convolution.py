#!/usr/bin/env python3
"""
Image Filtering and Convolution with OpenCV
This script demonstrates various filtering and convolution operations using OpenCV
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

class ImageFiltering:
    def __init__(self):
        """Initialize image filtering class"""
        pass
    
    def apply_convolution(self, image, kernel):
        """
        Apply convolution manually (for demonstration)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Get dimensions
        height, width = gray.shape
        k_height, k_width = kernel.shape
        
        # Calculate padding
        pad_height = k_height // 2
        pad_width = k_width // 2
        
        # Create padded image
        padded = cv2.copyMakeBorder(gray, pad_height, pad_height, pad_width, pad_width, 
                                  cv2.BORDER_REPLICATE)
        
        # Create output image
        output = np.zeros_like(gray)
        
        # Apply convolution
        for i in range(height):
            for j in range(width):
                # Extract region of interest
                roi = padded[i:i+k_height, j:j+k_width]
                # Apply kernel
                output[i, j] = np.sum(roi * kernel)
        
        # Normalize output
        output = cv2.normalize(output, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return output
    
    def apply_basic_filters(self, image):
        """Apply basic filtering operations"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Box filter (averaging)
        box_filter = cv2.blur(gray, (5, 5))
        
        # Gaussian filter
        gaussian = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Median filter
        median = cv2.medianBlur(gray, 5)
        
        # Bilateral filter
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)
        
        return {
            'original': gray,
            'box_filter': box_filter,
            'gaussian': gaussian,
            'median': median,
            'bilateral': bilateral
        }
    
    def apply_edge_filters(self, image):
        """Apply edge detection filters"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Sobel filters
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobelx = cv2.convertScaleAbs(sobelx)
        sobely = cv2.convertScaleAbs(sobely)
        sobel_combined = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
        
        # Scharr filters
        scharrx = cv2.Scharr(gray, cv2.CV_64F, 1, 0)
        scharry = cv2.Scharr(gray, cv2.CV_64F, 0, 1)
        scharrx = cv2.convertScaleAbs(scharrx)
        scharry = cv2.convertScaleAbs(scharry)
        scharr_combined = cv2.addWeighted(scharrx, 0.5, scharry, 0.5, 0)
        
        # Laplacian filter
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = cv2.convertScaleAbs(laplacian)
        
        # Canny edge detector
        canny = cv2.Canny(gray, 100, 200)
        
        return {
            'original': gray,
            'sobel_x': sobelx,
            'sobel_y': sobely,
            'sobel_combined': sobel_combined,
            'scharr_x': scharrx,
            'scharr_y': scharry,
            'scharr_combined': scharr_combined,
            'laplacian': laplacian,
            'canny': canny
        }
    
    def apply_custom_kernels(self, image):
        """Apply custom kernels for different effects"""
        # Define custom kernels
        kernels = {
            'identity': np.array([[0, 0, 0],
                                [0, 1, 0],
                                [0, 0, 0]]),
            
            'edge_detect': np.array([[-1, -1, -1],
                                   [-1,  8, -1],
                                   [-1, -1, -1]]),
            
            'sharpen': np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]]),
            
            'emboss': np.array([[-2, -1, 0],
                              [-1,  1, 1],
                              [0,  1, 2]]),
            
            'gaussian_blur': np.array([[1, 2, 1],
                                     [2, 4, 2],
                                     [1, 2, 1]]) / 16.0
        }
        
        # Apply each kernel
        results = {'original': image}
        
        for name, kernel in kernels.items():
            results[name] = cv2.filter2D(image, -1, kernel)
        
        return results
    
    def sharpen_image(self, image):
        """Apply different sharpening techniques"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Simple sharpening kernel
        kernel = np.array([[0, -1, 0],
                         [-1, 5, -1],
                         [0, -1, 0]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        
        # Unsharp masking
        gaussian = cv2.GaussianBlur(gray, (5, 5), 0)
        unsharp_mask = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)
        
        # Laplacian sharpening
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = cv2.convertScaleAbs(laplacian)
        laplacian_sharpened = cv2.addWeighted(gray, 1, laplacian, 0.5, 0)
        
        return {
            'original': gray,
            'simple_sharpen': sharpened,
            'unsharp_mask': unsharp_mask,
            'laplacian_sharpen': laplacian_sharpened
        }
    
    def frequency_domain_filtering(self, image):
        """Apply filtering in frequency domain"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Expand image to optimal size for DFT
        rows, cols = gray.shape
        optimal_rows = cv2.getOptimalDFTSize(rows)
        optimal_cols = cv2.getOptimalDFTSize(cols)
        padded = cv2.copyMakeBorder(gray, 0, optimal_rows - rows, 0, optimal_cols - cols,
                                  cv2.BORDER_CONSTANT, value=0)
        
        # Perform DFT
        dft = cv2.dft(np.float32(padded), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        
        # Magnitude spectrum for visualization
        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]) + 1)
        magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Create filters
        crow, ccol = optimal_rows // 2, optimal_cols // 2
        
        # Low pass filter
        low_pass_mask = np.zeros((optimal_rows, optimal_cols, 2), np.uint8)
        r = 30  # Radius
        center = [crow, ccol]
        x, y = np.ogrid[:optimal_rows, :optimal_cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
        low_pass_mask[mask_area] = 1
        
        # High pass filter
        high_pass_mask = np.ones((optimal_rows, optimal_cols, 2), np.uint8)
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
        high_pass_mask[mask_area] = 0
        
        # Apply filters
        low_pass = dft_shift * low_pass_mask
        high_pass = dft_shift * high_pass_mask
        
        # Inverse DFT
        low_pass_inverse = cv2.idft(np.fft.ifftshift(low_pass))
        high_pass_inverse = cv2.idft(np.fft.ifftshift(high_pass))
        
        # Compute magnitude
        low_pass_result = cv2.magnitude(low_pass_inverse[:,:,0], low_pass_inverse[:,:,1])
        high_pass_result = cv2.magnitude(high_pass_inverse[:,:,0], high_pass_inverse[:,:,1])
        
        # Normalize and crop to original size
        low_pass_result = cv2.normalize(low_pass_result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        high_pass_result = cv2.normalize(high_pass_result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        low_pass_result = low_pass_result[:rows, :cols]
        high_pass_result = high_pass_result[:rows, :cols]
        
        return {
            'original': gray,
            'magnitude_spectrum': magnitude_spectrum,
            'low_pass': low_pass_result,
            'high_pass': high_pass_result
        }

def create_sample_images():
    """Create sample images for testing"""
    # Create a simple image with shapes
    img1 = np.zeros((300, 300, 3), dtype=np.uint8)
    img1.fill(255)  # White background
    
    # Add some shapes
    cv2.rectangle(img1, (50, 50), (100, 100), (0, 0, 255), -1)  # Red rectangle
    cv2.circle(img1, (200, 150), 50, (0, 255, 0), -1)  # Green circle
    cv2.line(img1, (50, 200), (250, 200), (255, 0, 0), 5)  # Blue line
    
    # Create a noisy image
    img2 = np.zeros((300, 300), dtype=np.uint8)
    img2.fill(100)  # Gray background
    
    # Add salt and pepper noise
    noise = np.random.randint(0, 255, (300, 300))
    mask = np.random.randint(0, 100, (300, 300))
    img2[mask < 10] = 0  # Pepper
    img2[mask > 90] = 255  # Salt
    
    # Add text
    cv2.putText(img2, "OpenCV", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, 200, 3)
    
    # Create an image with edges
    img3 = np.zeros((300, 300), dtype=np.uint8)
    img3.fill(255)  # White background
    
    # Add some shapes with clear edges
    cv2.rectangle(img3, (50, 50), (250, 250), 0, 2)  # Black rectangle
    cv2.circle(img3, (150, 150), 80, 0, 2)  # Black circle
    cv2.line(img3, (50, 50), (250, 250), 0, 2)  # Black diagonal line
    
    return img1, img2, img3

def display_results(results, title, cols=3):
    """Display results using matplotlib"""
    n = len(results)
    rows = (n + cols - 1) // cols
    
    plt.figure(figsize=(15, 5 * rows))
    plt.suptitle(title, fontsize=16)
    
    for i, (name, img) in enumerate(results.items()):
        plt.subplot(rows, cols, i+1)
        
        # Handle different color spaces
        if len(img.shape) == 3:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(img, cmap='gray')
            
        plt.title(name)
        plt.axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

def main():
    """Main function"""
    print("Image Filtering and Convolution with OpenCV")
    print("==========================================")
    
    # Create or load sample images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), 'images')
    
    # Create sample images
    img1, img2, img3 = create_sample_images()
    
    # Save sample images if needed
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    cv2.imwrite(os.path.join(images_dir, 'filtering_sample1.jpg'), img1)
    cv2.imwrite(os.path.join(images_dir, 'filtering_sample2.jpg'), img2)
    cv2.imwrite(os.path.join(images_dir, 'filtering_sample3.jpg'), img3)
    
    # Initialize filtering class
    filtering = ImageFiltering()
    
    while True:
        print("\nSelect a demo:")
        print("1. Basic Filters")
        print("2. Edge Detection Filters")
        print("3. Custom Kernels")
        print("4. Image Sharpening")
        print("5. Frequency Domain Filtering")
        print("6. Manual Convolution")
        print("q. Quit")
        
        choice = input("\nEnter your choice: ")
        
        if choice == '1':
            print("\nBasic Filters Demo")
            results = filtering.apply_basic_filters(img2)
            display_results(results, "Basic Filtering Techniques")
            
        elif choice == '2':
            print("\nEdge Detection Filters Demo")
            results = filtering.apply_edge_filters(img3)
            display_results(results, "Edge Detection Filters", cols=3)
            
        elif choice == '3':
            print("\nCustom Kernels Demo")
            results = filtering.apply_custom_kernels(img1)
            display_results(results, "Custom Kernel Effects")
            
        elif choice == '4':
            print("\nImage Sharpening Demo")
            results = filtering.sharpen_image(img2)
            display_results(results, "Image Sharpening Techniques")
            
        elif choice == '5':
            print("\nFrequency Domain Filtering Demo")
            results = filtering.frequency_domain_filtering(img3)
            display_results(results, "Frequency Domain Filtering")
            
        elif choice == '6':
            print("\nManual Convolution Demo")
            # Define a simple kernel
            kernel = np.array([[-1, -1, -1],
                             [-1,  9, -1],
                             [-1, -1, -1]])
            result = filtering.apply_convolution(img1, kernel)
            
            # Display results
            plt.figure(figsize=(10, 5))
            plt.suptitle("Manual Convolution", fontsize=16)
            
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
            plt.title("Original")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(result, cmap='gray')
            plt.title(f"Convolved with Kernel:\n{kernel}")
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
            
        elif choice.lower() == 'q':
            break
        
        else:
            print("\nInvalid choice. Please try again.")
    
    print("\nDemo completed. Thank you!")

if __name__ == "__main__":
    main()