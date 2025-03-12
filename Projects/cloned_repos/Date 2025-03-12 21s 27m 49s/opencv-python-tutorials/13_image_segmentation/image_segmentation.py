#!/usr/bin/env python3
"""
Image Segmentation with OpenCV
This script demonstrates various image segmentation techniques using OpenCV
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

class ImageSegmentation:
    def __init__(self):
        """Initialize image segmentation class"""
        pass
    
    def basic_thresholding(self, image, threshold=127, max_val=255):
        """Apply basic thresholding techniques"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Simple thresholding
        ret, thresh_binary = cv2.threshold(gray, threshold, max_val, cv2.THRESH_BINARY)
        ret, thresh_binary_inv = cv2.threshold(gray, threshold, max_val, cv2.THRESH_BINARY_INV)
        ret, thresh_trunc = cv2.threshold(gray, threshold, max_val, cv2.THRESH_TRUNC)
        ret, thresh_tozero = cv2.threshold(gray, threshold, max_val, cv2.THRESH_TOZERO)
        ret, thresh_tozero_inv = cv2.threshold(gray, threshold, max_val, cv2.THRESH_TOZERO_INV)
        
        # Otsu's thresholding
        ret_otsu, thresh_otsu = cv2.threshold(gray, 0, max_val, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Adaptive thresholding
        thresh_adaptive_mean = cv2.adaptiveThreshold(gray, max_val, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                  cv2.THRESH_BINARY, 11, 2)
        thresh_adaptive_gaussian = cv2.adaptiveThreshold(gray, max_val, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                      cv2.THRESH_BINARY, 11, 2)
        
        return {
            'original': gray,
            'binary': thresh_binary,
            'binary_inv': thresh_binary_inv,
            'trunc': thresh_trunc,
            'tozero': thresh_tozero,
            'tozero_inv': thresh_tozero_inv,
            'otsu': thresh_otsu,
            'adaptive_mean': thresh_adaptive_mean,
            'adaptive_gaussian': thresh_adaptive_gaussian
        }
    
    def watershed_segmentation(self, image):
        """Apply watershed algorithm for segmentation"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Apply thresholding
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Noise removal
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Sure background area
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
        
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        ret, markers = cv2.connectedComponents(sure_fg)
        
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers + 1
        
        # Mark the region of unknown with zero
        markers[unknown == 255] = 0
        
        # Apply watershed algorithm
        markers = cv2.watershed(image, markers)
        image[markers == -1] = [255, 0, 0]  # Mark watershed boundaries in red
        
        return {
            'original': image.copy(),
            'thresholded': thresh,
            'sure_bg': sure_bg,
            'sure_fg': sure_fg,
            'distance_transform': cv2.normalize(dist_transform, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
            'markers': markers.astype(np.uint8) * 10,  # Multiply for better visualization
            'result': image
        }
    
    def grabcut_segmentation(self, image, rect=None, mask=None, iterCount=5):
        """Apply GrabCut algorithm for foreground extraction"""
        # Make a copy of the image
        img = image.copy()
        
        # If no rectangle is provided, use the center of the image
        if rect is None:
            h, w = img.shape[:2]
            rect = (w//4, h//4, w//2, h//2)
        
        # Initialize mask
        if mask is None:
            mask = np.zeros(img.shape[:2], np.uint8)
        
        # Initialize background and foreground models
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        
        # Apply GrabCut
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, iterCount, cv2.GC_INIT_WITH_RECT)
        
        # Create mask where sure and probable foreground are set to 1, else 0
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        
        # Multiply image with the mask to get the segmented image
        result = img * mask2[:, :, np.newaxis]
        
        return {
            'original': img,
            'mask': mask * 85,  # Scale for better visualization
            'result': result
        }
    
    def kmeans_segmentation(self, image, K=5):
        """Apply K-means clustering for segmentation"""
        # Reshape the image
        Z = image.reshape((-1, 3))
        
        # Convert to float32
        Z = np.float32(Z)
        
        # Define criteria, number of clusters and apply kmeans
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        # Convert back to uint8
        center = np.uint8(center)
        res = center[label.flatten()]
        result = res.reshape((image.shape))
        
        # Create a label image
        label_image = label.reshape((image.shape[0], image.shape[1]))
        
        return {
            'original': image,
            'result': result,
            'labels': label_image.astype(np.uint8) * (255 // K)  # Scale for visualization
        }
    
    def mean_shift_segmentation(self, image, spatial_radius=30, color_radius=30, min_density=50):
        """Apply mean shift segmentation"""
        # Apply mean shift filtering
        shifted = cv2.pyrMeanShiftFiltering(image, spatial_radius, color_radius, min_density)
        
        # Convert to grayscale for further processing
        gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
        
        # Threshold to get binary image
        ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours on the original image
        contour_img = image.copy()
        cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)
        
        return {
            'original': image,
            'shifted': shifted,
            'thresholded': thresh,
            'contours': contour_img
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
    
    # Create a more complex image with overlapping objects
    img2 = np.zeros((300, 300, 3), dtype=np.uint8)
    img2.fill(255)  # White background
    
    # Add overlapping circles
    cv2.circle(img2, (100, 100), 80, (255, 0, 0), -1)  # Blue circle
    cv2.circle(img2, (200, 150), 100, (0, 255, 0), -1)  # Green circle
    cv2.circle(img2, (150, 200), 70, (0, 0, 255), -1)  # Red circle
    
    # Create a gradient image
    img3 = np.zeros((300, 300), dtype=np.uint8)
    for i in range(300):
        img3[:, i] = i * 255 // 300
    img3 = cv2.cvtColor(img3, cv2.COLOR_GRAY2BGR)
    
    return img1, img2, img3

def display_results(results, title):
    """Display results using matplotlib"""
    n = len(results)
    plt.figure(figsize=(15, 5))
    plt.suptitle(title, fontsize=16)
    
    for i, (name, img) in enumerate(results.items()):
        plt.subplot(1, n, i+1)
        
        # Handle different color spaces
        if len(img.shape) == 3:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(img, cmap='gray')
            
        plt.title(name)
        plt.axis('off')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()

def main():
    """Main function"""
    print("Image Segmentation Techniques with OpenCV")
    print("========================================")
    
    # Create or load sample images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), 'images')
    
    # Create sample images
    img1, img2, img3 = create_sample_images()
    
    # Save sample images if needed
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    cv2.imwrite(os.path.join(images_dir, 'segmentation_sample1.jpg'), img1)
    cv2.imwrite(os.path.join(images_dir, 'segmentation_sample2.jpg'), img2)
    cv2.imwrite(os.path.join(images_dir, 'segmentation_sample3.jpg'), img3)
    
    # Initialize segmentation class
    segmentation = ImageSegmentation()
    
    while True:
        print("\nSelect a demo:")
        print("1. Basic Thresholding")
        print("2. Watershed Segmentation")
        print("3. GrabCut Segmentation")
        print("4. K-means Segmentation")
        print("5. Mean Shift Segmentation")
        print("q. Quit")
        
        choice = input("\nEnter your choice: ")
        
        if choice == '1':
            print("\nBasic Thresholding Demo")
            results = segmentation.basic_thresholding(img3[:,:,0])
            display_results(results, "Basic Thresholding Techniques")
            
        elif choice == '2':
            print("\nWatershed Segmentation Demo")
            results = segmentation.watershed_segmentation(img2)
            display_results(results, "Watershed Segmentation")
            
        elif choice == '3':
            print("\nGrabCut Segmentation Demo")
            # Define rectangle around the object of interest
            h, w = img1.shape[:2]
            rect = (50, 50, 200, 200)  # (x, y, width, height)
            results = segmentation.grabcut_segmentation(img1, rect)
            display_results(results, "GrabCut Segmentation")
            
        elif choice == '4':
            print("\nK-means Segmentation Demo")
            results = segmentation.kmeans_segmentation(img2, K=3)
            display_results(results, "K-means Segmentation (K=3)")
            
        elif choice == '5':
            print("\nMean Shift Segmentation Demo")
            results = segmentation.mean_shift_segmentation(img2)
            display_results(results, "Mean Shift Segmentation")
            
        elif choice.lower() == 'q':
            break
        
        else:
            print("\nInvalid choice. Please try again.")
    
    print("\nDemo completed. Thank you!")

if __name__ == "__main__":
    main()