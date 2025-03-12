#!/usr/bin/env python3
"""
Image Transformations with OpenCV
This script demonstrates various image transformation techniques using OpenCV
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

class ImageTransformations:
    def __init__(self):
        """Initialize image transformations class"""
        pass
    
    def resize_image(self, image, scale_factor=None, dimensions=None):
        """
        Resize image using different methods
        """
        if scale_factor is not None:
            # Resize by scale factor
            resized = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)
        elif dimensions is not None:
            # Resize to specific dimensions
            resized = cv2.resize(image, dimensions)
        else:
            raise ValueError("Either scale_factor or dimensions must be provided")
        
        return resized
    
    def rotate_image(self, image, angle, center=None, scale=1.0):
        """
        Rotate image by given angle
        """
        height, width = image.shape[:2]
        
        if center is None:
            center = (width // 2, height // 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        
        # Perform rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
        
        return rotated
    
    def translate_image(self, image, x, y):
        """
        Translate image by (x,y)
        """
        height, width = image.shape[:2]
        
        # Create translation matrix
        translation_matrix = np.float32([[1, 0, x],
                                       [0, 1, y]])
        
        # Perform translation
        translated = cv2.warpAffine(image, translation_matrix, (width, height))
        
        return translated
    
    def affine_transform(self, image, src_points=None, dst_points=None):
        """
        Apply affine transformation
        """
        height, width = image.shape[:2]
        
        if src_points is None or dst_points is None:
            # Default transformation: slight skew
            src_points = np.float32([[0, 0], [width - 1, 0], [0, height - 1]])
            dst_points = np.float32([[width * 0.1, height * 0.1], 
                                   [width * 0.9, height * 0.2], 
                                   [width * 0.2, height * 0.9]])
        
        # Get affine transform matrix
        affine_matrix = cv2.getAffineTransform(src_points, dst_points)
        
        # Apply transformation
        transformed = cv2.warpAffine(image, affine_matrix, (width, height))
        
        return transformed
    
    def perspective_transform(self, image, src_points=None, dst_points=None):
        """
        Apply perspective transformation
        """
        height, width = image.shape[:2]
        
        if src_points is None or dst_points is None:
            # Default transformation: perspective correction
            src_points = np.float32([[0, 0], [width - 1, 0], 
                                   [width - 1, height - 1], [0, height - 1]])
            dst_points = np.float32([[width * 0.1, height * 0.1], 
                                   [width * 0.9, height * 0.1],
                                   [width * 0.9, height * 0.9], 
                                   [width * 0.1, height * 0.9]])
        
        # Get perspective transform matrix
        perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Apply transformation
        transformed = cv2.warpPerspective(image, perspective_matrix, (width, height))
        
        return transformed
    
    def polar_warp(self, image, center=None, maxRadius=None):
        """
        Convert image to polar coordinates
        """
        height, width = image.shape[:2]
        
        if center is None:
            center = (width // 2, height // 2)
        
        if maxRadius is None:
            maxRadius = min(center[0], center[1])
        
        # Linear Polar
        linear_polar = cv2.linearPolar(image.astype(np.float32), center, maxRadius, cv2.WARP_FILL_OUTLIERS)
        linear_polar = linear_polar.astype(np.uint8)
        
        # Log Polar
        log_polar = cv2.logPolar(image.astype(np.float32), center, maxRadius, cv2.WARP_FILL_OUTLIERS)
        log_polar = log_polar.astype(np.uint8)
        
        return linear_polar, log_polar
    
    def remap_image(self, image):
        """
        Demonstrate image remapping
        """
        height, width = image.shape[:2]
        
        # Create maps for different effects
        
        # 1. Vertical flip
        map_x_flip = np.zeros((height, width), np.float32)
        map_y_flip = np.zeros((height, width), np.float32)
        
        for i in range(height):
            for j in range(width):
                map_x_flip[i, j] = j
                map_y_flip[i, j] = height - i - 1
        
        # 2. Horizontal wave
        map_x_wave = np.zeros((height, width), np.float32)
        map_y_wave = np.zeros((height, width), np.float32)
        
        for i in range(height):
            for j in range(width):
                map_x_wave[i, j] = j + 20 * np.sin(i / 10)
                map_y_wave[i, j] = i
        
        # 3. Magnify center
        map_x_magnify = np.zeros((height, width), np.float32)
        map_y_magnify = np.zeros((height, width), np.float32)
        
        center_x, center_y = width // 2, height // 2
        
        for i in range(height):
            for j in range(width):
                dx = j - center_x
                dy = i - center_y
                radius = np.sqrt(dx**2 + dy**2)
                
                if radius < min(width, height) // 4:
                    # Magnify center
                    theta = np.arctan2(dy, dx)
                    radius = radius * 0.7  # Magnification factor
                    map_x_magnify[i, j] = center_x + radius * np.cos(theta)
                    map_y_magnify[i, j] = center_y + radius * np.sin(theta)
                else:
                    map_x_magnify[i, j] = j
                    map_y_magnify[i, j] = i
        
        # Apply remapping
        flipped = cv2.remap(image, map_x_flip, map_y_flip, cv2.INTER_LINEAR)
        waved = cv2.remap(image, map_x_wave, map_y_wave, cv2.INTER_LINEAR)
        magnified = cv2.remap(image, map_x_magnify, map_y_magnify, cv2.INTER_LINEAR)
        
        return {
            'flipped': flipped,
            'waved': waved,
            'magnified': magnified
        }
    
    def compare_interpolation_methods(self, image, scale_factor=0.5):
        """
        Compare different interpolation methods
        """
        height, width = image.shape[:2]
        new_size = (int(width * scale_factor), int(height * scale_factor))
        
        methods = {
            'INTER_NEAREST': cv2.INTER_NEAREST,
            'INTER_LINEAR': cv2.INTER_LINEAR,
            'INTER_CUBIC': cv2.INTER_CUBIC,
            'INTER_LANCZOS4': cv2.INTER_LANCZOS4,
            'INTER_AREA': cv2.INTER_AREA
        }
        
        results = {'original': image}
        
        for name, method in methods.items():
            results[name] = cv2.resize(image, new_size, interpolation=method)
            # Resize back to original size for comparison
            results[name] = cv2.resize(results[name], (width, height), interpolation=method)
        
        return results

def create_sample_images():
    """Create sample images for testing"""
    # Create a simple image with a grid pattern
    img1 = np.zeros((300, 300, 3), dtype=np.uint8)
    img1.fill(255)  # White background
    
    # Add grid lines
    for i in range(0, 301, 30):
        cv2.line(img1, (0, i), (300, i), (0, 0, 0), 1)  # Horizontal lines
        cv2.line(img1, (i, 0), (i, 300), (0, 0, 0), 1)  # Vertical lines
    
    # Add some shapes
    cv2.rectangle(img1, (90, 90), (210, 210), (0, 0, 255), 2)  # Red rectangle
    cv2.circle(img1, (150, 150), 75, (0, 255, 0), 2)  # Green circle
    
    # Create an image with text
    img2 = np.zeros((300, 300, 3), dtype=np.uint8)
    img2.fill(255)  # White background
    
    # Add text
    cv2.putText(img2, "OpenCV", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    cv2.putText(img2, "Transforms", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
    
    # Create an image with a checkerboard pattern
    img3 = np.zeros((300, 300, 3), dtype=np.uint8)
    
    # Create checkerboard
    square_size = 30
    for i in range(0, 300, square_size):
        for j in range(0, 300, square_size):
            if (i // square_size + j // square_size) % 2 == 0:
                img3[i:i+square_size, j:j+square_size] = (255, 255, 255)
    
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
    print("Image Transformations with OpenCV")
    print("================================")
    
    # Create or load sample images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), 'images')
    
    # Create sample images
    img1, img2, img3 = create_sample_images()
    
    # Save sample images if needed
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    cv2.imwrite(os.path.join(images_dir, 'transform_sample1.jpg'), img1)
    cv2.imwrite(os.path.join(images_dir, 'transform_sample2.jpg'), img2)
    cv2.imwrite(os.path.join(images_dir, 'transform_sample3.jpg'), img3)
    
    # Initialize transformations class
    transformations = ImageTransformations()
    
    while True:
        print("\nSelect a demo:")
        print("1. Resize Image")
        print("2. Rotate Image")
        print("3. Translate Image")
        print("4. Affine Transform")
        print("5. Perspective Transform")
        print("6. Polar Warp")
        print("7. Image Remapping")
        print("8. Interpolation Methods")
        print("q. Quit")
        
        choice = input("\nEnter your choice: ")
        
        if choice == '1':
            print("\nResize Image Demo")
            
            # Resize with different scale factors
            results = {
                'original': img1,
                'scale_0.5': transformations.resize_image(img1, scale_factor=0.5),
                'scale_2.0': transformations.resize_image(img1, scale_factor=2.0),
                'size_150x150': transformations.resize_image(img1, dimensions=(150, 150))
            }
            
            display_results(results, "Image Resizing")
            
        elif choice == '2':
            print("\nRotate Image Demo")
            
            # Rotate by different angles
            results = {
                'original': img1,
                'rotate_45': transformations.rotate_image(img1, 45),
                'rotate_90': transformations.rotate_image(img1, 90),
                'rotate_180': transformations.rotate_image(img1, 180)
            }
            
            display_results(results, "Image Rotation")
            
        elif choice == '3':
            print("\nTranslate Image Demo")
            
            # Translate in different directions
            results = {
                'original': img1,
                'translate_right': transformations.translate_image(img1, 50, 0),
                'translate_down': transformations.translate_image(img1, 0, 50),
                'translate_diagonal': transformations.translate_image(img1, 50, 50)
            }
            
            display_results(results, "Image Translation")
            
        elif choice == '4':
            print("\nAffine Transform Demo")
            
            # Apply affine transformation
            height, width = img1.shape[:2]
            
            # Define different affine transformations
            # 1. Skew
            src_points1 = np.float32([[0, 0], [width - 1, 0], [0, height - 1]])
            dst_points1 = np.float32([[width * 0.1, height * 0.1], 
                                    [width * 0.9, height * 0.2], 
                                    [width * 0.2, height * 0.9]])
            
            # 2. Stretch
            src_points2 = np.float32([[0, 0], [width - 1, 0], [0, height - 1]])
            dst_points2 = np.float32([[0, 0], [width - 1, 0], [width * 0.3, height - 1]])
            
            results = {
                'original': img1,
                'affine_default': transformations.affine_transform(img1),
                'affine_skew': transformations.affine_transform(img1, src_points1, dst_points1),
                'affine_stretch': transformations.affine_transform(img1, src_points2, dst_points2)
            }
            
            display_results(results, "Affine Transformation")
            
        elif choice == '5':
            print("\nPerspective Transform Demo")
            
            # Apply perspective transformation
            height, width = img3.shape[:2]
            
            # Define different perspective transformations
            # 1. Trapezoid
            src_points1 = np.float32([[0, 0], [width - 1, 0], 
                                    [width - 1, height - 1], [0, height - 1]])
            dst_points1 = np.float32([[width * 0.1, 0], [width * 0.9, 0],
                                    [width - 1, height - 1], [0, height - 1]])
            
            # 2. Book perspective
            src_points2 = np.float32([[0, 0], [width - 1, 0], 
                                    [width - 1, height - 1], [0, height - 1]])
            dst_points2 = np.float32([[width * 0.2, height * 0.1], 
                                    [width * 0.8, height * 0.1],
                                    [width * 0.9, height * 0.9], 
                                    [width * 0.1, height * 0.9]])
            
            results = {
                'original': img3,
                'perspective_default': transformations.perspective_transform(img3),
                'perspective_trapezoid': transformations.perspective_transform(img3, src_points1, dst_points1),
                'perspective_book': transformations.perspective_transform(img3, src_points2, dst_points2)
            }
            
            display_results(results, "Perspective Transformation")
            
        elif choice == '6':
            print("\nPolar Warp Demo")
            
            # Apply polar warping
            linear_polar, log_polar = transformations.polar_warp(img1)
            
            results = {
                'original': img1,
                'linear_polar': linear_polar,
                'log_polar': log_polar
            }
            
            display_results(results, "Polar Warping")
            
        elif choice == '7':
            print("\nImage Remapping Demo")
            
            # Apply remapping
            remapped = transformations.remap_image(img2)
            
            results = {'original': img2}
            results.update(remapped)
            
            display_results(results, "Image Remapping")
            
        elif choice == '8':
            print("\nInterpolation Methods Demo")
            
            # Compare interpolation methods
            results = transformations.compare_interpolation_methods(img2)
            
            display_results(results, "Interpolation Methods Comparison")
            
        elif choice.lower() == 'q':
            break
        
        else:
            print("\nInvalid choice. Please try again.")
    
    print("\nDemo completed. Thank you!")

if __name__ == "__main__":
    main()