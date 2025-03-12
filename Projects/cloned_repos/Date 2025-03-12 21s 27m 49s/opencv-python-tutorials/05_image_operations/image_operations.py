#!/usr/bin/env python3
"""
Image Arithmetic and Bitwise Operations with OpenCV
This script demonstrates various image arithmetic and bitwise operations in OpenCV
"""

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def main():
    # Get the path to the images directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), 'images')
    
    # Try to load images from the images directory
    # If no suitable images are found, create sample images
    try:
        # Try to load two images for operations
        img1_path = os.path.join(images_dir, 'landscape.jpg')
        img2_path = os.path.join(images_dir, 'objects.jpg')
        
        if os.path.exists(img1_path) and os.path.exists(img2_path):
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
            # Resize the second image to match the first
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        else:
            # Try other potential images
            for img_name in ['shapes.jpg', 'faces.jpg', 'text.jpg']:
                img_path = os.path.join(images_dir, img_name)
                if os.path.exists(img_path):
                    img1 = cv2.imread(img_path)
                    # Create a second image by modifying the first
                    img2 = create_second_image(img1)
                    break
            else:
                # If no image is found, create sample images
                img1, img2 = create_sample_images()
    except:
        # Fallback to creating sample images
        img1, img2 = create_sample_images()
    
    # Ensure we have valid images
    if img1 is None or img2 is None or img1.size == 0 or img2.size == 0:
        img1, img2 = create_sample_images()
    
    # Get image dimensions
    height, width = img1.shape[:2]
    
    # 1. Image Addition
    print("\n1. IMAGE ADDITION")
    
    # Add the images
    added_img = cv2.add(img1, img2)
    
    # Display the result
    display_images([img1, img2, added_img], ['Image 1', 'Image 2', 'Added Image'])
    print("Press any key to continue...")
    cv2.waitKey(0)
    
    # 2. Weighted Addition (Blending)
    print("\n2. WEIGHTED ADDITION (BLENDING)")
    
    # Create a list of blended images with different weights
    blended_images = []
    titles = []
    
    for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
        beta = 1.0 - alpha
        blended = cv2.addWeighted(img1, alpha, img2, beta, 0)
        blended_images.append(blended)
        titles.append(f'alpha={alpha}, beta={beta}')
    
    # Display the results
    display_images(blended_images, titles)
    print("Press any key to continue...")
    cv2.waitKey(0)
    
    # 3. Subtraction
    print("\n3. SUBTRACTION")
    
    # Subtract img2 from img1
    subtracted_img1 = cv2.subtract(img1, img2)
    
    # Subtract img1 from img2
    subtracted_img2 = cv2.subtract(img2, img1)
    
    # Display the results
    display_images([img1, img2, subtracted_img1, subtracted_img2], 
                  ['Image 1', 'Image 2', 'Image 1 - Image 2', 'Image 2 - Image 1'])
    print("Press any key to continue...")
    cv2.waitKey(0)
    
    # 4. Multiplication and Division
    print("\n4. MULTIPLICATION AND DIVISION")
    
    # Multiply the images
    multiplied_img = cv2.multiply(img1, img2)
    
    # Divide img1 by img2 (avoid division by zero)
    img2_safe = np.copy(img2)
    img2_safe[img2_safe == 0] = 1  # Replace zeros with ones to avoid division by zero
    divided_img = cv2.divide(img1, img2_safe)
    
    # Display the results
    display_images([img1, img2, multiplied_img, divided_img], 
                  ['Image 1', 'Image 2', 'Multiplied Image', 'Divided Image'])
    print("Press any key to continue...")
    cv2.waitKey(0)
    
    # 5. Creating Masks
    print("\n5. CREATING MASKS")
    
    # Create a black image
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Draw a white circle in the middle
    center = (width // 2, height // 2)
    radius = min(width, height) // 4
    cv2.circle(mask, center, radius, 255, -1)
    
    # Create a rectangular mask
    rect_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.rectangle(rect_mask, (width//4, height//4), (3*width//4, 3*height//4), 255, -1)
    
    # Create a gradient mask
    gradient_mask = np.zeros((height, width), dtype=np.uint8)
    for i in range(width):
        gradient_mask[:, i] = i * 255 // width
    
    # Display the masks
    display_images([mask, rect_mask, gradient_mask], 
                  ['Circular Mask', 'Rectangular Mask', 'Gradient Mask'])
    print("Press any key to continue...")
    cv2.waitKey(0)
    
    # 6. Bitwise AND with Mask
    print("\n6. BITWISE AND WITH MASK")
    
    # Apply the circular mask using bitwise AND
    masked_img1 = cv2.bitwise_and(img1, img1, mask=mask)
    
    # Apply the rectangular mask
    masked_img2 = cv2.bitwise_and(img1, img1, mask=rect_mask)
    
    # Apply the gradient mask
    masked_img3 = cv2.bitwise_and(img1, img1, mask=gradient_mask)
    
    # Display the results
    display_images([img1, masked_img1, masked_img2, masked_img3], 
                  ['Original Image', 'Circular Mask Applied', 'Rectangular Mask Applied', 'Gradient Mask Applied'])
    print("Press any key to continue...")
    cv2.waitKey(0)
    
    # 7. Bitwise Operations
    print("\n7. BITWISE OPERATIONS")
    
    # Create another image with a different shape
    img3 = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(img3, (width//4, height//4), (3*width//4, 3*height//4), (0, 0, 255), -1)
    
    # Apply bitwise operations
    and_img = cv2.bitwise_and(img1, img3)
    or_img = cv2.bitwise_or(img1, img3)
    xor_img = cv2.bitwise_xor(img1, img3)
    not_img1 = cv2.bitwise_not(img1)
    
    # Display the results
    display_images([img1, img3, and_img, or_img, xor_img, not_img1], 
                  ['Image 1', 'Image 3', 'AND Operation', 'OR Operation', 'XOR Operation', 'NOT Operation (Image 1)'])
    print("Press any key to continue...")
    cv2.waitKey(0)
    
    # 8. Practical Application: Image Blending with Masks
    print("\n8. PRACTICAL APPLICATION: IMAGE BLENDING WITH MASKS")
    
    # Blend images using the gradient mask
    img1_masked = cv2.bitwise_and(img1, img1, mask=gradient_mask)
    img2_masked = cv2.bitwise_and(img2, img2, mask=cv2.bitwise_not(gradient_mask))
    blended_with_mask = cv2.add(img1_masked, img2_masked)
    
    # Display the result
    display_images([img1, img2, gradient_mask, img1_masked, img2_masked, blended_with_mask], 
                  ['Image 1', 'Image 2', 'Gradient Mask', 'Image 1 Masked', 'Image 2 Masked', 'Blended Result'])
    print("Press any key to continue...")
    cv2.waitKey(0)
    
    # 9. Practical Application: Background Removal
    print("\n9. PRACTICAL APPLICATION: BACKGROUND REMOVAL")
    
    # Use the rectangular mask as a foreground mask
    foreground_mask = rect_mask
    
    # Extract the foreground
    foreground = cv2.bitwise_and(img1, img1, mask=foreground_mask)
    
    # Create a colored background
    background = np.ones((height, width, 3), dtype=np.uint8) * [0, 255, 0]  # Green background
    background_mask = cv2.bitwise_not(foreground_mask)
    background = cv2.bitwise_and(background, background, mask=background_mask)
    
    # Combine foreground and new background
    result = cv2.add(foreground, background)
    
    # Display the result
    display_images([img1, foreground_mask, foreground, background, result], 
                  ['Original Image', 'Foreground Mask', 'Extracted Foreground', 'New Background', 'Final Result'])
    print("Press any key to continue...")
    cv2.waitKey(0)
    
    # 10. Practical Application: Logo Watermarking
    print("\n10. PRACTICAL APPLICATION: LOGO WATERMARKING")
    
    # Create a logo
    logo = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(logo, (50, 50), 40, (0, 0, 255), -1)
    cv2.putText(logo, 'CV', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Resize the logo
    logo_size = min(width, height) // 4
    logo_resized = cv2.resize(logo, (logo_size, logo_size))
    logo_height, logo_width = logo_resized.shape[:2]
    
    # Create a copy of the original image
    watermarked_img = img1.copy()
    
    # Create a region of interest (ROI) in the top-right corner
    roi = watermarked_img[0:logo_height, width-logo_width:width]
    
    # Create a mask of the logo and its inverse
    logo_gray = cv2.cvtColor(logo_resized, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(logo_gray, 10, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    
    # Black-out the area of the logo in ROI
    roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)
    
    # Take only the logo region from the logo image
    logo_fg = cv2.bitwise_and(logo_resized, logo_resized, mask=mask)
    
    # Put the logo in ROI and modify the original image
    dst = cv2.add(roi_bg, logo_fg)
    watermarked_img[0:logo_height, width-logo_width:width] = dst
    
    # Display the result
    display_images([img1, logo_resized, watermarked_img], 
                  ['Original Image', 'Logo', 'Watermarked Image'])
    print("Press any key to exit...")
    cv2.waitKey(0)
    
    # Clean up
    cv2.destroyAllWindows()
    print("\nAll operations completed successfully!")

def create_sample_images():
    """Create two sample images for arithmetic and bitwise operations"""
    # Create the first image with some shapes
    width, height = 600, 400
    img1 = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background
    
    # Add some shapes to the first image
    # Blue rectangle
    cv2.rectangle(img1, (50, 50), (200, 200), (255, 0, 0), -1)
    
    # Green circle
    cv2.circle(img1, (400, 100), 80, (0, 255, 0), -1)
    
    # Red triangle
    pts = np.array([[300, 300], [200, 200], [400, 200]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(img1, [pts], (0, 0, 255))
    
    # Create the second image with different shapes
    img2 = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background
    
    # Add some shapes to the second image
    # Yellow rectangle
    cv2.rectangle(img2, (100, 100), (250, 250), (0, 255, 255), -1)
    
    # Magenta circle
    cv2.circle(img2, (450, 150), 60, (255, 0, 255), -1)
    
    # Cyan triangle
    pts = np.array([[250, 350], [150, 250], [350, 250]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(img2, [pts], (255, 255, 0))
    
    return img1, img2

def create_second_image(img1):
    """Create a second image by modifying the first image"""
    # Create a copy of the first image
    img2 = img1.copy()
    
    # Apply some transformations to create a different image
    # Rotate the image
    height, width = img2.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, 30, 1.0)
    img2 = cv2.warpAffine(img2, rotation_matrix, (width, height))
    
    # Change the color balance
    img2 = cv2.convertScaleAbs(img2, alpha=1.2, beta=30)
    
    return img2

def display_images(images, titles):
    """Display multiple images in a grid using matplotlib"""
    n_images = len(images)
    n_cols = 3
    n_rows = (n_images + n_cols - 1) // n_cols  # Ceiling division
    
    plt.figure(figsize=(15, 5 * n_rows))
    
    for i in range(n_images):
        plt.subplot(n_rows, n_cols, i + 1)
        
        # Check if the image is grayscale or color
        if len(images[i].shape) == 2 or images[i].shape[2] == 1:
            plt.imshow(images[i], cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
            
        plt.title(titles[i])
        plt.xticks([])
        plt.yticks([])
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()