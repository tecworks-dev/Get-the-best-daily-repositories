#!/usr/bin/env python3
"""
Feature Detection and Matching with OpenCV
This script demonstrates various feature detection and matching techniques in OpenCV
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
        # Try to load images for feature matching
        img1_path = os.path.join(images_dir, 'panorama_1.jpg')
        img2_path = os.path.join(images_dir, 'panorama_2.jpg')
        
        if os.path.exists(img1_path) and os.path.exists(img2_path):
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
        else:
            # Try other potential images
            for img_name in ['landscape.jpg', 'objects.jpg', 'shapes.jpg']:
                img_path = os.path.join(images_dir, img_name)
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    # Create two versions of the same image with slight differences
                    img1 = img.copy()
                    img2 = rotate_and_scale(img.copy(), angle=15, scale=0.9)
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
    
    # Convert images to grayscale for feature detection
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # 1. Harris Corner Detector
    print("\n1. HARRIS CORNER DETECTOR")
    
    # Detect corners
    gray1_float = np.float32(gray1)
    corners = cv2.cornerHarris(gray1_float, blockSize=2, ksize=3, k=0.04)
    
    # Dilate to mark the corners
    corners = cv2.dilate(corners, None)
    
    # Create a copy of the image to draw on
    img1_harris = img1.copy()
    
    # Threshold for an optimal value, marking corners in red
    img1_harris[corners > 0.01 * corners.max()] = [0, 0, 255]
    
    # Display the result
    cv2.imshow('Harris Corner Detector', img1_harris)
    print("Press any key to continue...")
    cv2.waitKey(0)
    
    # 2. Shi-Tomasi Corner Detector
    print("\n2. SHI-TOMASI CORNER DETECTOR")
    
    # Detect corners
    corners = cv2.goodFeaturesToTrack(gray1, maxCorners=100, qualityLevel=0.01, minDistance=10)
    
    # Create a copy of the image to draw on
    img1_shi_tomasi = img1.copy()
    
    # Draw the corners
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(img1_shi_tomasi, (int(x), int(y)), 3, [0, 0, 255], -1)
    
    # Display the result
    cv2.imshow('Shi-Tomasi Corner Detector', img1_shi_tomasi)
    print("Press any key to continue...")
    cv2.waitKey(0)
    
    # 3. SIFT (Scale-Invariant Feature Transform)
    print("\n3. SIFT (SCALE-INVARIANT FEATURE TRANSFORM)")
    
    try:
        # Create SIFT detector
        sift = cv2.SIFT_create()
        
        # Detect keypoints and compute descriptors
        keypoints1 = sift.detect(gray1, None)
        
        # Draw keypoints
        img1_sift = cv2.drawKeypoints(img1, keypoints1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        # Display the result
        cv2.imshow('SIFT Keypoints', img1_sift)
        print("Press any key to continue...")
        cv2.waitKey(0)
        
        # Compute descriptors
        keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)
        
        # Match features using FLANN
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)
        
        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        
        # Draw good matches
        img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=2)
        
        # Display the result
        cv2.imshow('SIFT Matches', img_matches)
        print("Press any key to continue...")
        cv2.waitKey(0)
        
        # Find homography if enough matches are found
        MIN_MATCH_COUNT = 10
        if len(good_matches) > MIN_MATCH_COUNT:
            src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            # Draw outline of the object
            h, w = img1.shape[:2]
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            
            img2_homography = img2.copy()
            img2_homography = cv2.polylines(img2_homography, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
            
            # Display the result
            cv2.imshow('Homography', img2_homography)
            print("Press any key to continue...")
            cv2.waitKey(0)
    except Exception as e:
        print(f"SIFT not available or error occurred: {e}")
    
    # 4. ORB (Oriented FAST and Rotated BRIEF)
    print("\n4. ORB (ORIENTED FAST AND ROTATED BRIEF)")
    
    # Create ORB detector
    orb = cv2.ORB_create()
    
    # Detect keypoints and compute descriptors
    keypoints1 = orb.detect(gray1, None)
    
    # Draw keypoints
    img1_orb = cv2.drawKeypoints(img1, keypoints1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # Display the result
    cv2.imshow('ORB Keypoints', img1_orb)
    print("Press any key to continue...")
    cv2.waitKey(0)
    
    # Compute descriptors
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)
    
    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Match descriptors
    matches = bf.match(descriptors1, descriptors2)
    
    # Sort them in order of their distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Draw first 30 matches
    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:30], None, flags=2)
    
    # Display the result
    cv2.imshow('ORB Matches', img_matches)
    print("Press any key to continue...")
    cv2.waitKey(0)
    
    # 5. FAST (Features from Accelerated Segment Test)
    print("\n5. FAST (FEATURES FROM ACCELERATED SEGMENT TEST)")
    
    # Create FAST detector
    fast = cv2.FastFeatureDetector_create()
    
    # Detect keypoints
    keypoints = fast.detect(gray1, None)
    
    # Draw keypoints
    img1_fast = cv2.drawKeypoints(img1, keypoints, None, color=(0, 255, 0))
    
    # Display the result
    cv2.imshow('FAST Keypoints', img1_fast)
    print("Press any key to continue...")
    cv2.waitKey(0)
    
    # 6. Feature Matching Comparison
    print("\n6. FEATURE MATCHING COMPARISON")
    
    # Create a figure to display all feature detection methods
    plt.figure(figsize=(15, 10))
    
    # Harris
    plt.subplot(2, 3, 1)
    plt.imshow(cv2.cvtColor(img1_harris, cv2.COLOR_BGR2RGB))
    plt.title('Harris Corner Detector')
    plt.axis('off')
    
    # Shi-Tomasi
    plt.subplot(2, 3, 2)
    plt.imshow(cv2.cvtColor(img1_shi_tomasi, cv2.COLOR_BGR2RGB))
    plt.title('Shi-Tomasi Corner Detector')
    plt.axis('off')
    
    # SIFT
    try:
        plt.subplot(2, 3, 3)
        plt.imshow(cv2.cvtColor(img1_sift, cv2.COLOR_BGR2RGB))
        plt.title('SIFT Keypoints')
        plt.axis('off')
    except:
        pass
    
    # ORB
    plt.subplot(2, 3, 4)
    plt.imshow(cv2.cvtColor(img1_orb, cv2.COLOR_BGR2RGB))
    plt.title('ORB Keypoints')
    plt.axis('off')
    
    # FAST
    plt.subplot(2, 3, 5)
    plt.imshow(cv2.cvtColor(img1_fast, cv2.COLOR_BGR2RGB))
    plt.title('FAST Keypoints')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Clean up
    cv2.destroyAllWindows()
    print("\nAll operations completed successfully!")

def create_sample_images():
    """Create sample images for feature detection and matching demonstration"""
    # Create a base image with some shapes
    width, height = 600, 400
    img = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background
    
    # Add some shapes
    # Rectangle
    cv2.rectangle(img, (100, 100), (200, 200), (0, 0, 255), -1)
    
    # Circle
    cv2.circle(img, (350, 150), 50, (0, 255, 0), -1)
    
    # Triangle
    pts = np.array([[150, 300], [250, 250], [350, 300]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.fillPoly(img, [pts], (255, 0, 0))
    
    # Add some text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'OpenCV', (400, 300), font, 1, (0, 0, 0), 2)
    
    # Add some noise for more interesting features
    noise = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.randu(noise, 0, 30)
    img = cv2.add(img, noise)
    
    # Create a second image with slight rotation and scaling
    img2 = rotate_and_scale(img.copy(), angle=15, scale=0.9)
    
    return img, img2

def rotate_and_scale(img, angle=15, scale=0.9):
    """Rotate and scale an image"""
    height, width = img.shape[:2]
    center = (width // 2, height // 2)
    
    # Create rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    
    # Apply the rotation
    rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height))
    
    return rotated_img

if __name__ == "__main__":
    main()