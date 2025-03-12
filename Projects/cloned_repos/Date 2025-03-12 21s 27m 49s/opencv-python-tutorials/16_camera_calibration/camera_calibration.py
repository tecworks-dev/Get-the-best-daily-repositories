#!/usr/bin/env python3
"""
Camera Calibration with OpenCV
This script demonstrates camera calibration techniques using OpenCV
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import sys

class CameraCalibration:
    def __init__(self):
        """Initialize camera calibration class"""
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
    
    def calibrate_camera(self, images, pattern_size=(9, 6), square_size=1.0):
        """
        Calibrate camera using chessboard pattern
        
        Args:
            images: List of calibration images
            pattern_size: Number of inner corners (width, height)
            square_size: Size of chessboard squares in arbitrary units
        
        Returns:
            camera_matrix: Camera intrinsic matrix
            dist_coeffs: Distortion coefficients
            rvecs: Rotation vectors
            tvecs: Translation vectors
        """
        # Prepare object points
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        objp *= square_size
        
        # Arrays to store object points and image points
        objpoints = []  # 3D points in real world space
        imgpoints = []  # 2D points in image plane
        
        # Process each calibration image
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
            
            if ret:
                # Refine corner positions
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                
                objpoints.append(objp)
                imgpoints.append(corners2)
                
                # Draw and display corners
                img_corners = img.copy()
                cv2.drawChessboardCorners(img_corners, pattern_size, corners2, ret)
                
                # Display the image with corners
                plt.figure(figsize=(10, 8))
                plt.imshow(cv2.cvtColor(img_corners, cv2.COLOR_BGR2RGB))
                plt.title('Chessboard Corners')
                plt.axis('off')
                plt.show()
        
        # Calibrate camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None)
        
        # Store calibration results
        self.camera_matrix = mtx
        self.dist_coeffs = dist
        self.rvecs = rvecs
        self.tvecs = tvecs
        
        return mtx, dist, rvecs, tvecs
    
    def save_calibration(self, filename='calibration.npz'):
        """
        Save calibration parameters to file
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            raise ValueError("Camera not calibrated yet")
        
        np.savez(filename, 
                camera_matrix=self.camera_matrix, 
                dist_coeffs=self.dist_coeffs,
                rvecs=self.rvecs,
                tvecs=self.tvecs)
        
        print(f"Calibration saved to {filename}")
    
    def load_calibration(self, filename='calibration.npz'):
        """
        Load calibration parameters from file
        """
        data = np.load(filename)
        self.camera_matrix = data['camera_matrix']
        self.dist_coeffs = data['dist_coeffs']
        self.rvecs = data['rvecs']
        self.tvecs = data['tvecs']
        
        print(f"Calibration loaded from {filename}")
        
        return self.camera_matrix, self.dist_coeffs
    
    def undistort_image(self, image):
        """
        Undistort image using calibration parameters
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            raise ValueError("Camera not calibrated yet")
        
        h, w = image.shape[:2]
        
        # Get optimal new camera matrix
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h))
        
        # Undistort
        dst = cv2.undistort(image, self.camera_matrix, self.dist_coeffs, None, newcameramtx)
        
        # Crop the image
        x, y, w, h = roi
        if x >= 0 and y >= 0 and w > 0 and h > 0:
            dst = dst[y:y+h, x:x+w]
        
        return dst
    
    def calculate_reprojection_error(self, objpoints, imgpoints):
        """
        Calculate reprojection error
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            raise ValueError("Camera not calibrated yet")
        
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(
                objpoints[i], self.rvecs[i], self.tvecs[i], 
                self.camera_matrix, self.dist_coeffs)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            mean_error += error
        
        return mean_error / len(objpoints)
    
    def visualize_distortion(self, image_size=(640, 480)):
        """
        Visualize distortion pattern
        """
        if self.camera_matrix is None or self.dist_coeffs is None:
            raise ValueError("Camera not calibrated yet")
        
        # Create grid of points
        x, y = np.meshgrid(np.linspace(0, image_size[0], 20),
                         np.linspace(0, image_size[1], 20))
        points = np.float32(np.vstack((x.flatten(), y.flatten())).T)
        
        # Project points
        undistorted = cv2.undistortPoints(
            points.reshape(-1, 1, 2), self.camera_matrix, self.dist_coeffs, P=self.camera_matrix)
        undistorted = undistorted.reshape(-1, 2)
        
        # Create visualization
        plt.figure(figsize=(12, 10))
        plt.scatter(points[:, 0], points[:, 1], c='blue', s=20, label='Distorted')
        plt.scatter(undistorted[:, 0], undistorted[:, 1], c='red', s=20, label='Undistorted')
        
        # Draw lines connecting corresponding points
        for i in range(len(points)):
            plt.plot([points[i, 0], undistorted[i, 0]], 
                   [points[i, 1], undistorted[i, 1]], 'g-', alpha=0.5)
        
        plt.title('Distortion Pattern')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.show()

class StereoCalibration:
    def __init__(self):
        """Initialize stereo calibration class"""
        self.left_camera_matrix = None
        self.left_dist_coeffs = None
        self.right_camera_matrix = None
        self.right_dist_coeffs = None
        self.R = None  # Rotation matrix
        self.T = None  # Translation vector
        self.E = None  # Essential matrix
        self.F = None  # Fundamental matrix
        self.R1 = None  # Rectification transform for left camera
        self.R2 = None  # Rectification transform for right camera
        self.P1 = None  # Projection matrix for left camera
        self.P2 = None  # Projection matrix for right camera
        self.Q = None   # Disparity-to-depth mapping matrix
    
    def calibrate_stereo(self, left_images, right_images, pattern_size=(9, 6), square_size=1.0):
        """
        Calibrate stereo camera system
        
        Args:
            left_images: List of left camera images
            right_images: List of right camera images
            pattern_size: Number of inner corners (width, height)
            square_size: Size of chessboard squares in arbitrary units
        
        Returns:
            Stereo calibration parameters
        """
        # Calibrate each camera individually
        left_calibration = CameraCalibration()
        right_calibration = CameraCalibration()
        
        left_mtx, left_dist, left_rvecs, left_tvecs = left_calibration.calibrate_camera(
            left_images, pattern_size, square_size)
        
        right_mtx, right_dist, right_rvecs, right_tvecs = right_calibration.calibrate_camera(
            right_images, pattern_size, square_size)
        
        # Prepare object and image points
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        objp *= square_size
        
        objpoints = []
        imgpoints_left = []
        imgpoints_right = []
        
        # Find chessboard corners in stereo pairs
        for left_img, right_img in zip(left_images, right_images):
            gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
            
            ret_left, corners_left = cv2.findChessboardCorners(gray_left, pattern_size, None)
            ret_right, corners_right = cv2.findChessboardCorners(gray_right, pattern_size, None)
            
            if ret_left and ret_right:
                objpoints.append(objp)
                
                corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), 
                                             (-1, -1), criteria)
                corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), 
                                              (-1, -1), criteria)
                
                imgpoints_left.append(corners_left)
                imgpoints_right.append(corners_right)
        
        # Calibrate stereo cameras
        flags = 0
        flags |= cv2.CALIB_FIX_INTRINSIC
        
        ret, self.left_camera_matrix, self.left_dist_coeffs, \
            self.right_camera_matrix, self.right_dist_coeffs, \
            self.R, self.T, self.E, self.F = cv2.stereoCalibrate(
                objpoints, imgpoints_left, imgpoints_right,
                left_mtx, left_dist, right_mtx, right_dist,
                gray_left.shape[::-1], None, None, None, None, flags)
        
        return ret
    
    def compute_rectification(self, image_size):
        """
        Compute rectification transforms
        
        Args:
            image_size: Size of the images (width, height)
        
        Returns:
            Rectification parameters
        """
        if self.R is None or self.T is None:
            raise ValueError("Stereo system not calibrated yet")
        
        self.R1, self.R2, self.P1, self.P2, self.Q, roi1, roi2 = cv2.stereoRectify(
            self.left_camera_matrix, self.left_dist_coeffs,
            self.right_camera_matrix, self.right_dist_coeffs,
            image_size, self.R, self.T)
        
        # Compute rectification maps
        self.left_map1, self.left_map2 = cv2.initUndistortRectifyMap(
            self.left_camera_matrix, self.left_dist_coeffs, self.R1, self.P1,
            image_size, cv2.CV_32FC1)
        
        self.right_map1, self.right_map2 = cv2.initUndistortRectifyMap(
            self.right_camera_matrix, self.right_dist_coeffs, self.R2, self.P2,
            image_size, cv2.CV_32FC1)
        
        return roi1, roi2
    
    def rectify_stereo_images(self, left_img, right_img):
        """
        Rectify stereo image pair
        
        Args:
            left_img: Left camera image
            right_img: Right camera image
        
        Returns:
            Rectified left and right images
        """
        if self.left_map1 is None or self.right_map1 is None:
            raise ValueError("Rectification maps not computed yet")
        
        left_rectified = cv2.remap(left_img, self.left_map1, self.left_map2, cv2.INTER_LINEAR)
        right_rectified = cv2.remap(right_img, self.right_map1, self.right_map2, cv2.INTER_LINEAR)
        
        return left_rectified, right_rectified
    
    def save_calibration(self, filename='stereo_calibration.npz'):
        """
        Save stereo calibration parameters to file
        """
        if self.R is None or self.T is None:
            raise ValueError("Stereo system not calibrated yet")
        
        np.savez(filename, 
                left_camera_matrix=self.left_camera_matrix,
                left_dist_coeffs=self.left_dist_coeffs,
                right_camera_matrix=self.right_camera_matrix,
                right_dist_coeffs=self.right_dist_coeffs,
                R=self.R,
                T=self.T,
                E=self.E,
                F=self.F,
                R1=self.R1,
                R2=self.R2,
                P1=self.P1,
                P2=self.P2,
                Q=self.Q)
        
        print(f"Stereo calibration saved to {filename}")
    
    def load_calibration(self, filename='stereo_calibration.npz'):
        """
        Load stereo calibration parameters from file
        """
        data = np.load(filename)
        self.left_camera_matrix = data['left_camera_matrix']
        self.left_dist_coeffs = data['left_dist_coeffs']
        self.right_camera_matrix = data['right_camera_matrix']
        self.right_dist_coeffs = data['right_dist_coeffs']
        self.R = data['R']
        self.T = data['T']
        self.E = data['E']
        self.F = data['F']
        self.R1 = data['R1']
        self.R2 = data['R2']
        self.P1 = data['P1']
        self.P2 = data['P2']
        self.Q = data['Q']
        
        print(f"Stereo calibration loaded from {filename}")

def create_chessboard_image(pattern_size=(9, 6), square_size=50):
    """
    Create a chessboard image for calibration
    
    Args:
        pattern_size: Number of inner corners (width, height)
        square_size: Size of squares in pixels
    
    Returns:
        Chessboard image
    """
    width = (pattern_size[0] + 1) * square_size
    height = (pattern_size[1] + 1) * square_size
    
    # Create white image
    chessboard = np.ones((height, width), dtype=np.uint8) * 255
    
    # Draw black squares
    for i in range(pattern_size[1] + 1):
        for j in range(pattern_size[0] + 1):
            if (i + j) % 2 == 0:
                y1 = i * square_size
                y2 = (i + 1) * square_size
                x1 = j * square_size
                x2 = (j + 1) * square_size
                chessboard[y1:y2, x1:x2] = 0
    
    # Convert to BGR
    chessboard = cv2.cvtColor(chessboard, cv2.COLOR_GRAY2BGR)
    
    return chessboard

def simulate_distortion(image, camera_matrix, dist_coeffs):
    """
    Simulate lens distortion on an image
    
    Args:
        image: Input image
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
    
    Returns:
        Distorted image
    """
    h, w = image.shape[:2]
    
    # Create maps for distortion
    mapx, mapy = cv2.initUndistortRectifyMap(
        camera_matrix, -dist_coeffs, None, camera_matrix, (w, h), cv2.CV_32FC1)
    
    # Apply distortion
    distorted = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
    
    return distorted

def main():
    """Main function"""
    print("Camera Calibration with OpenCV")
    print("=============================")
    
    # Create or load sample images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), 'images')
    
    # Create sample chessboard image
    chessboard = create_chessboard_image()
    
    # Save sample image if needed
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    cv2.imwrite(os.path.join(images_dir, 'chessboard.jpg'), chessboard)
    
    # Initialize calibration class
    calibration = CameraCalibration()
    
    # Define sample camera matrix and distortion coefficients
    sample_camera_matrix = np.array([
        [800, 0, 320],
        [0, 800, 240],
        [0, 0, 1]
    ])
    
    # Different distortion types
    sample_dist_coeffs = {
        'no_distortion': np.zeros(5),
        'barrel': np.array([0.2, -0.1, 0, 0, 0]),
        'pincushion': np.array([-0.2, 0.1, 0, 0, 0]),
        'complex': np.array([0.1, -0.05, 0.01, 0.01, 0])
    }
    
    while True:
        print("\nSelect a demo:")
        print("1. Simulate Lens Distortion")
        print("2. Visualize Distortion Pattern")
        print("3. Undistort Image")
        print("q. Quit")
        
        choice = input("\nEnter your choice: ")
        
        if choice == '1':
            print("\nSimulate Lens Distortion Demo")
            
            # Simulate different types of distortion
            results = {'original': chessboard}
            
            for name, dist in sample_dist_coeffs.items():
                if name != 'no_distortion':
                    distorted = simulate_distortion(chessboard, sample_camera_matrix, dist)
                    results[name] = distorted
            
            # Display results
            plt.figure(figsize=(15, 10))
            plt.suptitle("Simulated Lens Distortion", fontsize=16)
            
            for i, (name, img) in enumerate(results.items()):
                plt.subplot(2, 2, i+1)
                plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                plt.title(name)
                plt.axis('off')
            
            plt.tight_layout()
            plt.show()
            
        elif choice == '2':
            print("\nVisualize Distortion Pattern Demo")
            
            # Set sample calibration parameters
            calibration.camera_matrix = sample_camera_matrix
            
            # Visualize different distortion patterns
            for name, dist in sample_dist_coeffs.items():
                print(f"\nDistortion type: {name}")
                calibration.dist_coeffs = dist
                calibration.visualize_distortion()
            
        elif choice == '3':
            print("\nUndistort Image Demo")
            
            # Set sample calibration parameters
            calibration.camera_matrix = sample_camera_matrix
            
            # Apply and remove distortion
            for name, dist in sample_dist_coeffs.items():
                if name != 'no_distortion':
                    print(f"\nDistortion type: {name}")
                    
                    # Apply distortion
                    distorted = simulate_distortion(chessboard, sample_camera_matrix, dist)
                    
                    # Set distortion coefficients
                    calibration.dist_coeffs = dist
                    
                    # Undistort
                    undistorted = calibration.undistort_image(distorted)
                    
                    # Display results
                    plt.figure(figsize=(15, 5))
                    plt.suptitle(f"Undistort Image - {name}", fontsize=16)
                    
                    plt.subplot(1, 3, 1)
                    plt.imshow(cv2.cvtColor(chessboard, cv2.COLOR_BGR2RGB))
                    plt.title("Original")
                    plt.axis('off')
                    
                    plt.subplot(1, 3, 2)
                    plt.imshow(cv2.cvtColor(distorted, cv2.COLOR_BGR2RGB))
                    plt.title("Distorted")
                    plt.axis('off')
                    
                    plt.subplot(1, 3, 3)
                    plt.imshow(cv2.cvtColor(undistorted, cv2.COLOR_BGR2RGB))
                    plt.title("Undistorted")
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