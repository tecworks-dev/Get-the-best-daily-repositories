# Camera Calibration with OpenCV

This tutorial covers camera calibration techniques using OpenCV, including intrinsic and extrinsic parameter estimation, distortion correction, and stereo camera calibration.

## Table of Contents
1. [Understanding Camera Calibration](#understanding-camera-calibration)
2. [Single Camera Calibration](#single-camera-calibration)
3. [Stereo Camera Calibration](#stereo-camera-calibration)
4. [Distortion Correction](#distortion-correction)
5. [Real-world Applications](#real-world-applications)

## Understanding Camera Calibration

Camera calibration is the process of estimating the parameters of a camera's imaging system. These parameters include:

- **Intrinsic Parameters**: Focal length, optical center, and lens distortion coefficients
- **Extrinsic Parameters**: Rotation and translation that describe camera position in world coordinates

## Single Camera Calibration

### Calibration Process

```python
import cv2
import numpy as np
import glob

def calibrate_camera(images_folder, pattern_size=(9,6), square_size=25.0):
    """
    Calibrate camera using chessboard pattern
    pattern_size: Number of inner corners (width, height)
    square_size: Size of chessboard squares in millimeters
    """
    # Prepare object points
    objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1,2)
    objp *= square_size
    
    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    # Get list of calibration images
    images = glob.glob(f'{images_folder}/*.jpg')
    
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        
        if ret:
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            
            objpoints.append(objp)
            imgpoints.append(corners2)
            
            # Draw and display corners
            cv2.drawChessboardCorners(img, pattern_size, corners2, ret)
            cv2.imshow('Corners', img)
            cv2.waitKey(500)
    
    cv2.destroyAllWindows()
    
    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, 
                                                      gray.shape[::-1], None, None)
    
    return mtx, dist, rvecs, tvecs

def save_calibration_params(mtx, dist, filename='calibration.npz'):
    """
    Save calibration parameters to file
    """
    np.savez(filename, mtx=mtx, dist=dist)

def load_calibration_params(filename='calibration.npz'):
    """
    Load calibration parameters from file
    """
    data = np.load(filename)
    return data['mtx'], data['dist']
```

### Undistortion

```python
def undistort_image(image, camera_matrix, dist_coeffs):
    """
    Undistort image using calibration parameters
    """
    h, w = image.shape[:2]
    
    # Get optimal new camera matrix
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, 
                                                     (w,h), 1, (w,h))
    
    # Undistort
    dst = cv2.undistort(image, camera_matrix, dist_coeffs, None, newcameramtx)
    
    # Crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    
    return dst
```

## Stereo Camera Calibration

```python
def calibrate_stereo_cameras(left_images, right_images, pattern_size=(9,6)):
    """
    Calibrate stereo camera system
    """
    # Calibrate each camera individually
    ret1, mtx1, dist1, rvecs1, tvecs1 = calibrate_camera(left_images, pattern_size)
    ret2, mtx2, dist2, rvecs2, tvecs2 = calibrate_camera(right_images, pattern_size)
    
    # Prepare object and image points
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((pattern_size[0]*pattern_size[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1,2)
    
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
            
            corners_left = cv2.cornerSubPix(gray_left, corners_left, (11,11), 
                                          (-1,-1), criteria)
            corners_right = cv2.cornerSubPix(gray_right, corners_right, (11,11), 
                                           (-1,-1), criteria)
            
            imgpoints_left.append(corners_left)
            imgpoints_right.append(corners_right)
    
    # Calibrate stereo cameras
    flags = 0
    flags |= cv2.CALIB_FIX_INTRINSIC
    
    ret, mtx1, dist1, mtx2, dist2, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        mtx1, dist1, mtx2, dist2,
        gray_left.shape[::-1], None, None, None, None, flags)
    
    return mtx1, dist1, mtx2, dist2, R, T, E, F
```

### Stereo Rectification

```python
def stereo_rectify(mtx1, dist1, mtx2, dist2, R, T, image_size):
    """
    Compute rectification transforms for stereo cameras
    """
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        mtx1, dist1, mtx2, dist2, image_size, R, T)
    
    # Compute rectification maps
    mapx1, mapy1 = cv2.initUndistortRectifyMap(mtx1, dist1, R1, P1, 
                                              image_size, cv2.CV_32FC1)
    mapx2, mapy2 = cv2.initUndistortRectifyMap(mtx2, dist2, R2, P2, 
                                              image_size, cv2.CV_32FC1)
    
    return mapx1, mapy1, mapx2, mapy2, Q

def rectify_stereo_images(left_img, right_img, mapx1, mapy1, mapx2, mapy2):
    """
    Rectify stereo image pair
    """
    rect_left = cv2.remap(left_img, mapx1, mapy1, cv2.INTER_LINEAR)
    rect_right = cv2.remap(right_img, mapx2, mapy2, cv2.INTER_LINEAR)
    
    return rect_left, rect_right
```

## Distortion Correction

### Types of Distortion

1. **Radial Distortion**
   - Barrel distortion
   - Pincushion distortion
   - Mustache distortion

2. **Tangential Distortion**
   - Due to misalignment of camera lens

```python
def analyze_distortion(camera_matrix, dist_coeffs, image_size):
    """
    Analyze and visualize distortion patterns
    """
    # Generate grid of points
    x, y = np.meshgrid(np.linspace(0, image_size[0], 20),
                       np.linspace(0, image_size[1], 20))
    points = np.float32(np.vstack((x.flatten(), y.flatten())).T)
    
    # Project points
    undistorted = cv2.undistortPoints(points, camera_matrix, dist_coeffs, P=camera_matrix)
    
    return points, undistorted.reshape(-1, 2)
```

## Best Practices and Tips

1. **Calibration Pattern**
   - Use high-quality chessboard pattern
   - Ensure good lighting conditions
   - Cover different angles and distances

2. **Image Collection**
   - Take multiple images (20+ recommended)
   - Vary pattern orientation
   - Include corners and edges of field of view

3. **Calibration Process**
   - Check reprojection error
   - Validate results with test images
   - Regular recalibration for accuracy

4. **Error Handling**
   - Verify pattern detection
   - Handle failed detections gracefully
   - Validate calibration results

## Applications

1. **3D Reconstruction**
   - Structure from Motion
   - Photogrammetry
   - Depth estimation

2. **Augmented Reality**
   - Marker tracking
   - Object placement
   - Scene understanding

3. **Industrial Inspection**
   - Measurement
   - Quality control
   - Robot vision

4. **Scientific Imaging**
   - Microscopy
   - Medical imaging
   - Satellite imaging

## Further Reading

1. [OpenCV Camera Calibration Documentation](https://docs.opencv.org/master/d9/d0c/group__calib3d.html)
2. Research papers on camera calibration
3. Advanced topics in computer vision
4. Multi-camera system calibration