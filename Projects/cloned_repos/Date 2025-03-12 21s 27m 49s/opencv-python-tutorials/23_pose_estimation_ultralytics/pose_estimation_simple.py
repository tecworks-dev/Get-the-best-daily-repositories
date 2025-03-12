#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple Pose Estimation with Ultralytics YOLOv8
--------------------------------------------
This script demonstrates real-time pose estimation using YOLOv8 and a webcam.
All parameters are set as variables at the top of the script for easy modification.
"""

import cv2
import time
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import math

# Configuration variables - modify these as needed
MODEL = "yolov8n-pose.pt"  # Options: "yolov8n-pose.pt", "yolov8s-pose.pt", "yolov8m-pose.pt", "yolov8l-pose.pt", "yolov8x-pose.pt"
DEVICE = 0  # Webcam index (usually 0 for built-in webcam)
CONFIDENCE_THRESHOLD = 0.25  # Minimum confidence for detections
SAVE_VIDEO = False  # Set to True to save the output video
SHOW_FPS = True  # Set to True to display FPS counter
SHOW_SKELETON = True  # Set to True to display skeleton connections
SHOW_ANGLES = True  # Set to True to calculate and display joint angles
KEYPOINT_CONF = 0.5  # Confidence threshold for keypoints
CUSTOM_VISUALIZATION = False  # Set to True to use custom visualization instead of built-in


def calculate_angle(a, b, c):
    """
    Calculate the angle between three points (in degrees).
    
    Args:
        a: First point [x, y]
        b: Mid point [x, y]
        c: End point [x, y]
    
    Returns:
        angle in degrees
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    # Calculate vectors
    ba = a - b
    bc = c - b
    
    # Calculate dot product
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    
    # Handle numerical errors
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    # Calculate angle in degrees
    angle = np.arccos(cosine_angle) * 180.0 / np.pi
    
    return angle


def custom_visualization(frame, results, show_skeleton=True, show_angles=False, keypoint_conf=0.5):
    """
    Custom visualization of pose estimation results.
    
    Args:
        frame: Original frame
        results: YOLOv8 results
        show_skeleton: Whether to draw skeleton connections
        show_angles: Whether to calculate and display joint angles
        keypoint_conf: Confidence threshold for keypoints
    
    Returns:
        Annotated frame with custom visualization
    """
    # Create a copy of the original frame
    annotated_frame = frame.copy()
    
    # Define the skeleton connections
    skeleton = [
        [16, 14], [14, 12], [17, 15], [15, 13], [12, 13], [6, 12], [7, 13],
        [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
        [2, 4], [3, 5], [4, 6], [5, 7]
    ]
    
    # Define joint angle configurations
    joint_angles = [
        # [point1_idx, mid_point_idx, point2_idx, label]
        [6, 8, 10, "R Elbow"],  # Right elbow
        [5, 7, 9, "L Elbow"],   # Left elbow
        [12, 14, 16, "R Knee"],  # Right knee
        [11, 13, 15, "L Knee"],  # Left knee
        [6, 12, 14, "R Hip"],   # Right hip
        [5, 11, 13, "L Hip"],   # Left hip
        [6, 7, 9, "Neck-LShoulder-LElbow"],  # Neck to left arm
        [5, 6, 8, "Neck-RShoulder-RElbow"]   # Neck to right arm
    ]
    
    # Define colors for keypoints and skeleton
    keypoint_color = (0, 255, 0)  # Green
    skeleton_colors = np.random.randint(0, 255, size=(len(skeleton), 3), dtype=np.uint8)
    
    # Process each person's keypoints
    if results[0].keypoints is not None:
        for person_idx, keypoints in enumerate(results[0].keypoints.data):
            # Draw keypoints
            for idx, kp in enumerate(keypoints):
                x, y, conf = int(kp[0]), int(kp[1]), kp[2]
                if conf > keypoint_conf:
                    cv2.circle(annotated_frame, (x, y), 5, keypoint_color, -1)
            
            # Draw skeleton
            if show_skeleton:
                for idx, (p1_idx, p2_idx) in enumerate(skeleton):
                    p1_idx -= 1  # Convert from 1-indexed to 0-indexed
                    p2_idx -= 1
                    
                    if (p1_idx < len(keypoints) and p2_idx < len(keypoints) and 
                        keypoints[p1_idx][2] > keypoint_conf and keypoints[p2_idx][2] > keypoint_conf):
                        
                        p1 = (int(keypoints[p1_idx][0]), int(keypoints[p1_idx][1]))
                        p2 = (int(keypoints[p2_idx][0]), int(keypoints[p2_idx][1]))
                        
                        cv2.line(annotated_frame, p1, p2, skeleton_colors[idx].tolist(), 2)
            
            # Calculate and display joint angles
            if show_angles:
                for angle_config in joint_angles:
                    p1_idx, mid_idx, p2_idx, label = angle_config
                    p1_idx -= 1  # Convert from 1-indexed to 0-indexed
                    mid_idx -= 1
                    p2_idx -= 1
                    
                    if (p1_idx < len(keypoints) and mid_idx < len(keypoints) and p2_idx < len(keypoints) and
                        keypoints[p1_idx][2] > keypoint_conf and 
                        keypoints[mid_idx][2] > keypoint_conf and 
                        keypoints[p2_idx][2] > keypoint_conf):
                        
                        p1 = (float(keypoints[p1_idx][0]), float(keypoints[p1_idx][1]))
                        mid = (float(keypoints[mid_idx][0]), float(keypoints[mid_idx][1]))
                        p2 = (float(keypoints[p2_idx][0]), float(keypoints[p2_idx][1]))
                        
                        # Calculate angle
                        angle = calculate_angle(p1, mid, p2)
                        
                        # Display angle
                        cv2.putText(
                            annotated_frame, 
                            f"{label}: {angle:.1f}°", 
                            (int(mid[0]), int(mid[1]) - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, 
                            (255, 255, 255), 
                            2
                        )
    
    return annotated_frame


def main():
    """Main function for real-time pose estimation."""
    # Load the model
    print(f"Loading model: {MODEL}...")
    model = YOLO(MODEL)
    
    # Open webcam
    print(f"Opening webcam at index: {DEVICE}...")
    cap = cv2.VideoCapture(DEVICE)
    
    if not cap.isOpened():
        print(f"Error: Could not open webcam at index {DEVICE}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create output video writer if saving is enabled
    output_writer = None
    if SAVE_VIDEO:
        output_path = f"output_{Path(MODEL).stem}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Saving output to: {output_path}")
    
    # Variables for FPS calculation
    frame_count = 0
    start_time = time.time()
    fps_display = 0
    
    print("Starting pose estimation. Press 'q' to quit.")
    
    # Main loop
    while cap.isOpened():
        # Read a frame
        success, frame = cap.read()
        
        if not success:
            print("Error: Failed to read frame")
            break
        
        # Update FPS calculation
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time >= 1.0:  # Update FPS every second
            fps_display = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
        
        # Run YOLOv8 inference
        results = model(
            frame, 
            conf=CONFIDENCE_THRESHOLD, 
            verbose=False
        )
        
        # Visualize the results on the frame
        if CUSTOM_VISUALIZATION:
            annotated_frame = custom_visualization(
                frame, 
                results, 
                show_skeleton=SHOW_SKELETON,
                show_angles=SHOW_ANGLES,
                keypoint_conf=KEYPOINT_CONF
            )
        else:
            annotated_frame = results[0].plot()
            
            # Add angle calculations if enabled and not using custom visualization
            if SHOW_ANGLES and results[0].keypoints is not None:
                # Define joint angle configurations
                joint_angles = [
                    # [point1_idx, mid_point_idx, point2_idx, label]
                    [6, 8, 10, "R Elbow"],  # Right elbow
                    [5, 7, 9, "L Elbow"],   # Left elbow
                    [12, 14, 16, "R Knee"],  # Right knee
                    [11, 13, 15, "L Knee"],  # Left knee
                ]
                
                for person_idx, keypoints in enumerate(results[0].keypoints.data):
                    for angle_config in joint_angles:
                        p1_idx, mid_idx, p2_idx, label = angle_config
                        p1_idx -= 1  # Convert from 1-indexed to 0-indexed
                        mid_idx -= 1
                        p2_idx -= 1
                        
                        if (p1_idx < len(keypoints) and mid_idx < len(keypoints) and p2_idx < len(keypoints) and
                            keypoints[p1_idx][2] > KEYPOINT_CONF and 
                            keypoints[mid_idx][2] > KEYPOINT_CONF and 
                            keypoints[p2_idx][2] > KEYPOINT_CONF):
                            
                            p1 = (float(keypoints[p1_idx][0]), float(keypoints[p1_idx][1]))
                            mid = (float(keypoints[mid_idx][0]), float(keypoints[mid_idx][1]))
                            p2 = (float(keypoints[p2_idx][0]), float(keypoints[p2_idx][1]))
                            
                            # Calculate angle
                            angle = calculate_angle(p1, mid, p2)
                            
                            # Display angle
                            cv2.putText(
                                annotated_frame, 
                                f"{label}: {angle:.1f}°", 
                                (int(mid[0]), int(mid[1]) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                0.5, 
                                (255, 255, 255), 
                                2
                            )
        
        # Add FPS counter if enabled
        if SHOW_FPS:
            cv2.putText(
                annotated_frame, 
                f"FPS: {fps_display:.1f}", 
                (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), 
                2
            )
        
        # Display the annotated frame
        cv2.imshow("YOLOv8 Pose Estimation", annotated_frame)
        
        # Save the frame if enabled
        if output_writer is not None:
            output_writer.write(annotated_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    if output_writer is not None:
        output_writer.release()
    cv2.destroyAllWindows()
    
    print("Pose estimation finished.")


if __name__ == "__main__":
    main()