#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple Object Detection with Ultralytics YOLOv8
----------------------------------------------
This script demonstrates real-time object detection using YOLOv8 and a webcam.
All parameters are set as variables at the top of the script for easy modification.
"""

import cv2
import time
import numpy as np
from ultralytics import YOLO
from pathlib import Path

# Configuration variables - modify these as needed
MODEL = "yolov8n.pt"  # Options: "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"
DEVICE = 0  # Webcam index (usually 0 for built-in webcam)
CONFIDENCE_THRESHOLD = 0.25  # Minimum confidence for detections
SAVE_VIDEO = False  # Set to True to save the output video
CLASSES = None  # Set to a list of class indices to filter, e.g., [0, 2, 3] for person, car, motorcycle
SHOW_FPS = True  # Set to True to display FPS counter
SHOW_LABELS = True  # Set to True to display class labels
SHOW_CONFIDENCE = True  # Set to True to display confidence scores


def main():
    """Main function for real-time object detection."""
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
    
    print("Starting detection. Press 'q' to quit.")
    
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
            classes=CLASSES,
            verbose=False
        )
        
        # Visualize the results on the frame
        annotated_frame = results[0].plot(
            conf=SHOW_CONFIDENCE,
            labels=SHOW_LABELS,
            line_width=2
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
        cv2.imshow("YOLOv8 Object Detection", annotated_frame)
        
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
    
    print("Detection finished.")


if __name__ == "__main__":
    main()