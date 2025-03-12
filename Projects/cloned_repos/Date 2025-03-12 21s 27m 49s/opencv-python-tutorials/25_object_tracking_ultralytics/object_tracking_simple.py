#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Simple Object Tracking with Ultralytics YOLOv8
--------------------------------------------
This script demonstrates real-time object tracking using YOLOv8 and a webcam.
All parameters are set as variables at the top of the script for easy modification.
"""

import cv2
import time
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from collections import defaultdict

# Configuration variables - modify these as needed
MODEL = "yolov8n.pt"  # Options: "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"
DEVICE = 0  # Webcam index (usually 0 for built-in webcam)
CONFIDENCE_THRESHOLD = 0.25  # Minimum confidence for detections
IOU_THRESHOLD = 0.5  # IoU threshold for NMS
SAVE_VIDEO = False  # Set to True to save the output video
CLASSES = None  # Set to a list of class indices to filter, e.g., [0, 2, 3] for person, car, motorcycle
SHOW_FPS = True  # Set to True to display FPS counter
SHOW_LABELS = True  # Set to True to display class labels
SHOW_TRAJECTORIES = True  # Set to True to display object trajectories
TRAJECTORY_LENGTH = 30  # Maximum length of trajectory history
TRACKER = "bytetrack"  # Options: "bytetrack", "botsort", "deepocsort", "ocsort", "strongsort"
PREDICT_TRAJECTORIES = True  # Set to True to predict and display future trajectories


def predict_trajectory(points, num_future_points=10):
    """
    Predict future trajectory based on past points.
    
    Args:
        points: List of past trajectory points [(x1, y1), (x2, y2), ...]
        num_future_points: Number of future points to predict
    
    Returns:
        List of predicted future points [(x1, y1), (x2, y2), ...]
    """
    if len(points) < 5:  # Need at least 5 points for a reasonable prediction
        return []
    
    # Extract x and y coordinates
    x = [p[0] for p in points]
    y = [p[1] for p in points]
    
    # Use linear regression for prediction
    # Calculate the slope and intercept
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xx = sum(xi*xi for xi in x)
    sum_xy = sum(xi*yi for xi, yi in zip(x, y))
    
    # Calculate slope (m) and intercept (b) for y = mx + b
    try:
        m = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
        b = (sum_y - m * sum_x) / n
        
        # Predict future points
        last_x = x[-1]
        step = (x[-1] - x[0]) / (len(x) - 1) if len(x) > 1 else 1
        
        future_points = []
        for i in range(1, num_future_points + 1):
            future_x = last_x + i * step
            future_y = m * future_x + b
            future_points.append((future_x, future_y))
        
        return future_points
    except ZeroDivisionError:
        # If division by zero, return empty list
        return []


def main():
    """Main function for real-time object tracking."""
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
        output_path = f"output_{Path(MODEL).stem}_{TRACKER}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Saving output to: {output_path}")
    
    # Variables for FPS calculation
    frame_count = 0
    start_time = time.time()
    fps_display = 0
    
    # Store the track history
    track_history = defaultdict(lambda: [])
    
    # Store colors for each track ID
    track_colors = {}
    
    print(f"Starting tracking with {TRACKER}. Press 'q' to quit.")
    
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
        
        # Run YOLOv8 tracking on the frame
        results = model.track(
            frame, 
            conf=CONFIDENCE_THRESHOLD, 
            iou=IOU_THRESHOLD,
            classes=CLASSES,
            tracker=TRACKER,
            persist=True,
            verbose=False
        )
        
        # Create a copy of the frame for annotation
        annotated_frame = frame.copy()
        
        # Process tracking results
        if results[0].boxes is not None and results[0].boxes.id is not None:
            # Get the boxes and track IDs
            boxes = results[0].boxes.xywh.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()
            
            # Process each tracked object
            for box, track_id, class_id, conf in zip(boxes, track_ids, class_ids, confs):
                x, y, w, h = box
                
                # Generate a color for this track ID if not already assigned
                if track_id not in track_colors:
                    track_colors[track_id] = tuple(np.random.randint(0, 255, 3).tolist())
                
                color = track_colors[track_id]
                
                # Draw bounding box
                x1, y1 = int(x - w/2), int(y - h/2)
                x2, y2 = int(x + w/2), int(y + h/2)
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label if enabled
                if SHOW_LABELS:
                    class_name = results[0].names[class_id]
                    label = f"{class_name} #{track_id} {conf:.2f}"
                    
                    # Calculate text size
                    text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    text_w, text_h = text_size
                    
                    # Draw label background
                    cv2.rectangle(
                        annotated_frame, 
                        (x1, y1 - text_h - 8), 
                        (x1 + text_w + 8, y1), 
                        color, 
                        -1
                    )
                    
                    # Draw label text
                    cv2.putText(
                        annotated_frame, 
                        label, 
                        (x1 + 4, y1 - 4), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        (255, 255, 255), 
                        2
                    )
                
                # Update track history
                center_point = (float(x), float(y))
                track_history[track_id].append(center_point)
                
                # Keep only the last N points
                track_history[track_id] = track_history[track_id][-TRAJECTORY_LENGTH:]
                
                # Draw trajectory if enabled
                if SHOW_TRAJECTORIES and len(track_history[track_id]) > 1:
                    # Convert track history to numpy array
                    points = np.array(track_history[track_id], dtype=np.int32).reshape((-1, 1, 2))
                    
                    # Draw trajectory line
                    cv2.polylines(annotated_frame, [points], False, color, 2)
                
                # Predict and draw future trajectory if enabled
                if PREDICT_TRAJECTORIES and len(track_history[track_id]) >= 5:
                    future_points = predict_trajectory(track_history[track_id])
                    
                    if future_points:
                        # Convert future points to numpy array
                        future_points_array = np.array(future_points, dtype=np.int32).reshape((-1, 1, 2))
                        
                        # Draw predicted trajectory with dashed line
                        for i in range(len(future_points_array) - 1):
                            pt1 = tuple(future_points_array[i][0])
                            pt2 = tuple(future_points_array[i+1][0])
                            cv2.line(annotated_frame, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)
        
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
        cv2.imshow(f"YOLOv8 Tracking ({TRACKER})", annotated_frame)
        
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
    
    print("Tracking finished.")


if __name__ == "__main__":
    main()