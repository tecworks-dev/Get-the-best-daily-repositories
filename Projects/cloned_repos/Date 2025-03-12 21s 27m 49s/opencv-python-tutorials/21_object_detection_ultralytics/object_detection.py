#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Object Detection with Ultralytics YOLOv8
----------------------------------------
This script demonstrates real-time object detection using YOLOv8 and a webcam.
It includes options for displaying bounding boxes, confidence scores, and saving results.
"""

import argparse
import cv2
import time
import numpy as np
from ultralytics import YOLO
from pathlib import Path


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="YOLOv8 real-time object detection with webcam")
    parser.add_argument("--model", type=str, default="yolov8n.pt", 
                        help="Model to use (yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt)")
    parser.add_argument("--device", type=str, default="0", 
                        help="Device to use (webcam index or video path)")
    parser.add_argument("--conf", type=float, default=0.25, 
                        help="Confidence threshold for detections")
    parser.add_argument("--save", action="store_true", 
                        help="Save the output video")
    parser.add_argument("--classes", nargs="+", type=int, 
                        help="Filter by class (e.g., --classes 0 2 3 for person, car, motorcycle)")
    parser.add_argument("--show-fps", action="store_true", 
                        help="Display FPS counter")
    parser.add_argument("--show-labels", action="store_true", default=True,
                        help="Display class labels")
    parser.add_argument("--show-conf", action="store_true", default=True,
                        help="Display confidence scores")
    return parser.parse_args()


def main():
    """Main function for real-time object detection."""
    # Parse arguments
    args = parse_arguments()
    
    # Load the model
    print(f"Loading model: {args.model}...")
    model = YOLO(args.model)
    
    # Open webcam or video file
    try:
        device = int(args.device)  # Try to convert to integer for webcam index
    except ValueError:
        device = args.device  # Use as string path for video file
    
    print(f"Opening video source: {device}...")
    cap = cv2.VideoCapture(device)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {device}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create output video writer if saving is enabled
    output_writer = None
    if args.save:
        output_path = f"output_{Path(args.model).stem}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
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
            conf=args.conf, 
            classes=args.classes,
            verbose=False
        )
        
        # Visualize the results on the frame
        annotated_frame = results[0].plot(
            conf=args.show_conf,
            labels=args.show_labels,
            line_width=2
        )
        
        # Add FPS counter if enabled
        if args.show_fps:
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