#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Classification with Ultralytics YOLOv8
--------------------------------------
This script demonstrates real-time image classification using YOLOv8 and a webcam.
It includes options for displaying top predictions, confidence scores, and saving results.
"""

import argparse
import cv2
import time
import numpy as np
from ultralytics import YOLO
from pathlib import Path


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="YOLOv8 real-time classification with webcam")
    parser.add_argument("--model", type=str, default="yolov8n-cls.pt", 
                        help="Model to use (yolov8n-cls.pt, yolov8s-cls.pt, yolov8m-cls.pt, yolov8l-cls.pt, yolov8x-cls.pt)")
    parser.add_argument("--device", type=str, default="0", 
                        help="Device to use (webcam index or video path)")
    parser.add_argument("--conf", type=float, default=0.25, 
                        help="Confidence threshold for predictions")
    parser.add_argument("--save", action="store_true", 
                        help="Save the output video")
    parser.add_argument("--show-fps", action="store_true", 
                        help="Display FPS counter")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Number of top predictions to display")
    parser.add_argument("--custom-visualization", action="store_true", default=True,
                        help="Use custom visualization instead of built-in")
    return parser.parse_args()


def custom_visualization(frame, results, top_k=3):
    """
    Custom visualization of classification results.
    
    Args:
        frame: Original frame
        results: YOLOv8 results
        top_k: Number of top predictions to display
    
    Returns:
        Annotated frame with custom visualization
    """
    # Create a copy of the original frame
    annotated_frame = frame.copy()
    
    # Get the probs from the first result
    probs = results[0].probs
    
    if probs is not None:
        # Get the top k class indices and their probabilities
        top_indices = probs.top5[:top_k]
        top_probs = probs.top5conf[:top_k]
        
        # Get class names
        class_names = results[0].names
        
        # Create a semi-transparent overlay for the prediction area
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (0, 0), (300, 30 + 30 * top_k), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, annotated_frame, 0.5, 0, annotated_frame)
        
        # Add title
        cv2.putText(
            annotated_frame, 
            "Top Predictions:", 
            (10, 25), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            (255, 255, 255), 
            2
        )
        
        # Add each prediction
        for i in range(len(top_indices)):
            class_idx = top_indices[i]
            prob = top_probs[i].item()
            class_name = class_names[class_idx]
            
            # Determine color based on confidence (green for high, yellow for medium, red for low)
            if prob > 0.7:
                color = (0, 255, 0)  # Green
            elif prob > 0.4:
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 0, 255)  # Red
            
            # Add text
            text = f"{i+1}. {class_name}: {prob:.2f}"
            cv2.putText(
                annotated_frame, 
                text, 
                (10, 55 + i * 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                color, 
                2
            )
            
            # Add confidence bar
            bar_length = int(200 * prob)
            cv2.rectangle(
                annotated_frame, 
                (220, 45 + i * 30), 
                (220 + bar_length, 55 + i * 30), 
                color, 
                -1
            )
    
    return annotated_frame


def main():
    """Main function for real-time classification."""
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
    
    # Variables for prediction smoothing
    smoothed_probs = None
    alpha = 0.7  # Smoothing factor (higher = more smoothing)
    
    print("Starting classification. Press 'q' to quit.")
    
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
            verbose=False
        )
        
        # Apply temporal smoothing to predictions
        if results[0].probs is not None:
            current_probs = results[0].probs.data.cpu().numpy()
            
            if smoothed_probs is None:
                smoothed_probs = current_probs
            else:
                smoothed_probs = alpha * smoothed_probs + (1 - alpha) * current_probs
                
                # Update the results with smoothed probabilities
                results[0].probs.data = torch.from_numpy(smoothed_probs).to(results[0].probs.data.device)
        
        # Visualize the results on the frame
        if args.custom_visualization:
            annotated_frame = custom_visualization(frame, results, top_k=args.top_k)
        else:
            # For classification, we need to create our own visualization
            annotated_frame = frame.copy()
            
            # Get the probs from the first result
            probs = results[0].probs
            
            if probs is not None:
                # Get the top k class indices and their probabilities
                top_indices = probs.top5[:args.top_k]
                top_probs = probs.top5conf[:args.top_k]
                
                # Get class names
                class_names = results[0].names
                
                # Add each prediction
                y_offset = 30
                for i in range(len(top_indices)):
                    class_idx = top_indices[i]
                    prob = top_probs[i].item()
                    class_name = class_names[class_idx]
                    
                    text = f"{class_name}: {prob:.2f}"
                    cv2.putText(
                        annotated_frame, 
                        text, 
                        (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, 
                        (0, 255, 0), 
                        2
                    )
                    y_offset += 30
        
        # Add FPS counter if enabled
        if args.show_fps:
            cv2.putText(
                annotated_frame, 
                f"FPS: {fps_display:.1f}", 
                (width - 150, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, 
                (0, 255, 0), 
                2
            )
        
        # Display the annotated frame
        cv2.imshow("YOLOv8 Classification", annotated_frame)
        
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
    
    print("Classification finished.")


if __name__ == "__main__":
    try:
        import torch  # Import torch for tensor operations
        main()
    except ImportError:
        print("Error: PyTorch is required for this script.")
        print("Please install it with: pip install torch")
    except Exception as e:
        print(f"Error: {e}")