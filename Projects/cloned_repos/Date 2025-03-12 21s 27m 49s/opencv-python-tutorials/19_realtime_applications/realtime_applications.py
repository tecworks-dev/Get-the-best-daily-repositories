#!/usr/bin/env python3
"""
Real-time Applications with OpenCV
This script demonstrates real-time video processing and optimization techniques using OpenCV
"""

import cv2
import numpy as np
import time
import threading
from queue import Queue
import os
import sys
import argparse

class VideoCapture:
    def __init__(self, source=0):
        """
        Initialize video capture
        
        Args:
            source: Camera index or video file path
        """
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source {source}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    def read_frame(self):
        """
        Read a single frame
        
        Returns:
            Frame or None if no frame is available
        """
        ret, frame = self.cap.read()
        if ret:
            return frame
        return None
    
    def release(self):
        """
        Release video capture
        """
        self.cap.release()

class ThreadedVideoCapture:
    def __init__(self, source=0, queue_size=128):
        """
        Initialize threaded video capture for better performance
        
        Args:
            source: Camera index or video file path
            queue_size: Maximum size of the frame queue
        """
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source {source}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.queue = Queue(maxsize=queue_size)
        self.stopped = False
        
        # Start frame capture thread
        self.thread = threading.Thread(target=self._update, args=())
        self.thread.daemon = True
        self.thread.start()
    
    def _update(self):
        """
        Update frame buffer continuously
        """
        while True:
            if self.stopped:
                return
            
            if not self.queue.full():
                ret, frame = self.cap.read()
                if ret:
                    self.queue.put(frame)
                else:
                    self.stopped = True
                    break
            else:
                time.sleep(0.01)  # Sleep if queue is full
    
    def read_frame(self):
        """
        Read frame from queue
        
        Returns:
            Frame or None if no frame is available
        """
        return self.queue.get() if not self.queue.empty() else None
    
    def release(self):
        """
        Stop video capture
        """
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join()
        self.cap.release()

class FrameProcessor:
    def __init__(self, max_workers=4):
        """
        Initialize frame processor with worker pool
        
        Args:
            max_workers: Number of worker threads
        """
        self.max_workers = max_workers
        self.processing_queue = Queue(maxsize=32)
        self.result_queue = Queue(maxsize=32)
        self.workers = []
        self.stopped = False
        self.start_workers()
    
    def start_workers(self):
        """
        Start worker threads
        """
        for _ in range(self.max_workers):
            worker = threading.Thread(target=self._process_frames)
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
    
    def _process_frames(self):
        """
        Worker thread function
        """
        while not self.stopped:
            if not self.processing_queue.empty():
                frame = self.processing_queue.get()
                if frame is None:
                    continue
                
                # Process frame
                processed_frame = self.process_single_frame(frame)
                
                if not self.result_queue.full():
                    self.result_queue.put(processed_frame)
            else:
                time.sleep(0.01)  # Sleep if queue is empty
    
    def process_single_frame(self, frame):
        """
        Process a single frame (override in subclass)
        """
        # Default implementation: no processing
        return frame
    
    def stop(self):
        """
        Stop all workers
        """
        self.stopped = True
        for worker in self.workers:
            if worker.is_alive():
                worker.join()

class EdgeDetectionProcessor(FrameProcessor):
    def __init__(self, max_workers=4):
        """
        Initialize edge detection processor
        """
        super().__init__(max_workers)
    
    def process_single_frame(self, frame):
        """
        Apply edge detection to frame
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Convert back to BGR for display
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Combine original and edges
        result = cv2.addWeighted(frame, 0.7, edges_bgr, 0.3, 0)
        
        return result

class MotionDetectionProcessor(FrameProcessor):
    def __init__(self, max_workers=4):
        """
        Initialize motion detection processor
        """
        super().__init__(max_workers)
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2()
        self.prev_frame = None
    
    def process_single_frame(self, frame):
        """
        Apply motion detection to frame
        """
        # Apply background subtraction
        fg_mask = self.background_subtractor.apply(frame)
        
        # Apply morphological operations to remove noise
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours on original frame
        result = frame.copy()
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter small contours
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Add text with number of detected objects
        num_objects = len([c for c in contours if cv2.contourArea(c) > 500])
        cv2.putText(result, f"Objects: {num_objects}", (10, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return result

class FaceDetectionProcessor(FrameProcessor):
    def __init__(self, max_workers=4):
        """
        Initialize face detection processor
        """
        super().__init__(max_workers)
        
        # Load face cascade
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Load eye cascade
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    def process_single_frame(self, frame):
        """
        Apply face detection to frame
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        # Draw rectangles around faces and detect eyes
        result = frame.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Region of interest for eyes
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = result[y:y + h, x:x + w]
            
            # Detect eyes
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        
        # Add text with number of detected faces
        cv2.putText(result, f"Faces: {len(faces)}", (10, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return result

class ColorTrackingProcessor(FrameProcessor):
    def __init__(self, max_workers=4, color='red'):
        """
        Initialize color tracking processor
        
        Args:
            max_workers: Number of worker threads
            color: Color to track ('red', 'green', 'blue', 'yellow')
        """
        super().__init__(max_workers)
        
        # Define color ranges in HSV
        self.color_ranges = {
            'red': ([0, 100, 100], [10, 255, 255]),
            'green': ([40, 100, 100], [80, 255, 255]),
            'blue': ([100, 100, 100], [140, 255, 255]),
            'yellow': ([20, 100, 100], [40, 255, 255])
        }
        
        self.color = color
    
    def process_single_frame(self, frame):
        """
        Apply color tracking to frame
        """
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Get color range
        lower, upper = self.color_ranges.get(self.color, ([0, 0, 0], [255, 255, 255]))
        lower = np.array(lower)
        upper = np.array(upper)
        
        # Create mask
        mask = cv2.inRange(hsv, lower, upper)
        
        # Apply morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours and calculate center
        result = frame.copy()
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Filter small contours
                # Draw contour
                cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)
                
                # Calculate center
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Draw center
                    cv2.circle(result, (cx, cy), 5, (0, 0, 255), -1)
                    
                    # Draw info
                    cv2.putText(result, f"Area: {int(area)}", (cx - 50, cy - 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Add text with color being tracked
        cv2.putText(result, f"Tracking: {self.color}", (10, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return result
    
    def set_color(self, color):
        """
        Set color to track
        
        Args:
            color: Color to track ('red', 'green', 'blue', 'yellow')
        """
        if color in self.color_ranges:
            self.color = color

class VideoProcessingApp:
    def __init__(self, source=0, processor_type='edge', threaded=True):
        """
        Initialize video processing application
        
        Args:
            source: Camera index or video file path
            processor_type: Type of frame processor to use
            threaded: Whether to use threaded video capture
        """
        # Initialize video capture
        if threaded:
            self.capture = ThreadedVideoCapture(source)
        else:
            self.capture = VideoCapture(source)
        
        # Initialize frame processor
        if processor_type == 'edge':
            self.processor = EdgeDetectionProcessor()
        elif processor_type == 'motion':
            self.processor = MotionDetectionProcessor()
        elif processor_type == 'face':
            self.processor = FaceDetectionProcessor()
        elif processor_type == 'color':
            self.processor = ColorTrackingProcessor()
        else:
            self.processor = FrameProcessor()
        
        # Initialize FPS counter
        self.fps_counter = FPSCounter()
        
        # Initialize window
        cv2.namedWindow('Video Processing', cv2.WINDOW_NORMAL)
        
        # Running flag
        self.running = False
    
    def start(self):
        """
        Start video processing
        """
        self.running = True
        self.fps_counter.start()
        
        while self.running:
            # Read frame
            frame = self.capture.read_frame()
            if frame is None:
                break
            
            # Add frame to processing queue
            if not self.processor.processing_queue.full():
                self.processor.processing_queue.put(frame)
            
            # Get processed frame
            if not self.processor.result_queue.empty():
                processed_frame = self.processor.result_queue.get()
                
                # Update FPS counter
                self.fps_counter.update()
                fps = self.fps_counter.fps()
                
                # Add FPS text
                cv2.putText(processed_frame, f"FPS: {fps:.2f}", (10, 70),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Display frame
                cv2.imshow('Video Processing', processed_frame)
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.stop()
                break
            elif key == ord('c') and isinstance(self.processor, ColorTrackingProcessor):
                # Cycle through colors
                colors = list(self.processor.color_ranges.keys())
                current_idx = colors.index(self.processor.color)
                next_idx = (current_idx + 1) % len(colors)
                self.processor.set_color(colors[next_idx])
    
    def stop(self):
        """
        Stop video processing
        """
        self.running = False
        self.processor.stop()
        self.capture.release()
        cv2.destroyAllWindows()

class FPSCounter:
    def __init__(self, num_frames=10):
        """
        Initialize FPS counter
        
        Args:
            num_frames: Number of frames to average over
        """
        self.num_frames = num_frames
        self.times = []
        self.start_time = None
    
    def start(self):
        """
        Start FPS counter
        """
        self.start_time = time.time()
    
    def update(self):
        """
        Update FPS counter
        """
        self.times.append(time.time() - self.start_time)
        self.start_time = time.time()
        
        # Keep only the last num_frames times
        if len(self.times) > self.num_frames:
            self.times.pop(0)
    
    def fps(self):
        """
        Calculate FPS
        
        Returns:
            Frames per second
        """
        if not self.times:
            return 0
        
        return len(self.times) / sum(self.times)

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Real-time Video Processing with OpenCV')
    parser.add_argument('--source', type=int, default=0,
                      help='Camera index or video file path')
    parser.add_argument('--processor', type=str, default='edge',
                      choices=['edge', 'motion', 'face', 'color'],
                      help='Type of frame processor to use')
    parser.add_argument('--threaded', action='store_true',
                      help='Use threaded video capture')
    args = parser.parse_args()
    
    print("Real-time Applications with OpenCV")
    print("=================================")
    print(f"Source: {args.source}")
    print(f"Processor: {args.processor}")
    print(f"Threaded: {args.threaded}")
    print("\nPress 'q' to quit")
    if args.processor == 'color':
        print("Press 'c' to cycle through colors")
    
    try:
        # Initialize and start video processing app
        app = VideoProcessingApp(args.source, args.processor, args.threaded)
        app.start()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        print("\nExiting...")

if __name__ == "__main__":
    main()