#!/usr/bin/env python3
"""
Video Basics Tutorial Script
This script demonstrates various video processing operations using OpenCV
"""

import cv2
import numpy as np
import os
import time
from datetime import datetime

class VideoPlayer:
    """Basic video player with controls"""
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.paused = False
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.current_frame = 0
    
    def create_controls(self):
        """Create window with controls"""
        cv2.namedWindow('Video Player')
        cv2.createTrackbar('Position', 'Video Player', 0, self.frame_count - 1, self.on_trackbar)
    
    def on_trackbar(self, pos):
        """Handle trackbar position change"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        self.current_frame = pos
    
    def play(self):
        """Play the video with controls"""
        self.create_controls()
        
        while self.cap.isOpened():
            if not self.paused:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Update trackbar position
                self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                cv2.setTrackbarPos('Position', 'Video Player', self.current_frame)
                
                # Display frame number and timestamp
                timestamp = self.current_frame / self.fps
                cv2.putText(frame, f'Frame: {self.current_frame}/{self.frame_count}', 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f'Time: {timestamp:.2f}s', 
                          (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow('Video Player', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(int(1000/self.fps)) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                self.paused = not self.paused
            elif key == ord('f'):  # Forward 10 frames
                new_pos = min(self.current_frame + 10, self.frame_count - 1)
                self.on_trackbar(new_pos)
            elif key == ord('b'):  # Backward 10 frames
                new_pos = max(self.current_frame - 10, 0)
                self.on_trackbar(new_pos)
        
        self.cap.release()
        cv2.destroyAllWindows()

class VideoRecorder:
    """Video recorder with effects"""
    def __init__(self, output_path, resolution=(640, 480), fps=30.0):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter(output_path, fourcc, fps, resolution)
        
        # Available effects
        self.effects = {
            'none': lambda x: x,
            'gray': lambda x: cv2.cvtColor(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR),
            'blur': lambda x: cv2.GaussianBlur(x, (15, 15), 0),
            'edge': lambda x: cv2.Canny(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), 100, 200),
            'sepia': self.sepia_effect
        }
        self.current_effect = 'none'
    
    def sepia_effect(self, frame):
        """Apply sepia effect to frame"""
        sepia_matrix = np.array([[0.393, 0.769, 0.189],
                               [0.349, 0.686, 0.168],
                               [0.272, 0.534, 0.131]])
        sepia = cv2.transform(frame, sepia_matrix)
        sepia[sepia > 255] = 255
        return np.array(sepia, dtype=np.uint8)
    
    def record(self):
        """Record video with effects"""
        recording = False
        start_time = None
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Apply current effect
            processed = self.effects[self.current_effect](frame.copy())
            
            # Add recording indicator and timer
            if recording:
                cv2.circle(processed, (30, 30), 10, (0, 0, 255), -1)
                elapsed = time.time() - start_time
                cv2.putText(processed, f'REC {elapsed:.1f}s', 
                          (50, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                self.out.write(processed)
            
            # Add effect name
            cv2.putText(processed, f'Effect: {self.current_effect}', 
                       (10, processed.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2)
            
            cv2.imshow('Recording', processed)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('e'):
                # Cycle through effects
                effects = list(self.effects.keys())
                current_idx = effects.index(self.current_effect)
                self.current_effect = effects[(current_idx + 1) % len(effects)]
            elif key == ord('r'):
                # Toggle recording
                recording = not recording
                if recording:
                    start_time = time.time()
                    print("Recording started...")
                else:
                    print("Recording stopped.")
        
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()

class VideoAnalyzer:
    """Real-time video analysis with multiple features"""
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.backSub = cv2.createBackgroundSubtractorMOG2()
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Analysis modes
        self.modes = {
            'normal': self.normal_mode,
            'motion': self.motion_mode,
            'background': self.background_mode,
            'face': self.face_detection_mode
        }
        self.current_mode = 'normal'
        
        # Motion detection parameters
        self.motion_threshold = 1000
        self.prev_frame = None
    
    def normal_mode(self, frame):
        """Display original frame with basic info"""
        result = frame.copy()
        cv2.putText(result, datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return result
    
    def motion_mode(self, frame):
        """Detect and highlight motion"""
        if self.prev_frame is None:
            self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return frame
        
        # Convert current frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate difference and threshold
        frame_diff = cv2.absdiff(self.prev_frame, gray)
        _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw rectangles around motion areas
        result = frame.copy()
        for contour in contours:
            if cv2.contourArea(contour) > self.motion_threshold:
                (x, y, w, h) = cv2.boundingRect(contour)
                cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Update previous frame
        self.prev_frame = gray
        
        return result
    
    def background_mode(self, frame):
        """Perform background subtraction"""
        fg_mask = self.backSub.apply(frame)
        result = cv2.bitwise_and(frame, frame, mask=fg_mask)
        return result
    
    def face_detection_mode(self, frame):
        """Detect and highlight faces"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        result = frame.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 2)
            # Add some measurements
            cv2.putText(result, f'Size: {w}x{h}px', 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        return result
    
    def analyze(self):
        """Main analysis loop"""
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Apply current analysis mode
            result = self.modes[self.current_mode](frame)
            
            # Add mode indicator
            cv2.putText(result, f'Mode: {self.current_mode}', 
                       (10, result.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2)
            
            cv2.imshow('Video Analysis', result)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('m'):
                # Cycle through modes
                modes = list(self.modes.keys())
                current_idx = modes.index(self.current_mode)
                self.current_mode = modes[(current_idx + 1) % len(modes)]
            elif key == ord('+'):
                self.motion_threshold += 500
                print(f"Motion threshold: {self.motion_threshold}")
            elif key == ord('-'):
                self.motion_threshold = max(500, self.motion_threshold - 500)
                print(f"Motion threshold: {self.motion_threshold}")
        
        self.cap.release()
        cv2.destroyAllWindows()

def create_sample_video():
    """Create a sample video file for testing"""
    output_path = 'sample_video.avi'
    fps = 30.0
    duration = 5  # seconds
    size = (640, 480)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, size)
    
    # Generate frames
    for i in range(int(fps * duration)):
        # Create a frame with some moving shapes
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        
        # Draw moving circle
        center = (int(size[0]/2 + 100*np.sin(i/fps)), int(size[1]/2))
        cv2.circle(frame, center, 50, (0, 255, 0), -1)
        
        # Draw moving rectangle
        x = int(size[0]/2 + 100*np.cos(i/fps))
        cv2.rectangle(frame, (x-30, 100), (x+30, 160), (255, 0, 0), -1)
        
        # Add frame number
        cv2.putText(frame, f'Frame: {i}/{int(fps * duration)}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    return output_path

def main():
    print("\nOpenCV Video Processing Examples")
    print("================================")
    
    # Create a sample video if needed
    if not os.path.exists('sample_video.avi'):
        print("\nCreating sample video...")
        video_path = create_sample_video()
        print(f"Sample video created: {video_path}")
    else:
        video_path = 'sample_video.avi'
    
    while True:
        print("\nSelect a demo:")
        print("1. Video Player")
        print("2. Video Recorder")
        print("3. Video Analyzer")
        print("q. Quit")
        
        choice = input("\nEnter your choice: ")
        
        if choice == '1':
            print("\nVideo Player Demo")
            print("Controls:")
            print("- 'p': Pause/Resume")
            print("- 'f': Forward 10 frames")
            print("- 'b': Backward 10 frames")
            print("- 'q': Quit")
            
            player = VideoPlayer(video_path)
            player.play()
        
        elif choice == '2':
            print("\nVideo Recorder Demo")
            print("Controls:")
            print("- 'r': Start/Stop recording")
            print("- 'e': Cycle through effects")
            print("- 'q': Quit")
            
            output_path = f'recorded_video_{int(time.time())}.avi'
            recorder = VideoRecorder(output_path)
            recorder.record()
            print(f"\nVideo saved as: {output_path}")
        
        elif choice == '3':
            print("\nVideo Analyzer Demo")
            print("Controls:")
            print("- 'm': Cycle through modes")
            print("- '+': Increase motion threshold")
            print("- '-': Decrease motion threshold")
            print("- 'q': Quit")
            
            analyzer = VideoAnalyzer()
            analyzer.analyze()
        
        elif choice.lower() == 'q':
            break
        
        else:
            print("\nInvalid choice. Please try again.")
    
    print("\nDemo completed. Thank you!")

if __name__ == "__main__":
    main()