#!/usr/bin/env python3
"""
Object Detection Tutorial Script
This script demonstrates various object detection techniques using OpenCV
"""

import cv2
import numpy as np
import os
import time
from datetime import datetime

class TemplateMatching:
    """Template matching for object detection"""
    def __init__(self, method=cv2.TM_CCOEFF_NORMED):
        self.method = method
    
    def detect(self, image, template):
        """Detect template in image"""
        # Apply template matching
        result = cv2.matchTemplate(image, template, self.method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # Get coordinates based on method
        if self.method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        
        h, w = template.shape[:2]
        bottom_right = (top_left[0] + w, top_left[1] + h)
        
        return top_left, bottom_right, max_val

class CascadeDetector:
    """Cascade classifier based detection"""
    def __init__(self):
        # Load cascades
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        self.body_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_fullbody.xml'
        )
    
    def detect_faces(self, frame, draw=True):
        """Detect faces and eyes"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if draw:
            for (x, y, w, h) in faces:
                # Draw face rectangle
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Detect eyes in face region
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = frame[y:y+h, x:x+w]
                eyes = self.eye_cascade.detectMultiScale(roi_gray)
                
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        
        return frame, faces
    
    def detect_bodies(self, frame, draw=True):
        """Detect full bodies"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        bodies = self.body_cascade.detectMultiScale(gray, 1.1, 3)
        
        if draw:
            for (x, y, w, h) in bodies:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        return frame, bodies

class ColorDetector:
    """Color-based object detection"""
    def __init__(self):
        # Predefined color ranges in HSV
        self.color_ranges = {
            'red': ([0, 100, 100], [10, 255, 255]),
            'blue': ([110, 50, 50], [130, 255, 255]),
            'green': ([50, 100, 100], [70, 255, 255]),
            'yellow': ([20, 100, 100], [30, 255, 255])
        }
    
    def detect_color(self, frame, color_name):
        """Detect objects of specified color"""
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Get color range
        lower, upper = self.color_ranges[color_name]
        lower = np.array(lower)
        upper = np.array(upper)
        
        # Create mask
        mask = cv2.inRange(hsv, lower, upper)
        
        # Remove noise
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Draw contours and calculate areas
        result = frame.copy()
        areas = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Filter small areas
                cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)
                areas.append(area)
        
        return result, contours, areas

class FeatureDetector:
    """Feature-based object detection"""
    def __init__(self, feature_type='sift'):
        if feature_type == 'sift':
            self.detector = cv2.SIFT_create()
        elif feature_type == 'orb':
            self.detector = cv2.ORB_create()
        else:
            raise ValueError("Unsupported feature type")
        
        self.matcher = cv2.BFMatcher()
    
    def detect_and_match(self, img1, img2, ratio=0.75):
        """Detect and match features between two images"""
        # Find keypoints and descriptors
        kp1, des1 = self.detector.detectAndCompute(img1, None)
        kp2, des2 = self.detector.detectAndCompute(img2, None)
        
        if des1 is None or des2 is None:
            return None, None, []
        
        # Match features
        matches = self.matcher.knnMatch(des1, des2, k=2)
        
        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < ratio * n.distance:
                good_matches.append(m)
        
        return kp1, kp2, good_matches
    
    def draw_matches(self, img1, kp1, img2, kp2, matches):
        """Draw matches between images"""
        return cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

class ObjectTracker:
    """Multi-object tracking system"""
    def __init__(self, tracker_type="CSRT"):
        self.tracker_types = {
            'BOOSTING': cv2.legacy.TrackerBoosting_create,
            'MIL': cv2.legacy.TrackerMIL_create,
            'KCF': cv2.legacy.TrackerKCF_create,
            'CSRT': cv2.legacy.TrackerCSRT_create,
            'MOSSE': cv2.legacy.TrackerMOSSE_create
        }
        self.tracker_type = tracker_type
        self.trackers = {}
        self.next_id = 0
    
    def add_tracker(self, frame, bbox):
        """Add a new tracker"""
        tracker = self.tracker_types[self.tracker_type]()
        success = tracker.init(frame, bbox)
        if success:
            self.trackers[self.next_id] = tracker
            self.next_id += 1
            return self.next_id - 1
        return None
    
    def update_all(self, frame):
        """Update all trackers"""
        results = {}
        failed_trackers = []
        
        for obj_id, tracker in self.trackers.items():
            success, bbox = tracker.update(frame)
            if success:
                results[obj_id] = bbox
            else:
                failed_trackers.append(obj_id)
        
        # Remove failed trackers
        for obj_id in failed_trackers:
            del self.trackers[obj_id]
        
        return results

def create_sample_images():
    """Create sample images for testing"""
    # Create main image with shapes
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    img.fill(255)  # White background
    
    # Draw some shapes
    cv2.rectangle(img, (100, 100), (200, 200), (255, 0, 0), -1)  # Blue rectangle
    cv2.circle(img, (300, 300), 50, (0, 255, 0), -1)  # Green circle
    cv2.circle(img, (400, 150), 30, (0, 0, 255), -1)  # Red circle
    
    # Create template (small blue rectangle)
    template = np.zeros((100, 100, 3), dtype=np.uint8)
    template.fill(255)
    cv2.rectangle(template, (25, 25), (75, 75), (255, 0, 0), -1)
    
    return img, template

def main():
    print("\nOpenCV Object Detection Examples")
    print("===============================")
    
    # Create or load sample images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), 'images')
    
    # Create sample images if needed
    img, template = create_sample_images()
    
    # Save sample images
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    cv2.imwrite(os.path.join(images_dir, 'sample_main.jpg'), img)
    cv2.imwrite(os.path.join(images_dir, 'sample_template.jpg'), template)
    
    while True:
        print("\nSelect a demo:")
        print("1. Template Matching")
        print("2. Face and Body Detection")
        print("3. Color Detection")
        print("4. Feature Detection")
        print("5. Object Tracking")
        print("q. Quit")
        
        choice = input("\nEnter your choice: ")
        
        if choice == '1':
            print("\nTemplate Matching Demo")
            
            # Perform template matching
            matcher = TemplateMatching()
            top_left, bottom_right, confidence = matcher.detect(img, template)
            
            # Draw result
            result = img.copy()
            cv2.rectangle(result, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(result, f"Confidence: {confidence:.2f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display results
            cv2.imshow('Template Matching', result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        elif choice == '2':
            print("\nFace and Body Detection Demo")
            print("Press 'q' to quit")
            
            detector = CascadeDetector()
            cap = cv2.VideoCapture(0)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect faces and bodies
                frame, faces = detector.detect_faces(frame)
                frame, bodies = detector.detect_bodies(frame)
                
                # Add info
                cv2.putText(frame, f"Faces: {len(faces)}, Bodies: {len(bodies)}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow('Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
        
        elif choice == '3':
            print("\nColor Detection Demo")
            print("Press 'c' to cycle through colors")
            print("Press 'q' to quit")
            
            detector = ColorDetector()
            cap = cv2.VideoCapture(0)
            colors = list(detector.color_ranges.keys())
            color_idx = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect current color
                color = colors[color_idx]
                result, contours, areas = detector.detect_color(frame, color)
                
                # Add info
                cv2.putText(result, f"Detecting: {color}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(result, f"Objects: {len(areas)}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow('Color Detection', result)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    color_idx = (color_idx + 1) % len(colors)
            
            cap.release()
            cv2.destroyAllWindows()
        
        elif choice == '4':
            print("\nFeature Detection Demo")
            print("Press 'q' to quit")
            
            detector = FeatureDetector()
            cap = cv2.VideoCapture(0)
            
            # Capture reference frame
            ret, ref_frame = cap.read()
            if not ret:
                print("Failed to capture reference frame")
                continue
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect and match features
                kp1, kp2, matches = detector.detect_and_match(ref_frame, frame)
                
                if kp1 is not None:
                    # Draw matches
                    result = detector.draw_matches(ref_frame, kp1, frame, kp2, matches[:10])
                    
                    # Add info
                    cv2.putText(result, f"Matches: {len(matches)}", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    cv2.imshow('Feature Matching', result)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
        
        elif choice == '5':
            print("\nObject Tracking Demo")
            print("Press 'r' to select region to track")
            print("Press 'c' to clear all trackers")
            print("Press 'q' to quit")
            
            tracker = ObjectTracker()
            cap = cv2.VideoCapture(0)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Update trackers
                boxes = tracker.update_all(frame)
                
                # Draw tracking boxes
                for obj_id, bbox in boxes.items():
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {obj_id}", p1,
                               cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                
                # Add info
                cv2.putText(frame, f"Objects: {len(boxes)}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imshow('Tracking', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    # Select ROI
                    bbox = cv2.selectROI('Tracking', frame, False)
                    if bbox[2] > 0 and bbox[3] > 0:
                        tracker.add_tracker(frame, bbox)
                elif key == ord('c'):
                    # Clear all trackers
                    tracker.trackers.clear()
            
            cap.release()
            cv2.destroyAllWindows()
        
        elif choice.lower() == 'q':
            break
        
        else:
            print("\nInvalid choice. Please try again.")
    
    print("\nDemo completed. Thank you!")

if __name__ == "__main__":
    main()