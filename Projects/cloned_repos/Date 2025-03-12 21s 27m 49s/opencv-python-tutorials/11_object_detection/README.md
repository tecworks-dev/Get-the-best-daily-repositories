# Object Detection with OpenCV

This tutorial covers various object detection techniques using OpenCV, from basic methods like template matching to more advanced approaches using pre-trained models.

## Introduction to Object Detection

Object detection is a computer vision technique that involves both locating and classifying objects in images or video streams. OpenCV provides several methods for object detection, ranging from simple to complex approaches.

## Template Matching

Template matching is a simple method to find areas of an image that match a template image:

```python
import cv2
import numpy as np

# Load image and template
img = cv2.imread('image.jpg')
template = cv2.imread('template.jpg')
h, w = template.shape[:2]

# Apply template matching
result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

# Draw rectangle around match
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
```

## Cascade Classifiers

Cascade classifiers are an effective way for object detection, especially for faces:

```python
# Load the cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Detect faces
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)
```

### Available Pre-trained Cascades

OpenCV comes with several pre-trained cascade classifiers:
- Face detection
- Eye detection
- Full body detection
- License plate detection
- etc.

## Color-based Object Detection

Detecting objects based on their color using HSV color space:

```python
# Convert to HSV
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# Define color range
lower_blue = np.array([110, 50, 50])
upper_blue = np.array([130, 255, 255])

# Create mask
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Find contours
contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
```

## Feature Detection and Matching

Feature detection can be used for object detection when dealing with complex objects:

```python
# Initialize SIFT detector
sift = cv2.SIFT_create()

# Find keypoints and descriptors
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

# Match features
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)
```

## Deep Learning-based Object Detection

OpenCV supports various deep learning frameworks for object detection:

### YOLO (You Only Look Once)

```python
# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load image and prepare blob
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)
```

### SSD (Single Shot Detector)

```python
# Load the model
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "model.caffemodel")

# Prepare input blob
blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104, 117, 123))
net.setInput(blob)
detections = net.forward()
```

## Object Tracking

Once objects are detected, they can be tracked across frames:

### 1. Simple Object Tracking

```python
def track_object(frame, bbox):
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Calculate histogram of object
    roi_hist = cv2.calcHist([hsv[bbox[1]:bbox[1]+bbox[3], 
                                bbox[0]:bbox[0]+bbox[2]]], 
                           [0], None, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    
    # Apply meanshift
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    _, bbox = cv2.meanShift(hsv, bbox, term_crit)
    
    return bbox
```

### 2. OpenCV Trackers

OpenCV provides several built-in trackers:

```python
# Initialize tracker
tracker = cv2.TrackerCSRT_create()  # or KCF, MOSSE, etc.

# Initialize tracker with first frame and bounding box
success = tracker.init(frame, bbox)

# Update tracker
success, bbox = tracker.update(frame)
```

## Practical Applications

### 1. Face Detection and Recognition System

```python
class FaceDetectionSystem:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
    
    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Detect eyes within face region
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = frame[y:y+h, x:x+w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray)
            
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
        
        return frame
```

### 2. Multi-Object Tracking System

```python
class MultiObjectTracker:
    def __init__(self):
        self.trackers = {}
        self.next_id = 0
    
    def add_object(self, frame, bbox):
        tracker = cv2.TrackerCSRT_create()
        success = tracker.init(frame, bbox)
        if success:
            self.trackers[self.next_id] = tracker
            self.next_id += 1
    
    def update(self, frame):
        # Update all trackers
        to_delete = []
        for obj_id, tracker in self.trackers.items():
            success, bbox = tracker.update(frame)
            if success:
                # Draw tracking box
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {obj_id}", p1, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            else:
                to_delete.append(obj_id)
        
        # Remove failed trackers
        for obj_id in to_delete:
            del self.trackers[obj_id]
        
        return frame
```

### 3. Object Counting System

```python
class ObjectCounter:
    def __init__(self):
        self.object_count = 0
        self.counted_objects = set()
        self.counting_line_y = None
    
    def setup_counting_line(self, frame_height):
        self.counting_line_y = frame_height // 2
    
    def count_objects(self, frame, detections):
        if self.counting_line_y is None:
            self.setup_counting_line(frame.shape[0])
        
        # Draw counting line
        cv2.line(frame, (0, self.counting_line_y), 
                (frame.shape[1], self.counting_line_y), (0, 255, 255), 2)
        
        for detection in detections:
            x, y, w, h = detection
            center_y = y + h//2
            
            # Check if object crosses the line
            if center_y > self.counting_line_y and id(detection) not in self.counted_objects:
                self.object_count += 1
                self.counted_objects.add(id(detection))
        
        # Display count
        cv2.putText(frame, f"Count: {self.object_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame
```

### 4. Real-time Object Detection System

```python
class RealTimeObjectDetector:
    def __init__(self, confidence_threshold=0.5):
        # Load YOLO
        self.net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
        self.classes = []
        with open("coco.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.confidence_threshold = confidence_threshold
        
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] 
                            for i in self.net.getUnconnectedOutLayers()]
    
    def detect_objects(self, frame):
        height, width = frame.shape[:2]
        
        # Detect objects
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), 
                                   True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        
        # Information to display on screen
        class_ids = []
        confidences = []
        boxes = []
        
        # Process detections
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.confidence_threshold:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maximum suppression
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        
        # Draw boxes
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                confidence = confidences[i]
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame
```

## Best Practices

1. **Pre-processing**:
   - Apply appropriate image preprocessing (noise reduction, contrast enhancement)
   - Consider resizing images for consistent processing
   - Use appropriate color spaces (HSV for color-based detection)

2. **Performance Optimization**:
   - Use appropriate detection methods based on requirements
   - Implement region of interest (ROI) to limit search area
   - Consider using GPU acceleration for deep learning models

3. **Robustness**:
   - Implement multiple detection methods when possible
   - Add validation steps to filter false positives
   - Consider environmental factors (lighting, occlusion)

4. **Real-time Applications**:
   - Optimize for speed vs accuracy based on requirements
   - Implement frame skipping if necessary
   - Use appropriate tracking methods to maintain performance

## Conclusion

Object detection is a fundamental task in computer vision with numerous applications. OpenCV provides a wide range of tools and methods for object detection, from basic template matching to advanced deep learning-based approaches. The choice of method depends on specific requirements such as accuracy, speed, and available computational resources.

In the next tutorial, we'll explore feature detection and matching techniques in more detail.