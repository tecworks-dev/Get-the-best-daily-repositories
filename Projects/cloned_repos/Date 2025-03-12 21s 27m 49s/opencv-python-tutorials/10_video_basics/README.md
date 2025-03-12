# Video Basics in OpenCV

This tutorial covers the fundamentals of working with video in OpenCV, including reading video files, capturing from cameras, and basic video processing techniques.

## Reading Video Files

OpenCV provides the `VideoCapture` class to read video files:

```python
import cv2

# Create a VideoCapture object
cap = cv2.VideoCapture('video.mp4')

while cap.isOpened():
    # Read a frame
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Process frame here
    cv2.imshow('Video', frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
```

### Video Properties

You can access various properties of the video:

```python
# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
```

## Capturing from Camera

To capture video from a camera:

```python
# 0 is usually the built-in webcam
cap = cv2.VideoCapture(0)

# Set camera properties
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
```

## Writing Video Files

To save video files:

```python
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Write the frame
    out.write(frame)

# Release resources
out.release()
```

## Basic Video Processing

### 1. Frame-by-Frame Processing

```python
def process_frame(frame):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Apply threshold
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    
    return thresh

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Process the frame
    processed = process_frame(frame)
    
    # Display result
    cv2.imshow('Processed', processed)
```

### 2. Background Subtraction

```python
# Create background subtractor
backSub = cv2.createBackgroundSubtractorMOG2()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Apply background subtraction
    fgMask = backSub.apply(frame)
    
    # Display result
    cv2.imshow('FG Mask', fgMask)
```

### 3. Motion Detection

```python
def detect_motion(frame1, frame2):
    # Calculate difference between frames
    diff = cv2.absdiff(frame1, frame2)
    
    # Convert to grayscale
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # Apply threshold
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw rectangles around motion areas
    for contour in contours:
        if cv2.contourArea(contour) > 500:
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    return frame1
```

## Advanced Video Processing

### 1. Frame Skipping and Seeking

```python
# Skip to specific frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 100)

# Skip every other frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Skip one frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) + 1)
```

### 2. Video Stabilization

```python
def stabilize_video(frame1, frame2):
    # Convert frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    # Find feature points
    points1 = cv2.goodFeaturesToTrack(gray1, 200, 0.3, 7)
    
    if points1 is not None:
        # Calculate optical flow
        points2, status, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, points1, None)
        
        # Filter good points
        good_points1 = points1[status == 1]
        good_points2 = points2[status == 1]
        
        # Find transformation matrix
        if len(good_points1) >= 3:
            transform = cv2.estimateAffinePartial2D(good_points1, good_points2)[0]
            
            # Apply transformation
            stabilized = cv2.warpAffine(frame2, transform, (frame2.shape[1], frame2.shape[0]))
            return stabilized
    
    return frame2
```

### 3. Frame Interpolation

```python
def interpolate_frames(frame1, frame2, factor):
    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(
        cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY),
        None, 0.5, 3, 15, 3, 5, 1.2, 0
    )
    
    # Create interpolated frame
    h, w = frame1.shape[:2]
    interpolated = np.zeros_like(frame1)
    
    # Create coordinate grid
    y, x = np.mgrid[0:h, 0:w].reshape(2, -1)
    coords = np.vstack((x, y))
    
    # Calculate new coordinates
    flow_coords = coords + factor * flow.reshape(2, -1)
    
    # Interpolate each channel
    for i in range(3):
        interpolated[..., i] = cv2.remap(
            frame1[..., i],
            flow_coords[0].reshape(h, w),
            flow_coords[1].reshape(h, w),
            cv2.INTER_LINEAR
        )
    
    return interpolated
```

## Practical Applications

### 1. Basic Video Player

```python
class VideoPlayer:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.paused = False
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
    
    def create_trackbar(self):
        cv2.namedWindow('Video Player')
        cv2.createTrackbar('Position', 'Video Player', 0, self.frame_count, self.on_trackbar)
    
    def on_trackbar(self, pos):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    
    def play(self):
        self.create_trackbar()
        
        while self.cap.isOpened():
            if not self.paused:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                cv2.imshow('Video Player', frame)
                current_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                cv2.setTrackbarPos('Position', 'Video Player', current_pos)
            
            key = cv2.waitKey(int(1000/self.fps)) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                self.paused = not self.paused
        
        self.cap.release()
        cv2.destroyAllWindows()
```

### 2. Video Recording with Effects

```python
class VideoRecorder:
    def __init__(self, output_path, resolution=(640, 480), fps=30.0):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter(output_path, fourcc, fps, resolution)
        
        self.effects = {
            'none': lambda x: x,
            'gray': lambda x: cv2.cvtColor(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR),
            'blur': lambda x: cv2.GaussianBlur(x, (15, 15), 0),
            'edge': lambda x: cv2.Canny(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), 100, 200)
        }
        self.current_effect = 'none'
    
    def record(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Apply effect
            processed = self.effects[self.current_effect](frame)
            
            # Write frame
            self.out.write(processed)
            
            # Display
            cv2.imshow('Recording', processed)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('e'):
                # Cycle through effects
                effects = list(self.effects.keys())
                current_idx = effects.index(self.current_effect)
                self.current_effect = effects[(current_idx + 1) % len(effects)]
        
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
```

### 3. Real-time Video Analysis

```python
class VideoAnalyzer:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.backSub = cv2.createBackgroundSubtractorMOG2()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    def analyze(self):
        prev_frame = None
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Motion detection
            if prev_frame is not None:
                motion = self.detect_motion(prev_frame, frame)
            
            # Background subtraction
            fg_mask = self.backSub.apply(frame)
            
            # Face detection
            faces = self.detect_faces(frame)
            
            # Display results
            cv2.imshow('Original', frame)
            cv2.imshow('Motion', motion if prev_frame is not None else frame)
            cv2.imshow('Background Subtraction', fg_mask)
            cv2.imshow('Face Detection', faces)
            
            prev_frame = frame.copy()
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()
    
    def detect_motion(self, prev, curr):
        diff = cv2.absdiff(prev, curr)
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=3)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        result = curr.copy()
        for c in contours:
            if cv2.contourArea(c) < 5000:
                continue
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        return result
    
    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        
        result = frame.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        return result
```

## Conclusion

This tutorial covered the basics of video processing with OpenCV, including:
- Reading and writing video files
- Capturing from cameras
- Basic frame processing
- Motion detection
- Background subtraction
- Video stabilization
- Frame interpolation
- Practical applications

In the next tutorial, we'll explore object detection techniques using OpenCV.