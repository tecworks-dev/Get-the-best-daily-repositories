# Real-time Applications with OpenCV

This tutorial covers building real-time applications using OpenCV, including video processing, webcam interaction, and performance optimization techniques.

## Table of Contents
1. [Video Capture and Processing](#video-capture-and-processing)
2. [Real-time Object Detection](#real-time-object-detection)
3. [Performance Optimization](#performance-optimization)
4. [Multi-threading and Parallel Processing](#multi-threading-and-parallel-processing)
5. [GUI Integration](#gui-integration)

## Video Capture and Processing

### Basic Video Capture

```python
import cv2
import numpy as np
import threading
from queue import Queue
import time

class VideoCapture:
    def __init__(self, source=0):
        """
        Initialize video capture
        source: camera index or video file path
        """
        self.cap = cv2.VideoCapture(source)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    def read_frame(self):
        """
        Read a single frame
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
        """
        self.cap = cv2.VideoCapture(source)
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
                time.sleep(0.1)  # Sleep if queue is full
    
    def read_frame(self):
        """
        Read frame from queue
        """
        return self.queue.get() if not self.queue.empty() else None
    
    def stop(self):
        """
        Stop video capture
        """
        self.stopped = True
        self.thread.join()
        self.cap.release()
```

## Real-time Object Detection

### Fast Object Detection Pipeline

```python
class RealTimeObjectDetector:
    def __init__(self, model_path, config_path, conf_threshold=0.5):
        """
        Initialize real-time object detector
        """
        self.net = cv2.dnn.readNet(model_path, config_path)
        self.conf_threshold = conf_threshold
        self.classes = []  # Load class names
        
        # Enable GPU if available
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    
    def detect(self, frame):
        """
        Detect objects in frame
        """
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416),
                                   swapRB=True, crop=False)
        self.net.setInput(blob)
        
        # Get detections
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        
        # Process detections
        boxes = []
        confidences = []
        class_ids = []
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > self.conf_threshold:
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    width = int(detection[2] * frame.shape[1])
                    height = int(detection[3] * frame.shape[0])
                    
                    x = int(center_x - width/2)
                    y = int(center_y - height/2)
                    
                    boxes.append([x, y, width, height])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(boxes, confidences,
                                 self.conf_threshold, 0.4)
        
        return boxes, confidences, class_ids, indices
```

## Performance Optimization

### Frame Processing Pipeline

```python
class FrameProcessor:
    def __init__(self, max_workers=4):
        """
        Initialize frame processor with worker pool
        """
        self.max_workers = max_workers
        self.processing_queue = Queue(maxsize=32)
        self.result_queue = Queue(maxsize=32)
        self.workers = []
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
        while True:
            frame = self.processing_queue.get()
            if frame is None:
                break
            
            # Process frame
            processed_frame = self.process_single_frame(frame)
            self.result_queue.put(processed_frame)
    
    def process_single_frame(self, frame):
        """
        Process a single frame (override in subclass)
        """
        raise NotImplementedError
    
    def stop(self):
        """
        Stop all workers
        """
        for _ in range(self.max_workers):
            self.processing_queue.put(None)
        for worker in self.workers:
            worker.join()
```

## Multi-threading and Parallel Processing

### Parallel Frame Processing

```python
class ParallelVideoProcessor:
    def __init__(self, source=0, num_workers=4):
        """
        Initialize parallel video processor
        """
        self.capture = ThreadedVideoCapture(source)
        self.processor = FrameProcessor(max_workers=num_workers)
        self.display_thread = None
        self.running = False
    
    def start(self):
        """
        Start video processing
        """
        self.running = True
        self.display_thread = threading.Thread(target=self._display_loop)
        self.display_thread.start()
    
    def _display_loop(self):
        """
        Main display loop
        """
        while self.running:
            frame = self.capture.read_frame()
            if frame is None:
                continue
            
            # Add frame to processing queue
            if not self.processor.processing_queue.full():
                self.processor.processing_queue.put(frame)
            
            # Display processed frame
            if not self.processor.result_queue.empty():
                processed_frame = self.processor.result_queue.get()
                cv2.imshow('Frame', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()
                break
    
    def stop(self):
        """
        Stop video processing
        """
        self.running = False
        self.capture.stop()
        self.processor.stop()
        if self.display_thread:
            self.display_thread.join()
        cv2.destroyAllWindows()
```

## GUI Integration

### Basic GUI Application

```python
import tkinter as tk
from PIL import Image, ImageTk

class VideoGUI:
    def __init__(self, window, video_source=0):
        """
        Initialize GUI application
        """
        self.window = window
        self.window.title("Real-time Video Processing")
        
        # Create video capture
        self.vid = VideoCapture(video_source)
        
        # Create canvas
        self.canvas = tk.Canvas(window, 
                              width=self.vid.width,
                              height=self.vid.height)
        self.canvas.pack()
        
        # Buttons
        self.btn_snapshot = tk.Button(window, 
                                    text="Snapshot",
                                    command=self.snapshot)
        self.btn_snapshot.pack(anchor=tk.CENTER, expand=True)
        
        # Start update loop
        self.update()
        
        self.window.mainloop()
    
    def snapshot(self):
        """
        Take snapshot of current frame
        """
        frame = self.vid.read_frame()
        if frame is not None:
            cv2.imwrite(f"frame-{time.strftime('%d-%m-%Y-%H-%M-%S')}.jpg",
                       cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    def update(self):
        """
        Update frame
        """
        frame = self.vid.read_frame()
        
        if frame is not None:
            self.photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        
        self.window.after(10, self.update)
```

## Best Practices and Tips

1. **Performance Optimization**
   - Use threading for I/O operations
   - Implement frame skipping when necessary
   - Optimize image processing operations
   - Use GPU acceleration when available

2. **Memory Management**
   - Implement proper cleanup
   - Monitor memory usage
   - Use appropriate queue sizes
   - Release resources properly

3. **Error Handling**
   - Handle camera disconnection
   - Implement timeout mechanisms
   - Log errors appropriately
   - Provide user feedback

4. **User Interface**
   - Keep UI responsive
   - Provide progress feedback
   - Implement proper controls
   - Handle window resizing

## Applications

1. **Surveillance Systems**
   - Motion detection
   - Object tracking
   - Activity recognition
   - Alert generation

2. **Human-Computer Interaction**
   - Gesture recognition
   - Face tracking
   - Eye tracking
   - Pose estimation

3. **Industrial Applications**
   - Quality control
   - Process monitoring
   - Defect detection
   - Assembly verification

4. **Augmented Reality**
   - Marker tracking
   - 3D overlay
   - Interactive displays
   - Real-time effects

## Further Reading

1. [OpenCV Real-time Processing Documentation](https://docs.opencv.org/)
2. Multi-threading and parallel processing
3. GPU acceleration techniques
4. Real-time system design principles