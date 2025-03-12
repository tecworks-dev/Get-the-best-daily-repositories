# Classification with Ultralytics

This tutorial covers image classification using the Ultralytics YOLOv8 framework, which provides accurate and efficient classification capabilities.

## Table of Contents
1. [Introduction to Image Classification](#introduction-to-image-classification)
2. [YOLOv8 for Classification](#yolov8-for-classification)
3. [Implementation](#implementation)
4. [Custom Training](#custom-training)
5. [Best Practices](#best-practices)
6. [Applications](#applications)

## Introduction to Image Classification

Image classification is a fundamental computer vision task that involves assigning a label or category to an entire image. Unlike object detection, which identifies and localizes multiple objects within an image, classification provides a single prediction for the entire image, answering the question "What is in this image?"

Key aspects of image classification:
- Assigns a single label to the entire image
- Typically outputs a probability distribution across all possible classes
- Forms the foundation for many computer vision applications
- Can be used for binary classification (two classes) or multi-class classification (multiple classes)

## YOLOv8 for Classification

While YOLO (You Only Look Once) is primarily known for object detection, Ultralytics YOLOv8 also offers specialized classification models that are designed for image classification tasks. These models are based on efficient architectures and are trained on large datasets like ImageNet.

YOLOv8 classification models:
- YOLOv8n-cls: Nano classification model
- YOLOv8s-cls: Small classification model
- YOLOv8m-cls: Medium classification model
- YOLOv8l-cls: Large classification model
- YOLOv8x-cls: Extra large classification model

## Implementation

### Basic Usage

```python
from ultralytics import YOLO
import cv2
import numpy as np

# Load a pretrained YOLOv8 classification model
model = YOLO('yolov8n-cls.pt')  # 'n' for nano, other options: 's', 'm', 'l', 'x'

# Run inference on an image
results = model('path/to/image.jpg')

# Process results
for result in results:
    # Get the original image
    original_img = result.orig_img
    
    # Get the probs tensor
    probs = result.probs
    
    if probs is not None:
        # Get the top 5 class indices and their probabilities
        top5_indices = probs.top5
        top5_probs = probs.top5conf
        
        # Get class names
        class_names = model.names
        
        # Create a copy of the original image for display
        display_img = original_img.copy()
        
        # Add classification results to the image
        y_offset = 30
        for i in range(len(top5_indices)):
            class_idx = top5_indices[i]
            prob = top5_probs[i].item()
            class_name = class_names[class_idx]
            
            text = f"{class_name}: {prob:.2f}"
            cv2.putText(display_img, text, (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
        
        # Display the result
        cv2.imshow("Classification Result", display_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
```

### Visualizing Results

```python
import cv2
from ultralytics import YOLO
import numpy as np

# Load the model
model = YOLO('yolov8n-cls.pt')

# Load image
image = cv2.imread('path/to/image.jpg')

# Run inference
results = model(image)

# Create a copy of the image for display
display_img = image.copy()

# Get the probs from the first result
probs = results[0].probs

# Get the top 5 class indices and their probabilities
top5_indices = probs.top5
top5_probs = probs.top5conf

# Get class names
class_names = model.names

# Add classification results to the image
y_offset = 30
for i in range(len(top5_indices)):
    class_idx = top5_indices[i]
    prob = top5_probs[i].item()
    class_name = class_names[class_idx]
    
    text = f"{class_name}: {prob:.2f}"
    cv2.putText(display_img, text, (10, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    y_offset += 30

# Display the image
cv2.imshow("YOLOv8 Classification", display_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Real-time Classification

```python
import cv2
from ultralytics import YOLO

# Load the model
model = YOLO('yolov8n-cls.pt')

# Open the video capture
cap = cv2.VideoCapture(0)  # Use 0 for webcam

while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    
    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)
        
        # Get the probs from the first result
        probs = results[0].probs
        
        # Get the top 3 class indices and their probabilities
        top3_indices = probs.top5[:3]  # Get only top 3
        top3_probs = probs.top5conf[:3]  # Get only top 3
        
        # Get class names
        class_names = model.names
        
        # Create a copy of the frame for display
        display_frame = frame.copy()
        
        # Add classification results to the frame
        y_offset = 30
        for i in range(len(top3_indices)):
            class_idx = top3_indices[i]
            prob = top3_probs[i].item()
            class_name = class_names[class_idx]
            
            text = f"{class_name}: {prob:.2f}"
            cv2.putText(display_frame, text, (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
        
        # Display the annotated frame
        cv2.imshow("YOLOv8 Classification", display_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
```

### Batch Processing

```python
from ultralytics import YOLO
import os
import cv2
import numpy as np
from pathlib import Path

# Load a pretrained YOLOv8 classification model
model = YOLO('yolov8n-cls.pt')

# Define input and output directories
input_dir = 'path/to/input/images'
output_dir = 'path/to/output/images'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Get all image files
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
image_files = [f for f in os.listdir(input_dir) if any(f.lower().endswith(ext) for ext in image_extensions)]

# Process images in batches
batch_size = 4
for i in range(0, len(image_files), batch_size):
    batch_files = image_files[i:i+batch_size]
    batch_paths = [os.path.join(input_dir, f) for f in batch_files]
    
    # Run inference on batch
    results = model(batch_paths)
    
    # Process each result
    for j, result in enumerate(results):
        # Get the original image
        original_img = result.orig_img
        
        # Get the probs tensor
        probs = result.probs
        
        if probs is not None:
            # Get the top 3 class indices and their probabilities
            top3_indices = probs.top5[:3]
            top3_probs = probs.top5conf[:3]
            
            # Get class names
            class_names = model.names
            
            # Create a copy of the original image for display
            display_img = original_img.copy()
            
            # Add classification results to the image
            y_offset = 30
            for k in range(len(top3_indices)):
                class_idx = top3_indices[k]
                prob = top3_probs[k].item()
                class_name = class_names[class_idx]
                
                text = f"{class_name}: {prob:.2f}"
                cv2.putText(display_img, text, (10, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 30
            
            # Save the result
            output_path = os.path.join(output_dir, f"classified_{batch_files[j]}")
            cv2.imwrite(output_path, display_img)
            
            print(f"Processed {batch_files[j]}: Top class is {class_names[top3_indices[0]]} with probability {top3_probs[0]:.2f}")
```

## Custom Training

Training a custom YOLOv8 classification model requires a dataset with labeled images organized in a specific folder structure.

```python
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-cls.pt')  # load a pretrained classification model

# Train the model on a custom dataset
results = model.train(
    data='path/to/dataset',
    epochs=100,
    imgsz=224,
    batch=16,
    name='yolov8n_cls_custom'
)
```

### Dataset Preparation

For classification, your dataset should be organized in the following structure:

```
dataset/
├── train/
│   ├── class1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── class2/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── ...
├── val/
│   ├── class1/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── class2/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── ...
└── test/ (optional)
    ├── class1/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── class2/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── ...
```

Each class should have its own folder, and images belonging to that class should be placed in the corresponding folder.

## Best Practices

1. **Model Selection**
   - Choose the appropriate classification model size based on your requirements
   - Larger models provide better accuracy but are slower

2. **Data Preparation**
   - Ensure balanced class distribution
   - Include diverse examples for each class
   - Use data augmentation to improve robustness
   - Remove duplicate or very similar images

3. **Training Tips**
   - Start with a pre-trained classification model
   - Use appropriate batch size based on available GPU memory
   - Monitor accuracy and loss during training
   - Use early stopping to save the best model
   - Consider learning rate scheduling

4. **Inference Optimization**
   - Adjust confidence thresholds for optimal results
   - Use batch processing for multiple images
   - Consider model quantization for deployment
   - Use hardware acceleration for real-time applications

## Applications

1. **Content Categorization**
   - Image sorting and organization
   - Content filtering
   - Media asset management
   - Automatic tagging

2. **Medical Imaging**
   - Disease classification
   - Medical image categorization
   - Diagnostic assistance
   - Pathology image analysis

3. **Industrial Inspection**
   - Product quality control
   - Defect classification
   - Material identification
   - Manufacturing process monitoring

4. **Agriculture**
   - Crop disease identification
   - Plant species classification
   - Fruit ripeness assessment
   - Weed detection

5. **Retail**
   - Product recognition
   - Visual search
   - Inventory management
   - Customer behavior analysis

## Further Reading

1. [Ultralytics Classification Documentation](https://docs.ultralytics.com/tasks/classify/)
2. [ImageNet Dataset](https://www.image-net.org/)
3. [Transfer Learning for Image Classification](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
4. [Data Augmentation Techniques](https://pytorch.org/vision/stable/transforms.html)