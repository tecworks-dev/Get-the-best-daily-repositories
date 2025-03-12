# OpenCV Python Tutorials

A comprehensive collection of tutorials and examples for computer vision using OpenCV and Python.

## Overview

This repository contains a series of tutorials covering various aspects of computer vision using OpenCV with Python. Each tutorial includes detailed explanations, code examples, and practical applications. The tutorials are designed to be accessible for beginners while also covering advanced topics for experienced developers.

## Requirements

- Python 3.6+
- OpenCV 4.x
- NumPy
- Matplotlib
- Additional requirements for specific tutorials are listed in their respective directories

You can install the basic requirements using:

```bash
pip install -r requirements.txt
```

## Tutorials

### Basic Operations
1. [Introduction to OpenCV](01_introduction/README.md)
2. [Image Basics](02_image_basics/README.md)
3. [Drawing and Writing on Images](03_drawing/README.md)
4. [Image Processing](04_image_processing/README.md)
5. [Image Arithmetic and Bitwise Operations](05_image_operations/README.md)

### Intermediate Techniques
6. [Image Thresholding](06_thresholding/README.md)
7. [Edge Detection](07_edge_detection/README.md)
8. [Contours](08_contours/README.md)
9. [Histograms](09_histograms/README.md)
10. [Video Basics](10_video_basics/README.md)
11. [Object Detection](11_object_detection/README.md)
12. [Feature Detection](12_feature_detection/README.md)
13. [Image Segmentation](13_image_segmentation/README.md)
14. [Image Filtering and Convolution](14_filtering/README.md)
15. [Image Transformations](15_transformations/README.md)
16. [Camera Calibration](16_camera_calibration/README.md)

### Advanced Topics
17. [Machine Learning with OpenCV](17_machine_learning/README.md)
18. [Deep Learning with OpenCV](18_deep_learning/README.md)
19. [Real-time Applications](19_realtime_applications/README.md)
20. [Advanced Topics](20_advanced_topics/README.md)

### Ultralytics YOLOv8 Integration
21. [Object Detection with Ultralytics](21_object_detection_ultralytics/README.md)
22. [Instance Segmentation with Ultralytics](22_instance_segmentation_ultralytics/README.md)
23. [Pose Estimation with Ultralytics](23_pose_estimation_ultralytics/README.md)
24. [Classification with Ultralytics](24_classification_ultralytics/README.md)
25. [Object Tracking with Ultralytics](25_object_tracking_ultralytics/README.md)

## Tutorial Structure

Each tutorial directory typically contains:

1. **README.md**: Detailed explanation of concepts, techniques, and theory
2. **Python scripts**: Practical implementations and examples
3. **Sample images**: Test images for the examples (when applicable)

## Running the Examples

To run any example script, navigate to its directory and execute:

```bash
python script_name.py
```

For scripts that use webcam input, you can typically run:

```bash
python script_name.py --device 0
```

Where `0` is the index of your webcam device.

## Ultralytics YOLOv8 Scripts

The Ultralytics YOLOv8 scripts provide real-time computer vision capabilities using a webcam. Each script includes command-line arguments for customization:

```bash
# Object Detection
python object_detection.py --model yolov8n.pt --device 0 --conf 0.25 --show-fps

# Instance Segmentation
python instance_segmentation.py --model yolov8n-seg.pt --device 0 --conf 0.25 --show-fps

# Pose Estimation
python pose_estimation.py --model yolov8n-pose.pt --device 0 --show-angles --show-fps

# Classification
python classification.py --model yolov8n-cls.pt --device 0 --top-k 3 --show-fps

# Object Tracking
python object_tracking.py --model yolov8n.pt --device 0 --tracker bytetrack --show-trajectories --show-fps
```

## Key Features

- **Comprehensive Coverage**: From basic image operations to advanced deep learning techniques
- **Practical Examples**: Real-world applications and use cases
- **Detailed Explanations**: Theory and implementation details for each topic
- **Code Quality**: Well-documented, readable code following best practices
- **Progressive Learning**: Structured from basic to advanced topics
- **State-of-the-art Integration**: Integration with modern frameworks like Ultralytics YOLOv8

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenCV team for the amazing library
- Ultralytics for the YOLOv8 framework
- The computer vision community for continuous innovation
