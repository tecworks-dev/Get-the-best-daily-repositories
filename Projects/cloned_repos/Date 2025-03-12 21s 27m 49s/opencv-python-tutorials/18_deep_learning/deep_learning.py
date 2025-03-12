#!/usr/bin/env python3
"""
Deep Learning with OpenCV
This script demonstrates how to use OpenCV's dnn module for various deep learning tasks
"""

import cv2
import numpy as np
import os
import sys
import time
import matplotlib.pyplot as plt
from urllib.request import urlretrieve

def main():
    # Get the path to the models directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, 'models')
    
    # Create models directory if it doesn't exist
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # Get the path to the images directory
    images_dir = os.path.join(os.path.dirname(script_dir), 'images')
    
    # Try to load an image from the images directory
    try:
        # Try to load an image with people
        img_path = os.path.join(images_dir, 'faces.jpg')
        if os.path.exists(img_path):
            image = cv2.imread(img_path)
        else:
            # Try other potential images
            for img_name in ['landscape.jpg', 'objects.jpg', 'shapes.jpg']:
                img_path = os.path.join(images_dir, img_name)
                if os.path.exists(img_path):
                    image = cv2.imread(img_path)
                    break
            else:
                # If no image is found, create a sample image
                image = create_sample_image()
    except:
        # Fallback to creating a sample image
        image = create_sample_image()
    
    # Ensure we have a valid image
    if image is None or image.size == 0:
        image = create_sample_image()
        cv2.imwrite('sample_image.jpg', image)
        print("Created and saved sample_image.jpg")
    
    # 1. Face Detection with SSD
    print("\n1. FACE DETECTION WITH SSD")
    
    # Download the model files if they don't exist
    prototxt_path = os.path.join(models_dir, 'deploy.prototxt')
    caffemodel_path = os.path.join(models_dir, 'res10_300x300_ssd_iter_140000.caffemodel')
    
    if not os.path.exists(prototxt_path) or not os.path.exists(caffemodel_path):
        print("Downloading face detection model files...")
        try:
            urlretrieve("https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt", prototxt_path)
            urlretrieve("https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel", caffemodel_path)
            print("Model files downloaded successfully")
        except Exception as e:
            print(f"Error downloading model files: {e}")
            print("Skipping face detection...")
            face_detection_success = False
        else:
            face_detection_success = True
    else:
        face_detection_success = True
    
    if face_detection_success:
        try:
            # Load the model
            net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
            
            # Get a copy of the image to draw on
            face_image = image.copy()
            height, width = face_image.shape[:2]
            
            # Pre-process the image
            blob = cv2.dnn.blobFromImage(face_image, 1.0, (300, 300), (104, 117, 123))
            net.setInput(blob)
            
            # Run forward pass
            detections = net.forward()
            
            # Process the detections
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                if confidence > 0.5:  # Confidence threshold
                    # Face detected
                    box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                    (x1, y1, x2, y2) = box.astype("int")
                    
                    # Draw the bounding box
                    cv2.rectangle(face_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(face_image, f"Confidence: {confidence:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Display the result
            cv2.imshow('Face Detection with SSD', face_image)
            print("Press any key to continue...")
            cv2.waitKey(0)
        except Exception as e:
            print(f"Error in face detection: {e}")
            print("Skipping face detection...")
    
    # 2. Object Detection with MobileNet-SSD
    print("\n2. OBJECT DETECTION WITH MOBILENET-SSD")
    
    # Download the model files if they don't exist
    prototxt_path = os.path.join(models_dir, 'MobileNetSSD_deploy.prototxt')
    caffemodel_path = os.path.join(models_dir, 'MobileNetSSD_deploy.caffemodel')
    
    if not os.path.exists(prototxt_path) or not os.path.exists(caffemodel_path):
        print("Downloading object detection model files...")
        try:
            urlretrieve("https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt", prototxt_path)
            urlretrieve("https://drive.google.com/uc?export=download&id=0B3gersZ2cHIxRm5PMWRoTkdHdHc", caffemodel_path)
            print("Model files downloaded successfully")
        except Exception as e:
            print(f"Error downloading model files: {e}")
            print("Skipping object detection...")
            object_detection_success = False
        else:
            object_detection_success = True
    else:
        object_detection_success = True
    
    if object_detection_success:
        try:
            # Load the model
            net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
            
            # Define the classes
            classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
                      "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
                      "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
            
            # Get a copy of the image to draw on
            object_image = image.copy()
            height, width = object_image.shape[:2]
            
            # Pre-process the image
            blob = cv2.dnn.blobFromImage(object_image, 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            
            # Run forward pass
            detections = net.forward()
            
            # Process the detections
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                if confidence > 0.2:  # Confidence threshold
                    # Object detected
                    class_id = int(detections[0, 0, i, 1])
                    
                    box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                    (x1, y1, x2, y2) = box.astype("int")
                    
                    # Draw the bounding box
                    cv2.rectangle(object_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{classes[class_id]}: {confidence:.2f}"
                    cv2.putText(object_image, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Display the result
            cv2.imshow('Object Detection with MobileNet-SSD', object_image)
            print("Press any key to continue...")
            cv2.waitKey(0)
        except Exception as e:
            print(f"Error in object detection: {e}")
            print("Skipping object detection...")
    
    # 3. Image Classification with GoogLeNet
    print("\n3. IMAGE CLASSIFICATION WITH GOOGLENET")
    
    # Download the model files if they don't exist
    prototxt_path = os.path.join(models_dir, 'bvlc_googlenet.prototxt')
    caffemodel_path = os.path.join(models_dir, 'bvlc_googlenet.caffemodel')
    synset_path = os.path.join(models_dir, 'synset_words.txt')
    
    if not os.path.exists(prototxt_path) or not os.path.exists(caffemodel_path) or not os.path.exists(synset_path):
        print("Downloading image classification model files...")
        try:
            urlretrieve("https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/bvlc_googlenet.prototxt", prototxt_path)
            urlretrieve("http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel", caffemodel_path)
            urlretrieve("https://raw.githubusercontent.com/HolzingerAlexander/deeplearning/master/data/synset_words.txt", synset_path)
            print("Model files downloaded successfully")
        except Exception as e:
            print(f"Error downloading model files: {e}")
            print("Skipping image classification...")
            classification_success = False
        else:
            classification_success = True
    else:
        classification_success = True
    
    if classification_success:
        try:
            # Load the model
            net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
            
            # Load the class names
            with open(synset_path, 'r') as f:
                class_names = [line.strip() for line in f.readlines()]
            
            # Get a copy of the image to draw on
            class_image = image.copy()
            
            # Pre-process the image
            blob = cv2.dnn.blobFromImage(class_image, 1.0, (224, 224), (104, 117, 123))
            net.setInput(blob)
            
            # Run forward pass
            start = time.time()
            output = net.forward()
            end = time.time()
            
            # Get the top 5 predictions
            top_indices = output[0].argsort()[-5:][::-1]
            top_confidences = output[0][top_indices]
            top_classes = [class_names[i] for i in top_indices]
            
            # Print the results
            print(f"Classification time: {end - start:.2f} seconds")
            for i in range(5):
                print(f"{i+1}. {top_classes[i]} ({top_confidences[i]:.4f})")
            
            # Create a visualization
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(class_image, cv2.COLOR_BGR2RGB))
            plt.title("Input Image")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            y_pos = np.arange(5)
            plt.barh(y_pos, top_confidences)
            plt.yticks(y_pos, [name.split(',')[0] for name in top_classes])
            plt.xlabel('Confidence')
            plt.title('Top 5 Predictions')
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error in image classification: {e}")
            print("Skipping image classification...")
    
    # 4. Neural Style Transfer
    print("\n4. NEURAL STYLE TRANSFER")
    
    # Download the model files if they don't exist
    model_path = os.path.join(models_dir, 'starry_night.t7')
    
    if not os.path.exists(model_path):
        print("Downloading neural style transfer model files...")
        try:
            # Create instance_norm directory if it doesn't exist
            instance_norm_dir = os.path.join(models_dir, 'instance_norm')
            if not os.path.exists(instance_norm_dir):
                os.makedirs(instance_norm_dir)
            
            urlretrieve("https://cs.stanford.edu/people/jcjohns/fast-neural-style/models/instance_norm/starry_night.t7", 
                       os.path.join(instance_norm_dir, 'starry_night.t7'))
            print("Model files downloaded successfully")
        except Exception as e:
            print(f"Error downloading model files: {e}")
            print("Skipping neural style transfer...")
            style_transfer_success = False
        else:
            style_transfer_success = True
    else:
        style_transfer_success = True
    
    if style_transfer_success:
        try:
            # Load the model
            model_path = os.path.join(models_dir, 'instance_norm', 'starry_night.t7')
            net = cv2.dnn.readNetFromTorch(model_path)
            
            # Get a copy of the image
            style_image = image.copy()
            height, width = style_image.shape[:2]
            
            # Pre-process the image
            blob = cv2.dnn.blobFromImage(style_image, 1.0, (width, height), 
                                        (103.939, 116.779, 123.68), swapRB=False, crop=False)
            net.setInput(blob)
            
            # Run forward pass
            output = net.forward()
            
            # Post-process the output
            output = output.reshape(3, output.shape[2], output.shape[3])
            output[0] += 103.939
            output[1] += 116.779
            output[2] += 123.68
            output /= 255.0
            output = output.transpose(1, 2, 0)
            
            # Ensure the output is in the valid range [0, 1]
            output = np.clip(output, 0, 1)
            
            # Display the result
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(cv2.cvtColor(style_image, cv2.COLOR_BGR2RGB))
            plt.title("Original Image")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(output)
            plt.title("Stylized Image (Starry Night)")
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error in neural style transfer: {e}")
            print("Skipping neural style transfer...")
    
    # Clean up
    cv2.destroyAllWindows()
    print("\nAll operations completed successfully!")

def create_sample_image():
    """Create a sample image for deep learning demonstrations"""
    # Create a blank image
    width, height = 640, 480
    img = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background
    
    # Add a person-like shape (stick figure)
    # Head
    cv2.circle(img, (width//2, height//4), 50, (0, 0, 0), 2)
    
    # Body
    cv2.line(img, (width//2, height//4 + 50), (width//2, height//4 + 200), (0, 0, 0), 2)
    
    # Arms
    cv2.line(img, (width//2, height//4 + 100), (width//2 - 100, height//4 + 50), (0, 0, 0), 2)
    cv2.line(img, (width//2, height//4 + 100), (width//2 + 100, height//4 + 50), (0, 0, 0), 2)
    
    # Legs
    cv2.line(img, (width//2, height//4 + 200), (width//2 - 70, height//4 + 350), (0, 0, 0), 2)
    cv2.line(img, (width//2, height//4 + 200), (width//2 + 70, height//4 + 350), (0, 0, 0), 2)
    
    # Add some objects
    # Table
    cv2.rectangle(img, (width//4, 3*height//4), (3*width//4, 3*height//4 + 30), (150, 75, 0), -1)
    cv2.line(img, (width//4 + 20, 3*height//4 + 30), (width//4 + 20, height - 20), (150, 75, 0), 10)
    cv2.line(img, (3*width//4 - 20, 3*height//4 + 30), (3*width//4 - 20, height - 20), (150, 75, 0), 10)
    
    # Bottle on the table
    cv2.rectangle(img, (width//2 - 15, 3*height//4 - 60), (width//2 + 15, 3*height//4), (0, 255, 0), -1)
    cv2.rectangle(img, (width//2 - 10, 3*height//4 - 80), (width//2 + 10, 3*height//4 - 60), (0, 255, 0), -1)
    
    # Add some text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'Sample Image', (width//4, 30), font, 1, (0, 0, 0), 2)
    
    return img

if __name__ == "__main__":
    main()