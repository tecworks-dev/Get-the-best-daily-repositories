#!/usr/bin/env python3
"""
Machine Learning with OpenCV
This script demonstrates machine learning techniques using OpenCV
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml, load_digits

class KNNClassifier:
    def __init__(self, k=3):
        """Initialize KNN classifier"""
        self.knn = cv2.ml.KNearest_create()
        self.k = k
    
    def train(self, features, labels):
        """
        Train KNN classifier
        
        Args:
            features: numpy array of shape (n_samples, n_features)
            labels: numpy array of shape (n_samples,)
        """
        self.knn.train(features.astype(np.float32), 
                      cv2.ml.ROW_SAMPLE,
                      labels.astype(np.float32))
    
    def predict(self, features):
        """
        Predict using trained KNN classifier
        
        Args:
            features: numpy array of shape (n_samples, n_features)
        
        Returns:
            Predicted labels
        """
        ret, results, neighbours, dist = self.knn.findNearest(
            features.astype(np.float32), self.k)
        return results
    
    def evaluate(self, features, true_labels):
        """
        Evaluate classifier performance
        
        Args:
            features: numpy array of shape (n_samples, n_features)
            true_labels: numpy array of shape (n_samples,)
        
        Returns:
            Accuracy score
        """
        predictions = self.predict(features)
        accuracy = np.mean(predictions.flatten() == true_labels)
        return accuracy

class SVMClassifier:
    def __init__(self, kernel_type=cv2.ml.SVM_RBF, C=1.0, gamma=0.5):
        """Initialize SVM classifier"""
        self.svm = cv2.ml.SVM_create()
        self.svm.setKernel(kernel_type)
        self.svm.setType(cv2.ml.SVM_C_SVC)
        self.svm.setC(C)
        self.svm.setGamma(gamma)
    
    def train(self, features, labels):
        """
        Train SVM classifier
        
        Args:
            features: numpy array of shape (n_samples, n_features)
            labels: numpy array of shape (n_samples,)
        """
        self.svm.train(features.astype(np.float32), 
                      cv2.ml.ROW_SAMPLE,
                      labels.astype(np.int32))
    
    def predict(self, features):
        """
        Predict using trained SVM
        
        Args:
            features: numpy array of shape (n_samples, n_features)
        
        Returns:
            Predicted labels
        """
        return self.svm.predict(features.astype(np.float32))[1]
    
    def evaluate(self, features, true_labels):
        """
        Evaluate classifier performance
        
        Args:
            features: numpy array of shape (n_samples, n_features)
            true_labels: numpy array of shape (n_samples,)
        
        Returns:
            Accuracy score
        """
        predictions = self.predict(features)
        accuracy = np.mean(predictions.flatten() == true_labels)
        return accuracy
    
    def save_model(self, filename):
        """
        Save SVM model to file
        """
        self.svm.save(filename)
    
    def load_model(self, filename):
        """
        Load SVM model from file
        """
        self.svm = cv2.ml.SVM_load(filename)

class DecisionTreeClassifier:
    def __init__(self, max_depth=10):
        """Initialize decision tree classifier"""
        self.tree = cv2.ml.DTrees_create()
        self.tree.setMaxDepth(max_depth)
    
    def train(self, features, labels):
        """
        Train decision tree classifier
        
        Args:
            features: numpy array of shape (n_samples, n_features)
            labels: numpy array of shape (n_samples,)
        """
        self.tree.train(features.astype(np.float32),
                       cv2.ml.ROW_SAMPLE,
                       labels.astype(np.int32))
    
    def predict(self, features):
        """
        Predict using trained decision tree
        
        Args:
            features: numpy array of shape (n_samples, n_features)
        
        Returns:
            Predicted labels
        """
        return self.tree.predict(features.astype(np.float32))[1]
    
    def evaluate(self, features, true_labels):
        """
        Evaluate classifier performance
        
        Args:
            features: numpy array of shape (n_samples, n_features)
            true_labels: numpy array of shape (n_samples,)
        
        Returns:
            Accuracy score
        """
        predictions = self.predict(features)
        accuracy = np.mean(predictions.flatten() == true_labels)
        return accuracy

class RandomForestClassifier:
    def __init__(self, n_trees=10, max_depth=10):
        """Initialize random forest classifier"""
        self.rf = cv2.ml.RTrees_create()
        self.rf.setMaxDepth(max_depth)
        self.rf.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, n_trees, 0.01))
    
    def train(self, features, labels):
        """
        Train random forest classifier
        
        Args:
            features: numpy array of shape (n_samples, n_features)
            labels: numpy array of shape (n_samples,)
        """
        self.rf.train(features.astype(np.float32),
                     cv2.ml.ROW_SAMPLE,
                     labels.astype(np.int32))
    
    def predict(self, features):
        """
        Predict using trained random forest
        
        Args:
            features: numpy array of shape (n_samples, n_features)
        
        Returns:
            Predicted labels
        """
        return self.rf.predict(features.astype(np.float32))[1]
    
    def evaluate(self, features, true_labels):
        """
        Evaluate classifier performance
        
        Args:
            features: numpy array of shape (n_samples, n_features)
            true_labels: numpy array of shape (n_samples,)
        
        Returns:
            Accuracy score
        """
        predictions = self.predict(features)
        accuracy = np.mean(predictions.flatten() == true_labels)
        return accuracy

class DeepLearningModel:
    def __init__(self, model_path, config_path=None, framework='caffe'):
        """
        Initialize deep learning model using OpenCV's DNN module
        
        Args:
            model_path: Path to model file
            config_path: Path to model configuration file (optional)
            framework: Framework used to train the model ('caffe', 'tensorflow', 'torch', etc.)
        """
        self.framework = framework
        
        if framework == 'caffe' and config_path:
            self.net = cv2.dnn.readNetFromCaffe(config_path, model_path)
        elif framework == 'tensorflow':
            self.net = cv2.dnn.readNetFromTensorflow(model_path, config_path)
        elif framework == 'torch':
            self.net = cv2.dnn.readNetFromTorch(model_path)
        elif framework == 'darknet':
            self.net = cv2.dnn.readNetFromDarknet(config_path, model_path)
        elif framework == 'onnx':
            self.net = cv2.dnn.readNetFromONNX(model_path)
        else:
            self.net = cv2.dnn.readNet(model_path, config_path, framework)
    
    def preprocess_image(self, image, size=(224, 224), scale=1.0/255.0, mean=None, swapRB=True):
        """
        Preprocess image for network input
        
        Args:
            image: Input image
            size: Target size (width, height)
            scale: Scaling factor
            mean: Mean values for normalization
            swapRB: Whether to swap red and blue channels
        
        Returns:
            Preprocessed blob
        """
        if mean is None:
            mean = (0, 0, 0)
        
        blob = cv2.dnn.blobFromImage(image, scale, size, mean, swapRB, crop=False)
        return blob
    
    def predict(self, image, size=(224, 224), scale=1.0/255.0, mean=None, swapRB=True):
        """
        Make prediction using the network
        
        Args:
            image: Input image
            size: Target size (width, height)
            scale: Scaling factor
            mean: Mean values for normalization
            swapRB: Whether to swap red and blue channels
        
        Returns:
            Network output
        """
        blob = self.preprocess_image(image, size, scale, mean, swapRB)
        self.net.setInput(blob)
        return self.net.forward()

class FeatureExtractor:
    def __init__(self, method='sift'):
        """
        Initialize feature detector and descriptor
        
        Args:
            method: Feature detection method ('sift', 'orb', etc.)
        """
        if method.lower() == 'sift':
            self.detector = cv2.SIFT_create()
        elif method.lower() == 'orb':
            self.detector = cv2.ORB_create()
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        self.method = method
        self.matcher = cv2.BFMatcher()
    
    def extract_features(self, image):
        """
        Extract features from image
        
        Args:
            image: Input image
        
        Returns:
            keypoints, descriptors
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        return keypoints, descriptors
    
    def match_features(self, desc1, desc2, ratio_thresh=0.75):
        """
        Match features using ratio test
        
        Args:
            desc1: First set of descriptors
            desc2: Second set of descriptors
            ratio_thresh: Ratio test threshold
        
        Returns:
            Good matches
        """
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        
        # Apply ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
        
        return good_matches
    
    def draw_matches(self, img1, kp1, img2, kp2, matches):
        """
        Draw matches between images
        
        Args:
            img1: First image
            kp1: Keypoints from first image
            img2: Second image
            kp2: Keypoints from second image
            matches: Matches between keypoints
        
        Returns:
            Image with drawn matches
        """
        return cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

def load_mnist_data():
    """
    Load MNIST dataset
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    try:
        # Try to load from sklearn
        digits = load_digits()
        X = digits.data
        y = digits.target
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test
    
    except:
        # Fallback: create synthetic data
        print("Could not load MNIST dataset. Creating synthetic data instead.")
        
        # Create synthetic data
        np.random.seed(42)
        X = np.random.rand(1000, 64) * 16
        y = np.random.randint(0, 10, 1000)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        
        return X_train, X_test, y_train, y_test

def create_sample_images():
    """Create sample images for testing"""
    # Create a simple image with shapes
    img1 = np.zeros((300, 300, 3), dtype=np.uint8)
    img1.fill(255)  # White background
    
    # Add some shapes
    cv2.rectangle(img1, (50, 50), (100, 100), (0, 0, 255), -1)  # Red rectangle
    cv2.circle(img1, (200, 150), 50, (0, 255, 0), -1)  # Green circle
    cv2.line(img1, (50, 200), (250, 200), (255, 0, 0), 5)  # Blue line
    
    # Create a second image with similar shapes but slightly moved
    img2 = np.zeros((300, 300, 3), dtype=np.uint8)
    img2.fill(255)  # White background
    
    # Add some shapes
    cv2.rectangle(img2, (60, 60), (110, 110), (0, 0, 255), -1)  # Red rectangle
    cv2.circle(img2, (210, 160), 50, (0, 255, 0), -1)  # Green circle
    cv2.line(img2, (60, 210), (260, 210), (255, 0, 0), 5)  # Blue line
    
    return img1, img2

def main():
    """Main function"""
    print("Machine Learning with OpenCV")
    print("==========================")
    
    # Create or load sample images
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), 'images')
    
    # Create sample images
    img1, img2 = create_sample_images()
    
    # Save sample images if needed
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    cv2.imwrite(os.path.join(images_dir, 'ml_sample1.jpg'), img1)
    cv2.imwrite(os.path.join(images_dir, 'ml_sample2.jpg'), img2)
    
    # Load MNIST data
    X_train, X_test, y_train, y_test = load_mnist_data()
    
    while True:
        print("\nSelect a demo:")
        print("1. K-Nearest Neighbors")
        print("2. Support Vector Machines")
        print("3. Decision Trees and Random Forests")
        print("4. Feature Detection and Matching")
        print("q. Quit")
        
        choice = input("\nEnter your choice: ")
        
        if choice == '1':
            print("\nK-Nearest Neighbors Demo")
            
            # Train KNN classifier
            knn = KNNClassifier(k=3)
            knn.train(X_train, y_train)
            
            # Evaluate on test set
            accuracy = knn.evaluate(X_test, y_test)
            print(f"KNN Accuracy: {accuracy:.4f}")
            
            # Visualize some predictions
            n_samples = 5
            indices = np.random.choice(len(X_test), n_samples, replace=False)
            
            plt.figure(figsize=(15, 3))
            plt.suptitle("KNN Predictions", fontsize=16)
            
            for i, idx in enumerate(indices):
                plt.subplot(1, n_samples, i+1)
                plt.imshow(X_test[idx].reshape(8, 8), cmap='gray')
                
                true_label = y_test[idx]
                pred_label = int(knn.predict(X_test[idx].reshape(1, -1))[0, 0])
                
                title = f"True: {true_label}\nPred: {pred_label}"
                color = 'green' if true_label == pred_label else 'red'
                
                plt.title(title, color=color)
                plt.axis('off')
            
            plt.tight_layout()
            plt.show()
            
        elif choice == '2':
            print("\nSupport Vector Machines Demo")
            
            # Train SVM classifier
            svm = SVMClassifier(C=10.0, gamma=0.01)
            svm.train(X_train, y_train)
            
            # Evaluate on test set
            accuracy = svm.evaluate(X_test, y_test)
            print(f"SVM Accuracy: {accuracy:.4f}")
            
            # Visualize some predictions
            n_samples = 5
            indices = np.random.choice(len(X_test), n_samples, replace=False)
            
            plt.figure(figsize=(15, 3))
            plt.suptitle("SVM Predictions", fontsize=16)
            
            for i, idx in enumerate(indices):
                plt.subplot(1, n_samples, i+1)
                plt.imshow(X_test[idx].reshape(8, 8), cmap='gray')
                
                true_label = y_test[idx]
                pred_label = int(svm.predict(X_test[idx].reshape(1, -1))[0, 0])
                
                title = f"True: {true_label}\nPred: {pred_label}"
                color = 'green' if true_label == pred_label else 'red'
                
                plt.title(title, color=color)
                plt.axis('off')
            
            plt.tight_layout()
            plt.show()
            
        elif choice == '3':
            print("\nDecision Trees and Random Forests Demo")
            
            # Train Decision Tree classifier
            dt = DecisionTreeClassifier(max_depth=10)
            dt.train(X_train, y_train)
            
            # Train Random Forest classifier
            rf = RandomForestClassifier(n_trees=10, max_depth=10)
            rf.train(X_train, y_train)
            
            # Evaluate on test set
            dt_accuracy = dt.evaluate(X_test, y_test)
            rf_accuracy = rf.evaluate(X_test, y_test)
            
            print(f"Decision Tree Accuracy: {dt_accuracy:.4f}")
            print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
            
            # Visualize some predictions
            n_samples = 5
            indices = np.random.choice(len(X_test), n_samples, replace=False)
            
            plt.figure(figsize=(15, 6))
            plt.suptitle("Decision Tree vs Random Forest", fontsize=16)
            
            for i, idx in enumerate(indices):
                # Decision Tree
                plt.subplot(2, n_samples, i+1)
                plt.imshow(X_test[idx].reshape(8, 8), cmap='gray')
                
                true_label = y_test[idx]
                dt_pred = int(dt.predict(X_test[idx].reshape(1, -1))[0, 0])
                
                title = f"True: {true_label}\nDT Pred: {dt_pred}"
                color = 'green' if true_label == dt_pred else 'red'
                
                plt.title(title, color=color)
                plt.axis('off')
                
                # Random Forest
                plt.subplot(2, n_samples, i+1+n_samples)
                plt.imshow(X_test[idx].reshape(8, 8), cmap='gray')
                
                rf_pred = int(rf.predict(X_test[idx].reshape(1, -1))[0, 0])
                
                title = f"True: {true_label}\nRF Pred: {rf_pred}"
                color = 'green' if true_label == rf_pred else 'red'
                
                plt.title(title, color=color)
                plt.axis('off')
            
            plt.tight_layout()
            plt.show()
            
        elif choice == '4':
            print("\nFeature Detection and Matching Demo")
            
            # Initialize feature extractor
            feature_extractor = FeatureExtractor(method='sift')
            
            # Extract features
            kp1, desc1 = feature_extractor.extract_features(img1)
            kp2, desc2 = feature_extractor.extract_features(img2)
            
            # Match features
            matches = feature_extractor.match_features(desc1, desc2)
            
            # Draw matches
            img_matches = feature_extractor.draw_matches(img1, kp1, img2, kp2, matches[:30])
            
            # Display results
            plt.figure(figsize=(15, 10))
            plt.suptitle("Feature Detection and Matching", fontsize=16)
            
            plt.subplot(2, 2, 1)
            plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
            plt.title("Image 1")
            plt.axis('off')
            
            plt.subplot(2, 2, 2)
            plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
            plt.title("Image 2")
            plt.axis('off')
            
            plt.subplot(2, 1, 2)
            plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
            plt.title(f"Feature Matches ({len(matches)} matches)")
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
            
        elif choice.lower() == 'q':
            break
        
        else:
            print("\nInvalid choice. Please try again.")
    
    print("\nDemo completed. Thank you!")

if __name__ == "__main__":
    main()