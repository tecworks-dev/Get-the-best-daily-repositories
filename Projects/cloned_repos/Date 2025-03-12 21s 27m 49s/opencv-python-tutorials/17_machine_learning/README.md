# Machine Learning with OpenCV

This tutorial covers machine learning techniques and algorithms available in OpenCV, including classification, clustering, and deep learning integration.

## Table of Contents
1. [K-Nearest Neighbors (KNN)](#k-nearest-neighbors)
2. [Support Vector Machines (SVM)](#support-vector-machines)
3. [Decision Trees and Random Forests](#decision-trees-and-random-forests)
4. [Deep Learning Integration](#deep-learning-integration)
5. [Feature Detection and Matching](#feature-detection-and-matching)

## K-Nearest Neighbors

```python
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

class KNNClassifier:
    def __init__(self, k=3):
        self.knn = cv2.ml.KNearest_create()
        self.k = k
    
    def train(self, features, labels):
        """
        Train KNN classifier
        features: numpy array of shape (n_samples, n_features)
        labels: numpy array of shape (n_samples,)
        """
        self.knn.train(features.astype(np.float32), 
                      cv2.ml.ROW_SAMPLE,
                      labels.astype(np.float32))
    
    def predict(self, features):
        """
        Predict using trained KNN classifier
        """
        ret, results, neighbours, dist = self.knn.findNearest(
            features.astype(np.float32), self.k)
        return results

def prepare_data(images, labels):
    """
    Prepare image data for KNN classification
    """
    # Flatten images
    features = np.array([img.flatten() for img in images])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test

# Example usage
# classifier = KNNClassifier(k=3)
# classifier.train(X_train, y_train)
# predictions = classifier.predict(X_test)
```

## Support Vector Machines

```python
class SVMClassifier:
    def __init__(self, kernel_type=cv2.ml.SVM_RBF):
        self.svm = cv2.ml.SVM_create()
        self.svm.setKernel(kernel_type)
        self.svm.setType(cv2.ml.SVM_C_SVC)
    
    def train(self, features, labels):
        """
        Train SVM classifier
        """
        self.svm.train(features.astype(np.float32), 
                      cv2.ml.ROW_SAMPLE,
                      labels.astype(np.int32))
    
    def predict(self, features):
        """
        Predict using trained SVM
        """
        return self.svm.predict(features.astype(np.float32))[1]
    
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

def optimize_svm_params(features, labels):
    """
    Optimize SVM parameters using grid search
    """
    svm = cv2.ml.SVM_create()
    svm.setKernel(cv2.ml.SVM_RBF)
    
    # Define parameter grid
    C_values = [0.1, 1, 10, 100]
    gamma_values = [0.1, 1, 10, 100]
    
    best_score = 0
    best_params = None
    
    for C in C_values:
        for gamma in gamma_values:
            svm.setC(C)
            svm.setGamma(gamma)
            
            # Perform cross-validation
            scores = cv2.ml.TrainData_create(
                features.astype(np.float32),
                cv2.ml.ROW_SAMPLE,
                labels.astype(np.int32)
            ).getDefaultSubsets()
            
            score = np.mean([
                svm.train(s[0], cv2.ml.ROW_SAMPLE, s[1]).predict(s[2])[1] == s[3]
                for s in scores
            ])
            
            if score > best_score:
                best_score = score
                best_params = (C, gamma)
    
    return best_params
```

## Decision Trees and Random Forests

```python
class DecisionTreeClassifier:
    def __init__(self, max_depth=10):
        self.tree = cv2.ml.DTrees_create()
        self.tree.setMaxDepth(max_depth)
    
    def train(self, features, labels):
        """
        Train decision tree classifier
        """
        self.tree.train(features.astype(np.float32),
                       cv2.ml.ROW_SAMPLE,
                       labels.astype(np.int32))
    
    def predict(self, features):
        """
        Predict using trained decision tree
        """
        return self.tree.predict(features.astype(np.float32))[1]

class RandomForestClassifier:
    def __init__(self, n_trees=10, max_depth=10):
        self.rf = cv2.ml.RTrees_create()
        self.rf.setMaxDepth(max_depth)
        self.rf.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, n_trees, 0.01))
    
    def train(self, features, labels):
        """
        Train random forest classifier
        """
        self.rf.train(features.astype(np.float32),
                     cv2.ml.ROW_SAMPLE,
                     labels.astype(np.int32))
    
    def predict(self, features):
        """
        Predict using trained random forest
        """
        return self.rf.predict(features.astype(np.float32))[1]
```

## Deep Learning Integration

```python
class DeepLearningModel:
    def __init__(self, model_path, config_path=None):
        """
        Initialize deep learning model using OpenCV's DNN module
        """
        if config_path:
            self.net = cv2.dnn.readNetFromCaffe(config_path, model_path)
        else:
            self.net = cv2.dnn.readNet(model_path)
    
    def preprocess_image(self, image, size=(224, 224)):
        """
        Preprocess image for network input
        """
        blob = cv2.dnn.blobFromImage(image, 1.0/255.0, size,
                                   (0, 0, 0), swapRB=True, crop=False)
        return blob
    
    def predict(self, image):
        """
        Make prediction using the network
        """
        blob = self.preprocess_image(image)
        self.net.setInput(blob)
        return self.net.forward()

def load_tensorflow_model(pb_path):
    """
    Load TensorFlow model using OpenCV DNN
    """
    return cv2.dnn.readNetFromTensorflow(pb_path)

def load_pytorch_model(model_path):
    """
    Load PyTorch model using OpenCV DNN
    """
    return cv2.dnn.readNetFromTorch(model_path)
```

## Feature Detection and Matching

```python
class FeatureExtractor:
    def __init__(self, method='sift'):
        """
        Initialize feature detector and descriptor
        """
        if method.lower() == 'sift':
            self.detector = cv2.SIFT_create()
        elif method.lower() == 'orb':
            self.detector = cv2.ORB_create()
        else:
            raise ValueError("Unsupported method")
        
        self.matcher = cv2.BFMatcher()
    
    def extract_features(self, image):
        """
        Extract features from image
        """
        keypoints, descriptors = self.detector.detectAndCompute(image, None)
        return keypoints, descriptors
    
    def match_features(self, desc1, desc2, ratio_thresh=0.75):
        """
        Match features using ratio test
        """
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        
        good_matches = []
        for m, n in matches:
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)
        
        return good_matches
```

## Best Practices and Tips

1. **Data Preparation**
   - Normalize input data
   - Handle imbalanced datasets
   - Use appropriate data augmentation
   - Split data properly (train/validation/test)

2. **Model Selection**
   - Choose appropriate algorithm for your task
   - Consider computational requirements
   - Balance accuracy vs. speed
   - Use cross-validation

3. **Parameter Tuning**
   - Use grid search or random search
   - Monitor overfitting
   - Validate results
   - Keep track of experiments

4. **Performance Optimization**
   - Use appropriate data types
   - Implement batch processing
   - Consider GPU acceleration
   - Optimize memory usage

## Applications

1. **Image Classification**
   - Object recognition
   - Texture classification
   - Document classification
   - Quality control

2. **Object Detection**
   - Face detection
   - Vehicle detection
   - Defect detection
   - Security systems

3. **Pattern Recognition**
   - Gesture recognition
   - Character recognition
   - Biometric systems
   - Anomaly detection

4. **Feature Matching**
   - Image retrieval
   - Object tracking
   - Scene recognition
   - 3D reconstruction

## Further Reading

1. [OpenCV Machine Learning Documentation](https://docs.opencv.org/master/d1/d69/tutorial_table_of_content_ml.html)
2. Deep learning frameworks integration
3. Advanced machine learning techniques
4. Research papers and case studies