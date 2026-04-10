"""
Image DeepFake Detection Module
Detects fake images using pretrained CNN models
"""

import os
import numpy as np

# Safe TensorFlow import - try to import but don't fail if not available
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
    print("✓ TensorFlow and Keras loaded successfully")
except ImportError as e:
    tf = None
    TENSORFLOW_AVAILABLE = False
    print(f"⚠️ TensorFlow/Keras not available: {e}")
    print("⚠️ AI model features will not work")

from utils.preprocessing import Preprocessor
from utils.helpers import load_model_weights, create_color_mapping

# Safe cv2 import - try to import but don't fail if not available
try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    _CV2_AVAILABLE = False
    print("⚠️ OpenCV not available in image detection module")


class ImageDeepFakeDetector:
    """Detect deepfakes in images"""
    
    def __init__(self, model_path='models/image_model.h5', img_size=(150, 150)):
        self.model_path = model_path
        self.img_size = img_size
        self.model = None
        self.preprocessor = Preprocessor()
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load pretrained model"""
        if os.path.exists(self.model_path):
            try:
                self.model = load_model(self.model_path)
                print(f"✓ Image detection model loaded from {self.model_path}")
            except Exception as e:
                print(f"✗ Error loading model: {e}")
                self.model = None
        else:
            print(f"⚠ Model not found at {self.model_path}")
            self.model = None
    
    def detect(self, image_input):
        """
        Detect if image is fake or real
        
        Args:
            image_input: Can be file path (str), URL, or numpy array
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            return {
                'success': False,
                'error': 'Model not loaded',
                'prediction': None,
                'confidence': None
            }
        
        try:
            # Load and preprocess image
            if isinstance(image_input, str):
                if os.path.exists(image_input):
                    img = self.preprocessor.preprocess_image(image_input, self.img_size)
                else:
                    return {
                        'success': False,
                        'error': f'File not found: {image_input}',
                        'prediction': None,
                        'confidence': None
                    }
            elif isinstance(image_input, np.ndarray):
                # Check if cv2 is available
                if not _CV2_AVAILABLE or cv2 is None:
                    return {
                        'success': False,
                        'error': 'OpenCV (cv2) is required for array input but not installed',
                        'prediction': None,
                        'confidence': None
                    }
                img = cv2.resize(image_input, self.img_size)
                img = img.astype(np.float32) / 255.0
            else:
                return {
                    'success': False,
                    'error': 'Invalid input type',
                    'prediction': None,
                    'confidence': None
                }
            
            # Add batch dimension
            img_batch = np.expand_dims(img, axis=0)
            
            # Predict
            prediction = self.model.predict(img_batch, verbose=0)[0][0]
            
            # Calculate probabilities
            # Model outputs probability of being FAKE (sigmoid activation)
            fake_probability = float(prediction)
            real_probability = 1 - fake_probability
            
            # Determine if fake or real based on threshold
            is_fake = fake_probability > 0.5
            label = 'Fake' if is_fake else 'Real'
            
            # Confidence is the probability of the predicted class
            confidence = fake_probability if is_fake else real_probability
            
            result = {
                'success': True,
                'prediction': label,
                'confidence': confidence,
                'is_fake': is_fake,
                'real_probability': real_probability,
                'fake_probability': fake_probability
            }
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'prediction': None,
                'confidence': None
            }
    
    def detect_batch(self, image_paths):
        """
        Detect deepfakes in multiple images
        
        Args:
            image_paths: List of image paths or arrays
            
        Returns:
            List of prediction results
        """
        results = []
        
        for img_input in image_paths:
            result = self.detect(img_input)
            results.append(result)
        
        return results
    
    def get_confidence_color(self, confidence):
        """Get color based on confidence (red for low, green for high)"""
        return create_color_mapping(confidence)


def test_image_detection():
    """Test image detection functionality"""
    detector = ImageDeepFakeDetector()
    
    # Test with sample image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = detector.detect(test_image)
    
    print("\nTest Result:")
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}" if result['confidence'] else "No confidence")
    print(f"Success: {result['success']}")
    
    return detector


if __name__ == "__main__":
    detector = test_image_detection()
