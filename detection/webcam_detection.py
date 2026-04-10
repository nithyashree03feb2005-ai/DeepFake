"""
Webcam Real-Time DeepFake Detection Module
Detects deepfakes in real-time using webcam
"""

import os
import numpy as np

# Safe cv2 import - try to import but don't fail if not available
try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    _CV2_AVAILABLE = False
    print("⚠️ OpenCV not available in webcam detection module")

# Safe TensorFlow import - try to import but don't fail if not available
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except ImportError as e:
    tf = None
    TENSORFLOW_AVAILABLE = False
    print(f"⚠️ TensorFlow/Keras not available in webcam detection module: {e}")

from utils.preprocessing import Preprocessor
from utils.feature_extraction import FeatureExtractor


class WebcamDeepFakeDetector:
    """Real-time deepfake detection using webcam"""
    
    def __init__(self, model_path='models/image_model.h5', img_size=(224, 224)):
        self.model_path = model_path
        self.img_size = img_size
        self.model = None
        self.preprocessor = Preprocessor()
        self.feature_extractor = FeatureExtractor()
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load pretrained model"""
        if os.path.exists(self.model_path):
            try:
                self.model = load_model(self.model_path)
                print(f"✓ Webcam detection model loaded from {self.model_path}")
            except Exception as e:
                print(f"✗ Error loading model: {e}")
                self.model = None
        else:
            print(f"⚠ Model not found at {self.model_path}")
            self.model = None
    
    def detect_frame(self, frame):
        """
        Detect if frame contains fake content
        
        Args:
            frame: BGR frame from webcam
            
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
            # Detect face
            face_bbox = self.feature_extractor.detect_face(frame)
            
            if face_bbox is None:
                return {
                    'success': False,
                    'error': 'No face detected',
                    'prediction': None,
                    'confidence': None
                }
            
            # Extract face region
            x1, y1, x2, y2 = map(int, face_bbox)
            face_region = frame[y1:y2, x1:x2]
            
            if face_region.size == 0:
                return {
                    'success': False,
                    'error': 'Invalid face region',
                    'prediction': None,
                    'confidence': None
                }
            
            # Preprocess face
            face_resized = cv2.resize(face_region, self.img_size)
            face_normalized = face_resized.astype(np.float32) / 255.0
            
            # Add batch dimension
            face_batch = np.expand_dims(face_normalized, axis=0)
            
            # Predict
            prediction = self.model.predict(face_batch, verbose=0)[0][0]
            confidence = float(prediction)
            
            # Determine if fake or real
            is_fake = confidence > 0.5
            label = 'Fake' if is_fake else 'Real'
            
            result = {
                'success': True,
                'prediction': label,
                'confidence': confidence,
                'is_fake': is_fake,
                'face_bbox': (x1, y1, x2, y2),
                'real_probability': 1 - confidence if is_fake else confidence,
                'fake_probability': confidence if is_fake else 1 - confidence
            }
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'prediction': None,
                'confidence': None
            }
    
    def start_webcam_detection(self, camera_id=0, threshold=0.5):
        """
        Start real-time webcam detection
        
        Args:
            camera_id: Camera device ID
            threshold: Confidence threshold
        """
        if self.model is None:
            print("Error: Model not loaded")
            return
        
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("Error: Cannot open webcam")
            return
        
        print("\nPress 'q' to quit")
        
        frame_count = 0
        detection_results = []
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            
            # Perform detection every 5 frames for performance
            if frame_count % 5 == 0:
                result = self.detect_frame(frame)
                detection_results.append(result)
                
                if result['success']:
                    # Draw bounding box
                    x1, y1, x2, y2 = result['face_bbox']
                    
                    # Choose color based on prediction
                    color = (0, 0, 255) if result['is_fake'] else (0, 255, 0)
                    
                    # Draw rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Put text
                    label = f"{result['prediction']}: {result['confidence']:.2%}"
                    cv2.putText(frame, label, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Display frame
            cv2.imshow('DeepFake Webcam Detection', frame)
            
            # Press q to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Calculate statistics
        if len(detection_results) > 0:
            successful_detections = [r for r in detection_results if r['success']]
            if len(successful_detections) > 0:
                avg_confidence = np.mean([r['confidence'] for r in successful_detections])
                fake_count = sum([r['is_fake'] for r in successful_detections])
                
                print(f"\nDetection Summary:")
                print(f"Total frames analyzed: {len(successful_detections)}")
                print(f"Average confidence: {avg_confidence:.2%}")
                print(f"Fake detections: {fake_count}")
                print(f"Real detections: {len(successful_detections) - fake_count}")
    
    def analyze_webcam_stream(self, camera_id=0, num_frames=30):
        """
        Analyze webcam stream and return detailed results
        
        Args:
            camera_id: Camera device ID
            num_frames: Number of frames to analyze
            
        Returns:
            Detailed analysis results
        """
        if self.model is None:
            return None
        
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            return {
                'success': False,
                'error': 'Cannot open webcam'
            }
        
        print(f"Analyzing {num_frames} frames...")
        
        predictions = []
        confidences = []
        
        for i in range(num_frames):
            ret, frame = cap.read()
            
            if not ret:
                break
            
            result = self.detect_frame(frame)
            
            if result['success']:
                predictions.append(result['is_fake'])
                confidences.append(result['confidence'])
            
            # Show progress
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{num_frames} frames")
        
        cap.release()
        
        if len(predictions) == 0:
            return {
                'success': False,
                'error': 'No faces detected'
            }
        
        # Aggregate results
        avg_confidence = np.mean(confidences)
        is_fake = avg_confidence > 0.5
        
        result = {
            'success': True,
            'prediction': 'Fake' if is_fake else 'Real',
            'confidence': float(avg_confidence),
            'is_fake': is_fake,
            'frames_analyzed': len(predictions),
            'predictions': predictions,
            'confidences': confidences,
            'consistency': 1 - np.std(confidences)
        }
        
        return result


def test_webcam_detection():
    """Test webcam detection functionality"""
    detector = WebcamDeepFakeDetector()
    
    print("\nWebcam DeepFake Detector initialized")
    print(f"Model loaded: {detector.model is not None}")
    print("\nTo start webcam detection, call: detector.start_webcam_detection()")
    
    return detector


if __name__ == "__main__":
    detector = test_webcam_detection()
