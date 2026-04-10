"""
Video DeepFake Detection Module
Detects fake videos using CNN + LSTM models
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
    print("⚠️ OpenCV not available in video detection module")

# Safe TensorFlow import - try to import but don't fail if not available
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except ImportError as e:
    tf = None
    TENSORFLOW_AVAILABLE = False
    print(f"⚠️ TensorFlow/Keras not available in video detection module: {e}")

# Safe tqdm import - try to import but don't fail if not available
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    tqdm = None
    TQDM_AVAILABLE = False
    print("⚠️ tqdm not available in video detection module")

from utils.preprocessing import Preprocessor


class VideoDeepFakeDetector:
    """Detect deepfakes in videos"""
    
    def __init__(self, model_path='models/video_model_fast.h5', img_size=(224, 224), max_frames=30):
        self.model_path = model_path
        self.img_size = img_size
        self.max_frames = max_frames
        self.model = None
        self.preprocessor = Preprocessor()
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load pretrained model"""
        if os.path.exists(self.model_path):
            try:
                self.model = load_model(self.model_path)
                print(f"✓ Video detection model loaded from {self.model_path}")
            except Exception as e:
                print(f"✗ Error loading model: {e}")
                self.model = None
        else:
            print(f"⚠ Model not found at {self.model_path}")
            self.model = None
    
    def _calibrate_confidence_mild(self, raw_score):
        """
        Apply MILD calibration to balance between false positives and false negatives.
        
        Uses lower temperature (1.5) to preserve more of the original signal
        while still preventing extreme overconfidence.
        
        Args:
            raw_score: Raw sigmoid output (0-1)
            
        Returns:
            Mildly calibrated confidence score
        """
        # Apply mild temperature scaling
        # Temperature = 1.5 provides balance - reduces overconfidence but keeps signal
        temperature = 1.5
        
        # Convert probability to logit
        epsilon = 1e-7
        raw_score = np.clip(raw_score, epsilon, 1 - epsilon)
        logit = np.log(raw_score / (1 - raw_score))
        
        # Apply temperature scaling
        scaled_logit = logit / temperature
        
        # Convert back to probability
        calibrated = 1 / (1 + np.exp(-scaled_logit))
        
        return calibrated
    
    def _calibrate_confidence(self, raw_score):
        """Legacy method - use _calibrate_confidence_mild instead"""
        return self._calibrate_confidence_mild(raw_score)
    
    def extract_frames(self, video_path, num_frames=None):
        """
        Extract frames from video
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract
            
        Returns:
            Array of frames
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            cap.release()
            raise ValueError("Video has no frames")
            
        if num_frames is None:
            num_frames = min(self.max_frames, total_frames)
        
        # Ensure we don't try to extract more frames than available
        num_frames = min(num_frames, total_frames)
        
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        frames = []
        
        for idx in tqdm(frame_indices, desc="Extracting frames", leave=False):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
                frame = self.preprocessor.preprocess_frame(frame, self.img_size)
                frames.append(frame)
            else:
                print(f"Warning: Could not read frame at index {idx}")
        
        cap.release()
        
        if len(frames) == 0:
            raise ValueError("No frames extracted from video")
        
        return np.array(frames)
    
    def detect(self, video_input):
        """
        Detect if video is fake or real
        
        Args:
            video_input: Video file path or frame sequence
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            return {
                'success': False,
                'error': 'Model not loaded',
                'prediction': None,
                'confidence': None,
                'frame_predictions': None
            }
        
        try:
            # Extract frames
            if isinstance(video_input, str):
                frames = self.extract_frames(video_input)
            elif isinstance(video_input, (list, np.ndarray)):
                frames = video_input
                if isinstance(frames, list):
                    frames = np.array([self.preprocessor.preprocess_frame(f, self.img_size) for f in frames])
            else:
                return {
                    'success': False,
                    'error': 'Invalid input type',
                    'prediction': None,
                    'confidence': None
                }
            
            # Ensure correct number of frames
            current_frames = frames.shape[0]
            
            if current_frames < self.max_frames:
                # Pad with zero frames
                padding = np.zeros((self.max_frames - current_frames, *frames.shape[1:]))
                frames = np.vstack([frames, padding])
                print(f"Padded video from {current_frames} to {self.max_frames} frames")
            elif current_frames > self.max_frames:
                # Take evenly spaced frames
                indices = np.linspace(0, current_frames - 1, self.max_frames, dtype=int)
                frames = frames[indices]
                print(f"Sampled {self.max_frames} frames from {current_frames} total frames")
            
            # Verify final shape
            if frames.shape[0] != self.max_frames:
                raise ValueError(f"Expected {self.max_frames} frames, got {frames.shape[0]}")
            
            # Add batch dimension (model expects: [batch, frames, height, width, channels])
            frames_batch = np.expand_dims(frames, axis=0)
            
            print(f"Input shape to model: {frames_batch.shape}")
            
            # Predict
            prediction = self.model.predict(frames_batch, verbose=0)[0][0]
            raw_confidence = float(prediction)
            
            # Smart detection: Use raw score directly since model is undertrained
            # Models trained on limited data produce scores close to 0.5
            # We need to interpret small deviations from 0.5 as meaningful signals
            
            raw_confidence = float(prediction)
            
            # Decision logic based on how far from random (0.5)
            # If model is uncertain (< 0.52 or > 0.48), consider it LOW confidence
            # But still make a prediction based on which side of 0.5
            
            is_fake = raw_confidence > 0.50
            label = 'Fake' if is_fake else 'Real'
            
            # Calculate confidence metrics
            # Distance from 0.5 indicates model certainty
            distance_from_random = abs(raw_confidence - 0.50)
            
            # Reliability based on how confident the model is
            if distance_from_random > 0.3:
                reliability = 'high'
                confidence_level = raw_confidence  # Trust the score
            elif distance_from_random > 0.15:
                reliability = 'medium'
                confidence_level = raw_confidence
            else:
                reliability = 'low'
                # Model is very uncertain - show this in confidence
                # Scale confidence to reflect uncertainty
                confidence_level = 0.50 + (distance_from_random * 0.5)  # Dampen extreme values
                if raw_confidence < 0.50:
                    confidence_level = 1.0 - confidence_level
            
            fake_prob = confidence_level if is_fake else 1 - confidence_level
            real_prob = 1 - fake_prob
            
            result = {
                'success': True,
                'prediction': label,
                'confidence': float(confidence_level),
                'is_fake': is_fake,
                'frames_analyzed': len(frames),
                'real_probability': float(real_prob),
                'fake_probability': float(fake_prob),
                'raw_score': float(raw_confidence),
                'reliability': reliability,
                'note': 'Model shows low confidence - predictions near 50% indicate uncertainty' if reliability == 'low' else ''
            }
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'prediction': None,
                'confidence': None
            }
    
    def detect_with_frame_analysis(self, video_path):
        """
        Detect deepfake with per-frame analysis
        
        Args:
            video_path: Path to video file
            
        Returns:
            Detailed results with frame-level predictions
        """
        if self.model is None:
            return None
        
        try:
            # Extract all frames
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            frame_predictions = []
            frame_confidences = []
            
            # Process frames in batches
            batch_size = self.max_frames
            frame_idx = 0
            
            while True:
                frames = []
                for _ in range(batch_size):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append(self.preprocessor.preprocess_frame(frame, self.img_size))
                
                if len(frames) == 0:
                    break
                
                # Pad to batch_size if needed
                if len(frames) < batch_size:
                    padding = np.zeros((batch_size - len(frames), *frames[0].shape))
                    frames.extend(padding.tolist())
                
                frames_array = np.array(frames)
                frames_array = np.expand_dims(frames_array, axis=0)
                
                # Predict
                pred = self.model.predict(frames_array, verbose=0)[0][0]
                
                frame_predictions.append(pred > 0.5)
                frame_confidences.append(float(pred))
                
                frame_idx += len(frames)
            
            cap.release()
            
            # Aggregate predictions
            avg_confidence = np.mean(frame_confidences)
            is_fake = avg_confidence > 0.5
            
            result = {
                'success': True,
                'prediction': 'Fake' if is_fake else 'Real',
                'confidence': float(avg_confidence),
                'is_fake': is_fake,
                'total_frames': frame_idx,
                'frame_predictions': frame_predictions,
                'frame_confidences': frame_confidences,
                'temporal_consistency': 1 - np.std(frame_confidences)
            }
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'prediction': None,
                'confidence': None
            }


def test_video_detection():
    """Test video detection functionality"""
    detector = VideoDeepFakeDetector()
    
    print("\nVideo DeepFake Detector initialized")
    print(f"Model loaded: {detector.model is not None}")
    
    return detector


if __name__ == "__main__":
    detector = test_video_detection()
