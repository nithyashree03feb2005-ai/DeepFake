"""
Audio DeepFake Detection Module
Detects fake audio using CNN on Mel Spectrograms
"""

import os
import numpy as np

# Safe TensorFlow import - try to import but don't fail if not available
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except ImportError as e:
    tf = None
    TENSORFLOW_AVAILABLE = False
    print(f"⚠️ TensorFlow/Keras not available in audio detection module: {e}")

# Safe librosa import - try to import but don't fail if not available
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError as e:
    librosa = None
    LIBROSA_AVAILABLE = False
    print(f"⚠️ Librosa not available in audio detection module: {e}")

from utils.preprocessing import Preprocessor


class AudioDeepFakeDetector:
    """Detect deepfakes in audio"""
    
    def __init__(self, model_path='models/audio_model.h5', sr=16000, duration=5):
        self.model_path = model_path
        self.sr = sr
        self.duration = duration
        self.model = None
        self.preprocessor = Preprocessor()
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load pretrained model"""
        if os.path.exists(self.model_path):
            try:
                self.model = load_model(self.model_path)
                print(f"✓ Audio detection model loaded from {self.model_path}")
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
    
    def extract_spectrogram(self, audio_input):
        """
        Extract mel spectrogram from audio
        
        Args:
            audio_input: Audio file path or audio array
            
        Returns:
            Mel spectrogram
        """
        if isinstance(audio_input, str):
            # Load from file
            y, _ = librosa.load(audio_input, sr=self.sr, duration=self.duration)
        elif isinstance(audio_input, np.ndarray):
            # Use provided audio array
            y = audio_input
            # Trim to duration
            max_samples = int(self.sr * self.duration)
            if len(y) > max_samples:
                y = y[:max_samples]
            else:
                y = np.pad(y, (0, max_samples - len(y)), mode='constant')
        else:
            raise ValueError("Invalid audio input type")
        
        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
        mel_spec_db = np.expand_dims(mel_spec_db, axis=-1)
        
        return mel_spec_db
    
    def detect(self, audio_input):
        """
        Detect if audio is fake or real
        
        Args:
            audio_input: Audio file path or audio array
            
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
            # Extract spectrogram
            spectrogram = self.extract_spectrogram(audio_input)
            
            # Add batch dimension
            spec_batch = np.expand_dims(spectrogram, axis=0)
            
            # Predict
            prediction = self.model.predict(spec_batch, verbose=0)[0][0]
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
                'duration_analyzed': self.duration,
                'sample_rate': self.sr,
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
    
    def detect_with_segments(self, audio_path, segment_duration=2):
        """
        Detect deepfake by analyzing audio segments
        
        Args:
            audio_path: Path to audio file
            segment_duration: Duration of each segment in seconds
            
        Returns:
            Detailed results with segment-level predictions
        """
        if self.model is None:
            return None
        
        try:
            # Load full audio
            y, _ = librosa.load(audio_path, sr=self.sr)
            
            # Calculate number of segments
            segment_samples = int(self.sr * segment_duration)
            num_segments = len(y) // segment_samples
            
            segment_predictions = []
            segment_confidences = []
            
            for i in range(num_segments):
                start_idx = i * segment_samples
                end_idx = start_idx + segment_samples
                
                segment = y[start_idx:end_idx]
                
                # Extract spectrogram for segment
                mel_spec = librosa.feature.melspectrogram(y=segment, sr=self.sr, n_mels=128)
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
                mel_spec_db = np.expand_dims(mel_spec_db, axis=-1)
                
                # Predict
                spec_batch = np.expand_dims(mel_spec_db, axis=0)
                pred = self.model.predict(spec_batch, verbose=0)[0][0]
                
                segment_predictions.append(pred > 0.5)
                segment_confidences.append(float(pred))
            
            # Aggregate predictions
            avg_confidence = np.mean(segment_confidences)
            is_fake = avg_confidence > 0.5
            
            result = {
                'success': True,
                'prediction': 'Fake' if is_fake else 'Real',
                'confidence': float(avg_confidence),
                'is_fake': is_fake,
                'num_segments': num_segments,
                'segment_predictions': segment_predictions,
                'segment_confidences': segment_confidences,
                'temporal_consistency': 1 - np.std(segment_confidences)
            }
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'prediction': None,
                'confidence': None
            }
    
    def get_audio_features(self, audio_input):
        """
        Extract additional audio features for analysis
        
        Args:
            audio_input: Audio file path or array
            
        Returns:
            Dictionary of audio features
        """
        if isinstance(audio_input, str):
            y, _ = librosa.load(audio_input, sr=self.sr)
        else:
            y = audio_input
        
        # Extract features
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0].mean()
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=self.sr)[0].mean()
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=self.sr)[0].mean()
        mfccs = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=13)
        mfcc_mean = mfccs.mean(axis=1)
        
        features = {
            'zero_crossing_rate': float(zero_crossing_rate),
            'spectral_centroid': float(spectral_centroid),
            'spectral_rolloff': float(spectral_rolloff),
            'mfccs': mfcc_mean.tolist(),
            'duration': len(y) / self.sr
        }
        
        return features


def test_audio_detection():
    """Test audio detection functionality"""
    detector = AudioDeepFakeDetector()
    
    print("\nAudio DeepFake Detector initialized")
    print(f"Model loaded: {detector.model is not None}")
    
    return detector


if __name__ == "__main__":
    detector = test_audio_detection()
