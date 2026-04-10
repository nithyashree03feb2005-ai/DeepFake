"""
Enhanced Audio DeepFake Detection Module
Improved detection with better preprocessing and segment analysis
"""

import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')


class EnhancedAudioDeepFakeDetector:
    """Enhanced deepfake detector for audio with improved accuracy"""
    
    def __init__(self, model_path='models/audio_model.h5', sr=16000, duration=5):
        self.model_path = model_path
        self.sr = sr
        self.duration = duration
        self.model = None
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load pretrained model"""
        if os.path.exists(self.model_path):
            try:
                self.model = load_model(self.model_path)
                print(f"✅ Audio detection model loaded from {self.model_path}")
            except Exception as e:
                print(f"❌ Error loading model: {e}")
                self.model = None
        else:
            print(f"⚠️ Model not found at {self.model_path}")
            self.model = None
    
    def extract_spectrogram(self, audio_input, duration=None):
        """
        Extract mel spectrogram from audio with improved preprocessing
        
        Args:
            audio_input: Audio file path or audio array
            duration: Override default duration
            
        Returns:
            Mel spectrogram
        """
        dur = duration if duration else self.duration
        
        if isinstance(audio_input, str):
            # Load from file
            y, _ = librosa.load(audio_input, sr=self.sr, duration=dur)
        elif isinstance(audio_input, np.ndarray):
            # Use provided audio array
            y = audio_input
            # Trim to duration
            max_samples = int(self.sr * dur)
            if len(y) > max_samples:
                y = y[:max_samples]
            else:
                y = np.pad(y, (0, max_samples - len(y)), mode='constant')
        else:
            raise ValueError("Invalid audio input type")
        
        # Extract mel spectrogram with optimized parameters
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=self.sr, 
            n_mels=128,
            fmin=20,
            fmax=self.sr//2,
            n_fft=2048,
            hop_length=512
        )
        
        # Convert to dB
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to 0-1 range
        mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
        
        # Expand dimensions for model input
        mel_spec_db = np.expand_dims(mel_spec_db, axis=-1)
        
        return mel_spec_db
    
    def detect(self, audio_input, use_segments=True):
        """
        Enhanced detection with segment analysis
        
        Args:
            audio_input: Audio file path or audio array
            use_segments: Whether to use segment analysis for better accuracy
            
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
            print("\n🎵 Starting audio analysis...")
            
            # Step 1: Full audio analysis
            print("   📊 Analyzing full audio...")
            spectrogram = self.extract_spectrogram(audio_input)
            spec_batch = np.expand_dims(spectrogram, axis=0)
            
            full_prediction = self.model.predict(spec_batch, verbose=0)[0][0]
            print(f"   Full audio score: {full_prediction:.4f}")
            
            # Step 2: Segment analysis (if enabled and we have a file)
            if use_segments and isinstance(audio_input, str) and os.path.exists(audio_input):
                print("   🔍 Performing segment analysis...")
                segment_result = self._analyze_segments(audio_input)
                
                if segment_result:
                    # Combine full and segment predictions
                    combined_score = (full_prediction + segment_result['avg_score']) / 2
                    print(f"   Combined score: {combined_score:.4f}")
                    
                    final_score = combined_score
                    used_segments = True
                else:
                    final_score = full_prediction
                    used_segments = False
            else:
                final_score = full_prediction
                used_segments = False
            
            # Step 3: Make decision with improved thresholds
            if final_score > 0.60:
                is_fake = True
                confidence = final_score
                reliability = 'high' if final_score > 0.75 else 'medium'
            elif final_score < 0.40:
                is_fake = False
                confidence = 1 - final_score
                reliability = 'high' if final_score < 0.25 else 'medium'
            else:
                # Uncertain region - use segment info if available
                if used_segments:
                    # Trust segment analysis
                    is_fake = final_score > 0.5
                    confidence = max(final_score, 1 - final_score)
                    reliability = 'medium'
                else:
                    # Single prediction uncertainty
                    is_fake = final_score > 0.5
                    confidence = 0.5 + abs(final_score - 0.5) * 0.8
                    reliability = 'low'
            
            label = 'Fake' if is_fake else 'Real'
            
            # Calculate probabilities
            fake_prob = confidence if is_fake else 1 - confidence
            real_prob = 1 - fake_prob
            
            result = {
                'success': True,
                'prediction': label,
                'confidence': float(confidence),
                'is_fake': is_fake,
                'duration_analyzed': self.duration,
                'sample_rate': self.sr,
                'real_probability': float(real_prob),
                'fake_probability': float(fake_prob),
                'raw_score': float(final_score),
                'reliability': reliability,
                'used_segments': used_segments,
                'note': ''
            }
            
            print(f"\n✅ Analysis complete!")
            print(f"   Result: {label}")
            print(f"   Confidence: {confidence*100:.1f}%")
            print(f"   Reliability: {reliability}")
            
            return result
            
        except Exception as e:
            print(f"   ❌ Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'prediction': None,
                'confidence': None
            }
    
    def _analyze_segments(self, audio_path, segment_duration=2):
        """
        Analyze audio in segments for more robust detection
        
        Args:
            audio_path: Path to audio file
            segment_duration: Duration of each segment
            
        Returns:
            Dictionary with segment analysis results
        """
        try:
            # Load full audio
            y, _ = librosa.load(audio_path, sr=self.sr)
            
            # Calculate segments
            segment_samples = int(self.sr * segment_duration)
            num_segments = max(1, len(y) // segment_samples)
            
            if num_segments < 2:
                # Audio too short for segments
                return None
            
            segment_scores = []
            
            for i in range(num_segments):
                start_idx = i * segment_samples
                end_idx = min(start_idx + segment_samples, len(y))
                
                segment = y[start_idx:end_idx]
                
                # Skip silent segments
                if np.abs(segment).max() < 0.01:
                    continue
                
                # Extract features
                mel_spec = self.extract_spectrogram(segment, duration=segment_duration)
                spec_batch = np.expand_dims(mel_spec, axis=0)
                
                # Predict
                pred = self.model.predict(spec_batch, verbose=0)[0][0]
                segment_scores.append(pred)
            
            if not segment_scores:
                return None
            
            avg_score = np.mean(segment_scores)
            std_score = np.std(segment_scores)
            
            return {
                'avg_score': avg_score,
                'std_score': std_score,
                'num_segments': len(segment_scores),
                'scores': segment_scores
            }
            
        except Exception as e:
            print(f"   Segment analysis failed: {str(e)}")
            return None
    
    def get_detailed_analysis(self, audio_path):
        """
        Get detailed audio analysis with multiple features
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Comprehensive analysis dictionary
        """
        try:
            # Load audio
            y, _ = librosa.load(audio_path, sr=self.sr)
            
            # Basic stats
            duration = len(y) / self.sr
            rms_energy = librosa.feature.rms(y=y)[0].mean()
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0].mean()
            
            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=self.sr)[0].mean()
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=self.sr)[0].mean()
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=self.sr)[0].mean()
            
            # MFCCs
            mfccs = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=13)
            mfcc_mean = mfccs.mean(axis=1)
            
            # Run detection
            detection_result = self.detect(audio_path, use_segments=True)
            
            return {
                'audio_properties': {
                    'duration_seconds': duration,
                    'sample_rate': self.sr,
                    'rms_energy': float(rms_energy),
                    'zero_crossing_rate': float(zero_crossing_rate),
                    'spectral_centroid': float(spectral_centroid),
                    'spectral_bandwidth': float(spectral_bandwidth),
                    'spectral_rolloff': float(spectral_rolloff)
                },
                'mfcc_features': mfcc_mean.tolist(),
                'detection': detection_result
            }
            
        except Exception as e:
            return {
                'error': str(e)
            }


# Test function
def test_enhanced_detector():
    """Test the enhanced detector"""
    detector = EnhancedAudioDeepFakeDetector()
    print("\n✅ Enhanced Audio Detector initialized")
    print(f"Model loaded: {detector.model is not None}")
    return detector


if __name__ == "__main__":
    detector = test_enhanced_detector()
