"""
Lip-Sync Mismatch Detection Module
Detects inconsistencies between lip movements and audio
"""

import numpy as np
import librosa
from scipy import signal
from utils.feature_extraction import FeatureExtractor

# Safe cv2 import - try to import but don't fail if not available
try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    _CV2_AVAILABLE = False
    print("⚠️ OpenCV not available in lip sync detection module")


class LipSyncDetector:
    """Detect lip-sync mismatches in videos"""
    
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.lip_indices = list(range(61, 68)) + list(range(291, 298))
    
    def detect_lip_sync_mismatch(self, video_path, audio_path=None):
        """
        Detect lip-sync mismatch in video
        
        Args:
            video_path: Path to video file
            audio_path: Path to audio file (optional, extracts from video if None)
            
        Returns:
            Dictionary with lip-sync analysis results
        """
        try:
            # Extract video frames
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Extract audio
            if audio_path:
                y, sr = librosa.load(audio_path)
            else:
                # Extract audio from video (placeholder - requires moviepy or similar)
                print("Audio extraction from video requires additional libraries")
                return {
                    'success': False,
                    'error': 'Audio extraction not implemented',
                    'mismatch_detected': False
                }
            
            # Extract lip movements
            lip_movements = []
            frame_times = []
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                landmarks = self.feature_extractor.extract_facial_landmarks(frame)
                
                if landmarks is not None:
                    # Calculate mouth aspect ratio
                    mouth_points = landmarks[self.lip_indices]
                    mouth_ar = self._calculate_mouth_aspect_ratio(mouth_points)
                    lip_movements.append(mouth_ar)
                    frame_times.append(frame_count / fps)
                
                frame_count += 1
            
            cap.release()
            
            if len(lip_movements) == 0:
                return {
                    'success': False,
                    'error': 'No face detected in video',
                    'mismatch_detected': False
                }
            
            # Convert to numpy arrays
            lip_movements = np.array(lip_movements)
            
            # Extract audio features at same time intervals
            audio_features = self._extract_audio_envelope(y, sr, frame_times)
            
            if len(audio_features) != len(lip_movements):
                min_len = min(len(audio_features), len(lip_movements))
                audio_features = audio_features[:min_len]
                lip_movements = lip_movements[:min_len]
            
            # Calculate correlation
            correlation = self._calculate_correlation(lip_movements, audio_features)
            
            # Detect mismatch
            mismatch_threshold = 0.3  # Adjust based on testing
            mismatch_detected = correlation < mismatch_threshold
            
            result = {
                'success': True,
                'correlation': float(correlation),
                'mismatch_detected': mismatch_detected,
                'frames_analyzed': len(lip_movements),
                'audio_duration': len(y) / sr,
                'confidence': float(1 - correlation) if mismatch_detected else float(correlation)
            }
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'mismatch_detected': False
            }
    
    def _calculate_mouth_aspect_ratio(self, mouth_points):
        """Calculate mouth aspect ratio"""
        # Vertical distance
        top_lip = mouth_points[2:4]  # Upper lip
        bottom_lip = mouth_points[5:7]  # Lower lip
        
        vertical_dist = np.mean(np.linalg.norm(top_lip - bottom_lip[0], axis=1))
        
        # Horizontal distance
        left_corner = mouth_points[0]
        right_corner = mouth_points[4]
        horizontal_dist = np.linalg.norm(left_corner - right_corner)
        
        if horizontal_dist > 0:
            return vertical_dist / horizontal_dist
        return 0
    
    def _extract_audio_envelope(self, y, sr, times):
        """
        Extract audio amplitude envelope
        
        Args:
            y: Audio signal
            sr: Sample rate
            times: Time points
            
        Returns:
            Amplitude values at specified times
        """
        # Compute short-time energy
        hop_length = 512
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        
        # Get time axis for RMS
        rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
        
        # Interpolate to match video frame times
        audio_envelope = np.interp(times, rms_times, rms)
        
        return audio_envelope
    
    def _calculate_correlation(self, lip_movements, audio_envelope):
        """
        Calculate correlation between lip movements and audio
        
        Args:
            lip_movements: Mouth aspect ratio sequence
            audio_envelope: Audio amplitude envelope
            
        Returns:
            Correlation coefficient
        """
        # Normalize signals
        lip_norm = (lip_movements - np.mean(lip_movements)) / (np.std(lip_movements) + 1e-8)
        audio_norm = (audio_envelope - np.mean(audio_envelope)) / (np.std(audio_envelope) + 1e-8)
        
        # Cross-correlation
        correlation = np.corrcoef(lip_norm, audio_norm)[0, 1]
        
        # Handle NaN
        if np.isnan(correlation):
            correlation = 0.0
        
        return abs(correlation)  # Use absolute value
    
    def analyze_phoneme_viseme_alignment(self, video_path):
        """
        Analyze alignment between phonemes (audio) and visemes (visual)
        
        Args:
            video_path: Path to video
            
        Returns:
            Phoneme-viseme alignment analysis
        """
        # This is a placeholder for advanced phoneme-viseme analysis
        # Full implementation would require:
        # 1. Speech recognition to get phonemes
        # 2. Viseme classification from lip movements
        # 3. Temporal alignment analysis
        
        return {
            'success': False,
            'error': 'Phoneme-viseme analysis requires additional models',
            'note': 'This is a placeholder for future implementation'
        }
    
    def detect_temporal_drift(self, video_path):
        """
        Detect temporal drift between audio and video
        
        Args:
            video_path: Path to video
            
        Returns:
            Drift analysis results
        """
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Extract lip movements
            lip_movements = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                landmarks = self.feature_extractor.extract_facial_landmarks(frame)
                
                if landmarks is not None:
                    mouth_points = landmarks[self.lip_indices]
                    mouth_ar = self._calculate_mouth_aspect_ratio(mouth_points)
                    lip_movements.append(mouth_ar)
            
            cap.release()
            
            if len(lip_movements) == 0:
                return {'success': False, 'error': 'No face detected'}
            
            # Analyze timing consistency
            lip_diff = np.diff(lip_movements)
            
            # Calculate variance in movement timing
            movement_peaks = self._find_peaks(np.abs(lip_diff))
            
            if len(movement_peaks) < 2:
                return {
                    'success': True,
                    'drift_detected': False,
                    'note': 'Insufficient lip movement for drift analysis'
                }
            
            # Calculate inter-peak intervals
            intervals = np.diff(movement_peaks)
            interval_std = np.std(intervals)
            
            # High variance suggests temporal inconsistency
            drift_detected = interval_std > np.mean(intervals) * 0.5
            
            result = {
                'success': True,
                'drift_detected': drift_detected,
                'interval_std': float(interval_std),
                'num_movements': len(movement_peaks),
                'temporal_consistency': float(1.0 / (1.0 + interval_std))
            }
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _find_peaks(self, signal_array, threshold=None):
        """Find peaks in signal"""
        if threshold is None:
            threshold = np.mean(signal_array) + np.std(signal_array)
        
        peaks = []
        for i in range(1, len(signal_array) - 1):
            if signal_array[i] > signal_array[i-1] and signal_array[i] > signal_array[i+1]:
                if signal_array[i] > threshold:
                    peaks.append(i)
        
        return np.array(peaks)


def test_lipsync_detection():
    """Test lip-sync detection functionality"""
    detector = LipSyncDetector()
    
    print("\nLip-Sync Detector initialized")
    print("Note: Full functionality requires audio extraction from video")
    
    return detector


if __name__ == "__main__":
    detector = test_lipsync_detection()
