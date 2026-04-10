"""
Eye Blink Detection Module
Detects unnatural blinking patterns in videos
"""

import numpy as np
from scipy.spatial import distance as dist
from utils.feature_extraction import FeatureExtractor

# Safe cv2 import - try to import but don't fail if not available
try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    _CV2_AVAILABLE = False
    print("⚠️ OpenCV not available in eye blink detection module")


class EyeBlinkDetector:
    """Detect eye blink anomalies in videos"""
    
    def __init__(self, ear_threshold=0.25, consecutive_frames=3):
        self.ear_threshold = ear_threshold
        self.consecutive_frames = consecutive_frames
        self.feature_extractor = FeatureExtractor()
        
        # Eye landmark indices (MediaPipe)
        self.left_eye_indices = [33, 133, 160, 159, 158, 157, 173]
        self.right_eye_indices = [362, 263, 385, 386, 387, 388, 466]
    
    def detect_blink_anomalies(self, video_path):
        """
        Detect unnatural blinking patterns in video
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with blink analysis results
        """
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            if fps == 0:
                return {
                    'success': False,
                    'error': 'Invalid video FPS'
                }
            
            # Extract eye aspect ratios for all frames
            left_ears = []
            right_ears = []
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                landmarks = self.feature_extractor.extract_facial_landmarks(frame)
                
                if landmarks is not None:
                    # Calculate EAR for both eyes
                    left_eye = landmarks[self.left_eye_indices]
                    right_eye = landmarks[self.right_eye_indices]
                    
                    left_ear = self.feature_extractor.calculate_eye_aspect_ratio(left_eye[:6])
                    right_ear = self.feature_extractor.calculate_eye_aspect_ratio(right_eye[:6])
                    
                    left_ears.append(left_ear)
                    right_ears.append(right_ear)
                else:
                    left_ears.append(None)
                    right_ears.append(None)
                
                frame_count += 1
            
            cap.release()
            
            if len([e for e in left_ears if e is not None]) < 10:
                return {
                    'success': False,
                    'error': 'Insufficient frames with detected faces'
                }
            
            # Detect blinks
            left_blinks = self._detect_blinks(left_ears, fps)
            right_blinks = self._detect_blinks(right_ears, fps)
            
            # Analyze blink patterns
            blink_analysis = self._analyze_blink_pattern(left_blinks, right_blinks, fps)
            
            # Detect anomalies
            anomalies = self._detect_anomalies(blink_analysis, fps)
            
            result = {
                'success': True,
                'left_eye_blinks': left_blinks,
                'right_eye_blinks': right_blinks,
                'blink_analysis': blink_analysis,
                'anomalies_detected': anomalies['has_anomalies'],
                'anomaly_details': anomalies,
                'frames_analyzed': frame_count
            }
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _detect_blinks(self, ears, fps):
        """
        Detect blinks from eye aspect ratio sequence
        
        Args:
            ears: List of EAR values
            fps: Video FPS
            
        Returns:
            List of blink events
        """
        blinks = []
        in_blink = False
        blink_start = None
        
        for i, ear in enumerate(ears):
            if ear is None:
                if in_blink:
                    # End blink if we lose tracking
                    blink_end = i - 1
                    if blink_end - blink_start >= self.consecutive_frames:
                        blinks.append({
                            'start_frame': blink_start,
                            'end_frame': blink_end,
                            'duration': (blink_end - blink_start) / fps,
                            'complete': False
                        })
                    in_blink = False
                continue
            
            if ear < self.ear_threshold and not in_blink:
                # Start of blink
                in_blink = True
                blink_start = i
            elif ear >= self.ear_threshold and in_blink:
                # End of blink
                in_blink = False
                blink_end = i
                
                if blink_end - blink_start >= self.consecutive_frames:
                    blinks.append({
                        'start_frame': blink_start,
                        'end_frame': blink_end,
                        'duration': (blink_end - blink_start) / fps,
                        'complete': True
                    })
        
        return blinks
    
    def _analyze_blink_pattern(self, left_blinks, right_blinks, fps):
        """
        Analyze overall blink pattern
        
        Args:
            left_blinks: Left eye blink events
            right_blinks: Right eye blink events
            fps: Video FPS
            
        Returns:
            Blink pattern analysis
        """
        total_blinks = len(left_blinks) + len(right_blinks)
        
        # Calculate blink rate (blinks per minute)
        if len(left_blinks) > 1:
            time_span = (left_blinks[-1]['end_frame'] - left_blinks[0]['start_frame']) / fps
            blink_rate = (len(left_blinks) / time_span) * 60 if time_span > 0 else 0
        else:
            blink_rate = 0
        
        # Average blink duration
        all_durations = [b['duration'] for b in left_blinks + right_blinks if b['complete']]
        avg_duration = np.mean(all_durations) if all_durations else 0
        
        # Synchronization between eyes
        sync_score = self._calculate_eye_synchronization(left_blinks, right_blinks)
        
        analysis = {
            'total_blinks': total_blinks,
            'left_eye_blink_count': len(left_blinks),
            'right_eye_blink_count': len(right_blinks),
            'blink_rate_per_minute': float(blink_rate),
            'average_blink_duration': float(avg_duration),
            'eye_synchronization_score': float(sync_score),
            'normal_blink_rate_range': "10-20 blinks/min",
            'normal_blink_duration_range': "0.1-0.4 seconds"
        }
        
        return analysis
    
    def _calculate_eye_synchronization(self, left_blinks, right_blinks):
        """
        Calculate synchronization between left and right eye blinks
        
        Args:
            left_blinks: Left eye blink events
            right_blinks: Right eye blink events
            
        Returns:
            Synchronization score (0-1)
        """
        if len(left_blinks) == 0 or len(right_blinks) == 0:
            return 0.0
        
        synchronized = 0
        tolerance_frames = 3  # Frames tolerance for synchronization
        
        for left_blink in left_blinks:
            for right_blink in right_blinks:
                # Check if blinks overlap or are close in time
                if abs(left_blink['start_frame'] - right_blink['start_frame']) <= tolerance_frames:
                    synchronized += 1
                    break
        
        sync_score = synchronized / max(len(left_blinks), len(right_blinks))
        return sync_score
    
    def _detect_anomalies(self, blink_analysis, fps):
        """
        Detect blink anomalies
        
        Args:
            blink_analysis: Blink pattern analysis
            fps: Video FPS
            
        Returns:
            Anomaly detection results
        """
        anomalies = {
            'has_anomalies': False,
            'details': []
        }
        
        # Check blink rate
        blink_rate = blink_analysis['blink_rate_per_minute']
        if blink_rate < 5 or blink_rate > 30:
            anomalies['has_anomalies'] = True
            anomalies['details'].append({
                'type': 'abnormal_blink_rate',
                'value': blink_rate,
                'normal_range': '10-20 blinks/min',
                'severity': 'high' if blink_rate < 3 or blink_rate > 40 else 'medium'
            })
        
        # Check blink duration
        avg_duration = blink_analysis['average_blink_duration']
        if avg_duration < 0.05 or avg_duration > 0.6:
            anomalies['has_anomalies'] = True
            anomalies['details'].append({
                'type': 'abnormal_blink_duration',
                'value': avg_duration,
                'normal_range': '0.1-0.4 seconds',
                'severity': 'medium'
            })
        
        # Check eye synchronization
        sync_score = blink_analysis['eye_synchronization_score']
        if sync_score < 0.5:
            anomalies['has_anomalies'] = True
            anomalies['details'].append({
                'type': 'poor_eye_synchronization',
                'value': sync_score,
                'normal_range': '> 0.8',
                'severity': 'high' if sync_score < 0.3 else 'medium'
            })
        
        # Check for no blinking
        if blink_analysis['total_blinks'] == 0:
            anomalies['has_anomalies'] = True
            anomalies['details'].append({
                'type': 'no_blinking_detected',
                'severity': 'high'
            })
        
        return anomalies
    
    def analyze_from_camera(self, camera_id=0, duration_seconds=10):
        """
        Analyze eye blinks from live camera feed
        
        Args:
            camera_id: Camera device ID
            duration_seconds: Recording duration
            
        Returns:
            Blink analysis results
        """
        cap = cv2.VideoCapture(camera_id)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if fps == 0:
            fps = 30  # Default FPS
        
        total_frames = int(fps * duration_seconds)
        
        print(f"Recording for {duration_seconds} seconds...")
        
        left_ears = []
        right_ears = []
        
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            landmarks = self.feature_extractor.extract_facial_landmarks(frame)
            
            if landmarks is not None:
                left_eye = landmarks[self.left_eye_indices]
                right_eye = landmarks[self.right_eye_indices]
                
                left_ear = self.feature_extractor.calculate_eye_aspect_ratio(left_eye[:6])
                right_ear = self.feature_extractor.calculate_eye_aspect_ratio(right_eye[:6])
                
                left_ears.append(left_ear)
                right_ears.append(right_ear)
            
            if i % 30 == 0:
                print(f"Frame {i}/{total_frames}")
        
        cap.release()
        
        # Analyze blinks
        left_blinks = self._detect_blinks(left_ears, fps)
        right_blinks = self._detect_blinks(right_ears, fps)
        blink_analysis = self._analyze_blink_pattern(left_blinks, right_blinks, fps)
        anomalies = self._detect_anomalies(blink_analysis, fps)
        
        result = {
            'success': True,
            'source': 'webcam',
            'duration': duration_seconds,
            'blink_analysis': blink_analysis,
            'anomalies_detected': anomalies['has_anomalies'],
            'anomaly_details': anomalies
        }
        
        return result


def test_eye_blink_detection():
    """Test eye blink detection functionality"""
    detector = EyeBlinkDetector()
    
    print("\nEye Blink Detector initialized")
    print("To analyze a video, call: detector.detect_blink_anomalies('video.mp4')")
    
    return detector


if __name__ == "__main__":
    detector = test_eye_blink_detection()
