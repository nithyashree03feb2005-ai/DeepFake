"""
Facial Landmark Analysis Module
Analyzes facial landmarks for deepfake detection
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
    print("⚠️ OpenCV not available in facial landmarks module")


class FacialLandmarkAnalyzer:
    """Analyze facial landmarks for inconsistencies"""
    
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        
        # Define landmark indices for specific features
        self.left_eye_indices = [33, 133, 160, 159, 158, 157, 173]
        self.right_eye_indices = [362, 263, 385, 386, 387, 388, 466]
        self.mouth_indices = list(range(61, 68)) + list(range(291, 298))
        self.nose_indices = [1, 4, 5, 274, 275, 281, 44, 45, 51, 220]
    
    def analyze_landmarks(self, image_bgr):
        """
        Analyze facial landmarks in image
        
        Args:
            image_bgr: BGR image from OpenCV
            
        Returns:
            Dictionary with landmark analysis results
        """
        # Extract landmarks
        landmarks = self.feature_extractor.extract_facial_landmarks(image_bgr)
        
        if landmarks is None:
            return {
                'success': False,
                'error': 'No face detected',
                'landmarks': None
            }
        
        # Calculate various metrics
        eye_asymmetry = self.calculate_eye_asymmetry(landmarks)
        mouth_symmetry = self.calculate_mouth_symmetry(landmarks)
        nose_symmetry = self.calculate_nose_symmetry(landmarks)
        facial_proportions = self.calculate_facial_proportions(landmarks)
        
        # Overall symmetry score
        symmetry_score = (eye_asymmetry['score'] + mouth_symmetry['score'] + 
                         nose_symmetry['score']) / 3
        
        result = {
            'success': True,
            'landmarks': landmarks,
            'eye_asymmetry': eye_asymmetry,
            'mouth_symmetry': mouth_symmetry,
            'nose_symmetry': nose_symmetry,
            'facial_proportions': facial_proportions,
            'overall_symmetry_score': float(symmetry_score),
            'anomaly_detected': symmetry_score < 0.85  # Threshold for anomaly
        }
        
        return result
    
    def calculate_eye_asymmetry(self, landmarks):
        """
        Calculate eye asymmetry
        
        Args:
            landmarks: Facial landmarks
            
        Returns:
            Eye asymmetry metrics
        """
        left_eye = landmarks[self.left_eye_indices]
        right_eye = landmarks[self.right_eye_indices]
        
        # Calculate eye aspect ratios
        left_ear = self.feature_extractor.calculate_eye_aspect_ratio(left_eye[:6])
        right_ear = self.feature_extractor.calculate_eye_aspect_ratio(right_eye[:6])
        
        # Calculate eye sizes
        left_size = np.linalg.norm(left_eye[0] - left_eye[3])
        right_size = np.linalg.norm(right_eye[0] - right_eye[3])
        
        # Asymmetry score (1.0 = perfectly symmetric)
        ear_diff = abs(left_ear - right_ear) / max(left_ear, right_ear)
        size_diff = abs(left_size - right_size) / max(left_size, right_size)
        
        symmetry_score = 1.0 - (ear_diff + size_diff) / 2
        
        return {
            'left_ear': float(left_ear),
            'right_ear': float(right_ear),
            'ear_difference': float(ear_diff),
            'left_size': float(left_size),
            'right_size': float(right_size),
            'size_difference': float(size_diff),
            'symmetry_score': float(symmetry_score)
        }
    
    def calculate_mouth_symmetry(self, landmarks):
        """
        Calculate mouth symmetry
        
        Args:
            landmarks: Facial landmarks
            
        Returns:
            Mouth symmetry metrics
        """
        mouth_points = landmarks[self.mouth_indices]
        
        # Split into left and right halves
        mid_point = len(mouth_points) // 2
        left_half = mouth_points[:mid_point]
        right_half = mouth_points[mid_point:]
        
        # Calculate distances from center
        mouth_center = np.mean(mouth_points, axis=0)
        
        left_distances = [np.linalg.norm(p - mouth_center) for p in left_half]
        right_distances = [np.linalg.norm(p - mouth_center) for p in right_half]
        
        # Symmetry score
        avg_left = np.mean(left_distances)
        avg_right = np.mean(right_distances)
        symmetry_score = 1.0 - abs(avg_left - avg_right) / max(avg_left, avg_right)
        
        return {
            'left_side_size': float(avg_left),
            'right_side_size': float(avg_right),
            'symmetry_score': float(symmetry_score)
        }
    
    def calculate_nose_symmetry(self, landmarks):
        """
        Calculate nose symmetry
        
        Args:
            landmarks: Facial landmarks
            
        Returns:
            Nose symmetry metrics
        """
        nose_points = landmarks[self.nose_indices]
        
        # Find nose center line
        nose_bridge = nose_points[:3]
        nose_tip = nose_points[3]
        
        # Calculate symmetry
        nose_center_x = np.mean(nose_bridge[:, 0])
        
        left_points = nose_points[nose_points[:, 0] < nose_center_x]
        right_points = nose_points[nose_points[:, 0] > nose_center_x]
        
        if len(left_points) == 0 or len(right_points) == 0:
            return {'symmetry_score': 0.5}
        
        left_area = self._calculate_convex_hull_area(left_points[:, :2])
        right_area = self._calculate_convex_hull_area(right_points[:, :2])
        
        symmetry_score = 1.0 - abs(left_area - right_area) / max(left_area, right_area)
        
        return {
            'left_area': float(left_area),
            'right_area': float(right_area),
            'symmetry_score': float(symmetry_score)
        }
    
    def calculate_facial_proportions(self, landmarks):
        """
        Calculate facial proportions (golden ratio analysis)
        
        Args:
            landmarks: Facial landmarks
            
        Returns:
            Facial proportion metrics
        """
        # Key points
        chin = landmarks[152]
        nose_tip = landmarks[1]
        left_eye_center = np.mean(landmarks[self.left_eye_indices], axis=0)
        right_eye_center = np.mean(landmarks[self.right_eye_indices], axis=0)
        mouth_center = landmarks[13]
        
        # Distances
        face_height = np.linalg.norm(chin - nose_tip)
        eye_distance = np.linalg.norm(left_eye_center - right_eye_center)
        nose_to_mouth = np.linalg.norm(nose_tip - mouth_center)
        
        # Ratios
        eye_to_face_ratio = eye_distance / face_height if face_height > 0 else 0
        nose_to_eye_ratio = nose_to_mouth / eye_distance if eye_distance > 0 else 0
        
        proportions = {
            'face_height': float(face_height),
            'eye_distance': float(eye_distance),
            'nose_to_mouth_distance': float(nose_to_mouth),
            'eye_to_face_ratio': float(eye_to_face_ratio),
            'nose_to_eye_ratio': float(nose_to_eye_ratio)
        }
        
        return proportions
    
    def _calculate_convex_hull_area(self, points):
        """Calculate convex hull area of 2D points"""
        from scipy.spatial import ConvexHull
        
        if len(points) < 3:
            return 0.0
        
        try:
            hull = ConvexHull(points)
            return hull.volume  # In 2D, volume is area
        except:
            return 0.0
    
    def detect_landmark_anomalies(self, landmarks_sequence):
        """
        Detect anomalies in sequence of landmarks (for video)
        
        Args:
            landmarks_sequence: List of landmark arrays
            
        Returns:
            Anomaly detection results
        """
        if len(landmarks_sequence) < 2:
            return {'success': False, 'error': 'Insufficient frames'}
        
        # Calculate frame-to-frame differences
        differences = []
        
        for i in range(1, len(landmarks_sequence)):
            prev_landmarks = landmarks_sequence[i-1]
            curr_landmarks = landmarks_sequence[i]
            
            diff = np.mean(np.linalg.norm(curr_landmarks - prev_landmarks, axis=1))
            differences.append(diff)
        
        # Statistics
        mean_diff = np.mean(differences)
        std_diff = np.std(differences)
        max_diff = np.max(differences)
        
        # Detect sudden jumps (potential artifacts)
        threshold = mean_diff + 2 * std_diff
        anomalies = [i for i, diff in enumerate(differences) if diff > threshold]
        
        result = {
            'success': True,
            'mean_difference': float(mean_diff),
            'std_difference': float(std_diff),
            'max_difference': float(max_diff),
            'num_anomalies': len(anomalies),
            'anomaly_indices': anomalies,
            'temporal_consistency': float(1.0 / (1.0 + std_diff))
        }
        
        return result


def test_landmark_analysis():
    """Test facial landmark analysis"""
    analyzer = FacialLandmarkAnalyzer()
    
    # Create test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    print("\nTesting Facial Landmark Analyzer...")
    result = analyzer.analyze_landmarks(test_image)
    
    print(f"Success: {result['success']}")
    if result['success']:
        print(f"Overall symmetry score: {result['overall_symmetry_score']:.2f}")
        print(f"Anomaly detected: {result['anomaly_detected']}")
    
    return analyzer


if __name__ == "__main__":
    analyzer = test_landmark_analysis()
