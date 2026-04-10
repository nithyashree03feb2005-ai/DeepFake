"""
Biometric Mismatch Detection Module
Detects inconsistencies in biometric features
"""

import numpy as np
from utils.feature_extraction import FeatureExtractor

# Safe cv2 import - try to import but don't fail if not available
try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    _CV2_AVAILABLE = False
    print("⚠️ OpenCV not available in biometric mismatch module")


class BiometricMismatchDetector:
    """Detect biometric mismatches and inconsistencies"""
    
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
    
    def detect_face_identity_mismatch(self, image1, image2):
        """
        Detect if two images show different identities
        
        Args:
            image1: First face image
            image2: Second face image
            
        Returns:
            Dictionary with mismatch analysis
        """
        try:
            # Extract embeddings (placeholder - requires FaceNet or similar)
            embedding1 = self._extract_face_embedding(image1)
            embedding2 = self._extract_face_embedding(image2)
            
            if embedding1 is None or embedding2 is None:
                return {
                    'success': False,
                    'error': 'Could not extract face embeddings'
                }
            
            # Calculate similarity
            similarity = self._calculate_cosine_similarity(embedding1, embedding2)
            
            # Threshold for identity match (adjust based on model)
            threshold = 0.6
            is_same_person = similarity > threshold
            
            result = {
                'success': True,
                'similarity_score': float(similarity),
                'is_same_person': is_same_person,
                'mismatch_detected': not is_same_person,
                'confidence': float(abs(similarity - threshold) / threshold)
            }
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _extract_face_embedding(self, image):
        """
        Extract face embedding vector
        
        Args:
            image: Face image
            
        Returns:
            Embedding vector or None
        """
        # Placeholder - In practice, use FaceNet or ArcFace
        # This would load a pretrained model and extract 512-d embedding
        
        try:
            # Detect face
            landmarks = self.feature_extractor.extract_facial_landmarks(image)
            
            if landmarks is None:
                return None
            
            # For now, use flattened landmarks as pseudo-embedding
            # Replace with actual FaceNet embedding extraction
            embedding = landmarks.flatten()[:512]  # Truncate to 512-d
            
            # Normalize
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            
            return embedding
            
        except Exception as e:
            print(f"Error extracting embedding: {e}")
            return None
    
    def _calculate_cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)
    
    def analyze_facial_feature_consistency(self, image):
        """
        Analyze consistency of facial features
        
        Args:
            image: Face image
            
        Returns:
            Consistency analysis results
        """
        try:
            landmarks = self.feature_extractor.extract_facial_landmarks(image)
            
            if landmarks is None:
                return {
                    'success': False,
                    'error': 'No face detected'
                }
            
            # Analyze left-right symmetry
            left_right_symmetry = self._analyze_left_right_symmetry(landmarks)
            
            # Analyze vertical proportions
            vertical_proportions = self._analyze_vertical_proportions(landmarks)
            
            # Analyze feature alignment
            feature_alignment = self._analyze_feature_alignment(landmarks)
            
            # Overall consistency score
            consistency_score = (left_right_symmetry['score'] + 
                                vertical_proportions['score'] + 
                                feature_alignment['score']) / 3
            
            result = {
                'success': True,
                'consistency_score': float(consistency_score),
                'left_right_symmetry': left_right_symmetry,
                'vertical_proportions': vertical_proportions,
                'feature_alignment': feature_alignment,
                'anomaly_detected': consistency_score < 0.7
            }
            
            return result
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _analyze_left_right_symmetry(self, landmarks):
        """Analyze left-right facial symmetry"""
        # Split face into left and right halves
        nose_bridge = landmarks[1]
        mid_x = nose_bridge[0]
        
        left_points = landmarks[landmarks[:, 0] < mid_x]
        right_points = landmarks[landmarks[:, 0] > mid_x]
        
        if len(left_points) == 0 or len(right_points) == 0:
            return {'score': 0.0}
        
        # Mirror left points
        left_mirrored = left_points.copy()
        left_mirrored[:, 0] = mid_x + (mid_x - left_mirrored[:, 0])
        
        # Find closest points and calculate distances
        distances = []
        for left_pt in left_mirrored:
            min_dist = min([np.linalg.norm(left_pt - right_pt) for right_pt in right_points])
            distances.append(min_dist)
        
        avg_distance = np.mean(distances)
        score = 1.0 / (1.0 + avg_distance)
        
        return {
            'symmetry_score': float(score),
            'average_deviation': float(avg_distance)
        }
    
    def _analyze_vertical_proportions(self, landmarks):
        """Analyze vertical facial proportions"""
        # Three main sections: forehead, nose, mouth+chin
        chin = landmarks[152]
        nose_tip = landmarks[1]
        mouth_center = landmarks[13]
        
        # Estimate forehead center (approximate)
        forehead_center = landmarks[10]
        
        # Calculate section heights
        upper_third = np.linalg.norm(forehead_center - nose_tip)
        middle_third = np.linalg.norm(nose_tip - mouth_center)
        lower_third = np.linalg.norm(mouth_center - chin)
        
        # Ideal proportions (approximately equal thirds)
        total_height = upper_third + middle_third + lower_third
        
        if total_height == 0:
            return {'score': 0.0}
        
        upper_ratio = upper_third / total_height
        middle_ratio = middle_third / total_height
        lower_ratio = lower_third / total_height
        
        # Ideal ratio is ~1/3 for each
        ideal_ratio = 1/3
        deviation = (abs(upper_ratio - ideal_ratio) + 
                    abs(middle_ratio - ideal_ratio) + 
                    abs(lower_ratio - ideal_ratio)) / 3
        
        score = 1.0 / (1.0 + deviation * 10)
        
        return {
            'proportion_score': float(score),
            'upper_third_ratio': float(upper_ratio),
            'middle_third_ratio': float(middle_ratio),
            'lower_third_ratio': float(lower_ratio)
        }
    
    def _analyze_feature_alignment(self, landmarks):
        """Analyze alignment of facial features"""
        # Eye alignment
        left_eye_center = np.mean(landmarks[[33, 133, 160, 159, 158, 157, 173]], axis=0)
        right_eye_center = np.mean(landmarks[[362, 263, 385, 386, 387, 388, 466]], axis=0)
        
        # Check if eyes are at same height
        eye_y_diff = abs(left_eye_center[1] - right_eye_center[1])
        eye_distance = np.linalg.norm(left_eye_center - right_eye_center)
        
        eye_alignment_score = 1.0 - (eye_y_diff / (eye_distance + 1e-8)) if eye_distance > 0 else 0
        
        # Mouth alignment
        mouth_left = landmarks[61]
        mouth_right = landmarks[291]
        mouth_y_diff = abs(mouth_left[1] - mouth_right[1])
        mouth_width = np.linalg.norm(mouth_left - mouth_right)
        
        mouth_alignment_score = 1.0 - (mouth_y_diff / (mouth_width + 1e-8)) if mouth_width > 0 else 0
        
        # Nose alignment
        nose_top = landmarks[10]
        nose_bottom = landmarks[1]
        nose_x_mid = (nose_top[0] + nose_bottom[0]) / 2
        face_mid_x = (landmarks[:, 0].min() + landmarks[:, 0].max()) / 2
        
        nose_deviation = abs(nose_x_mid - face_mid_x)
        face_width = landmarks[:, 0].max() - landmarks[:, 0].min()
        
        nose_alignment_score = 1.0 - (nose_deviation / (face_width + 1e-8)) if face_width > 0 else 0
        
        overall_score = (eye_alignment_score + mouth_alignment_score + nose_alignment_score) / 3
        
        return {
            'alignment_score': float(overall_score),
            'eye_alignment': float(eye_alignment_score),
            'mouth_alignment': float(mouth_alignment_score),
            'nose_alignment': float(nose_alignment_score)
        }
    
    def detect_texture_inconsistencies(self, image):
        """
        Detect texture inconsistencies in skin
        
        Args:
            image: Face image
            
        Returns:
            Texture analysis results
        """
        try:
            landmarks = self.feature_extractor.extract_facial_landmarks(image)
            
            if landmarks is None:
                return {
                    'success': False,
                    'error': 'No face detected'
                }
            
            # Extract cheek regions (typically smooth skin)
            left_cheek_indices = [116, 117, 118, 119, 120]
            right_cheek_indices = [345, 346, 347, 348, 349]
            
            left_cheek = landmarks[left_cheek_indices]
            right_cheek = landmarks[right_cheek_indices]
            
            # In practice, would extract patches and analyze texture
            # This is a placeholder
            
            return {
                'success': True,
                'note': 'Texture analysis requires image patch extraction',
                'texture_consistency': 0.8  # Placeholder value
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }


def test_biometric_mismatch_detection():
    """Test biometric mismatch detection functionality"""
    detector = BiometricMismatchDetector()
    
    print("\nBiometric Mismatch Detector initialized")
    print("Note: Full functionality requires FaceNet or similar embedding model")
    
    return detector


if __name__ == "__main__":
    detector = test_biometric_mismatch_detection()
