"""
Feature extraction utilities for DeepFake Detection System
"""

import numpy as np
from scipy.spatial import distance as dist

# Safe cv2 import - try to import but don't fail if not available
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False
    print("⚠️ OpenCV not available in feature extraction module")

# Safe mediapipe import
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    mp = None
    MEDIAPIPE_AVAILABLE = False

# Optional dlib import (requires manual installation)
try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    dlib = None


class FeatureExtractor:
    """Extract facial features and landmarks"""
    
    def __init__(self):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
        
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        
        # Initialize Dlib predictor (requires dlib shape predictor file)
        # self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        # self.detector = dlib.get_frontal_face_detector()
    
    def extract_facial_landmarks(self, image_bgr):
        """
        Extract 468 facial landmarks using MediaPipe
        
        Args:
            image_bgr: BGR image from OpenCV
            
        Returns:
            Array of landmark coordinates or None
        """
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            landmark_array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
            return landmark_array
        return None
    
    def detect_face(self, image_bgr):
        """
        Detect face in image
        
        Args:
            image_bgr: BGR image
            
        Returns:
            Bounding box coordinates or None
        """
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(image_rgb)
        
        if results.detections:
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            h, w = image_bgr.shape[:2]
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)
            return (x, y, x + bw, y + bh)
        return None
    
    def extract_eye_region(self, landmarks, eye_indices):
        """
        Extract eye region from landmarks
        
        Args:
            landmarks: Facial landmarks array
            eye_indices: Indices of eye landmarks
            
        Returns:
            Eye aspect ratio or eye region
        """
        eye_points = landmarks[eye_indices]
        return eye_points
    
    def calculate_eye_aspect_ratio(self, eye_landmarks):
        """
        Calculate Eye Aspect Ratio (EAR) for blink detection
        
        Args:
            eye_landmarks: 6 eye landmark points
            
        Returns:
            EAR value
        """
        # Vertical distances
        v1 = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
        v2 = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
        
        # Horizontal distance
        h = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
        
        # EAR formula
        ear = (v1 + v2) / (2.0 * h)
        return ear
    
    def extract_lip_region(self, landmarks):
        """
        Extract lip region from facial landmarks
        
        Args:
            landmarks: Facial landmarks
            
        Returns:
            Lip landmark points
        """
        # MediaPipe lip indices (approximate)
        lip_indices = list(range(61, 68)) + list(range(291, 298))
        lip_points = landmarks[lip_indices]
        return lip_points
    
    def extract_face_embedding(self, face_image, model=None):
        """
        Extract face embedding using FaceNet or similar
        
        Args:
            face_image: Cropped face image
            model: Pretrained embedding model
            
        Returns:
            Face embedding vector
        """
        # Placeholder for FaceNet embedding extraction
        # In practice, load FaceNet model and extract 512-d embedding
        pass
    
    def get_landmark_distances(self, landmarks):
        """
        Calculate distances between key facial landmarks
        
        Args:
            landmarks: Facial landmarks array
            
        Returns:
            Dictionary of distances
        """
        distances = {}
        
        # Example: Distance between eyes
        left_eye_center = np.mean(landmarks[33:40], axis=0)
        right_eye_center = np.mean(landmarks[263:270], axis=0)
        distances['eye_distance'] = dist.euclidean(left_eye_center[:2], right_eye_center[:2])
        
        # Nose to mouth distance
        nose_tip = landmarks[1]
        mouth_center = landmarks[13]
        distances['nose_mouth_distance'] = dist.euclidean(nose_tip[:2], mouth_center[:2])
        
        return distances


class VideoFeatureExtractor:
    """Extract temporal features from video"""
    
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
    
    def extract_frame_sequence_features(self, frames):
        """
        Extract features from sequence of frames
        
        Args:
            frames: List of video frames
            
        Returns:
            Sequence of feature vectors
        """
        features = []
        for frame in frames:
            landmarks = self.feature_extractor.extract_facial_landmarks(frame)
            if landmarks is not None:
                # Flatten landmarks for feature vector
                feature_vector = landmarks.flatten()
                features.append(feature_vector)
            else:
                features.append(np.zeros(468 * 3))  # Zero padding if no face detected
        
        return np.array(features)
    
    def extract_temporal_inconsistencies(self, frames):
        """
        Detect temporal inconsistencies in video
        
        Args:
            frames: Video frames
            
        Returns:
            Inconsistency metrics
        """
        landmarks_seq = self.extract_frame_sequence_features(frames)
        
        # Calculate frame-to-frame differences
        differences = np.diff(landmarks_seq, axis=0)
        
        # Statistics of differences
        mean_diff = np.mean(np.abs(differences))
        std_diff = np.std(differences)
        max_diff = np.max(np.abs(differences))
        
        return {
            'mean_difference': mean_diff,
            'std_difference': std_diff,
            'max_difference': max_diff
        }
