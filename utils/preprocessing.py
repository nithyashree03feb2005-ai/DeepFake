"""
Preprocessing Utilities for DeepFake Detection System
"""

import numpy as np

# Safe TensorFlow import
try:
    from tensorflow.keras.preprocessing import image
    TENSORFLOW_AVAILABLE = True
except ImportError:
    image = None
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow/Keras not available in preprocessing module")

# Safe cv2 import - try to import but don't fail if not available
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False
    print("⚠️ OpenCV not available in preprocessing module")

# Safe librosa import - try to import but don't fail if not available
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    librosa = None
    LIBROSA_AVAILABLE = False
    print("⚠️ Librosa not available in preprocessing module")


class Preprocessor:
    """Image, video, and audio preprocessing utilities"""
    
    @staticmethod
    def preprocess_image(img_path, img_size=(224, 224)):
        """
        Preprocess image for model input
        
        Args:
            img_path: Path to image file
            img_size: Target size
            
        Returns:
            Preprocessed image array
        """
        img = image.load_img(img_path, target_size=img_size)
        img_array = image.img_to_array(img)
        img_array = img_array / 255.0
        return img_array
    
    @staticmethod
    def preprocess_frame(frame, img_size=(224, 224)):
        """
        Preprocess video frame
        
        Args:
            frame: BGR frame from OpenCV
            img_size: Target size
            
        Returns:
            Preprocessed frame array
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, img_size)
        frame_normalized = frame_resized.astype(np.float32) / 255.0
        return frame_normalized
    
    @staticmethod
    def extract_face_region(frame, face_detector):
        """
        Extract face region from frame using MediaPipe or Dlib
        
        Args:
            frame: Input frame
            face_detector: Face detection model
            
        Returns:
            Cropped face region
        """
        # Implementation depends on face detector used
        # This is a placeholder for face extraction logic
        pass
    
    @staticmethod
    def preprocess_audio(audio_path, sr=16000, duration=5):
        """
        Preprocess audio file and extract spectrogram
        
        Args:
            audio_path: Path to audio file
            sr: Sample rate
            duration: Duration in seconds
            
        Returns:
            Mel spectrogram
        """
        y, _ = librosa.load(audio_path, sr=sr, duration=duration)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
        mel_spec_db = np.expand_dims(mel_spec_db, axis=-1)
        
        return mel_spec_db
    
    @staticmethod
    def augment_image(img, rotation_range=10, zoom_range=0.1, shift_range=0.1):
        """
        Apply data augmentation to image
        
        Args:
            img: Input image
            rotation_range: Rotation range in degrees
            zoom_range: Zoom range
            shift_range: Shift range
            
        Returns:
            Augmented image
        """
        rows, cols = img.shape[:2]
        
        # Random rotation
        if np.random.rand() > 0.5:
            angle = np.random.uniform(-rotation_range, rotation_range)
            M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
            img = cv2.warpAffine(img, M, (cols, rows))
        
        # Random zoom
        if np.random.rand() > 0.5:
            zoom = np.random.uniform(1 - zoom_range, 1 + zoom_range)
            img = cv2.resize(img, None, fx=zoom, fy=zoom)
            img = cv2.resize(img, (cols, rows))
        
        # Random shift
        if np.random.rand() > 0.5:
            tx = np.random.uniform(-shift_range * cols, shift_range * cols)
            ty = np.random.uniform(-shift_range * rows, shift_range * rows)
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            img = cv2.warpAffine(img, M, (cols, rows))
        
        return img
    
    @staticmethod
    def normalize_audio(audio, target_db=-20):
        """
        Normalize audio volume
        
        Args:
            audio: Audio array
            target_db: Target dB level
            
        Returns:
            Normalized audio
        """
        rms = np.sqrt(np.mean(audio**2))
        if rms > 0:
            scalar = 10**(target_db / 20) / rms
            audio = audio * scalar
        return audio
