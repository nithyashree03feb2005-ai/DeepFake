"""
Dataset Loader for DeepFake Detection System
Loads and preprocesses images, videos, and audio from dataset directory
"""

import os
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Safe tqdm import - try to import but don't fail if not available
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    tqdm = None
    TQDM_AVAILABLE = False
    print("⚠️ tqdm not available in dataset loader module")

import random

# Safe cv2 import - try to import but don't fail if not available
try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    _CV2_AVAILABLE = False
    print("⚠️ OpenCV not available in dataset loader module")


class DatasetLoader:
    """Load and preprocess deepfake detection datasets"""
    
    def __init__(self, dataset_path='dataset', img_size=(224, 224), max_frames=300):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.max_frames = max_frames
        
    def load_image_dataset(self, batch_size=32, validation_split=0.2, use_subset=False, subset_size=5000):
        """
        Load image dataset for training
        
        Args:
            batch_size: Batch size for training
            validation_split: Validation split ratio
            use_subset: Use subset of data for faster training
            subset_size: Number of images to load if using subset (default: 5000)
            
        Returns:
            X_train, X_val, y_train, y_val: Train and validation splits
        """
        images = []
        labels = []
        
        # Load real images
        real_path = os.path.join(self.dataset_path, 'Train', 'Real')
        if os.path.exists(real_path):
            print("Loading real images...")
            count = 0
            for img_name in tqdm(os.listdir(real_path)):
                if use_subset and count >= subset_size // 2:
                    break
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(real_path, img_name)
                    img = self._load_and_preprocess_image(img_path)
                    if img is not None:
                        images.append(img)
                        labels.append(0)  # Real = 0
                        count += 1
                        
        # Load fake images
        fake_path = os.path.join(self.dataset_path, 'Train', 'Fake')
        if os.path.exists(fake_path):
            print("Loading fake images...")
            count = 0
            for img_name in tqdm(os.listdir(fake_path)):
                if use_subset and count >= subset_size // 2:
                    break
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    img_path = os.path.join(fake_path, img_name)
                    img = self._load_and_preprocess_image(img_path)
                    if img is not None:
                        images.append(img)
                        labels.append(1)  # Fake = 1
                        count += 1
        
        X = np.array(images)
        y = np.array(labels)
        
        # Split dataset
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        print(f"Loaded {len(X)} images: {np.sum(labels==0)} real, {np.sum(labels==1)} fake")
        print(f"Training set: {len(X_train)}, Validation set: {len(X_val)}")
        
        return X_train, X_val, y_train, y_val
    
    def _load_and_preprocess_image(self, img_path):
        """Load and preprocess a single image"""
        try:
            img = tf.keras.preprocessing.image.load_img(img_path, target_size=self.img_size)
            img_array = tf.keras.preprocessing.image.img_to_array(img)
            
            # Normalize pixel values
            img_array = img_array / 255.0
            
            return img_array
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None
    
    def load_video_dataset(self, validation_split=0.2):
        """
        Load video dataset for training
        
        Returns:
            X_train, X_val, y_train, y_val: Train and validation splits
        """
        sequences = []
        labels = []
        
        # Support both dataset/Train/{Real,Fake} and dataset/videos/{real,fake}
        real_candidates = [
            os.path.join(self.dataset_path, 'Train', 'Real'),
            os.path.join(self.dataset_path, 'videos', 'real'),
            os.path.join(self.dataset_path, 'videos', 'videos_real'),
            os.path.join(self.dataset_path, 'Videos', 'Real'),
            os.path.join(self.dataset_path, 'Real'),
            os.path.join(self.dataset_path, 'real'),
        ]
        fake_candidates = [
            os.path.join(self.dataset_path, 'Train', 'Fake'),
            os.path.join(self.dataset_path, 'videos', 'fake'),
            os.path.join(self.dataset_path, 'videos', 'videos_fake'),
            os.path.join(self.dataset_path, 'Videos', 'Fake'),
            os.path.join(self.dataset_path, 'Fake'),
            os.path.join(self.dataset_path, 'fake'),
        ]
        valid_exts = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
        
        def pick_video_dir(candidates):
            for path in candidates:
                if not os.path.isdir(path):
                    continue
                for name in os.listdir(path):
                    if name.lower().endswith(valid_exts):
                        return path
            return None

        real_path = pick_video_dir(real_candidates)
        fake_path = pick_video_dir(fake_candidates)

        # Load real videos
        if real_path:
            print(f"Loading real videos from: {real_path}")
            for video_name in tqdm(os.listdir(real_path)):
                if video_name.lower().endswith(valid_exts):
                    video_path = os.path.join(real_path, video_name)
                    frames = self._extract_frames_from_video(video_path)
                    if len(frames) > 0:
                        sequences.append(frames)
                        labels.append(0)
        else:
            print("No real video directory found.")

        # Load fake videos
        if fake_path:
            print(f"Loading fake videos from: {fake_path}")
            for video_name in tqdm(os.listdir(fake_path)):
                if video_name.lower().endswith(valid_exts):
                    video_path = os.path.join(fake_path, video_name)
                    frames = self._extract_frames_from_video(video_path)
                    if len(frames) > 0:
                        sequences.append(frames)
                        labels.append(1)
        else:
            print("No fake video directory found.")
        
        X = np.array(sequences)
        y = np.array(labels)
        
        if len(X) == 0:
            raise ValueError(
                "No videos were loaded. Ensure your dataset has real/fake videos in "
                "either dataset/Train/{Real,Fake} or dataset/videos/{real,fake}."
            )

        # Split dataset
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        print(f"Loaded {len(X)} videos")
        print(f"Training set: {len(X_train)}, Validation set: {len(X_val)}")
        
        return X_train, X_val, y_train, y_val
    
    def _extract_frames_from_video(self, video_path, num_frames=None):
        """Extract frames from video"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return np.array([])

        # Ensure a fixed-length sequence for every video so tensors have uniform shape.
        target_frames = self.max_frames if num_frames is None else num_frames
        sample_count = min(target_frames, total_frames)
        frame_indices = np.linspace(0, total_frames - 1, sample_count, dtype=int)
        
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, self.img_size)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = frame.astype(np.float32) / 255.0
                frames.append(frame)
        
        cap.release()
        
        if len(frames) == 0:
            return np.array([])

        # Pad short videos by repeating the last valid frame.
        while len(frames) < target_frames:
            frames.append(frames[-1].copy())

        # Safety truncate if needed.
        if len(frames) > target_frames:
            frames = frames[:target_frames]

        return np.array(frames, dtype=np.float32)
    
    def load_audio_dataset(self, validation_split=0.2, sr=16000, duration=5):
        """
        Load audio dataset for training
        
        Args:
            sr: Sample rate
            duration: Duration in seconds
            
        Returns:
            X_train, X_val, y_train, y_val: Train and validation splits
        """
        spectrograms = []
        labels = []
        
        # Load real audio
        real_path = os.path.join(self.dataset_path, 'audio', 'wav_real')
        if os.path.exists(real_path):
            print("Loading real audio...")
            for audio_name in tqdm(os.listdir(real_path)):
                if audio_name.endswith(('.wav', '.mp3', '.flac')):
                    audio_path = os.path.join(real_path, audio_name)
                    spec = self._extract_spectrogram(audio_path, sr, duration)
                    if spec is not None:
                        spectrograms.append(spec)
                        labels.append(0)
                        
        # Load fake audio
        fake_path = os.path.join(self.dataset_path, 'audio', 'wav_fake')
        if os.path.exists(fake_path):
            print("Loading fake audio...")
            for audio_name in tqdm(os.listdir(fake_path)):
                if audio_name.endswith(('.wav', '.mp3', '.flac')):
                    audio_path = os.path.join(fake_path, audio_name)
                    spec = self._extract_spectrogram(audio_path, sr, duration)
                    if spec is not None:
                        spectrograms.append(spec)
                        labels.append(1)
        
        X = np.array(spectrograms)
        y = np.array(labels)
        
        # Split dataset
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        print(f"Loaded {len(X)} audio files")
        print(f"Training set: {len(X_train)}, Validation set: {len(X_val)}")
        
        return X_train, X_val, y_train, y_val
    
    def _extract_spectrogram(self, audio_path, sr, duration):
        """Extract mel spectrogram from audio file"""
        try:
            y, _ = librosa.load(audio_path, sr=sr, duration=duration)
            
            # Extract mel spectrogram
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize
            mel_spec_db = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
            
            # Ensure consistent time frames by padding or truncating
            target_frames = int(sr * duration / 512)  # Approximate number of frames
            current_frames = mel_spec_db.shape[1]
            
            if current_frames < target_frames:
                # Pad with zeros
                pad_width = target_frames - current_frames
                mel_spec_db = np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
            elif current_frames > target_frames:
                # Truncate
                mel_spec_db = mel_spec_db[:, :target_frames]
            
            # Expand dimensions for CNN input (height, width, channels)
            mel_spec_db = np.expand_dims(mel_spec_db, axis=-1)
            
            return mel_spec_db
        except Exception as e:
            print(f"Error processing audio {audio_path}: {e}")
            return None


def create_sample_dataset_structure(base_path='dataset'):
    """Create sample dataset directory structure"""
    directories = [
        os.path.join(base_path, 'images', 'real'),
        os.path.join(base_path, 'images', 'fake'),
        os.path.join(base_path, 'videos', 'real'),
        os.path.join(base_path, 'videos', 'fake'),
        os.path.join(base_path, 'audio', 'real'),
        os.path.join(base_path, 'audio', 'fake'),
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created: {directory}")
    
    print("\nDataset structure created!")
    print("Please add your dataset files to the respective directories.")
    print("\nSupported datasets:")
    print("- FaceForensics++")
    print("- CelebDF")
    print("- DFDC dataset")
    print("- FakeAVCeleb")


if __name__ == "__main__":
    # Create dataset structure
    create_sample_dataset_structure()
    
    # Example usage
    # loader = DatasetLoader()
    # X_train, X_val, y_train, y_val = loader.load_image_dataset()
