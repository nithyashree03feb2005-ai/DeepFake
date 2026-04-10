"""
Helper functions for DeepFake Detection System
"""

import os
import numpy as np
from datetime import datetime

# Safe cv2 import - try to import but don't fail if not available
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False
    print("⚠️ OpenCV not available in helpers module")

# Safe matplotlib import - try to import but don't fail if not available
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    MATPLOTLIB_AVAILABLE = False
    print("⚠️ Matplotlib not available in helpers module")


def ensure_directory_exists(path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")


def get_file_extension(filename):
    """Get file extension"""
    return os.path.splitext(filename)[1].lower()


def is_image_file(filename):
    """Check if file is an image"""
    ext = get_file_extension(filename)
    return ext in ['.jpg', '.jpeg', '.png', '.bmp']


def is_video_file(filename):
    """Check if file is a video"""
    ext = get_file_extension(filename)
    return ext in ['.mp4', '.avi', '.mov', '.mkv']


def is_audio_file(filename):
    """Check if file is an audio file"""
    ext = get_file_extension(filename)
    return ext in ['.wav', '.mp3', '.flac', '.ogg']


def resize_image(img, max_size=1024):
    """Resize image if it exceeds max size"""
    h, w = img.shape[:2]
    if h > max_size or w > max_size:
        scale = max_size / max(h, w)
        new_size = (int(w * scale), int(h * scale))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    return img


def plot_confidence_bar(confidence, title="Confidence Score"):
    """
    Plot confidence score bar
    
    Args:
        confidence: Confidence value (0-1)
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 2))
    
    colors = ['red' if confidence < 0.5 else 'green']
    ax.barh(['Confidence'], [confidence * 100], color=colors)
    ax.set_xlim(0, 100)
    ax.set_xlabel('Percentage (%)')
    ax.set_title(title)
    
    # Add value label
    for i, v in enumerate([confidence * 100]):
        ax.text(v + 1, i, f'{v:.1f}%', va='center', fontweight='bold')
    
    plt.tight_layout()
    return fig


def format_timestamp(timestamp=None):
    """Format timestamp for display"""
    if timestamp is None:
        timestamp = datetime.now()
    return timestamp.strftime("%Y-%m-%d %H:%M:%S")


def calculate_fps(video_path):
    """Calculate frames per second of video"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps


def get_video_duration(video_path):
    """Get video duration in seconds"""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return frame_count / fps if fps > 0 else 0


def safe_divide(a, b, default=0.0):
    """Safe division to avoid division by zero"""
    return a / b if b != 0 else default


def normalize_array(arr):
    """Normalize array to 0-1 range"""
    arr_min = arr.min()
    arr_max = arr.max()
    if arr_max - arr_min > 0:
        return (arr - arr_min) / (arr_max - arr_min)
    return np.zeros_like(arr)


def load_model_weights(model, model_path):
    """
    Load model weights if file exists
    
    Args:
        model: Keras/Torch model
        model_path: Path to weights file
        
    Returns:
        Model with loaded weights or None if file doesn't exist
    """
    if os.path.exists(model_path):
        try:
            model.load_weights(model_path)
            print(f"Loaded weights from {model_path}")
            return model
        except Exception as e:
            print(f"Error loading weights: {e}")
            return None
    else:
        print(f"Weights file not found: {model_path}")
        return None


def create_color_mapping(value, min_val=0, max_val=1):
    """
    Create color mapping based on value (red to green gradient)
    
    Args:
        value: Value to map
        min_val: Minimum value
        max_val: Maximum value
        
    Returns:
        RGB color tuple
    """
    # Normalize value
    normalized = (value - min_val) / (max_val - min_val)
    normalized = np.clip(normalized, 0, 1)
    
    # Red (fake) to Green (real) gradient
    r = int(255 * (1 - normalized))
    g = int(255 * normalized)
    b = 0
    
    return (r, g, b)


def extract_filename(filepath):
    """Extract filename without extension"""
    return os.path.splitext(os.path.basename(filepath))[0]


def get_media_info(filepath):
    """
    Get media file information
    
    Args:
        filepath: Path to media file
        
    Returns:
        Dictionary with file information
    """
    info = {
        'filename': extract_filename(filepath),
        'extension': get_file_extension(filepath),
        'size_bytes': os.path.getsize(filepath) if os.path.exists(filepath) else 0,
        'type': 'unknown'
    }
    
    if is_image_file(filepath):
        info['type'] = 'image'
        img = cv2.imread(filepath)
        if img is not None:
            info['dimensions'] = img.shape[:2]
    elif is_video_file(filepath):
        info['type'] = 'video'
        cap = cv2.VideoCapture(filepath)
        if cap.isOpened():
            info['fps'] = cap.get(cv2.CAP_PROP_FPS)
            info['frame_count'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            info['duration'] = info['frame_count'] / info['fps'] if info['fps'] > 0 else 0
            cap.release()
    elif is_audio_file(filepath):
        info['type'] = 'audio'
    
    return info
