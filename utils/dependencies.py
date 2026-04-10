"""
Central Dependency Management for DeepFake Detection System
Handles all critical imports with graceful error handling
"""

import sys

# ============================================================================
# OPENCV (cv2)
# ============================================================================
try:
    import cv2
    CV2_AVAILABLE = True
    print(f"✓ OpenCV {cv2.__version__} loaded successfully")
except ImportError as e:
    cv2 = None
    CV2_AVAILABLE = False
    print(f"⚠️ OpenCV not available: {e}")
    print("⚠️ Image/video processing features will be limited")

# ============================================================================
# TENSORFLOW / KERAS
# ============================================================================
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
    TF_VERSION = tf.__version__
    
    # Try to import Keras
    try:
        from tensorflow.keras.models import load_model
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
        KERAS_AVAILABLE = True
        print(f"✓ TensorFlow {TF_VERSION} and Keras loaded successfully")
    except ImportError as e:
        KERAS_AVAILABLE = False
        print(f"⚠️ Keras not available: {e}")
        
except ImportError as e:
    tf = None
    TENSORFLOW_AVAILABLE = False
    KERAS_AVAILABLE = False
    TF_VERSION = None
    print(f"⚠️ TensorFlow not available: {e}")
    print("⚠️ AI model loading and prediction will not work")

# ============================================================================
# PYTORCH
# ============================================================================
try:
    import torch
    import torchvision
    PYTORCH_AVAILABLE = True
    TORCH_VERSION = torch.__version__
    print(f"✓ PyTorch {TORCH_VERSION} loaded successfully")
except ImportError as e:
    torch = None
    torchvision = None
    PYTORCH_AVAILABLE = False
    TORCH_VERSION = None
    print(f"⚠️ PyTorch not available: {e}")

# ============================================================================
# LIBROSA (Audio Processing)
# ============================================================================
try:
    import librosa
    LIBROSA_AVAILABLE = True
    print(f"✓ Librosa {librosa.__version__} loaded successfully")
except ImportError as e:
    librosa = None
    LIBROSA_AVAILABLE = False
    print(f"⚠️ Librosa not available: {e}")
    print("⚠️ Audio processing features will be limited")

# ============================================================================
# MEDIAPIPE (Face Detection)
# ============================================================================
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    print(f"✓ MediaPipe loaded successfully")
except ImportError as e:
    mp = None
    MEDIAPIPE_AVAILABLE = False
    print(f"⚠️ MediaPipe not available: {e}")
    print("⚠️ Face detection features will be limited")

# ============================================================================
# FACERECOGNITION (dlib-based face recognition)
# ============================================================================
try:
    # First check if dlib is available (required dependency)
    try:
        import dlib
        DLIB_CHECK_PASSED = True
    except ImportError:
        DLIB_CHECK_PASSED = False
        raise ImportError("dlib is required but not installed")
    
    # Try to import face_recognition
    import face_recognition
    
    # Verify models are available by doing a simple test
    try:
        # Quick test to ensure models are loaded (import numpy locally)
        import numpy as np_test
        test_image = np_test.zeros((100, 100, 3), dtype=np.uint8)
        # This will fail if models aren't properly installed
        face_recognition.face_locations(test_image)
        FACE_RECOGNITION_AVAILABLE = True
        print(f"✓ face_recognition library loaded successfully with models")
    except Exception as model_error:
        # Models might not be installed incorrectly
        FACE_RECOGNITION_AVAILABLE = False
        face_recognition = None
        print(f"⚠️ face_recognition models not properly installed: {model_error}")
        print("⚠️ Trying to continue without face_recognition...")
        
except ImportError as e:
    face_recognition = None
    FACE_RECOGNITION_AVAILABLE = False
    print(f"⚠️ face_recognition not available: {e}")
    print("⚠️ Advanced face recognition features will be limited")

# ============================================================================
# DLIB (Required for face_recognition)
# ============================================================================
try:
    import dlib
    DLIB_AVAILABLE = True
    print(f"✓ dlib {dlib.__version__} loaded successfully")
except ImportError as e:
    dlib = None
    DLIB_AVAILABLE = False
    print(f"⚠️ dlib not available: {e}")

# ============================================================================
# PILLOW (Image Processing)
# ============================================================================
try:
    from PIL import Image
    PILLOW_AVAILABLE = True
    print(f"✓ Pillow loaded successfully")
except ImportError as e:
    Image = None
    PILLOW_AVAILABLE = False
    print(f"⚠️ Pillow not available: {e}")

# ============================================================================
# NUMPY (Critical - Almost Always Available)
# ============================================================================
try:
    import numpy as np
    NUMPY_AVAILABLE = True
    print(f"✓ NumPy {np.__version__} loaded successfully")
except ImportError as e:
    np = None
    NUMPY_AVAILABLE = False
    print(f"⚠️ NumPy not available: {e}")
    print("⚠️ CRITICAL: Most features will not work without NumPy!")

# ============================================================================
# SCIPY (Scientific Computing)
# ============================================================================
try:
    import scipy
    SCIPY_AVAILABLE = True
    print(f"✓ SciPy {scipy.__version__} loaded successfully")
except ImportError as e:
    scipy = None
    SCIPY_AVAILABLE = False
    print(f"⚠️ SciPy not available: {e}")

# ============================================================================
# STREAMLIT (Web Framework)
# ============================================================================
try:
    import streamlit as st
    STREAMLIT_AVAILABLE = True
    print(f"✓ Streamlit {st.__version__} loaded successfully")
except ImportError as e:
    st = None
    STREAMLIT_AVAILABLE = False
    print(f"⚠️ Streamlit not available: {e}")
    print("⚠️ CRITICAL: Web interface will not work!")

# ============================================================================
# SUMMARY FUNCTION
# ============================================================================

def get_dependency_summary():
    """Get a summary of all available dependencies"""
    
    summary = {
        'core': {
            'numpy': NUMPY_AVAILABLE,
            'scipy': SCIPY_AVAILABLE,
            'pillow': PILLOW_AVAILABLE,
        },
        'ml_frameworks': {
            'tensorflow': TENSORFLOW_AVAILABLE,
            'keras': KERAS_AVAILABLE,
            'pytorch': PYTORCH_AVAILABLE,
        },
        'computer_vision': {
            'opencv': CV2_AVAILABLE,
            'mediapipe': MEDIAPIPE_AVAILABLE,
            'dlib': DLIB_AVAILABLE,
            'face_recognition': FACE_RECOGNITION_AVAILABLE,
        },
        'audio': {
            'librosa': LIBROSA_AVAILABLE,
        },
        'web': {
            'streamlit': STREAMLIT_AVAILABLE,
        }
    }
    
    return summary


def print_dependency_status():
    """Print detailed dependency status"""
    
    summary = get_dependency_summary()
    
    print("\n" + "="*60)
    print("DEPENDENCY STATUS")
    print("="*60)
    
    for category, deps in summary.items():
        print(f"\n{category.upper().replace('_', ' ')}:")
        for dep, available in deps.items():
            status = "✅" if available else "❌"
            print(f"  {status} {dep.upper()}")
    
    print("\n" + "="*60)
    
    # Check critical dependencies
    critical_missing = []
    
    if not STREAMLIT_AVAILABLE:
        critical_missing.append("Streamlit")
    if not NUMPY_AVAILABLE:
        critical_missing.append("NumPy")
    if not TENSORFLOW_AVAILABLE and not PYTORCH_AVAILABLE:
        critical_missing.append("TensorFlow or PyTorch")
    
    if critical_missing:
        print(f"\n⚠️ CRITICAL: Missing essential dependencies: {', '.join(critical_missing)}")
        print("The application may not function properly.")
    else:
        print("\n✅ All critical dependencies available!")
    
    print("="*60 + "\n")
    
    return summary


# Export everything
__all__ = [
    # Availability flags
    'CV2_AVAILABLE',
    'TENSORFLOW_AVAILABLE', 
    'KERAS_AVAILABLE',
    'PYTORCH_AVAILABLE',
    'LIBROSA_AVAILABLE',
    'MEDIAPIPE_AVAILABLE',
    'FACE_RECOGNITION_AVAILABLE',
    'DLIB_AVAILABLE',
    'PILLOW_AVAILABLE',
    'NUMPY_AVAILABLE',
    'SCIPY_AVAILABLE',
    'STREAMLIT_AVAILABLE',
    
    # Module references (may be None)
    'cv2',
    'tf',
    'torch',
    'librosa',
    'mp',
    'face_recognition',
    'dlib',
    'Image',
    'np',
    'scipy',
    'st',
    
    # Helper functions
    'get_dependency_summary',
    'print_dependency_status',
]

# Print status on import (helpful for debugging)
if __name__ == "__main__":
    print_dependency_status()
