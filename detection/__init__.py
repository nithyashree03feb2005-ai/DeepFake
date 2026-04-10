"""
Detection module for DeepFake Detection System
"""

# Try to import detection modules with error handling
try:
    from .image_detection import ImageDeepFakeDetector
except ImportError as e:
    print(f"⚠️ Image detection module failed to load: {e}")
    ImageDeepFakeDetector = None

try:
    from .video_detection import VideoDeepFakeDetector
except ImportError as e:
    print(f"⚠️ Video detection module failed to load: {e}")
    VideoDeepFakeDetector = None

try:
    from .audio_detection import AudioDeepFakeDetector
except ImportError as e:
    print(f"⚠️ Audio detection module failed to load: {e}")
    AudioDeepFakeDetector = None

try:
    from .webcam_detection import WebcamDeepFakeDetector
except ImportError as e:
    print(f"⚠️ Webcam detection module failed to load: {e}")
    WebcamDeepFakeDetector = None

__all__ = [
    'ImageDeepFakeDetector',
    'VideoDeepFakeDetector',
    'AudioDeepFakeDetector',
    'WebcamDeepFakeDetector'
]
