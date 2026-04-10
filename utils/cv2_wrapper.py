"""
OpenCV Wrapper Module
Provides safe import and usage of OpenCV with fallback handling for Streamlit Cloud
"""

import sys

# Try to import cv2
cv2 = None
_cv2_import_error = None

try:
    import cv2
    _CV2_AVAILABLE = True
    print(f"✓ OpenCV {cv2.__version__} loaded successfully")
except ImportError as e:
    _CV2_AVAILABLE = False
    _cv2_import_error = str(e)
    print(f"⚠️ OpenCV not available: {e}")
    print("⚠️ Some features requiring image processing may not work")


def get_cv2():
    """
    Get cv2 module if available
    
    Returns:
        module or None: cv2 module if available, None otherwise
    """
    return cv2


def is_cv2_available():
    """
    Check if OpenCV is available
    
    Returns:
        bool: True if cv2 is available, False otherwise
    """
    return _CV2_AVAILABLE


def require_cv2(func):
    """
    Decorator to check if cv2 is available before running a function
    
    Args:
        func: Function to wrap
        
    Returns:
        Wrapped function that checks cv2 availability
    """
    def wrapper(*args, **kwargs):
        if not _CV2_AVAILABLE:
            raise RuntimeError(
                "OpenCV (cv2) is required for this operation but is not installed. "
                "Please ensure system dependencies are installed on Streamlit Cloud "
                "(libgl1-mesa-glx, libglib2.0-0) via apt.txt"
            )
        return func(*args, **kwargs)
    
    wrapper.__name__ = func.__name__
    wrapper.__doc__ = func.__doc__
    return wrapper


class CV2Wrapper:
    """
    Safe wrapper for OpenCV operations
    
    Provides methods that gracefully handle missing cv2 module
    """
    
    def __init__(self):
        self._cv2 = cv2
        self._available = _CV2_AVAILABLE
    
    @property
    def available(self):
        """Check if cv2 is available"""
        return self._available
    
    @property
    def version(self):
        """Get OpenCV version if available"""
        if self._available and hasattr(self._cv2, '__version__'):
            return self._cv2.__version__
        return None
    
    def imread(self, *args, **kwargs):
        """Safe cv2.imread"""
        if not self._available:
            raise RuntimeError("cv2 not available")
        return self._cv2.imread(*args, **kwargs)
    
    def imwrite(self, *args, **kwargs):
        """Safe cv2.imwrite"""
        if not self._available:
            raise RuntimeError("cv2 not available")
        return self._cv2.imwrite(*args, **kwargs)
    
    def resize(self, *args, **kwargs):
        """Safe cv2.resize"""
        if not self._available:
            raise RuntimeError("cv2 not available")
        return self._cv2.resize(*args, **kwargs)
    
    def cvtColor(self, *args, **kwargs):
        """Safe cv2.cvtColor"""
        if not self._available:
            raise RuntimeError("cv2 not available")
        return self._cv2.cvtColor(*args, **kwargs)
    
    def applyColorMap(self, *args, **kwargs):
        """Safe cv2.applyColorMap"""
        if not self._available:
            raise RuntimeError("cv2 not available")
        return self._cv2.applyColorMap(*args, **kwargs)
    
    def findContours(self, *args, **kwargs):
        """Safe cv2.findContours"""
        if not self._available:
            raise RuntimeError("cv2 not available")
        return self._cv2.findContours(*args, **kwargs)
    
    def rectangle(self, *args, **kwargs):
        """Safe cv2.rectangle"""
        if not self._available:
            raise RuntimeError("cv2 not available")
        return self._cv2.rectangle(*args, **kwargs)
    
    def putText(self, *args, **kwargs):
        """Safe cv2.putText"""
        if not self._available:
            raise RuntimeError("cv2 not available")
        return self._cv2.putText(*args, **kwargs)
    
    def getRotationMatrix2D(self, *args, **kwargs):
        """Safe cv2.getRotationMatrix2D"""
        if not self._available:
            raise RuntimeError("cv2 not available")
        return self._cv2.getRotationMatrix2D(*args, **kwargs)
    
    def warpAffine(self, *args, **kwargs):
        """Safe cv2.warpAffine"""
        if not self._available:
            raise RuntimeError("cv2 not available")
        return self._cv2.warpAffine(*args, **kwargs)
    
    def flip(self, *args, **kwargs):
        """Safe cv2.flip"""
        if not self._available:
            raise RuntimeError("cv2 not available")
        return self._cv2.flip(*args, **kwargs)
    
    def GaussianBlur(self, *args, **kwargs):
        """Safe cv2.GaussianBlur"""
        if not self._available:
            raise RuntimeError("cv2 not available")
        return self._cv2.GaussianBlur(*args, **kwargs)
    
    def Canny(self, *args, **kwargs):
        """Safe cv2.Canny"""
        if not self._available:
            raise RuntimeError("cv2 not available")
        return self._cv2.Canny(*args, **kwargs)
    
    def cascade_CascadeClassifier(self, *args, **kwargs):
        """Safe cv2.CascadeClassifier"""
        if not self._available:
            raise RuntimeError("cv2 not available")
        return self._cv2.CascadeClassifier(*args, **kwargs)
    
    def face_FaceRecognizer(self, *args, **kwargs):
        """Safe cv2.face.FaceRecognizer"""
        if not self._available:
            raise RuntimeError("cv2 not available")
        try:
            return self._cv2.face.FaceRecognizer(*args, **kwargs)
        except Exception as e:
            raise RuntimeError(f"FaceRecognizer not available: {e}")


# Create global instance
safe_cv2 = CV2Wrapper()

# Export for easy access
__all__ = ['cv2', 'get_cv2', 'is_cv2_available', 'require_cv2', 'safe_cv2']
