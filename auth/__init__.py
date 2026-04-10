"""
Authentication module for DeepFake Detection System
"""

from .database import DatabaseManager
from .login import AuthenticationManager

__all__ = [
    'DatabaseManager',
    'AuthenticationManager'
]
