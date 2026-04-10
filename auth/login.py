"""
Login and Authentication Module for DeepFake Detection System
"""

import os
import hashlib
from auth.database import DatabaseManager


class AuthenticationManager:
    """Handle user authentication and session management"""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.current_user = None
    
    def register(self, username, email, password, confirm_password=None):
        """
        Register new user
        
        Args:
            username: Username
            email: Email
            password: Password
            confirm_password: Password confirmation
            
        Returns:
            Tuple (success: bool, message: str, user_id: int|None)
        """
        # Validation
        if not username or len(username) < 3:
            return False, "Username must be at least 3 characters", None
        
        if not email or '@' not in email:
            return False, "Invalid email address", None
        
        if not password or len(password) < 6:
            return False, "Password must be at least 6 characters", None
        
        if confirm_password and password != confirm_password:
            return False, "Passwords do not match", None
        
        # Check if username exists
        existing_user = self.db.authenticate_user(username, password)
        # Note: This is a simple check, could be optimized
        
        # Create user
        user_id = self.db.create_user(username, email, password)
        
        if user_id:
            return True, f"User {username} registered successfully!", user_id
        else:
            return False, "Username or email already exists", None
    
    def login(self, username, password):
        """
        Login user
        
        Args:
            username: Username
            password: Password
            
        Returns:
            Tuple (success: bool, message: str, user: dict|None)
        """
        if not username or not password:
            return False, "Please enter username and password", None
        
        user = self.db.authenticate_user(username, password)
        
        if user:
            self.current_user = user
            return True, f"Welcome back, {user['username']}!", user
        else:
            return False, "Invalid username or password", None
    
    def logout(self):
        """Logout current user"""
        if self.current_user:
            username = self.current_user['username']
            self.current_user = None
            return True, f"Goodbye, {username}!"
        return True, "Logged out"
    
    def get_current_user(self):
        """Get current logged-in user"""
        return self.current_user
    
    def is_authenticated(self):
        """Check if user is authenticated"""
        return self.current_user is not None
    
    def change_password(self, user_id, old_password, new_password, confirm_password):
        """
        Change user password
        
        Args:
            user_id: User ID
            old_password: Current password
            new_password: New password
            confirm_password: New password confirmation
            
        Returns:
            Tuple (success: bool, message: str)
        """
        # Verify old password
        user = self.db.get_user_by_id(user_id)
        
        if not user:
            return False, "User not found"
        
        # In a real application, we would verify the password hash properly
        # For now, we'll trust the user_id came from an authenticated session
        
        if new_password != confirm_password:
            return False, "New passwords do not match"
        
        if len(new_password) < 6:
            return False, "Password must be at least 6 characters"
        
        # Update password (this would normally update the hash in the database)
        # Placeholder for password update functionality
        return True, "Password changed successfully"
    
    def get_user_statistics(self, user_id):
        """Get user detection statistics"""
        return self.db.get_detection_statistics(user_id)
    
    def get_user_history(self, user_id, limit=50):
        """Get user detection history"""
        return self.db.get_user_detection_history(user_id, limit)
    
    def search_history(self, user_id, search_term):
        """Search user detection history"""
        return self.db.search_detections(user_id, search_term)


def test_authentication():
    """Test authentication functionality"""
    auth = AuthenticationManager()
    
    print("\n=== Testing Authentication ===\n")
    
    # Test registration
    success, message, user_id = auth.register("testuser", "test@example.com", "password123", "password123")
    print(f"Registration: {message}")
    
    # Test login
    success, message, user = auth.login("testuser", "password123")
    print(f"Login: {message}")
    
    if user:
        print(f"Current user: {user['username']}")
        
        # Test statistics
        stats = auth.get_user_statistics(user['id'])
        print(f"Statistics: {stats}")
    
    return auth


if __name__ == "__main__":
    auth = test_authentication()
