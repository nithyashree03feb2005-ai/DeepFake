"""
Database module for DeepFake Detection System
Handles user authentication and detection history
"""

import sqlite3
import hashlib
import os
from datetime import datetime
import bcrypt


class DatabaseManager:
    """Manage SQLite database for users and detection history"""
    
    def __init__(self, db_path='history/detection_history.db'):
        self.db_path = db_path
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        ''')
        
        # Create detection history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detection_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                file_name TEXT NOT NULL,
                file_type TEXT NOT NULL,
                file_path TEXT,
                prediction TEXT NOT NULL,
                confidence REAL NOT NULL,
                is_fake BOOLEAN NOT NULL,
                analysis_details TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Create reports table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                detection_id INTEGER NOT NULL,
                report_path TEXT NOT NULL,
                generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id),
                FOREIGN KEY (detection_id) REFERENCES detection_history (id)
            )
        ''')
        
        # Create indexes for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_user_id ON detection_history(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON detection_history(created_at)')
        
        conn.commit()
        conn.close()
        print(f"Database initialized at {self.db_path}")
    
    def _get_connection(self):
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    # User Management
    
    def create_user(self, username, email, password):
        """
        Create new user
        
        Args:
            username: Username
            email: Email
            password: Password
            
        Returns:
            User ID if successful, None otherwise
        """
        try:
            # Hash password
            password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO users (username, email, password_hash)
                VALUES (?, ?, ?)
            ''', (username, email, password_hash))
            
            user_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            print(f"User created: {username} (ID: {user_id})")
            return user_id
            
        except sqlite3.IntegrityError as e:
            print(f"User creation failed: {e}")
            return None
        except Exception as e:
            print(f"Error creating user: {e}")
            return None
    
    def authenticate_user(self, username, password):
        """
        Authenticate user
        
        Args:
            username: Username
            password: Password
            
        Returns:
            User dict if successful, None otherwise
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM users WHERE username = ? AND is_active = 1
            ''', (username,))
            
            user = cursor.fetchone()
            
            if user:
                # Verify password
                password_match = bcrypt.checkpw(
                    password.encode('utf-8'),
                    user['password_hash'].encode('utf-8')
                )
                
                if password_match:
                    # Update last login
                    cursor.execute('''
                        UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?
                    ''', (user['id'],))
                    conn.commit()
                    
                    user_dict = dict(user)
                    conn.close()
                    
                    print(f"User authenticated: {username}")
                    return user_dict
                else:
                    conn.close()
                    print("Invalid password")
                    return None
            else:
                conn.close()
                print("User not found")
                return None
                
        except Exception as e:
            print(f"Error authenticating user: {e}")
            return None
    
    def get_user_by_id(self, user_id):
        """Get user by ID"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
            user = cursor.fetchone()
            
            conn.close()
            
            return dict(user) if user else None
            
        except Exception as e:
            print(f"Error getting user: {e}")
            return None
    
    def update_user(self, user_id, **kwargs):
        """Update user information"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Build update query dynamically
            updates = []
            values = []
            
            for key, value in kwargs.items():
                if key in ['username', 'email', 'is_active']:
                    updates.append(f"{key} = ?")
                    values.append(value)
            
            if updates:
                values.append(user_id)
                query = f"UPDATE users SET {', '.join(updates)} WHERE id = ?"
                
                cursor.execute(query, values)
                conn.commit()
                print(f"User {user_id} updated")
            
            conn.close()
            return True
            
        except Exception as e:
            print(f"Error updating user: {e}")
            return False
    
    # Detection History Management
    
    def add_detection_record(self, user_id, file_name, file_type, prediction, confidence, 
                            is_fake, file_path=None, analysis_details=None):
        """
        Add detection record to history
        
        Args:
            user_id: User ID
            file_name: Name of analyzed file
            file_type: Type (image/video/audio)
            prediction: Prediction (Real/Fake)
            confidence: Confidence score
            is_fake: Boolean indicating if fake
            file_path: Path to file
            analysis_details: Additional analysis details (JSON string)
            
        Returns:
            Detection ID if successful
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO detection_history 
                (user_id, file_name, file_type, file_path, prediction, confidence, is_fake, analysis_details)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (user_id, file_name, file_type, file_path, prediction, confidence, is_fake, analysis_details))
            
            detection_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            print(f"Detection record added (ID: {detection_id})")
            return detection_id
            
        except Exception as e:
            print(f"Error adding detection record: {e}")
            return None
    
    def get_user_detection_history(self, user_id, limit=50):
        """
        Get user's detection history
        
        Args:
            user_id: User ID
            limit: Maximum records to return
            
        Returns:
            List of detection records
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM detection_history 
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            ''', (user_id, limit))
            
            records = cursor.fetchall()
            conn.close()
            
            return [dict(record) for record in records]
            
        except Exception as e:
            print(f"Error getting detection history: {e}")
            return []
    
    def get_detection_by_id(self, detection_id):
        """Get specific detection record"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM detection_history WHERE id = ?', (detection_id,))
            record = cursor.fetchone()
            
            conn.close()
            
            return dict(record) if record else None
            
        except Exception as e:
            print(f"Error getting detection record: {e}")
            return None
    
    def search_detections(self, user_id, search_term):
        """Search detection history by filename"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT * FROM detection_history 
                WHERE user_id = ? AND file_name LIKE ?
                ORDER BY created_at DESC
            ''', (user_id, f'%{search_term}%'))
            
            records = cursor.fetchall()
            conn.close()
            
            return [dict(record) for record in records]
            
        except Exception as e:
            print(f"Error searching detections: {e}")
            return []
    
    def get_detection_statistics(self, user_id):
        """
        Get detection statistics for user
        
        Args:
            user_id: User ID
            
        Returns:
            Statistics dictionary
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Total detections
            cursor.execute('''
                SELECT COUNT(*) as total FROM detection_history WHERE user_id = ?
            ''', (user_id,))
            total = cursor.fetchone()['total']
            
            # Fake detections
            cursor.execute('''
                SELECT COUNT(*) as fake_count FROM detection_history 
                WHERE user_id = ? AND is_fake = 1
            ''', (user_id,))
            fake_count = cursor.fetchone()['fake_count']
            
            # Real detections
            real_count = total - fake_count
            
            # Average confidence
            cursor.execute('''
                SELECT AVG(confidence) as avg_confidence FROM detection_history 
                WHERE user_id = ?
            ''', (user_id,))
            avg_confidence = cursor.fetchone()['avg_confidence'] or 0
            
            # By file type
            cursor.execute('''
                SELECT file_type, COUNT(*) as count FROM detection_history 
                WHERE user_id = ?
                GROUP BY file_type
            ''', (user_id,))
            by_type = {row['file_type']: row['count'] for row in cursor.fetchall()}
            
            conn.close()
            
            stats = {
                'total_detections': total,
                'fake_count': fake_count,
                'real_count': real_count,
                'average_confidence': avg_confidence,
                'by_file_type': by_type,
                'fake_percentage': (fake_count / total * 100) if total > 0 else 0
            }
            
            return stats
            
        except Exception as e:
            print(f"Error getting statistics: {e}")
            return None
    
    # Report Management
    
    def add_report_record(self, user_id, detection_id, report_path):
        """Add generated report record"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO reports (user_id, detection_id, report_path)
                VALUES (?, ?, ?)
            ''', (user_id, detection_id, report_path))
            
            report_id = cursor.lastrowid
            conn.commit()
            conn.close()
            
            print(f"Report record added (ID: {report_id})")
            return report_id
            
        except Exception as e:
            print(f"Error adding report: {e}")
            return None
    
    def get_user_reports(self, user_id):
        """Get all reports for user"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT r.*, d.file_name, d.prediction 
                FROM reports r
                JOIN detection_history d ON r.detection_id = d.id
                WHERE r.user_id = ?
                ORDER BY r.generated_at DESC
            ''', (user_id,))
            
            reports = cursor.fetchall()
            conn.close()
            
            return [dict(report) for report in reports]
            
        except Exception as e:
            print(f"Error getting reports: {e}")
            return []
    
    def delete_detection_record(self, detection_id):
        """Delete detection record"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM detection_history WHERE id = ?', (detection_id,))
            conn.commit()
            conn.close()
            
            print(f"Detection record {detection_id} deleted")
            return True
            
        except Exception as e:
            print(f"Error deleting detection record: {e}")
            return False


def test_database():
    """Test database functionality"""
    db = DatabaseManager()
    
    # Test user creation
    user_id = db.create_user("testuser", "test@example.com", "password123")
    
    if user_id:
        # Test authentication
        user = db.authenticate_user("testuser", "password123")
        print(f"Authenticated user: {user['username'] if user else None}")
        
        # Test detection history
        detection_id = db.add_detection_record(
            user_id=user_id,
            file_name="test_video.mp4",
            file_type="video",
            prediction="Fake",
            confidence=0.92,
            is_fake=True
        )
        
        # Get history
        history = db.get_user_detection_history(user_id)
        print(f"Detection history: {len(history)} records")
        
        # Get statistics
        stats = db.get_detection_statistics(user_id)
        print(f"Statistics: {stats}")
    
    return db


if __name__ == "__main__":
    db = test_database()
