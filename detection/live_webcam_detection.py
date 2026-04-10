"""
Live Webcam Real-Time DeepFake Detection
Streams webcam feed with instant frame-by-frame analysis
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
from utils.preprocessing import Preprocessor


class LiveWebcamDetector:
    """Real-time webcam detection with live video streaming"""
    
    def __init__(self, model_path='models/webcam_model.h5', img_size=(224, 224)):
        self.model_path = model_path
        self.img_size = img_size
        self.model = None
        self.preprocessor = Preprocessor()
        
        # Detection settings
        self.detection_interval = 10  # Detect every N frames
        self.confidence_threshold = 0.5
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load trained model"""
        if os.path.exists(self.model_path):
            try:
                self.model = load_model(self.model_path)
                print(f"✓ Live webcam model loaded from {self.model_path}")
            except Exception as e:
                print(f"✗ Error loading model: {e}")
                self.model = None
        else:
            print(f"⚠ Model not found at {self.model_path}")
            print("  Using default image model instead...")
            # Fallback to image model
            default_path = 'models/image_model.h5'
            if os.path.exists(default_path):
                self.model = load_model(default_path)
                self.model_path = default_path
            else:
                self.model = None
    
    def detect_face(self, frame):
        """
        Simple face detection using OpenCV Haar Cascades
        Returns face bounding box or None
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Load Haar cascade (built into OpenCV)
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Detect faces
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        if len(faces) > 0:
            # Return largest face
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            return (x, y, x + w, y + h)
        
        return None
    
    def predict_frame(self, frame, face_bbox):
        """
        Predict if face region is real or fake
        
        Args:
            frame: BGR frame
            face_bbox: (x1, y1, x2, y2) bounding box
            
        Returns:
            prediction, confidence
        """
        if self.model is None or face_bbox is None:
            return None, None
        
        try:
            # Extract face region
            x1, y1, x2, y2 = map(int, face_bbox)
            face_region = frame[y1:y2, x1:x2]
            
            if face_region.size == 0:
                return None, None
            
            # Preprocess
            face_resized = cv2.resize(face_region, self.img_size)
            face_normalized = face_resized.astype(np.float32) / 255.0
            
            # Add batch dimension
            face_batch = np.expand_dims(face_normalized, axis=0)
            
            # Predict
            prediction = self.model.predict(face_batch, verbose=0)[0][0]
            confidence = float(prediction)
            
            return prediction > 0.5, confidence
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return None, None
    
    def start_live_detection(self, camera_id=0):
        """
        Start live webcam detection with real-time overlay
        
        Args:
            camera_id: Camera device ID
        """
        
        if self.model is None:
            print("Error: Model not loaded!")
            return
        
        # Open webcam
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("Error: Cannot open webcam")
            return
        
        print("\n" + "=" * 70)
        print("LIVE Webcam Detection Started")
        print("=" * 70)
        print("\nControls:")
        print("  - Press 'q' to quit")
        print("  - Press 's' to save current frame")
        print("  - Press 'space' to pause/resume")
        print("=" * 70 + "\n")
        
        frame_count = 0
        last_prediction = None
        last_confidence = None
        last_bbox = None
        paused = False
        
        # Statistics
        predictions_list = []
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            
            if not paused:
                # Detect faces
                face_bbox = self.detect_face(frame)
                
                # Perform detection every N frames or when face changes
                if face_bbox is not None:
                    # Check if face moved significantly
                    if last_bbox is None or self._bbox_changed(last_bbox, face_bbox):
                        # Run detection
                        is_fake, confidence = self.predict_frame(frame, face_bbox)
                        
                        if is_fake is not None:
                            last_prediction = is_fake
                            last_confidence = confidence
                            last_bbox = face_bbox
                            predictions_list.append(confidence)
                
                # Draw results
                self._draw_overlay(frame, last_bbox, last_prediction, last_confidence)
            
            # Display statistics
            self._draw_stats(frame, frame_count, predictions_list)
            
            # Show frame
            cv2.imshow('LIVE DeepFake Detection - Press q to quit', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save frame
                filename = f"webcam_capture_{frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"✓ Saved: {filename}")
            elif key == ord(' '):
                paused = not paused
                print(f"{'⏸ Paused' if paused else '▶ Resumed'}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Print summary
        if len(predictions_list) > 0:
            avg_confidence = np.mean(predictions_list)
            fake_count = sum(1 for p in predictions_list if p > 0.5)
            
            print("\n" + "=" * 70)
            print("Detection Summary")
            print("=" * 70)
            print(f"Total detections: {len(predictions_list)}")
            print(f"Average confidence: {avg_confidence*100:.2f}%")
            print(f"Fake frames: {fake_count} ({fake_count/len(predictions_list)*100:.1f}%)")
            print(f"Real frames: {len(predictions_list) - fake_count} ({(len(predictions_list) - fake_count)/len(predictions_list)*100:.1f}%)")
            print("=" * 70)
    
    def _bbox_changed(self, bbox1, bbox2, threshold=20):
        """Check if bounding box changed significantly"""
        if bbox1 is None or bbox2 is None:
            return True
        
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate center points
        cx1 = (x1_1 + x2_1) / 2
        cy1 = (y1_1 + y2_1) / 2
        cx2 = (x1_2 + x2_2) / 2
        cy2 = (y1_2 + y2_2) / 2
        
        # Check distance
        distance = np.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)
        
        return distance > threshold
    
    def _draw_overlay(self, frame, bbox, is_fake, confidence):
        """Draw detection overlay on frame"""
        
        if bbox is None:
            cv2.putText(frame, "No face detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            return
        
        x1, y1, x2, y2 = map(int, bbox)
        
        # Choose color
        if is_fake is None:
            color = (0, 255, 255)  # Yellow (unknown)
        elif is_fake:
            color = (0, 0, 255)  # Red (fake)
        else:
            color = (0, 255, 0)  # Green (real)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        # Draw label background
        if is_fake is not None and confidence is not None:
            label = f"FAKE" if is_fake else "REAL"
            conf_text = f"{confidence*100:.1f}%"
            
            # Calculate text size
            (text_w, text_h), baseline = cv2.getTextSize(
                f"{label} {conf_text}",
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                2
            )
            
            # Draw background
            cv2.rectangle(
                frame,
                (x1, y2 - text_h - baseline - 10),
                (x1 + text_w + 10, y2),
                color,
                -1
            )
            
            # Draw text
            cv2.putText(
                frame,
                f"{label} {conf_text}",
                (x1 + 5, y2 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
    
    def _draw_stats(self, frame, frame_count, predictions):
        """Draw statistics overlay"""
        
        # Frame counter
        cv2.putText(frame, f"Frame: {frame_count}", (10, frame.shape[0] - 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Detection count
        det_count = len(predictions)
        cv2.putText(frame, f"Detections: {det_count}", (10, frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Recent average
        if len(predictions) >= 5:
            recent_avg = np.mean(predictions[-5:])
            status = "FAKE" if recent_avg > 0.5 else "REAL"
            color = (0, 0, 255) if recent_avg > 0.5 else (0, 255, 0)
            
            cv2.putText(frame, f"Status: {status}", (frame.shape[1] - 200, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def test_live_detector():
    """Test live detector"""
    detector = LiveWebcamDetector()
    
    if detector.model is None:
        print("✗ Model not loaded. Please train webcam model first.")
        return
    
    print("\nStarting live detection test...")
    detector.start_live_detection(camera_id=0)


if __name__ == "__main__":
    test_live_detector()
