"""
Real-Time Webcam DeepFake Detection - HIGH PERFORMANCE
Optimized for 20+ FPS with instant face detection and analysis
"""

import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime
import time
from threading import Thread, Lock
import collections

# ============== OPTIMIZATION CONFIGURATION ==============

# Frame processing optimization
PROCESS_EVERY_NTH_FRAME = 3  # Process every 3rd frame for speed
TARGET_WIDTH = 640  # Resize width
TARGET_HEIGHT = 480  # Resize height

# Model optimization - use GPU if available
GPU_MEMORY_FRACTION = 0.5  # Use 50% of GPU memory
ALLOW_GPU_GROWTH = True

# Detection optimization
CONFIDENCE_THRESHOLD = 0.50  # Minimum confidence for display
FPS_DISPLAY_POSITION = (10, 30)  # Top-left corner

# ============== FACE DETECTOR (Haar Cascade - FAST) ==============

class FastFaceDetector:
    """Ultra-fast Haar Cascade face detector"""
    
    def __init__(self):
        # Load Haar Cascade (very fast, CPU-based)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Multi-scale detection parameters (optimized for speed)
        self.scaleFactor = 1.1
        self.minNeighbors = 5
        self.minSize = (30, 30)
        self.flags = cv2.CASCADE_SCALE_IMAGE
    
    def detect(self, frame_bgr):
        """
        Detect faces in frame
        Returns: List of bounding boxes [(x, y, w, h), ...]
        """
        # Convert to grayscale (required for Haar Cascade)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scaleFactor,
            minNeighbors=self.minNeighbors,
            minSize=self.minSize,
            flags=self.flags
        )
        
        return faces
    
    def draw_boxes(self, frame, faces, color=(0, 255, 0), thickness=2):
        """Draw bounding boxes on frame"""
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
        return frame


# ============== DEEPFAKE CLASSIFIER (Lightweight) ==============

class LightweightDeepFakeClassifier:
    """Optimized DeepFake classifier using MobileNet/EfficientNet"""
    
    def __init__(self, model_path=None):
        """
        Initialize classifier
        
        Args:
            model_path: Path to trained model (.h5 file)
        """
        self.model = None
        self.model_path = model_path
        self.input_size = (224, 224)  # Standard input size
        
        # Configure GPU
        self._configure_gpu()
        
        # Load model
        if model_path:
            self.load_model(model_path)
    
    def _configure_gpu(self):
        """Configure TensorFlow GPU settings"""
        gpus = tf.config.list_physical_devices('GPU')
        
        if gpus:
            try:
                # Configure GPU memory
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, ALLOW_GPU_GROWTH)
                    tf.config.experimental.set_virtual_device_configuration(
                        gpu,
                        [tf.config.experimental.VirtualDeviceConfiguration(
                            memory_limit=int(4096 * GPU_MEMORY_FRACTION)  # MB
                        )]
                    )
                print(f"✓ GPU configured: {len(gpus)} device(s)")
            except RuntimeError as e:
                print(f"⚠ GPU config error: {e}")
        else:
            print("ℹ No GPU detected, using CPU")
    
    def load_model(self, model_path):
        """Load pretrained model"""
        try:
            # Load model with optimizations
            self.model = tf.keras.models.load_model(
                model_path,
                compile=False  # Faster loading
            )
            
            # Optimize for inference
            self.model.compile(optimizer='adam', loss='binary_crossentropy')
            
            # Warm up (first inference is slow)
            dummy_input = np.random.rand(1, *self.input_size, 3).astype(np.float32)
            _ = self.model.predict(dummy_input, verbose=0)
            
            print(f"✓ Model loaded: {model_path}")
            print(f"  Input shape: {self.model.input_shape}")
            print(f"  Output shape: {self.model.output_shape}")
            
        except Exception as e:
            print(f"❌ Model load error: {e}")
            self.model = None
    
    def preprocess(self, image_bgr):
        """
        Preprocess image for model
        
        Args:
            image_bgr: OpenCV BGR image
            
        Returns:
            Preprocessed tensor
        """
        # Resize to model input size
        resized = cv2.resize(image_bgr, self.input_size)
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        # Add batch dimension
        batched = np.expand_dims(normalized, axis=0)
        
        return batched
    
    def predict(self, image_bgr):
        """
        Run inference on face image
        
        Args:
            image_bgr: Face crop in BGR format
            
        Returns:
            dict with prediction results
        """
        if self.model is None:
            return {'success': False, 'error': 'Model not loaded'}
        
        # Preprocess
        input_tensor = self.preprocess(image_bgr)
        
        # Run inference
        start_time = time.time()
        predictions = self.model.predict(input_tensor, verbose=0)
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Parse predictions
        fake_prob = float(predictions[0][0])
        real_prob = 1.0 - fake_prob
        
        is_fake = fake_prob > CONFIDENCE_THRESHOLD
        confidence = max(fake_prob, real_prob)
        
        result = {
            'success': True,
            'is_fake': is_fake,
            'fake_probability': fake_prob,
            'real_probability': real_prob,
            'confidence': confidence,
            'inference_time_ms': inference_time
        }
        
        return result


# ============== THREADED VIDEO CAPTURE ==============

class ThreadedWebcamCapture:
    """Non-blocking video capture using threading"""
    
    def __init__(self, src=0):
        """
        Initialize webcam capture
        
        Args:
            src: Camera source (0 = default webcam)
        """
        self.src = src
        self.cap = None
        self.ret = False
        self.frame = None
        self.lock = Lock()
        self.stopped = False
        self.thread = None
        
        # Performance tracking
        self.fps_counter = collections.deque(maxlen=30)  # Last 30 frames
    
    def start(self):
        """Start threaded capture"""
        if self.cap is None:
            # Initialize camera
            self.cap = cv2.VideoCapture(self.src)
            
            # Set optimal camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
            
            if not self.cap.isOpened():
                raise IOError(f"Cannot open webcam {self.src}")
        
        # Start thread
        self.thread = Thread(target=self._update, args=(), daemon=True)
        self.thread.start()
        
        print(f"✓ Webcam started: {TARGET_WIDTH}x{TARGET_HEIGHT}@30fps")
        return self
    
    def _update(self):
        """Thread worker - continuously grab frames"""
        while not self.stopped:
            if self.cap is not None:
                ret, frame = self.cap.read()
                
                with self.lock:
                    self.ret = ret
                    self.frame = frame
                    
                    # Track FPS
                    if ret:
                        self.fps_counter.append(time.time())
    
    def read(self):
        """Get latest frame (non-blocking)"""
        with self.lock:
            if self.ret:
                return self.ret, self.frame.copy()
            else:
                return False, None
    
    def get_fps(self):
        """Calculate current FPS"""
        if len(self.fps_counter) < 2:
            return 0.0
        
        time_diff = self.fps_counter[-1] - self.fps_counter[0]
        if time_diff <= 0:
            return 0.0
        
        fps = (len(self.fps_counter) - 1) / time_diff
        return round(fps, 1)
    
    def stop(self):
        """Stop capture and release resources"""
        self.stopped = True
        
        if self.thread:
            self.thread.join(timeout=1.0)
        
        if self.cap:
            self.cap.release()
        
        print("✓ Webcam stopped")


# ============== MAIN REAL-TIME DETECTOR ==============

class RealTimeDeepFakeDetector:
    """Main real-time DeepFake detection system"""
    
    def __init__(self, model_path=None):
        """
        Initialize detector
        
        Args:
            model_path: Path to trained DeepFake model
        """
        print("="*60)
        print("REAL-TIME WEBCAM DEEPFAKE DETECTOR")
        print("="*60)
        
        # Initialize components
        self.face_detector = FastFaceDetector()
        self.classifier = LightweightDeepFakeClassifier(model_path)
        self.webcam = None
        
        # Processing state
        self.frame_count = 0
        self.process_this_frame = False
        
        # Results caching (smooth display)
        self.last_result = None
        self.result_lock = Lock()
        
        # Performance metrics
        self.total_frames = 0
        self.processed_frames = 0
        self.detection_times = collections.deque(maxlen=30)
    
    def start(self, webcam_id=0):
        """Start real-time detection"""
        print("\n🚀 Starting real-time detection...")
        print(f"   Model: {self.classifier.model_path or 'None'}")
        print(f"   Process every {PROCESS_EVERY_NTH_FRAME}rd frame")
        print(f"   Target resolution: {TARGET_WIDTH}x{TARGET_HEIGHT}")
        print(f"   Confidence threshold: {CONFIDENCE_THRESHOLD}")
        print("="*60 + "\n")
        
        # Start webcam
        try:
            self.webcam = ThreadedWebcamCapture(src=webcam_id)
            self.webcam.start()
            time.sleep(1.0)  # Wait for camera warm-up
        except Exception as e:
            print(f"❌ Failed to start webcam: {e}")
            return False
        
        # Main loop
        self._run_detection_loop()
        
        return True
    
    def _run_detection_loop(self):
        """Main detection loop"""
        print("Press 'q' to quit, 's' to save screenshot\n")
        
        while True:
            # Get frame from webcam
            ret, frame = self.webcam.read()
            
            if not ret:
                print("⚠️ Failed to grab frame")
                break
            
            self.total_frames += 1
            self.frame_count += 1
            
            # Decide whether to process this frame
            self.process_this_frame = (self.frame_count % PROCESS_EVERY_NTH_FRAME == 0)
            
            # Start timing
            frame_start = time.time()
            
            if self.process_this_frame:
                # Process frame
                frame_display, result = self._process_frame(frame)
                
                # Update cached result
                with self.result_lock:
                    self.last_result = result
                
                self.processed_frames += 1
            else:
                # Just display frame without processing
                frame_display = frame.copy()
                
                # Show last known result
                with self.result_lock:
                    if self.last_result:
                        self._draw_overlay(frame_display, self.last_result)
            
            # Calculate frame processing time
            frame_time = (time.time() - frame_start) * 1000  # ms
            self.detection_times.append(frame_time)
            
            # Display FPS
            current_fps = self.webcam.get_fps()
            avg_time = sum(self.detection_times) / len(self.detection_times) if self.detection_times else 0
            
            self._draw_fps(frame_display, current_fps, avg_time)
            
            # Show frame
            cv2.imshow('Real-Time DeepFake Detection - Press Q to Quit', frame_display)
            
            # Check for key press
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\n⏹️ Stopping...")
                break
            elif key == ord('s'):
                self._save_screenshot(frame)
        
        # Cleanup
        self.stop()
    
    def _process_frame(self, frame):
        """
        Process single frame for DeepFake detection
        
        Args:
            frame: Original BGR frame
            
        Returns:
            (display_frame, result_dict)
        """
        display_frame = frame.copy()
        
        # Detect faces
        faces = self.face_detector.detect(frame)
        
        if len(faces) == 0:
            # No face detected
            self.face_detector.draw_boxes(display_frame, [], color=(0, 0, 255))
            cv2.putText(display_frame, "NO FACE DETECTED", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return display_frame, None
        
        # Process first face (or batch if you want multiple)
        face_rect = faces[0]
        x, y, w, h = face_rect
        
        # Extract face ROI
        face_roi = frame[y:y+h, x:x+w]
        
        # Ensure face ROI has minimum size
        if face_roi.shape[0] < 20 or face_roi.shape[1] < 20:
            return display_frame, None
        
        # Run DeepFake classification
        result = self.classifier.predict(face_roi)
        
        # Draw bounding box with result
        if result['success']:
            # Choose color based on prediction
            if result['is_fake']:
                box_color = (0, 0, 255)  # Red (BGR)
                label = f"FAKE {result['confidence']*100:.1f}%"
            else:
                box_color = (0, 255, 0)  # Green (BGR)
                label = f"REAL {result['confidence']*100:.1f}%"
            
            # Draw box
            self.face_detector.draw_boxes(display_frame, [face_rect], 
                                         color=box_color, thickness=3)
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                        0.8, 2)[0]
            cv2.rectangle(display_frame, 
                         (x, y - label_size[1] - 10),
                         (x + label_size[0], y),
                         box_color, -1)
            
            # Draw label text
            cv2.putText(display_frame, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add inference time
            time_label = f"{result['inference_time_ms']:.1f}ms"
            cv2.putText(display_frame, time_label, (x, y + h + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1)
        else:
            # Detection failed
            self.face_detector.draw_boxes(display_frame, [face_rect], 
                                         color=(0, 0, 255))
        
        return display_frame, result
    
    def _draw_overlay(self, frame, result):
        """Draw result overlay on frame"""
        if result is None:
            return
        
        # Draw info panel
        panel_y = frame.shape[0] - 80
        
        cv2.rectangle(frame, (0, panel_y), (frame.shape[1], frame.shape[0]),
                     (0, 0, 0), -1)
        
        if result['is_fake']:
            status_text = "STATUS: FAKE DETECTED"
            status_color = (0, 0, 255)
        else:
            status_text = "STATUS: REAL VERIFIED"
            status_color = (0, 255, 0)
        
        cv2.putText(frame, status_text, (10, panel_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        conf_text = f"Confidence: {result['confidence']*100:.1f}%"
        cv2.putText(frame, conf_text, (10, panel_y + 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    def _draw_fps(self, frame, fps, avg_time):
        """Draw FPS counter"""
        fps_text = f"FPS: {fps:.1f}"
        time_text = f"Frame Time: {avg_time:.1f}ms"
        
        cv2.putText(frame, fps_text, FPS_DISPLAY_POSITION,
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.putText(frame, time_text, (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    
    def _save_screenshot(self, frame):
        """Save current frame as screenshot"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        print(f"✓ Screenshot saved: {filename}")
    
    def stop(self):
        """Stop detector and cleanup"""
        if self.webcam:
            self.webcam.stop()
        
        cv2.destroyAllWindows()
        
        # Print statistics
        if self.total_frames > 0:
            print(f"\n📊 Session Statistics:")
            print(f"   Total frames: {self.total_frames}")
            print(f"   Processed frames: {self.processed_frames}")
            print(f"   Processing rate: {self.processed_frames/self.total_frames*100:.1f}%")
            if self.detection_times:
                print(f"   Avg frame time: {sum(self.detection_times)/len(self.detection_times):.1f}ms")


# ============== COMMAND LINE INTERFACE ==============

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-Time Webcam DeepFake Detector')
    parser.add_argument('--model', type=str, default='models/image_model_fast.h5',
                       help='Path to trained DeepFake model (.h5 file)')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera ID (default: 0)')
    
    args = parser.parse_args()
    
    # Create detector
    detector = RealTimeDeepFakeDetector(model_path=args.model)
    
    # Start detection
    detector.start(webcam_id=args.camera)


if __name__ == '__main__':
    main()
