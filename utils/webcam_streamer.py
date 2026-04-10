"""
Enhanced Real-Time Webcam Streaming for Streamlit
Uses OpenCV direct webcam access with automatic frame processing
"""

import numpy as np
import streamlit as st
from PIL import Image
import time
import tempfile
import os

# Safe cv2 import - try to import but don't fail if not available
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    CV2_AVAILABLE = False
    print("⚠️ OpenCV not available in webcam streamer module")


class EnhancedWebcamStreamer:
    """Enhanced webcam streaming with real-time detection"""
    
    def __init__(self, detector=None):
        self.detector = detector
        self.cap = None
        self.is_running = False
        self.frame_count = 0
        self.detection_interval = 2.0  # seconds
        
    def open_webcam(self, camera_id=0):
        """Open webcam"""
        try:
            self.cap = cv2.VideoCapture(camera_id)
            if not self.cap.isOpened():
                return False
            
            # Set camera properties for better quality
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            return True
        except Exception as e:
            print(f"Error opening webcam: {e}")
            return False
    
    def capture_frame(self):
        """Capture single frame from webcam"""
        if self.cap is None or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        self.frame_count += 1
        return frame
    
    def detect_in_frame(self, frame_bgr):
        """Run detection on frame"""
        if self.detector is None:
            return None
        
        try:
            result = self.detector.detect(frame_bgr)
            return result
        except Exception as e:
            print(f"Detection error: {e}")
            return None
    
    def release(self):
        """Release webcam"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_running = False


def run_enhanced_webcam_detection(detector=None):
    """
    Run enhanced webcam detection with Streamlit
    
    Args:
        detector: DeepFake detector instance
    """
    st.markdown("""
    ### 🎥 Enhanced Real-Time Webcam Detection
    
    **Features:**
    - ✅ Direct webcam access (no browser prompts after first allow)
    - ✅ Automatic frame capture every 2 seconds
    - ✅ Instant AI analysis
    - ✅ Real-time results display
    - ✅ Continuous monitoring
    """)
    
    # Session state
    if 'webcam_streamer' not in st.session_state:
        st.session_state.webcam_streamer = None
    if 'streaming_active' not in st.session_state:
        st.session_state.streaming_active = False
    if 'last_detection_time' not in st.session_state:
        st.session_state.last_detection_time = 0
    if 'detection_results' not in st.session_state:
        st.session_state.detection_results = []
    
    # Controls
    col1, col2 = st.columns([3, 1])
    
    with col1:
        start_btn = st.button(
            "🎬 START LIVE STREAMING", 
            type="primary", 
            use_container_width=True,
            disabled=st.session_state.streaming_active,
            key="start_webcam"
        )
    
    with col2:
        stop_btn = st.button(
            "⏹️ STOP", 
            use_container_width=True,
            disabled=not st.session_state.streaming_active,
            key="stop_webcam"
        )
    
    # Handle start
    if start_btn:
        streamer = EnhancedWebcamStreamer(detector)
        if streamer.open_webcam():
            st.session_state.webcam_streamer = streamer
            st.session_state.streaming_active = True
            st.session_state.last_detection_time = time.time()
            st.success("✅ Webcam started! Processing frames...")
            st.rerun()
        else:
            st.error("❌ Failed to open webcam. Check permissions.")
            return
    
    # Handle stop
    if stop_btn:
        if st.session_state.webcam_streamer:
            st.session_state.webcam_streamer.release()
        st.session_state.streaming_active = False
        st.session_state.webcam_streamer = None
        st.info("⏹️ Streaming stopped")
        st.rerun()
    
    # Main streaming loop
    if st.session_state.streaming_active and st.session_state.webcam_streamer:
        streamer = st.session_state.webcam_streamer
        
        # Status display
        elapsed = time.time() - st.session_state.last_detection_time
        next_capture = max(0, streamer.detection_interval - elapsed)
        
        status_col = st.columns([2, 1])[0]
        with status_col:
            st.info(f"🔴 **LIVE** | Frame #{streamer.frame_count} | Next capture in {next_capture:.1f}s")
        
        # Check if time to detect
        current_time = time.time()
        should_detect = (current_time - st.session_state.last_detection_time) >= streamer.detection_interval
        
        if should_detect:
            # Capture frame
            frame = streamer.capture_frame()
            
            if frame is not None:
                # Convert BGR to RGB for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Display frame
                st.image(frame_rgb, caption=f"📹 Live Frame #{streamer.frame_count}", width=640)
                
                # Run detection
                with st.spinner("🔍 Analyzing..."):
                    result = streamer.detect_in_frame(frame)
                    
                    if result and result.get('success'):
                        # Store result
                        st.session_state.detection_results.append({
                            'frame': streamer.frame_count,
                            'time': current_time,
                            'result': result
                        })
                        
                        # Display result
                        if result['is_fake']:
                            st.error(f"""
                            **⚠️ FAKE DETECTED!**
                            
                            - Confidence: **{result['confidence']*100:.1f}%**
                            - Fake Probability: **{result['fake_probability']*100:.1f}%**
                            - Frame: #{streamer.frame_count}
                            """)
                        else:
                            st.success(f"""
                            **✓ REAL VERIFIED!**
                            
                            - Confidence: **{result['confidence']*100:.1f}%**
                            - Real Probability: **{result['real_probability']*100:.1f}%**
                            - Frame: #{streamer.frame_count}
                            """)
                        
                        # Save to history
                        try:
                            from auth.database import DatabaseManager
                            db = DatabaseManager()
                            db.add_detection_record(
                                user_id=st.session_state.current_user['id'],
                                file_name=f"live_frame_{streamer.frame_count}.jpg",
                                file_type="webcam_live",
                                prediction=result['prediction'],
                                confidence=result['confidence'],
                                is_fake=result['is_fake']
                            )
                            st.caption("✓ Saved to history")
                        except:
                            pass
                        
                        # Update last detection time
                        st.session_state.last_detection_time = current_time
                        
                        # Auto-continue streaming
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.warning("⚠️ Detection failed or no face detected")
                        st.session_state.last_detection_time = current_time
                        st.rerun()
            else:
                st.warning("⚠️ Failed to capture frame")
                st.rerun()
        else:
            # Waiting between detections
            countdown = int(streamer.detection_interval - (current_time - st.session_state.last_detection_time))
            st.info(f"⏳ Next detection in {countdown} second(s)...")
            time.sleep(0.5)
            st.rerun()
    
    elif not st.session_state.streaming_active:
        st.info("👆 Click '🎬 START LIVE STREAMING' to begin")
        
        st.markdown("""
        ---
        ### How It Works:
        
        1. **START** - Click button to activate webcam
        2. **ALLOW** - Grant camera permission when prompted (first time only)
        3. **AUTO-CAPTURE** - Frames captured automatically every 2 seconds
        4. **INSTANT ANALYSIS** - AI detects fake/real immediately
        5. **RESULTS** - Displayed in real-time with confidence scores
        6. **CONTINUOUS** - Runs until you click STOP
        
        ### Tips for Best Results:
        
        - ✅ Good lighting on your face
        - ✅ Face the camera directly
        - ✅ Stay relatively still
        - ✅ Face should be clearly visible
        """)


if __name__ == "__main__":
    # Test without detector
    run_enhanced_webcam_detection(detector=None)
