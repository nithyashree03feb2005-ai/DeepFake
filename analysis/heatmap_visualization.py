"""
Heatmap Visualization Module using Grad-CAM
Visualizes manipulated regions in images
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

# Safe TensorFlow import - try to import but don't fail if not available
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model, load_model
    TENSORFLOW_AVAILABLE = True
except ImportError as e:
    tf = None
    TENSORFLOW_AVAILABLE = False
    print(f"⚠️ TensorFlow/Keras not available in heatmap module: {e}")


class HeatmapVisualizer:
    """Generate Grad-CAM heatmaps for deepfake detection"""
    
    def __init__(self, model_path='models/image_model.h5', img_size=(150, 150)):
        self.model_path = model_path
        self.img_size = img_size
        self.last_conv_layer = None
        self.model = None
        self.heatmap_model = None
        
        # Load model
        self._load_model()
    
    def _load_model(self):
        """Load pretrained model and identify last convolutional layer"""
        try:
            if tf.io.gfile.exists(self.model_path):
                self.model = load_model(self.model_path)
                print(f"✓ Model loaded from {self.model_path}")
                
                # Find last convolutional layer
                for layer in reversed(self.model.layers):
                    if 'conv' in layer.name or 'Conv' in layer.name:
                        self.last_conv_layer = layer.name
                        print(f"✓ Using last conv layer: {self.last_conv_layer}")
                        break
                
                if self.last_conv_layer is None:
                    print("⚠ No convolutional layer found, using default")
                    self.last_conv_layer = 'conv5_block3_out'
                
                # Create heatmap model
                self.heatmap_model = Model(
                    inputs=self.model.input,
                    outputs=[
                        self.model.get_layer(self.last_conv_layer).output,
                        self.model.output
                    ]
                )
            else:
                print(f"⚠ Model not found at {self.model_path}")
                self.model = None
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            self.model = None
    
    def generate_grad_cam(self, image_input, class_idx=0):
        """
        Generate Grad-CAM heatmap
        
        Args:
            image_input: Image path or array
            class_idx: Class index (0 for real, 1 for fake)
            
        Returns:
            Heatmap overlay on original image
        """
        if self.model is None:
            print("⚠ Heatmap generation failed: Model not loaded")
            return {
                'success': False,
                'error': 'Model not loaded'
            }
        
        try:
            # Load and preprocess image
            if isinstance(image_input, str):
                img = cv2.imread(image_input)
                if img is None:
                    return {
                        'success': False,
                        'error': f'Failed to load image: {image_input}'
                    }
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif isinstance(image_input, np.ndarray):
                img_rgb = (image_input * 255).astype(np.uint8)
                img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            else:
                return {
                    'success': False,
                    'error': 'Invalid input type'
                }
            
            print(f"🔍 Generating Grad-CAM heatmap...")
            print(f"   Image shape: {img.shape}")
            print(f"   Target size: {self.img_size}")
            
            # Resize for model input
            img_resized = cv2.resize(img, self.img_size)
            img_normalized = img_resized.astype(np.float32) / 255.0
            img_batch = np.expand_dims(img_normalized, axis=0)
            
            # Get conv feature maps and predictions
            print(f"   Running model prediction...")
            conv_outputs, predictions = self.heatmap_model.predict(img_batch, verbose=0)
            print(f"   Predictions: {predictions}")
            
            # Simple Grad-CAM: Use weighted average of convolutional outputs
            # Get the last convolutional layer output
            conv_output = conv_outputs[0]
            
            # Get weights from the final dense layer for the predicted class
            # For binary classification with sigmoid, we use the prediction directly
            pred_class_idx = 0 if predictions[0, 0] < 0.5 else 1
            
            # Compute weights as mean of each channel (simplified approach)
            pooled_conv = np.mean(conv_output, axis=(0, 1))
            
            # Weight each channel by its importance
            weighted_conv = conv_output * pooled_conv[np.newaxis, np.newaxis, :]
            
            # Generate heatmap by summing weighted channels
            heatmap = np.sum(weighted_conv, axis=-1)
            heatmap = np.maximum(heatmap, 0)  # ReLU
            
            # Normalize heatmap
            if heatmap.max() > 0:
                heatmap /= heatmap.max()
            
            print(f"   Heatmap generated: min={heatmap.min():.3f}, max={heatmap.max():.3f}")
            
            # Resize heatmap to match image size
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            
            # Apply colormap
            heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
            
            # Overlay heatmap on original image
            superimposed = cv2.addWeighted(img_rgb, 0.6, heatmap_color, 0.4, 0)
            
            result = {
                'success': True,
                'heatmap': heatmap,
                'heatmap_color': heatmap_color,
                'overlay': superimposed,
                'prediction': float(predictions[0, class_idx]),
                'manipulation_regions': self._identify_manipulation_regions(heatmap, prediction_score=predictions[0, 0])
            }
            
            print(f"✓ Heatmap generated successfully!")
            print(f"   Prediction score: {predictions[0, 0]:.4f}")
            print(f"   Manipulation severity: {result['manipulation_regions']['severity']}")
            return result
            
        except Exception as e:
            print(f"✗ Error generating heatmap: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e)
            }
    
    def _identify_manipulation_regions(self, heatmap, prediction_score=None, threshold=0.3):
        """
        Identify regions with high manipulation probability
        
        Args:
            heatmap: Heatmap array (shows feature importance)
            prediction_score: Model prediction score (0-1, >0.5 = fake)
            threshold: Threshold for manipulation detection
            
        Returns:
            Dictionary with manipulation region info
        """
        # Adjust interpretation based on prediction
        if prediction_score is not None:
            # If prediction says REAL (< 0.5), heatmap shows authentic features
            # If prediction says FAKE (> 0.5), heatmap shows manipulated features
            is_fake = prediction_score > 0.5
            
            if not is_fake:
                # For real images, low heatmap values mean authentic
                # Use inverse thresholding
                mask = heatmap < (1 - threshold)
                severity_label = 'low'  # Real images have low manipulation
                manipulation_ratio = 1 - (np.sum(heatmap > threshold) / heatmap.size)
            else:
                # For fake images, high heatmap values show manipulation
                mask = heatmap > threshold
                manipulation_ratio = np.sum(mask) / heatmap.size
                severity_label = 'high' if manipulation_ratio > 0.3 else 'medium' if manipulation_ratio > 0.1 else 'low'
        else:
            # Fallback to original behavior
            mask = heatmap > threshold
            manipulation_ratio = np.sum(mask) / heatmap.size
            severity_label = 'high' if manipulation_ratio > 0.3 else 'medium' if manipulation_ratio > 0.1 else 'low'
        
        # Find contours of manipulated regions
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                regions.append({
                    'x': int(x),
                    'y': int(y),
                    'width': int(w),
                    'height': int(h),
                    'area': int(cv2.contourArea(contour))
                })
        
        # Override severity based on prediction confidence if available
        if prediction_score is not None:
            is_fake = prediction_score > 0.5
            confidence = abs(prediction_score - 0.5) * 2  # Normalize to 0-1
            
            if not is_fake and confidence > 0.9:
                severity_label = 'none'  # Very confident real image
            elif is_fake and confidence < 0.6:
                severity_label = 'low'  # Uncertain fake prediction
        
        return {
            'manipulation_ratio': float(manipulation_ratio),
            'num_regions': len(regions),
            'regions': regions,
            'severity': severity_label
        }
    
    def visualize_multiple_images(self, image_paths, save_path=None):
        """
        Visualize heatmaps for multiple images
        
        Args:
            image_paths: List of image paths
            save_path: Path to save visualization (optional)
            
        Returns:
            Matplotlib figure
        """
        n_images = len(image_paths)
        fig, axes = plt.subplots(n_images, 2, figsize=(12, 6 * n_images))
        
        if n_images == 1:
            axes = axes.reshape(1, -1)
        
        for i, img_path in enumerate(image_paths):
            result = self.generate_grad_cam(img_path)
            
            if result['success']:
                # Original image
                img = cv2.imread(img_path)
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[i, 0].imshow(img_rgb)
                axes[i, 0].set_title(f'Original Image\nPrediction: {result["prediction"]:.2%}')
                axes[i, 0].axis('off')
                
                # Heatmap overlay
                axes[i, 1].imshow(result['overlay'])
                axes[i, 1].set_title(f'Grad-CAM Heatmap\nManipulation: {result["manipulation_regions"]["severity"].upper()}')
                axes[i, 1].axis('off')
            else:
                axes[i, 0].text(0.5, 0.5, f'Error: {result.get("error", "Unknown")}', 
                               ha='center', va='center')
                axes[i, 0].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        return fig
    
    def create_comparison_visualization(self, real_image, fake_image, save_path=None):
        """
        Create side-by-side comparison of real vs fake
        
        Args:
            real_image: Real image path
            fake_image: Fake image path
            save_path: Save path (optional)
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Real image
        real_result = self.generate_grad_cam(real_image, class_idx=0)
        real_img = cv2.imread(real_image)
        real_rgb = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)
        
        axes[0, 0].imshow(real_rgb)
        axes[0, 0].set_title('Real Image')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(real_result['overlay'])
        axes[0, 1].set_title(f'Real Image Heatmap\nConfidence: {real_result["prediction"]:.2%}')
        axes[0, 1].axis('off')
        
        # Fake image
        fake_result = self.generate_grad_cam(fake_image, class_idx=1)
        fake_img = cv2.imread(fake_image)
        fake_rgb = cv2.cvtColor(fake_img, cv2.COLOR_BGR2RGB)
        
        axes[1, 0].imshow(fake_rgb)
        axes[1, 0].set_title('Fake Image')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(fake_result['overlay'])
        axes[1, 1].set_title(f'Fake Image Heatmap\nConfidence: {fake_result["prediction"]:.2%}')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison saved to {save_path}")
        
        return fig


def test_heatmap_visualization():
    """Test heatmap visualization functionality"""
    visualizer = HeatmapVisualizer()
    
    print("\nHeatmap Visualizer initialized")
    print("Note: Requires trained model for actual heatmap generation")
    
    return visualizer


if __name__ == "__main__":
    visualizer = test_heatmap_visualization()
