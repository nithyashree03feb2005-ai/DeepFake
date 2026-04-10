"""
Train Webcam DeepFake Detection Model
Uses available webcam dataset with various attack types
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Safe tqdm import - try to import but don't fail if not available
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    tqdm = None
    TQDM_AVAILABLE = False
    print("⚠️ tqdm not available in webcam training module")

# Safe cv2 import - try to import but don't fail if not available
try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    cv2 = None
    _CV2_AVAILABLE = False
    print("⚠️ OpenCV not available in webcam training module")

# Set environment variables
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def prepare_webcam_dataset(dataset_path='Dataset/webcam/files', img_size=(224, 224)):
    """
    Prepare webcam dataset for training
    
    Dataset structure:
    webcam/files/
        ├── real/ (genuine face videos)
        ├── mask/ (mask attack videos)
        ├── print/ (print attack videos)
        ├── print_cut/ (cut print attack videos)
        ├── outline/ (outline attack videos)
        ├── monitor/ (replay attack videos)
        └── silicone/ (silicone mask attack videos)
    
    Extracts frames from videos for training
    """
    
    print("=" * 70)
    print("Preparing Webcam Dataset")
    print("=" * 70)
    
    # Define classes
    # Real = 0, Fake = 1 (all non-real types are fake)
    real_dir = os.path.join(dataset_path, 'real')
    fake_dirs = [
        os.path.join(dataset_path, 'mask'),
        os.path.join(dataset_path, 'print'),
        os.path.join(dataset_path, 'print_cut'),
        os.path.join(dataset_path, 'outline'),
        os.path.join(dataset_path, 'monitor'),
        os.path.join(dataset_path, 'silicone')
    ]
    
    # Extract frames from video
    def extract_frames_from_video(video_path, max_frames=10):
        """Extract frames from video file"""
        frames = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return frames
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            # Sample frames evenly
            indices = np.linspace(0, min(frame_count - 1, max_frames - 1), max_frames, dtype=int)
            
            for idx in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, img_size)
                    frame = frame.astype(np.float32) / 255.0
                    frames.append(frame)
            
            cap.release()
            
        except Exception as e:
            print(f"Error extracting frames from {video_path}: {e}")
        
        return frames
    
    # Load videos and extract frames
    def load_videos_from_dir(directory, label):
        all_frames = []
        all_labels = []
        
        if not os.path.exists(directory):
            print(f"⚠ Directory not found: {directory}")
            return all_frames, all_labels
        
        for filename in os.listdir(directory):
            if filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_path = os.path.join(directory, filename)
                
                # Extract frames
                frames = extract_frames_from_video(video_path, max_frames=10)
                
                if len(frames) > 0:
                    all_frames.extend(frames)
                    all_labels.extend([label] * len(frames))
                    print(f"  - {filename}: extracted {len(frames)} frames")
        
        return all_frames, all_labels
    
    # Load real videos and extract frames
    print("\nLoading REAL videos...")
    real_frames, real_labels = load_videos_from_dir(real_dir, 0)
    print(f"✓ Extracted {len(real_frames)} real frames")
    
    # Load fake videos and extract frames
    print("\nLoading FAKE videos...")
    fake_frames = []
    fake_labels = []
    
    for fake_dir in fake_dirs:
        dir_name = os.path.basename(fake_dir)
        frames, labels = load_videos_from_dir(fake_dir, 1)
        fake_frames.extend(frames)
        fake_labels.extend(labels)
        if len(frames) > 0:
            print(f"  - {dir_name}: extracted {len(frames)} frames")
    
    print(f"\n✓ Total FAKE frames: {len(fake_frames)}")
    
    # Combine datasets
    all_frames = real_frames + fake_frames
    all_labels = real_labels + fake_labels
    
    if len(all_frames) == 0:
        print("✗ No frames extracted! Check dataset path and video files.")
        return None, None
    
    # Convert to numpy arrays
    X = np.array(all_frames)
    y = np.array(all_labels)
    
    # Shuffle data
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Split into train and validation
    split_idx = int(len(X) * 0.8)
    
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_val = X[split_idx:]
    y_val = y[split_idx:]
    
    print(f"\n✓ Training samples: {len(X_train)} (Real: {sum(y_train)==0}, Fake: {sum(y_train)==1})")
    print(f"✓ Validation samples: {len(X_val)} (Real: {sum(y_val)==0}, Fake: {sum(y_val)==1})")
    
    return (X_train, y_train), (X_val, y_val)


def create_webcam_model(img_size=(224, 224)):
    """
    Create CNN model optimized for webcam face analysis
    Based on MobileNetV2 for speed (important for real-time detection)
    """
    
    print("\nCreating Webcam Detection Model...")
    print("=" * 70)
    
    # Use transfer learning with MobileNetV2 (faster than ResNet for real-time)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(img_size[0], img_size[1], 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Build model
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(img_size[0], img_size[1], 3)),
        
        # Base model
        base_model,
        
        # Custom classification head
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        
        # Output layer
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    print("\nModel Architecture:")
    model.summary()
    
    return model


def train_webcam_model(epochs=30, batch_size=32, img_size=(224, 224)):
    """
    Train webcam detection model
    """
    
    print("\n" + "=" * 70)
    print("Webcam DeepFake Detection Model Training")
    print("=" * 70)
    
    # Clear session
    tf.keras.backend.clear_session()
    
    # Prepare dataset
    train_data, val_data = prepare_webcam_dataset(img_size=img_size)
    
    if train_data is None:
        print("✗ Training failed - no data")
        return None
    
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2]
    )
    datagen.fit(X_train)
    
    # Create model
    model = create_webcam_model(img_size)
    
    # Callbacks
    callbacks = [
        ModelCheckpoint(
            'models/webcam_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train
    print("\n" + "=" * 70)
    print("Starting Training...")
    print("=" * 70 + "\n")
    
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        validation_data=(X_val, y_val),
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\n" + "=" * 70)
    print("Evaluating Model...")
    print("=" * 70)
    
    eval_results = model.evaluate(X_val, y_val, verbose=0)
    print(f"\nValidation Accuracy: {eval_results[1]*100:.2f}%")
    print(f"Validation Loss: {eval_results[0]:.4f}")
    print(f"Precision: {eval_results[2]*100:.2f}%")
    print(f"Recall: {eval_results[3]*100:.2f}%")
    
    # Save final model
    model.save('models/webcam_model_final.h5')
    print("\n✓ Model saved to models/webcam_model_final.h5")
    
    # Plot training history
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Train Acc')
        plt.plot(history.history['val_accuracy'], label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Accuracy')
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss')
        
        plt.tight_layout()
        plt.savefig('models/webcam_training_history.png', dpi=150)
        print("✓ Training history plot saved")
        
    except Exception as e:
        print(f"⚠ Could not save plot: {e}")
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print("\n✅ WEBCAM TRAINING SUCCESSFULLY COMPLETED!")
    print("=" * 70 + "\n")
    
    return model


if __name__ == "__main__":
    model = train_webcam_model(epochs=30, batch_size=32)
    print("\n✓ Webcam model training completed!")
    print("\n" + "=" * 70)
    print("🎉 WEBCAM TRAINING SUCCESSFULLY FINISHED!")
    print("=" * 70)
    print("\nTrained Models:")
    print("  - models/webcam_model.h5 (best validation accuracy)")
    print("  - models/webcam_model_final.h5 (final weights)")
    print("\nTraining History:")
    print("  - models/webcam_training_history.png")
    print("\n" + "=" * 70 + "\n")
