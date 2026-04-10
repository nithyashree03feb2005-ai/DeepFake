"""
FAST Video DeepFake Detection Training
Optimized for quick completion with minimal logging
"""

import os
import sys

# Set environment variables BEFORE importing tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress ALL TF logs
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.dataset_loader import DatasetLoader


def create_video_model(img_size=(224, 224), max_frames=30):
    """Create simplified CNN + LSTM model"""
    
    # Simpler architecture for speed
    cnn_base = tf.keras.Sequential([
        tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
            input_shape=(max_frames, img_size[0], img_size[1], 3)
        ),
        tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization()),
        tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2))),
        
        tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')),
        tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization()),
        tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2))),
        
        tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D())
    ])
    
    lstm_layers = tf.keras.Sequential([
        tf.keras.layers.LSTM(128, return_sequences=False),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    model = tf.keras.Sequential([cnn_base, lstm_layers])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def train_fast():
    """Fast training with minimal overhead"""
    print("=" * 60)
    print("FAST Video DeepFake Training")
    print("=" * 60)
    
    # Clear session
    tf.keras.backend.clear_session()
    
    # Load dataset with reduced frames
    print("\nLoading videos (max 30 frames each)...")
    loader = DatasetLoader(dataset_path='dataset', img_size=(224, 224), max_frames=30)
    X_train, X_val, y_train, y_val = loader.load_video_dataset()
    
    print(f"✓ Loaded {len(X_train)} training, {len(X_val)} validation samples")
    
    # Create model
    print("\nCreating model...")
    model = create_video_model(img_size=(224, 224), max_frames=30)
    model.summary()
    
    # Callbacks
    callbacks = [
        ModelCheckpoint('models/video_model_fast.h5', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7, verbose=1)
    ]
    
    # Train - FAST!
    print("\n" + "=" * 60)
    print("Starting Training (Fast Mode - 10 epochs max)")
    print("=" * 60 + "\n")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=4,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save
    model.save('models/video_model_final_fast.h5')
    print("\n✓ Training Complete!")
    print(f"✓ Model saved to models/video_model_final_fast.h5")
    
    return model, history


if __name__ == "__main__":
    try:
        model, history = train_fast()
        print("\n🎉 SUCCESS! Video training completed!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
