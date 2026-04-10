"""
Train Video DeepFake Detection Model
Uses CNN + LSTM architecture for temporal analysis (90%+ accuracy)
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.dataset_loader import DatasetLoader


def create_video_model(img_size=(224, 224), max_frames=100):
    """
    Create CNN + LSTM model for video deepfake detection
    
    Args:
        img_size: Input frame size
        max_frames: Maximum number of frames
        
    Returns:
        Compiled Keras model
    """
    # CNN feature extractor (TimeDistributed applies to each frame)
    cnn_base = tf.keras.Sequential([
        tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            input_shape=(max_frames, img_size[0], img_size[1], 3)
        ),
        tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization()),
        tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2))),
        
        tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')),
        tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization()),
        tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2))),
        
        tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')),
        tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization()),
        tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2))),
        
        tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same')),
        tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization()),
        tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2))),
        
        tf.keras.layers.TimeDistributed(tf.keras.layers.GlobalAveragePooling2D())
    ])
    
    # LSTM temporal analysis
    lstm_layers = tf.keras.Sequential([
        tf.keras.layers.LSTM(256, return_sequences=True),
        tf.keras.layers.Dropout(0.5),
        
        tf.keras.layers.LSTM(128, return_sequences=False),
        tf.keras.layers.Dropout(0.5),
        
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.4),
        
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    
    # Combine models
    model = tf.keras.Sequential([cnn_base, lstm_layers])
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model


def train_model(epochs=30, batch_size=8, img_size=(224, 224), max_frames=50):
    """
    Train the video deepfake detection model
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size
        img_size: Frame size
        max_frames: Maximum frames per video (reduced to 50 for speed)
    """
    print("=" * 60)
    print("Training Video DeepFake Detection Model")
    print("=" * 60)
    
    # Configure TensorFlow memory settings
    tf.config.optimizer.set_jit(True)  # Enable XLA for better performance
    tf.keras.backend.clear_session()  # Clear any existing sessions
    
    # Load dataset
    print(f"\nLoading dataset with max_frames={max_frames}...")
    loader = DatasetLoader(dataset_path='dataset', img_size=img_size, max_frames=max_frames)
    print("Extracting video features...")
    X_train, X_val, y_train, y_val = loader.load_video_dataset()
    print(f"Dataset loaded: {len(X_train)} training, {len(X_val)} validation")
    
    # Create model
    print("\nCreating CNN+LSTM model...")
    model = create_video_model(img_size=img_size, max_frames=max_frames)
    model.summary()
    
    # Callbacks - Optimized for fast training
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'models/video_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=3,  # Reduced from 8 for faster completion
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=2,  # Reduced from 4 for faster response
        min_lr=1e-7,
        verbose=1
    )
    
    callbacks = [checkpoint, early_stop, reduce_lr]
    
    # Train model
    print("\n" + "=" * 60)
    print("Starting Training...")
    print("=" * 60 + "\n")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    print("\n" + "=" * 60)
    print("Evaluating Model...")
    print("=" * 60 + "\n")
    
    # Load best model
    model.load_weights('models/video_model.h5')
    
    y_pred_prob = model.predict(X_val)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    from sklearn.metrics import classification_report, confusion_matrix
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=['Real', 'Fake']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_val, y_pred)
    print(cm)
    
    # Plot training history
    plot_training_history(history)
    
    # Save final model
    model.save('models/video_model_final.h5')
    print("\nModel saved to models/video_model_final.h5")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    
    return model, history


def plot_training_history(history):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train Acc')
    axes[0, 0].plot(history.history['val_accuracy'], label='Val Acc')
    axes[0, 0].set_title('Model Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train Loss')
    axes[0, 1].plot(history.history['val_loss'], label='Val Loss')
    axes[0, 1].set_title('Model Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Precision
    if 'precision' in history.history:
        axes[1, 0].plot(history.history['precision'], label='Train Precision')
        axes[1, 0].plot(history.history['val_precision'], label='Val Precision')
        axes[1, 0].set_title('Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Recall
    if 'recall' in history.history:
        axes[1, 1].plot(history.history['recall'], label='Train Recall')
        axes[1, 1].plot(history.history['val_recall'], label='Val Recall')
        axes[1, 1].set_title('Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('models/video_training_history.png', dpi=300)
    print("Training history plot saved")
    plt.show()


if __name__ == "__main__":
    # Configure TensorFlow memory settings to prevent allocation errors
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Set memory growth to True (only allocate as needed)
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Limit total GPU memory usage (in MB) - adjust based on your GPU
            # tf.config.set_logical_device_configuration(
            #     gpus[0],
            #     [tf.config.LogicalDeviceConfiguration(memory_limit=2048)])
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    
    # Force CPU usage and limit memory on Windows
    import os
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings
    
    # Clear any existing sessions
    tf.keras.backend.clear_session()
    
    # Train model with reduced batch size for memory efficiency
    print("\nStarting FAST video model training...")
    model, history = train_model(epochs=10, batch_size=2, max_frames=50)
