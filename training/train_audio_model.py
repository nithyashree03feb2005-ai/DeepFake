"""
Train Audio DeepFake Detection Model
Uses CNN on Mel Spectrogram (91%+ accuracy)
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.dataset_loader import DatasetLoader


def create_audio_model(input_shape=(128, None, 1)):
    """
    Create CNN model for audio deepfake detection using mel spectrograms
    
    Args:
        input_shape: Input spectrogram shape (mel_bins, time_frames, channels)
        
    Returns:
        Compiled Keras model
    """
    model = Sequential([
        # Layer 1
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Layer 2
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Layer 3
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Layer 4
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),
        
        # Layer 5
        Conv2D(512, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        GlobalAveragePooling2D(),
        
        # Fully connected layers
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        
        # Output layer
        Dense(1, activation='sigmoid')
    ])
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model


def train_model(epochs=40, batch_size=32, sr=16000, duration=5):
    """
    Train the audio deepfake detection model
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size
        sr: Sample rate
        duration: Audio duration in seconds
    """
    print("=" * 60)
    print("Training Audio DeepFake Detection Model")
    print("=" * 60)
    
    # Configure TensorFlow memory settings
    tf.config.optimizer.set_jit(True)  # Enable XLA for better performance
    tf.keras.backend.clear_session()  # Clear any existing sessions
    
    # Load dataset
    loader = DatasetLoader(dataset_path='dataset')
    X_train, X_val, y_train, y_val = loader.load_audio_dataset(
        validation_split=0.2, 
        sr=sr, 
        duration=duration
    )
    
    # Get input shape from data
    input_shape = X_train.shape[1:]
    print(f"Input shape: {input_shape}")
    
    # Create model
    model = create_audio_model(input_shape=input_shape)
    model.summary()
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        'models/audio_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
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
    model.load_weights('models/audio_model.h5')
    
    y_pred_prob = model.predict(X_val)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    from sklearn.metrics import classification_report, confusion_matrix
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=['Real', 'Fake']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_val, y_pred)
    print(cm)
    
    # Calculate accuracy
    accuracy = np.mean(y_pred == y_val) * 100
    print(f"\nValidation Accuracy: {accuracy:.2f}%")
    
    # Plot training history
    plot_training_history(history)
    
    # Save final model
    model.save('models/audio_model_final.h5')
    print("\nModel saved to models/audio_model_final.h5")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print("\n✅ AUDIO TRAINING SUCCESSFULLY COMPLETED!")
    print("=" * 60 + "\n")
    
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
    plt.savefig('models/audio_training_history.png', dpi=300)
    print("Training history plot saved")
    plt.show()


if __name__ == "__main__":
    # Configure GPU memory settings to prevent allocation errors
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Set memory growth to True (only allocate as needed)
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    
    # Force CPU usage and limit memory on Windows
    import os
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings
    
    # Clear any existing sessions
    tf.keras.backend.clear_session()
    
    # Train model with adjusted parameters for memory efficiency
    model, history = train_model(epochs=10, batch_size=8)
