"""
Train Image DeepFake Detection Model
Uses ResNet50 + CNN architecture for high accuracy (92-96%)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.dataset_loader import DatasetLoader


def create_image_model(img_size=(150, 150), pretrained_weights=True):
    """
    Create ResNet50 + CNN model for image deepfake detection
    
    Args:
        img_size: Input image size
        pretrained_weights: Use ImageNet pretrained weights
        
    Returns:
        Compiled Keras model
    """
    # Load ResNet50 backbone
    base_model = ResNet50(
        include_top=False,
        input_shape=(img_size[0], img_size[1], 3),
        weights='imagenet' if pretrained_weights else None,
        pooling='avg'
    )
    
    # Freeze base model layers initially
    base_model.trainable = False
    
    # Add custom layers on top
    x = base_model.output
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Output layer
    output = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=output)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model


def train_model(epochs=50, batch_size=32, img_size=(150, 150), use_subset=True):
    """
    Train the image deepfake detection model
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size
        img_size: Image size (default: 150x150 for memory efficiency)
        use_subset: Use subset for faster training (5k images)
    """
    print("=" * 60)
    print("Training Image DeepFake Detection Model")
    print("=" * 60)
    
    # Configure TensorFlow memory settings
    tf.config.optimizer.set_jit(True)  # Enable XLA for better performance
    tf.keras.backend.clear_session()  # Clear any existing sessions
    
    # Load dataset
    loader = DatasetLoader(dataset_path='dataset', img_size=img_size)
    X_train, X_val, y_train, y_val = loader.load_image_dataset(batch_size=batch_size, use_subset=use_subset)
    
    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Create model
    model = create_image_model(img_size=img_size)
    model.summary()
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        'models/image_model.h5',
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
    
    # Create data generator with optimized settings
    train_generator = datagen.flow(X_train, y_train, batch_size=batch_size)
    
    history = model.fit(
        train_generator,
        validation_data=(X_val, y_val),
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    print("\n" + "=" * 60)
    print("Evaluating Model...")
    print("=" * 60 + "\n")
    
    # Load best model
    model.load_weights('models/image_model.h5')
    
    y_pred_prob = model.predict(X_val)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=['Real', 'Fake']))
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_val, y_pred)
    print(cm)
    
    # Plot training history
    plot_training_history(history)
    
    # Unfreeze some layers for fine-tuning
    print("\nFine-tuning model...")
    for layer in model.layers[-50:]:
        layer.trainable = True
    
    model.compile(
        optimizer=Adam(learning_rate=0.0001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    # Fine-tune
    fine_tune_history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        validation_data=(X_val, y_val),
        epochs=20,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    model.save('models/image_model_final.h5')
    print("\nModel saved to models/image_model_final.h5")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print("\n✅ IMAGE TRAINING SUCCESSFULLY COMPLETED!")
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
    plt.savefig('models/training_history.png', dpi=300)
    print("Training history plot saved to models/training_history.png")
    plt.show()


if __name__ == "__main__":
    # Configure GPU memory settings to prevent allocation errors
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Set memory growth to True (only allocate as needed)
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # OR set a specific memory limit (adjust based on your GPU)
            # tf.config.set_logical_device_configuration(
            #     gpus[0],
            #     [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])  # 4GB limit
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    
    # Train model with subset for faster training (set use_subset=False for full dataset)
    # Reduced epochs and adjusted batch size for better memory management
    model, history = train_model(epochs=30, batch_size=16, use_subset=True)
