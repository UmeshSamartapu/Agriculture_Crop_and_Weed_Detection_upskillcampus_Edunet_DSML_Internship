#!/usr/bin/env python3
"""
Crop and Weed Detection System using YOLO-formatted Dataset
Improved version with bug fixes and optimizations
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from PIL import Image
import glob
import random
import argparse

# Constants
DEFAULT_IMAGE_SIZE = (512, 512)
BATCH_SIZE = 16  # Reduced from 32 to help with memory usage
EPOCHS = 25
LEARNING_RATE = 0.0001
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.15
RANDOM_SEED = 42

def setup_directories():
    """Create necessary directories for outputs"""
    os.makedirs('sample_visualizations', exist_ok=True)
    os.makedirs('predictions', exist_ok=True)
    os.makedirs('models', exist_ok=True)

def load_data(data_dir):
    """Load and validate dataset"""
    image_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(data_dir, ext)))
    
    # Verify we found images
    if not image_files:
        raise FileNotFoundError(f"No images found in {data_dir} with extensions {image_extensions}")
    
    # Load corresponding label files
    label_files = []
    for img_path in image_files:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(data_dir, f"{base_name}.txt")
        if not os.path.exists(label_path):
            print(f"Warning: Missing label file for {img_path}")
            continue
        label_files.append(label_path)
    
    # Verify we have matching labels
    if len(image_files) != len(label_files):
        print(f"Warning: Found {len(image_files)} images but {len(label_files)} label files")
    
    return image_files, label_files

def parse_yolo_label(label_path, img_width, img_height):
    """Parse YOLO format label file with error handling"""
    objects = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) != 5:
                    print(f"Warning: Invalid line format in {label_path}: {line}")
                    continue
                
                try:
                    class_id, x_center, y_center, width, height = map(float, parts)
                except ValueError:
                    print(f"Warning: Couldn't convert values to float in {label_path}: {line}")
                    continue
                
                # Convert YOLO format to pixel coordinates
                x_center = max(0, min(1, x_center)) * img_width
                y_center = max(0, min(1, y_center)) * img_height
                width = max(0, min(1, width)) * img_width
                height = max(0, min(1, height)) * img_height
                
                x_min = int(max(0, x_center - width/2))
                y_min = int(max(0, y_center - height/2))
                x_max = int(min(img_width, x_center + width/2))
                y_max = int(min(img_height, y_center + height/2))
                
                objects.append({
                    'class_id': int(class_id),
                    'x_min': x_min,
                    'y_min': y_min,
                    'x_max': x_max,
                    'y_max': y_max
                })
    except Exception as e:
        print(f"Error reading {label_path}: {str(e)}")
    
    return objects

def visualize_sample(image_path, label_path, classes, save_path=None):
    """Visualize image with bounding boxes"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}")
            return
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_height, img_width = image.shape[:2]
        
        objects = parse_yolo_label(label_path, img_width, img_height)
        
        for obj in objects:
            try:
                class_name = classes[obj['class_id']]
                color = (0, 255, 0) if class_name.lower() == 'crop' else (255, 0, 0)
                
                cv2.rectangle(image, (obj['x_min'], obj['y_min']), 
                            (obj['x_max'], obj['y_max']), color, 2)
                cv2.putText(image, class_name, (obj['x_min'], obj['y_min']-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            except IndexError:
                print(f"Warning: Invalid class ID {obj['class_id']} in {label_path}")
                continue
        
        plt.figure(figsize=(8, 8))
        plt.imshow(image)
        plt.axis('off')
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()
    except Exception as e:
        print(f"Error visualizing {image_path}: {str(e)}")

def create_dataframe(image_files, label_files, classes):
    """Create dataframe with image paths and labels"""
    data = []
    for img_path, label_path in zip(image_files, label_files):
        try:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            if not os.path.exists(label_path):
                print(f"Warning: Label file missing for {img_path}")
                continue
            
            with Image.open(img_path) as img:
                width, height = img.size
            
            objects = parse_yolo_label(label_path, width, height)
            has_weed = any(classes[obj['class_id']].lower() == 'weed' for obj in objects)
            
            data.append({
                'image_path': img_path,
                'label_path': label_path,
                'has_weed': 'yes' if has_weed else 'no',  # <<== change done here
                'num_objects': len(objects),
                'width': width,
                'height': height
            })
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
    
    return pd.DataFrame(data)


def build_model(input_shape):
    """Build CNN model with improved architecture"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Dropout(0.3),
        
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

def main(data_dir, classes_file):
    """Main execution function"""
    setup_directories()
    
    # Load class names
    try:
        with open(classes_file, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
    except Exception as e:
        print(f"Error loading classes file: {str(e)}")
        return
    
    if len(classes) < 2:
        print("Error: Need at least 2 classes (crop and weed)")
        return
    
    # Load and validate data
    try:
        image_files, label_files = load_data(data_dir)
        print(f"\nFound {len(image_files)} images and {len(label_files)} label files")
        
        # Visualize samples
        for i in range(min(3, len(image_files))):
            idx = random.randint(0, len(image_files)-1)
            visualize_sample(
                image_files[idx], label_files[idx], classes,
                f"sample_visualizations/sample_{i+1}.png"
            )
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return
    
    # Create dataframe
    df = create_dataframe(image_files, label_files, classes)
    if df.empty:
        print("Error: No valid images found after processing")
        return
    
    print("\nDataset summary:")
    print(df[['has_weed', 'num_objects', 'width', 'height']].describe())
    
    # Class distribution
    plt.figure(figsize=(8, 5))
    sns.countplot(x='has_weed', data=df)
    plt.title('Class Distribution (0: Crop only, 1: Contains Weed)')
    plt.savefig('class_distribution.png', bbox_inches='tight')
    plt.close()
    
    # Train-test split with stratification
    train_df, test_df = train_test_split(
        df, test_size=TEST_SPLIT, 
        random_state=RANDOM_SEED, 
        stratify=df['has_weed']
    )
    train_df, val_df = train_test_split(
        train_df, test_size=VALIDATION_SPLIT/(1-TEST_SPLIT),
        random_state=RANDOM_SEED, 
        stratify=train_df['has_weed']
    )
    
    print(f"\nTraining samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.15,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    # Data generators
    def create_generator(datagen, dataframe, shuffle=False):
        return datagen.flow_from_dataframe(
            dataframe=dataframe,
            x_col='image_path',
            y_col='has_weed',
            target_size=DEFAULT_IMAGE_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='binary',
            shuffle=shuffle,
            interpolation='bilinear'
        )
    
    train_generator = create_generator(train_datagen, train_df, shuffle=True)
    val_generator = create_generator(val_datagen, val_df)
    test_generator = create_generator(test_datagen, test_df)
    
    # Build model
    model = build_model((*DEFAULT_IMAGE_SIZE, 3))
    print("\nModel summary:")
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(
            'models/best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max'
        )
    ]
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=EPOCHS,
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.legend()
    
    plt.savefig('training_history.png', bbox_inches='tight')
    plt.close()
    
    # Load best model
    from tensorflow.keras.models import load_model
    best_model = load_model('models/best_model.h5')
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = best_model.evaluate(test_generator, steps=len(test_generator))
    print(f"\nTest Accuracy: {test_results[1]:.4f}")
    print(f"Test Loss: {test_results[0]:.4f}")
    print(f"Precision: {test_results[2]:.4f}")
    print(f"Recall: {test_results[3]:.4f}")
    
    # Save final model
    best_model.save('crop_weed_classifier.h5')
    print("\nModel saved as 'crop_weed_classifier.h5'")
    
    # Sample predictions
    def predict_image(model, image_path, classes, save_path=None):
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"Warning: Could not read image {image_path}")
                return None, 0
            
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img, DEFAULT_IMAGE_SIZE)
            img_array = np.expand_dims(img_resized, axis=0) / 255.0
            
            prediction = model.predict(img_array)[0][0]
            predicted_class = "Weed" if prediction > 0.5 else "Crop"
            confidence = prediction if predicted_class == "Weed" else 1 - prediction
            
            plt.figure(figsize=(8, 8))
            plt.imshow(img)
            plt.title(f"Predicted: {predicted_class} ({confidence:.2f})")
            plt.axis('off')
            if save_path:
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                plt.close()
            
            return predicted_class, confidence
        except Exception as e:
            print(f"Error predicting {image_path}: {str(e)}")
            return None, 0
    
    print("\nMaking sample predictions...")
    for i in range(min(5, len(test_df))):
        sample = test_df.iloc[i]
        print(f"\nImage: {os.path.basename(sample['image_path'])}")
        print(f"Actual: {'Weed' if sample['has_weed'] else 'Crop'}")
        pred_class, confidence = predict_image(
            best_model,
            sample['image_path'],
            classes,
            f"predictions/prediction_{i+1}.png"
        )
        print(f"Predicted: {pred_class} (confidence: {confidence:.2f})")
    
    print("\nScript completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Crop and Weed Detection System')
    parser.add_argument('--data_dir', default='DataSet/', help='Path to dataset directory')
    parser.add_argument('--classes', default='classes.txt', help='Path to classes file')
    args = parser.parse_args()
    
    # Add TensorFlow import here to avoid import before argument parsing
    import tensorflow as tf
    
    main(args.data_dir, args.classes)