#!/usr/bin/env python3
"""
Crop and Weed Detection System using YOLO-formatted Dataset
PyTorch version with .pt model saving
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from PIL import Image
import glob
import random
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Constants
DEFAULT_IMAGE_SIZE = (512, 512)
BATCH_SIZE = 16
EPOCHS = 25
LEARNING_RATE = 0.0001
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.15
RANDOM_SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    
    if not image_files:
        raise FileNotFoundError(f"No images found in {data_dir} with extensions {image_extensions}")
    
    label_files = []
    for img_path in image_files:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(data_dir, f"{base_name}.txt")
        if not os.path.exists(label_path):
            print(f"Warning: Missing label file for {img_path}")
            continue
        label_files.append(label_path)
    
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

class CropWeedDataset(Dataset):
    """Custom PyTorch dataset for crop/weed classification"""
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform or transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(DEFAULT_IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image = cv2.imread(row['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = 1 if row['has_weed'] == 'yes' else 0
        
        if self.transform:
            image = self.transform(image)
            
        return image, torch.tensor(label, dtype=torch.float32)

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
                'has_weed': 'yes' if has_weed else 'no',
                'num_objects': len(objects),
                'width': width,
                'height': height
            })
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
    
    return pd.DataFrame(data)

class CropWeedModel(nn.Module):
    """CNN model for crop/weed classification"""
    def __init__(self):
        super().__init__()
        self.base_model = resnet18(pretrained=True)
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.base_model(x)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs):
    """Training loop"""
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            predicted = (outputs.squeeze() > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = correct / total
        
        # Validation
        val_loss, val_acc = evaluate_model(model, val_loader, criterion)
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/best_model.pt')
            print("Saved new best model")
    
    return history

def evaluate_model(model, loader, criterion):
    """Evaluate model on validation/test set"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels)
            
            running_loss += loss.item() * images.size(0)
            predicted = (outputs.squeeze() > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    loss = running_loss / len(loader.dataset)
    acc = correct / total
    return loss, acc

def predict_image(model, image_path, classes, save_path=None):
    """Make prediction on single image"""
    try:
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}")
            return None, 0
        
        orig_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(orig_image, DEFAULT_IMAGE_SIZE)
        image = transforms.ToTensor()(image)
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
        image = image.unsqueeze(0).to(DEVICE)
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            output = model(image)
            prediction = output.squeeze().item()
        
        predicted_class = "Weed" if prediction > 0.5 else "Crop"
        confidence = prediction if predicted_class == "Weed" else 1 - prediction
        
        # Visualization
        plt.figure(figsize=(8, 8))
        plt.imshow(orig_image)
        plt.title(f"Predicted: {predicted_class} ({confidence:.2f})")
        plt.axis('off')
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()
        
        return predicted_class, confidence
    except Exception as e:
        print(f"Error predicting {image_path}: {str(e)}")
        return None, 0

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
    
    # Train-test split
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
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(DEFAULT_IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(25),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(DEFAULT_IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = CropWeedDataset(train_df, train_transform)
    val_dataset = CropWeedDataset(val_df, val_transform)
    test_dataset = CropWeedDataset(test_df, val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Initialize model
    model = CropWeedModel().to(DEVICE)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1, verbose=True)
    
    print("\nModel architecture:")
    print(model)
    
    # Train model
    print("\nTraining model...")
    history = train_model(
        model, train_loader, val_loader, 
        criterion, optimizer, scheduler, EPOCHS
    )
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy Over Epochs')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.legend()
    
    plt.savefig('training_history.png', bbox_inches='tight')
    plt.close()
    
    # Load best model
    model.load_state_dict(torch.load('models/best_model.pt'))
    model.eval()
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loss, test_acc = evaluate_model(model, test_loader, criterion)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Save final model
    torch.save(model.state_dict(), 'crop_weed_classifier.pt')
    print("\nModel saved as 'crop_weed_classifier.pt'")
    
    # Sample predictions
    print("\nMaking sample predictions...")
    for i in range(min(5, len(test_df))):
        sample = test_df.iloc[i]
        print(f"\nImage: {os.path.basename(sample['image_path'])}")
        print(f"Actual: {'Weed' if sample['has_weed'] == 'yes' else 'Crop'}")
        pred_class, confidence = predict_image(
            model,
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
    
    main(args.data_dir, args.classes)