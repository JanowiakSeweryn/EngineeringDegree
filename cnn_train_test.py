"""
CNN Training and Testing Script
Trains a CNN on raw camera images instead of MediaPipe landmarks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import cv2
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd

from get_data import GESTURES

# Configuration
UPDATE_WEIGHTS = False
IMG_SIZE = 128  # Resize images to 128x128 for faster training
BATCH_SIZE = 32
NUM_EPOCHS = 3
LEARNING_RATE = 0.001
APPLY_BACKGROUND_FILTER = True  # Apply skin detection filter to isolate hand

module_dir = os.path.dirname(__file__)
GESTURES_DIR = os.path.join(module_dir, 'Gestures')
CNN_WEIGHTS_FILE = os.path.join(module_dir, 'cnn_weights_raw_images.pth')

def apply_background_filter(image):
    """
    Apply background subtraction filter to detect hand/object from background.
    Uses skin color detection in YCrCb colorspace combined with morphological operations.
    
    Args:
        image: BGR image (numpy array)
    
    Returns:
        Filtered image with background removed (hand region preserved)
    """
    # Convert to YCrCb colorspace (better for skin detection)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    
    # Define skin color range in YCrCb
    # These values work for most skin tones under various lighting
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)
    
    # Create binary mask for skin regions
    skin_mask = cv2.inRange(ycrcb, lower_skin, upper_skin)
    
    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Apply Gaussian blur to smooth edges
    skin_mask = cv2.GaussianBlur(skin_mask, (5, 5), 0)
    
    # Apply mask to original image
    result = cv2.bitwise_and(image, image, mask=skin_mask)
    
    return result



class GestureImageDataset(Dataset):
    """Dataset for loading gesture images."""
    
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Ensure path is a proper string
        if not isinstance(img_path, str):
            img_path = str(img_path)
        
        image = cv2.imread(img_path)
        
        # Check if image loaded successfully
        if image is None:
            raise ValueError(f"Failed to load image at index {idx}. Path: {img_path}, Type: {type(img_path)}")
        
        # Apply background filter to isolate hand from background
        if APPLY_BACKGROUND_FILTER:
            image = apply_background_filter(image)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return image, label


def load_image_data():
    """
    Load images from Gestures folder.
    Returns paths and labels (one-hot encoded).
    """
    image_paths = []
    labels = []
    
    # Exclude six-seven folder as requested
    gestures_to_use = [g for g in GESTURES if g != 'six-seven' and os.path.exists(os.path.join(GESTURES_DIR, g))]
    
    print(f"Loading images from {len(gestures_to_use)} gesture classes...")
    
    for gesture_idx, gesture_name in enumerate(gestures_to_use):
        gesture_folder = os.path.join(GESTURES_DIR, gesture_name)
        
        if not os.path.isdir(gesture_folder):
            print(f"Warning: {gesture_folder} not found, skipping...")
            continue
        
        # Get all PNG fileszin this folder
        files = [f for f in os.listdir(gesture_folder) if f.endswith('.png')]
        
        print(f"  {gesture_name}: {len(files)} images")
        
        for filename in files:
            img_path = os.path.join(gesture_folder, filename)
            image_paths.append(img_path)
            
            # Create one-hot encoded label
            label = [0] * len(gestures_to_use)
            label[gesture_idx] = 1
            labels.append(label)
    
    print(f"\nTotal images loaded: {len(image_paths)}")
    return image_paths, labels, gestures_to_use


def stratified_split_paths(paths, labels, train_ratio):
    """
    Stratified split for image paths (avoids numpy conversion issues).
    
    Args:
        paths: List of image paths (strings)
        labels: List of one-hot encoded labels
        train_ratio: Ratio of data for training (e.g., 0.8 for 80%)
    
    Returns:
        paths_train, labels_train, paths_test, labels_test
    """
    from collections import defaultdict
    import random
    
    # Group paths and labels by class
    class_data = defaultdict(list)
    
    for path, label in zip(paths, labels):
        class_idx = label.index(1)  # Get class from one-hot encoding
        class_data[class_idx].append((path, label))
    
    paths_train = []
    labels_train = []
    paths_test = []
    labels_test = []
    
    # Split each class separately
    for class_idx, samples in class_data.items():
        # Shuffle samples for this class
        random.shuffle(samples)
        
        # Calculate split point
        n_train = int(len(samples) * train_ratio)
        
        # Split
        train_samples = samples[:n_train]
        test_samples = samples[n_train:]
        
        # Add to respective lists
        for path, label in train_samples:
            paths_train.append(path)
            labels_train.append(label)
        
        for path, label in test_samples:
            paths_test.append(path)
            labels_test.append(label)
    
    # Shuffle the combined data
    combined_train = list(zip(paths_train, labels_train))
    combined_test = list(zip(paths_test, labels_test))
    
    random.shuffle(combined_train)
    random.shuffle(combined_test)
    
    paths_train, labels_train = zip(*combined_train) if combined_train else ([], [])
    paths_test, labels_test = zip(*combined_test) if combined_test else ([], [])
    
    # Convert back to lists
    paths_train = list(paths_train)
    paths_test = list(paths_test)
    labels_train = list(labels_train)
    labels_test = list(labels_test)
    
    return paths_train, labels_train, paths_test, labels_test


class ImageCNN(nn.Module):
    """CNN for raw image classification."""
    
    def __init__(self, num_classes=10):
        super(ImageCNN, self).__init__()
        
        # Conv layers for 128x128 input
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # -> 32x128x128
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  # -> 32x64x64
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # -> 64x64x64
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)  # -> 64x32x32
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # -> 128x32x32
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)  # -> 128x16x16
        
        # FC layers
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        x = x.view(x.size(0), -1)  # Flatten
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def train_model(model, train_loader, criterion, optimizer, epoch, num_epochs):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], '
                  f'Loss: {loss.item():.4f}')
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def validate_model(model, val_loader, criterion):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def plot_confusion_matrix(model, test_loader, gesture_names, filename='cnn_confusion_matrix.png'):
    """Generate and plot confusion matrix."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot
    plt.figure(figsize=(12, 10))
    df = pd.DataFrame(cm, index=gesture_names, columns=gesture_names)
    sns.heatmap(df, annot=True, cmap="Blues", fmt='d')
    plt.title("CNN Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Confusion matrix saved to {filename}")
    
    return cm


# Main training script
if __name__ == "__main__":
    print("="*60)
    print("CNN Training on Raw Images")
    print("="*60)
    
    # Load data
    image_paths, labels, gesture_names = load_image_data()
    
    # Split data using custom stratified split for paths
    paths_train, labels_train, paths_test, labels_test = stratified_split_paths(image_paths, labels, 0.8)
    paths_train, labels_train, paths_val, labels_val = stratified_split_paths(paths_train, labels_train, 0.75)
    
    print(f"\nDataset split:")
    print(f"  Training: {len(paths_train)} images")
    print(f"  Validation: {len(paths_val)} images")
    print(f"  Test: {len(paths_test)} images")
    
    # Print class distribution in test set
    print("\nClass distribution in test set:")
    temp_labels = [labels_test[i].index(1) for i in range(len(labels_test))]
    for i, g in enumerate(gesture_names):
        count = temp_labels.count(i)
        print(f"  {g}: {count}")
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets and dataloaders
    train_labels_idx = [l.index(1) for l in labels_train]
    val_labels_idx = [l.index(1) for l in labels_val]
    test_labels_idx = [l.index(1) for l in labels_test]
    
    train_dataset = GestureImageDataset(paths_train, train_labels_idx, transform)
    val_dataset = GestureImageDataset(paths_val, val_labels_idx, transform)
    test_dataset = GestureImageDataset(paths_test, test_labels_idx, transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Create model
    num_classes = len(gesture_names)
    model = ImageCNN(num_classes=num_classes)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print("\nStarting training...")
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, epoch, NUM_EPOCHS)
        val_loss, val_acc = validate_model(model, val_loader, criterion)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f'Epoch [{epoch}/{NUM_EPOCHS}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%\n')
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('cnn_training_curves.png')
    print("Training curves saved to cnn_training_curves.png")
    
    # Test the model
    test_loss, test_acc = validate_model(model, test_loader, criterion)
    print(f'\nTest Accuracy: {test_acc:.2f}%')
    
    # Generate confusion matrix
    cm = plot_confusion_matrix(model, test_loader, gesture_names)
    
    # Save model if requested
    if UPDATE_WEIGHTS:
        torch.save(model.state_dict(), CNN_WEIGHTS_FILE)
        print(f"\nModel weights saved to {CNN_WEIGHTS_FILE}")
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    print("\nGenerated files:")
    print(f"  - cnn_training_curves.png (training/validation curves)")
    print(f"  - cnn_confusion_matrix.png (confusion matrix heatmap)")
    if UPDATE_WEIGHTS:
        print(f"  - {CNN_WEIGHTS_FILE} (model weights)")

