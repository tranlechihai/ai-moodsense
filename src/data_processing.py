"""
Data Processing Script for FER2013 Dataset
Processes FER2013 CSV data and prepares it for training
"""

import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import os

# Emotion label mapping (FER2013 standard)
EMOTION_MAP = {
    0: 'angry',      # Negative
    1: 'disgust',    # Negative
    2: 'fear',       # Negative
    3: 'happy',      # Positive
    4: 'sad',        # Negative
    5: 'surprise',   # Positive
    6: 'neutral'     # Neutral
}

# Reduced emotion groups
REDUCED_EMOTIONS = {
    'Positive': ['happy', 'surprise'],
    'Neutral': ['neutral'],
    'Negative': ['angry', 'disgust', 'fear', 'sad']
}

def load_fer2013_data(csv_path):
    """
    Load FER2013 dataset from CSV file
    
    Args:
        csv_path: Path to fer2013.csv file
        
    Returns:
        images: numpy array of images (N, 48, 48)
        labels: numpy array of labels (N,)
    """
    print("Loading FER2013 dataset...")
    df = pd.read_csv(csv_path)
    
    # Extract pixels and emotions
    pixels = df['pixels'].values
    emotions = df['emotion'].values
    
    # Convert pixel strings to numpy arrays
    images = []
    for pixel_str in pixels:
        pixel_array = np.array([int(p) for p in pixel_str.split()], dtype=np.uint8)
        image = pixel_array.reshape(48, 48)
        images.append(image)
    
    images = np.array(images, dtype=np.uint8)
    
    print(f"Loaded {len(images)} images")
    print(f"Image shape: {images.shape}")
    print(f"Emotion distribution: {np.bincount(emotions)}")
    
    return images, emotions

def reduce_emotion_labels(emotions):
    """
    Reduce 7 emotion labels to 3 groups
    
    Args:
        emotions: numpy array of original emotion labels (0-6)
        
    Returns:
        reduced_labels: numpy array of reduced labels (0: Negative, 1: Neutral, 2: Positive)
        label_names: list of label names
    """
    # Create mapping from original to reduced labels (CORRECTED FER2013 mapping)
    original_to_reduced = {
        0: 0,  # angry -> Negative
        1: 0,  # disgust -> Negative
        2: 0,  # fear -> Negative
        3: 2,  # happy -> Positive
        4: 0,  # sad -> Negative (CORRECTED: was incorrectly mapped as neutral)
        5: 2,  # surprise -> Positive
        6: 1   # neutral -> Neutral (CORRECTED: was incorrectly mapped as sad)
    }
    
    reduced_labels = np.array([original_to_reduced[e] for e in emotions])
    label_names = ['Negative', 'Neutral', 'Positive']
    
    return reduced_labels, label_names

def preprocess_images(images):
    """
    Preprocess images: normalize to [0, 1] and add channel dimension
    
    Args:
        images: numpy array of images (N, 48, 48)
        
    Returns:
        processed_images: numpy array (N, 48, 48, 1) normalized
    """
    # Normalize to [0, 1]
    images = images.astype(np.float32) / 255.0
    
    # Add channel dimension
    images = np.expand_dims(images, axis=-1)
    
    return images

def prepare_data(csv_path, test_size=0.2, val_size=0.1):
    """
    Prepare data for training
    
    Args:
        csv_path: Path to fer2013.csv
        test_size: Test set proportion
        val_size: Validation set proportion (from training set)
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, label_names
    """
    # Load data
    images, emotions = load_fer2013_data(csv_path)
    
    # Reduce emotion labels
    reduced_labels, label_names = reduce_emotion_labels(emotions)
    
    # Preprocess images
    processed_images = preprocess_images(images)
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        processed_images, reduced_labels,
        test_size=test_size,
        random_state=42,
        stratify=reduced_labels
    )
    
    # Split train into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=val_size,
        random_state=42,
        stratify=y_train
    )
    
    print(f"\nData split:")
    print(f"Train: {X_train.shape[0]} samples")
    print(f"Validation: {X_val.shape[0]} samples")
    print(f"Test: {X_test.shape[0]} samples")
    
    print(f"\nLabel distribution (Train):")
    unique, counts = np.unique(y_train, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  {label_names[label]}: {count}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, label_names

if __name__ == "__main__":
    # Example usage
    csv_path = "../data/fer2013.csv"
    
    if os.path.exists(csv_path):
        X_train, X_val, X_test, y_train, y_val, y_test, label_names = prepare_data(csv_path)
        print("\nData preparation complete!")
    else:
        print(f"Please download fer2013.csv and place it at: {csv_path}")

