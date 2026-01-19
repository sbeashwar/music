import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from utils import AudioFeatureExtractor

def train_raga_detector(data_dir, model_save_path):
    """
    Train a raga detection model using audio features.
    
    Args:
        data_dir (str): Directory containing audio files organized by raga
        model_save_path (str): Path to save the trained model
    """
    # Initialize feature extractor
    feature_extractor = AudioFeatureExtractor()
    
    print("Extracting features from audio files...")
    X, y = feature_extractor.process_audio_directory(data_dir)
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Train model
    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Save model and label encoder
    joblib.dump({
        'model': model,
        'label_encoder': le,
        'feature_extractor': feature_extractor
    }, model_save_path)
    
    print(f"\nModel saved to {model_save_path}")
    return model, le

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train a raga detection model')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing audio files organized by raga')
    parser.add_argument('--model_path', type=str, default='models/raga_detector.pkl',
                        help='Path to save the trained model')
    
    args = parser.parse_args()
    
    # Create model directory if it doesn't exist
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    
    # Train the model
    train_raga_detector(args.data_dir, args.model_path)
