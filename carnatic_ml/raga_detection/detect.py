import os
import joblib
import argparse
import numpy as np
from utils import AudioFeatureExtractor

def detect_raga(audio_path, model_path):
    """
    Detect the raga of an audio file using a trained model.
    
    Args:
        audio_path (str): Path to the audio file
        model_path (str): Path to the trained model
        
    Returns:
        tuple: (predicted_raga, confidence, probabilities)
    """
    # Load model and related objects
    model_data = joblib.load(model_path)
    model = model_data['model']
    le = model_data['label_encoder']
    feature_extractor = model_data['feature_extractor']
    
    # Extract features
    features = feature_extractor.extract_features(audio_path)
    if features is None:
        return None, 0.0, {}
    
    # Reshape for single prediction
    features = features.reshape(1, -1)
    
    # Predict
    probas = model.predict_proba(features)[0]
    pred_idx = np.argmax(probas)
    confidence = probas[pred_idx]
    predicted_raga = le.inverse_transform([pred_idx])[0]
    
    # Get probabilities for all ragas
    raga_probs = {raga: prob for raga, prob in zip(le.classes_, probas)}
    
    return predicted_raga, confidence, raga_probs

def main():
    parser = argparse.ArgumentParser(description='Detect raga from audio file')
    parser.add_argument('audio_path', type=str, help='Path to audio file')
    parser.add_argument('--model_path', type=str, default='models/raga_detector.pkl',
                       help='Path to trained model')
    args = parser.parse_args()
    
    if not os.path.exists(args.audio_path):
        print(f"Error: Audio file not found: {args.audio_path}")
        return
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model not found: {args.model_path}")
        print("Please train the model first using train.py")
        return
    
    print(f"Analyzing {args.audio_path}...")
    raga, confidence, probs = detect_raga(args.audio_path, args.model_path)
    
    if raga is None:
        print("Error processing audio file")
        return
    
    print(f"\nDetected Raga: {raga} (Confidence: {confidence:.2f})")
    
    # Sort ragas by probability
    sorted_ragas = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    
    print("\nTop 5 predicted ragas:")
    for raga, prob in sorted_ragas[:5]:
        print(f"  {raga}: {prob:.4f}")

if __name__ == "__main__":
    main()
