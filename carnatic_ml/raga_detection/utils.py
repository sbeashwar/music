import os
import numpy as np
import librosa
from tqdm import tqdm
import pandas as pd

class AudioFeatureExtractor:
    def __init__(self, sample_rate=22050, n_fft=2048, hop_length=512, n_mels=128):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

    def extract_features(self, audio_path):
        """Extract audio features for raga detection."""
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Extract features
            features = {}
            
            # Mel-frequency cepstral coefficients (MFCCs)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features['mfcc_mean'] = np.mean(mfcc, axis=1)
            features['mfcc_std'] = np.std(mfcc, axis=1)
            
            # Chroma features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            features['chroma_mean'] = np.mean(chroma, axis=1)
            features['chroma_std'] = np.std(chroma, axis=1)
            
            # Spectral contrast
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            features['contrast_mean'] = np.mean(contrast, axis=1)
            features['contrast_std'] = np.std(contrast, axis=1)
            
            # Tonnetz features
            tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(y), sr=sr)
            features['tonnetz_mean'] = np.mean(tonnetz, axis=1)
            features['tonnetz_std'] = np.std(tonnetz, axis=1)
            
            # Flatten all features
            flat_features = []
            for key in sorted(features.keys()):
                flat_features.append(features[key])
                
            return np.concatenate(flat_features)
            
        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
            return None

    def process_audio_directory(self, directory):
        """Process all audio files in a directory and extract features."""
        features = []
        labels = []
        
        # Get all audio files
        audio_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.wav', '.mp3', '.flac')):
                    audio_files.append((os.path.join(root, file), os.path.basename(root)))
        
        # Process each file
        for audio_path, label in tqdm(audio_files, desc="Processing audio files"):
            feature = self.extract_features(audio_path)
            if feature is not None:
                features.append(feature)
                labels.append(label)
        
        return np.array(features), np.array(labels)
