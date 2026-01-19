import os
import joblib
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

class RagaGeneratorTrainer:
    def __init__(self, input_shape, num_ragas):
        """
        Initialize the raga generator trainer.
        
        Args:
            input_shape (tuple): Shape of input sequences (seq_length, num_features)
            num_ragas (int): Number of ragas in the dataset
        """
        self.input_shape = input_shape
        self.num_ragas = num_ragas
        self.model = self._build_model()
    
    def _build_model(self):
        """Build the LSTM-based raga generation model."""
        model = Sequential([
            # First LSTM layer
            LSTM(512, 
                 input_shape=self.input_shape,
                 return_sequences=True,
                 dropout=0.2,
                 recurrent_dropout=0.2),
            BatchNormalization(),
            
            # Second LSTM layer
            LSTM(512,
                 return_sequences=True,
                 dropout=0.2,
                 recurrent_dropout=0.2),
            BatchNormalization(),
            
            # Third LSTM layer
            LSTM(512,
                 return_sequences=False,
                 dropout=0.2,
                 recurrent_dropout=0.2),
            BatchNormalization(),
            
            # Dense layers
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(self.num_ragas, activation='softmax')
        ])
        
        # Compile the model
        optimizer = Adam(learning_rate=0.001)
        model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=64, model_save_path='models'):
        """
        Train the raga generation model.
        
        Args:
            X_train: Training sequences
            y_train: Training labels (one-hot encoded)
            X_val: Validation sequences
            y_val: Validation labels (one-hot encoded)
            epochs: Number of training epochs
            batch_size: Batch size for training
            model_save_path: Directory to save the best model
            
        Returns:
            Training history
        """
        # Create model directory if it doesn't exist
        os.makedirs(model_save_path, exist_ok=True)
        
        # Define callbacks
        checkpoint = ModelCheckpoint(
            os.path.join(model_save_path, 'raga_generator_best.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        )
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[checkpoint, early_stopping],
            verbose=1
        )
        
        return history

def prepare_training_data(data_dir, seq_length=100):
    """
    Prepare training data for the raga generation model.
    
    Args:
        data_dir: Directory containing audio files organized by raga
        seq_length: Length of input sequences
        
    Returns:
        X: Input sequences
        y: Target sequences (one-hot encoded)
        raga_to_idx: Mapping from raga names to indices
    """
    # In a real implementation, you would:
    # 1. Load audio files
    # 2. Extract features (pitch, duration, etc.)
    # 3. Create sequences of fixed length
    # 4. One-hot encode the raga labels
    
    # This is a placeholder implementation
    print("Preparing training data...")
    
    # Get list of ragas (subdirectories in data_dir)
    ragas = [d for d in os.listdir(data_dir) 
             if os.path.isdir(os.path.join(data_dir, d))]
    raga_to_idx = {raga: i for i, raga in enumerate(ragas)}
    
    # In a real implementation, you would process the audio files here
    # For now, we'll create some dummy data
    num_samples = 1000
    num_features = 128  # Example: MFCC features
    
    X = np.random.rand(num_samples, seq_length, num_features)
    y = np.random.randint(0, len(ragas), num_samples)
    y = tf.keras.utils.to_categorical(y, num_classes=len(ragas))
    
    return X, y, raga_to_idx

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train a raga generation model')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Directory containing audio files organized by raga')
    parser.add_argument('--model_dir', type=str, default='models',
                       help='Directory to save the trained model')
    parser.add_argument('--seq_length', type=int, default=100,
                       help='Length of input sequences')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    
    args = parser.parse_args()
    
    # Prepare training data
    X, y, raga_to_idx = prepare_training_data(args.data_dir, args.seq_length)
    
    # Split into training and validation sets
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize and train the model
    trainer = RagaGeneratorTrainer(
        input_shape=(args.seq_length, X_train.shape[2]),
        num_ragas=len(raga_to_idx)
    )
    
    print("Training model...")
    history = trainer.train(
        X_train, y_train,
        X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_save_path=args.model_dir
    )
    
    # Save the raga to index mapping
    joblib.dump(
        raga_to_idx,
        os.path.join(args.model_dir, 'raga_to_idx.pkl')
    )
    
    print("\nTraining complete!")
    print(f"Model saved to {args.model_dir}")

if __name__ == "__main__":
    main()
