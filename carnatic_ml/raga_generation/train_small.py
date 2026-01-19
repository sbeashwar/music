import os
import json
import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.utils import to_categorical
from raga_generation.tokenizer import SwaraTokenizer

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
os.makedirs(MODEL_DIR, exist_ok=True)


def load_sequences():
    sequences = []
    for f in os.listdir(DATA_DIR):
        if f.endswith('_sequences.json'):
            with open(os.path.join(DATA_DIR, f), 'r', encoding='utf-8') as fh:
                data = json.load(fh)
                sequences.extend(data)
    return sequences


def build_dataset(sequences, tokenizer, seq_len=8):
    X = []
    y = []
    for seq in sequences:
        ids = tokenizer.encode(seq)
        # break into sliding windows
        for i in range(0, len(ids) - seq_len):
            X.append(ids[i:i+seq_len])
            y.append(ids[i+seq_len])
    return np.array(X), np.array(y)


def train():
    sequences = load_sequences()
    if not sequences:
        print('No sequences found. Run prepare_dataset.py first.')
        return

    tokenizer = SwaraTokenizer()
    X, y = build_dataset(sequences, tokenizer, seq_len=6)
    vocab_size = len(tokenizer.vocab)
    y_cat = to_categorical(y, num_classes=vocab_size)

    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=32, input_length=6),
        LSTM(128, return_sequences=False),
        Dense(vocab_size, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y_cat, epochs=3, batch_size=32)
    model.save(os.path.join(MODEL_DIR, 'small_raga_gen.h5'))
    tokenizer.save(os.path.join(MODEL_DIR, 'tokenizer.json'))
    print('Model and tokenizer saved to', MODEL_DIR)


if __name__ == '__main__':
    train()
