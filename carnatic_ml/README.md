# Carnatic Music ML

This project provides tools for working with Carnatic music using machine learning, focusing on two main tasks:
1. **Raga Generation**: Generate alapanas and melodies based on specific ragas
2. **Raga Detection**: Identify ragas from audio samples

## Project Structure

```
carnatic_ml/
├── raga_generation/       # For generating alapanas/melodies
│   ├── models/           # Model architectures
│   ├── data/             # Training data for generation
│   ├── train.py          # Training script for generation
│   ├── generate.py       # Generate new alapanas
│   └── utils.py          # Helper functions
│
├── raga_detection/      # For identifying ragas from audio
│   ├── models/           # Model architectures
│   ├── data/             # Training/validation data
│   ├── train.py          # Training script
│   ├── detect.py         # Detect raga from audio
│   └── utils.py          # Feature extraction, etc.
│
├── shared/               # Shared components
│   ├── audio/            # Audio processing utilities
│   ├── raga_definitions/  # Raga scales, arohanam/avarohanam
│   └── visualization/    # Common visualization code
│
└── requirements.txt     # Project dependencies
```

## Setup

1. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Raga Generation

1. Place your training data in `raga_generation/data/`
2. Train the model:
   ```bash
   cd raga_generation
   python train.py
   ```
3. Generate new alapanas:
   ```bash
   python generate.py --raga "mohanam" --duration 30
   ```

### Raga Detection

1. Place your training data in `raga_detection/data/`
2. Train the model:
   ```bash
   cd raga_detection
   python train.py
   ```
3. Detect raga from audio:
   ```bash
   python detect.py --audio path/to/audio.wav
   ```

## Data Organization

- Store audio samples in their respective raga folders under `data/`
- Use consistent naming: `[raga]_[artist]_[duration].wav`
- Supported formats: WAV, MP3, FLAC

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
