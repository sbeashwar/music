# Carnatic ML

**Raga Detection and Generation for Carnatic Music**

A practical toolkit for:
1. **Raga Detection** - Identify the ragam from a 30-second audio clip
2. **Raga Generation** - Generate melodies in any of 2500+ ragas

## Quick Start

```bash
# Install dependencies
pip install numpy librosa pretty_midi soundfile

# List available ragas
python -m carnatic list

# Get raga info
python -m carnatic info mohanam

# Generate a melody
python -m carnatic generate mohanam -o mohanam.mid

# Detect raga from audio
python -m carnatic detect sample.wav
```

## How It Works

### Raga Detection (Audio → Ragam)

Unlike ML approaches that need large labeled datasets, we use **pitch analysis + rule matching**:

1. **Extract pitches** from audio using librosa
2. **Estimate tonic (Sa)** - the drone frequency
3. **Map pitches to swaras** relative to Sa
4. **Match swara set** against 2500+ raga definitions
5. **Return ranked matches** with confidence scores

```python
from carnatic import RagaDetector

detector = RagaDetector()
results = detector.detect_from_file("sample.wav")

for r in results[:3]:
    print(f"{r.raga.name}: {r.confidence:.0%}")
```

### Raga Generation (Ragam → Melody)

Uses **grammar-based generation** following actual raga rules:

1. **Load raga definition** (arohanam, avarohanam, characteristic phrases)
2. **Build Markov transitions** weighted by raga structure
3. **Generate phrases** following melodic movement rules
4. **Export to MIDI** (can be converted to audio with FluidSynth)

```python
from carnatic import RagaGenerator

gen = RagaGenerator()
notes = gen.generate("kalyani", duration_beats=64, style='alapana')
gen.to_midi(notes, "kalyani_alapana.mid")
```

## Why This Approach?

### Traditional ML Problems:
- **Detection**: Needs thousands of labeled audio samples per raga
- **Generation**: Neural networks often produce "plausible but wrong" notes

### Our Approach:
- **Detection**: Uses the actual rules - ragas ARE defined by their scale/movement
- **Generation**: Respects raga grammar - every note is valid for the raga
- **No training data required** - works immediately with 2500+ ragas
- **Interpretable** - you can see exactly why a raga was detected

## Project Structure

```
carnatic_ml/
├── carnatic/               # Clean rule-based implementation
│   ├── __init__.py
│   ├── __main__.py         # CLI interface
│   ├── raga_db.py          # Raga database and lookups
│   ├── detector.py         # Audio → Raga detection
│   └── generator.py        # Raga → Melody generation
│
├── shared/
│   └── ragas_metadata/     # 2500+ raga JSON definitions
│
├── raga_detection/         # (Legacy) ML-based detection
├── raga_generation/        # (Legacy) LSTM-based generation
└── requirements.txt
```

## CLI Commands

```bash
# Detection
python -m carnatic detect audio.wav              # Auto-detect tonic
python -m carnatic detect audio.wav --tonic 261  # Specify Sa frequency
python -m carnatic detect audio.wav -v           # Verbose output

# Generation
python -m carnatic generate mohanam              # Default 32 beats
python -m carnatic generate kalyani -d 64        # 64 beats
python -m carnatic generate shankarabharanam -s tana  # Fast style
python -m carnatic generate bhairavi -o out.mid  # Specify output

# Info
python -m carnatic list                          # All ragas
python -m carnatic list --melakartas             # 72 parent ragas
python -m carnatic list --search bhairavi        # Search
python -m carnatic info mohanam                  # Detailed info
```

## Swara Reference

| Swara | Variants | Semitones from Sa |
|-------|----------|-------------------|
| S (Sa) | - | 0 |
| R (Ri) | R1, R2, R3 | 1, 2, 3 |
| G (Ga) | G1, G2, G3 | 2, 3, 4 |
| M (Ma) | M1, M2 | 5, 6 |
| P (Pa) | - | 7 |
| D (Dha) | D1, D2, D3 | 8, 9, 10 |
| N (Ni) | N1, N2, N3 | 9, 10, 11 |

## Future Enhancements

- [ ] MIDI to WAV synthesis (FluidSynth integration)
- [ ] Real-time detection from microphone
- [ ] Gamaka (ornament) patterns in generation
- [ ] Web interface
- [ ] Optional ML enhancement for improved detection
