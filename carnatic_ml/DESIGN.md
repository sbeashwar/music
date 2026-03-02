# Carnatic Raga Detection — Design Document

## Architecture Overview

This project has evolved through three major approaches to raga detection:

| | **v2** (`carnatic/detector_v2.py`) | **v3 ML** (`carnatic/detector_v3.py`) | **v3 Arohanam** (`raga_detection/`) |
|---|---|---|---|
| **Approach** | Pitch tracking + rule-based | Chroma ML classifier | Scale sequence matching |
| **Input** | Any audio | Any audio | Clean scale rendition |
| **Database** | 5,321 ragas (`carnatic/raga_db.py`) | 51 ragas (trained) | 5,321 ragas (`shared/ragas_metadata/`) |
| **Accuracy** | Poor on real recordings | 69.5% test, poor on new audio | 7/7 generated, works on clean voice |
| **Status** | Deprecated | Abandoned (bad sample quality) | **Active** |

---

## v2: Pitch Tracking + Rule-Based (`carnatic/detector_v2.py`)

### How it works
1. **Extract pitches** using `librosa.pyin` (monophonic pitch tracker)
2. **Find Sa and Pa** — the two fixed reference pitches (7 semitones apart)
3. **Map all pitches to swaras** relative to the detected Sa
4. **Count swara occurrences** and filter by frequency threshold
5. **Match against raga DB** by comparing detected swara set vs arohanam/avarohanam

### Why it failed
- **Gamakas (ornaments)** spread pitch energy across neighboring swaras — a clean G3 might
  register as both G2 and G3 due to the characteristic oscillation in Carnatic music
- **Too many swaras detected** — even a 5-note raga (pentatonic) would yield 7-8 detected
  swaras because transitional pitches between notes get counted
- **Tonic detection unreliable** when audio contains tambura drone, multiple instruments,
  or the singer's Sa is not prominent

### Key files
- `carnatic/detector_v2.py` — `RagaDetectorV2` class (789 lines)
- `carnatic/raga_db.py` — `RagaDB` class with 5,321 ragas
- `carnatic/gui.py` — Original GUI (uses v2 only)

---

## v3 ML: Chroma Feature Classifier (`carnatic/detector_v3.py`)

### How it works
1. **Extract features** (~130 dimensions per audio clip):
   - 12-bin chroma profile: mean, std, max, skew (48 dims)
   - Tonic-normalized chroma: mean, std (24 dims)
   - Pitch class histogram (12 dims)
   - Spectral features: centroid, bandwidth, rolloff, flatness (8 dims)
   - MFCC summary statistics (26 dims)
   - Tonnetz (12 dims)
2. **Train a RandomForest/SVM** on labeled recordings from `shared/samples/`
3. **Re-rank** predictions using raga metadata (penalize foreign notes, reward coverage)

### Why it was abandoned
- **Sample quality is poor** — the ~1400 MP3 samples from `shared/samples/Songs/` are
  full compositions with accompaniment, not clean scale renditions
- **Chroma profiles are nearly FLAT** for Carnatic music — gamakas spread pitch energy
  uniformly across all 12 semitones, making ragas indistinguishable by chroma alone
- Only 51 ragas had enough training samples; the model essentially memorized timbral
  differences between recordings rather than learning raga structure
- Test accuracy was 69.5%, but real-world accuracy on new recordings was near zero

### Key insight
> Chroma-based features are fundamentally unsuited for Carnatic music. The continuous
> pitch modulations (gamakas) that define Carnatic style also destroy the discrete
> pitch-class information that chroma features rely on.

---

## v3 Arohanam: Scale Sequence Matching (`raga_detection/`)

This is the **current active approach**. It works by detecting the specific note sequence
from a clean scale rendition and matching it against the 5,321-raga database.

### Design Philosophy

Instead of trying to identify ragas from complex compositions (which even human experts
can find difficult), this approach focuses on the simplest expression of a raga:

> **"Sing the arohanam and avarohanam"** — every Carnatic music student can do this, and
> it's the most canonical representation of a raga's identity.

### Architecture

```
Audio (WAV/MP3/mic) 
    │
    ▼
┌─────────────────────┐
│  ArohanamDetector    │  pYIN pitch tracking → tonic detection →
│                      │  semitone quantization → note segmentation →
│  (arohanam_detector) │  enharmonic disambiguation → direction detection
└──────────┬──────────┘
           │ ArohanamResult (swaras, semitones, raw_sequence, direction)
           ▼
┌─────────────────────┐
│ SwaraSequenceMatcher │  Loads 5,321 ragas from shared/ragas_metadata/ →
│                      │  semitone-based Jaccard matching →
│  (swara_matcher)     │  asymmetric aro/ava scoring → order matching
└──────────┬──────────┘
           │ List[RagaMatch] (ranked by score)
           ▼
       GUI / CLI
```

### Module Details

#### `raga_detection/arohanam_detector.py` — ArohanamDetector

**Purpose:** Extract the swara sequence from audio.

**Pipeline:**
1. **Pitch extraction** — `librosa.pyin` (fmin=80 Hz, fmax=800 Hz, hop_length=512)
2. **Tonic (Sa) detection** — Finds the most stable low-frequency pitch class; uses
   the median of the first and last portions of the recording
3. **Semitone quantization** — `12 * log2(freq / tonic_hz)` → round to nearest integer mod 12
4. **Note segmentation** — Groups consecutive frames with similar pitch into discrete notes.
   Minimum note duration configurable (0.08s for clean, 0.20s for voice mode)
5. **Transitional note filtering** (voice mode only) — Removes brief pitch excursions that
   occur during gamakas/slides. A note shorter than 40% of the median duration AND positioned
   between its neighbors (in pitch) is considered transitional
6. **Enharmonic disambiguation** — Resolves ambiguous semitones (e.g., semitone 2 can be R2
   or G1) using context rules: what comes before/after
7. **Direction detection** — Classifies as ascending/descending/mixed based on pitch transition ratios

**Voice mode** (`voice_mode=True`):
- Longer minimum note duration (0.20s vs 0.08s)
- Higher pitch confidence threshold (0.6 vs 0.5)
- Wider pitch tolerance (0.6 vs 0.4 semitones)
- Transitional note filtering enabled

**Key classes:**
- `DetectedNote` — individual note with start_time, duration, frequency, semitone, swara, confidence
- `ArohanamResult` — detection result with detected_swaras, raw_sequence, tonic_hz, direction, semitones

#### `raga_detection/swara_matcher.py` — SwaraSequenceMatcher

**Purpose:** Match detected swaras against the 5,321-raga database.

**Raga database:** 5,321 ragas loaded from `shared/ragas_metadata/*.json`. Each raga has:
- `arohanam` / `avarohanam` — ordered swara lists (e.g., `["S", "R2", "G3", "P", "D2", "S"]`)
- `is_melakarta` — whether it's one of the 72 parent scales
- `parent_melakarta` — which melakarta this janya derives from

**Matching algorithm (`match_swaras`):**

1. **Semitone-based Jaccard similarity** (primary score):
   - Convert detected swaras and raga swaras to semitone sets
   - `score = |intersection| / |union|`
   - This handles enharmonic equivalents (R2/G1 both = semitone 2)

2. **Match type classification:**
   - `exact` — all raga semitones matched, no extras → `score = 1.0`
   - `superset` — all raga semitones matched + extras → `score = jaccard × 0.9`
   - `subset` — all detected in raga, some raga notes missing → `score = jaccard × 0.85`
   - `partial` — some overlap → `score = jaccard × 0.7`

3. **Swara name bonus** — +0.005 if swara names (not just semitones) match exactly.
   This differentiates R2-based ragas from G1-based ragas that share the same semitone.

4. **Sequence order scoring** — Compares note order against the raga's arohanam sequence
   using concordant pair counting (like Kendall's tau). Works at the semitone level to
   avoid enharmonic naming issues.

5. **Asymmetric arohanam/avarohanam scoring** — When `raw_sequence` is provided and
   direction is "mixed" (typical: user sings arohanam then avarohanam):
   - Splits the raw sequence at the turning point (highest pitch)
   - Compares ascending half vs raga's arohanam (Jaccard on semitone sets)
   - Compares descending half vs raga's avarohanam
   - Average of both → asymmetry score
   - This is critical for asymmetric ragas like Mand (5-note arohanam, 7-note avarohanam)
     where 237 ragas share the same overall swara set but differ in direction structure.

6. **Melakarta bonus** — +0.01 for melakartas (slight preference when scores are tied)

**Final score:** `score = base × 0.65 + asymmetric × 0.35` (when asymmetry data available)

**Indices for fast lookup:**
- By swara count (5-note, 6-note, 7-note ragas)
- By arohanam set (frozenset of swaras → raga IDs)
- By semitone set (sorted semitone tuple → raga IDs)

#### `raga_detection/raga_player.py` — Raga Scale Generator

**Purpose:** Generate MIDI and WAV audio for any raga's scale.

**MIDI generation:**
- Uses `midiutil` library
- Maps swaras to MIDI note numbers relative to configurable tonic
- Handles ascending (arohanam) then descending (avarohanam) with proper octave wrapping
- Configurable tempo, instrument (GM), tonic

**WAV synthesis:**
- Simple sine wave with harmonics (fundamental + 0.3×2nd + 0.15×3rd + 0.05×4th)
- ADSR envelope (attack=0.05s, decay=0.05s, sustain=0.8, release=0.1s)
- 22050 Hz sample rate, float32

**Avarohanam handling:** When generating the descending portion, starts from upper Sa and
applies octave adjustments to ensure the sequence descends properly.

#### `raga_detection/gui.py` — Tkinter GUI

**Two-tab interface:**

1. **Detect Raga tab:**
   - Record from microphone (30 seconds) or load audio file
   - Source mode toggle: Voice (filters gamakas) vs Clean (for generated/instrument)
   - Shows: tonic, direction, detected swaras, raw sequence, top 15 matches with scores

2. **Play Raga tab:**
   - Search any of 5,321 ragas by name (partial match, case-insensitive)
   - Select from results list, shows arohanam/avarohanam/melakarta info
   - Generate as WAV or MIDI with configurable tonic (C3-C5)
   - Auto-plays WAV if `sounddevice` + `soundfile` are installed

### Test Results

**Generated scale detection (7/7):**
| Raga | Rank | Score |
|------|------|-------|
| Mohanam | #1 | 1.003 |
| Kalyani | #2 | 1.003 |
| Bahudari | #1 | 0.991 |
| Shankarabharanam | #4 | 1.013 |
| Hamsadhwani | #1 | 1.003 |
| Kharaharapriya | #1 | 0.955 |
| Todi | #1 | 0.965 |

Note: Ragas ranked #2-4 are tied with other ragas that share the same swara set. For
example, Shankarabharanam ties with Dheerasankarabharanam (its melakarta parent name).

**Voice recording:** Works with voice mode enabled, though accuracy depends on clean
singing with held notes and minimal gamakas.

### Key Design Decisions

1. **Semitone-first comparison** — Eliminates enharmonic false negatives. R2/G1 both map
   to semitone 2, so a detector naming it "G1" when the raga says "R2" isn't penalized.

2. **Asymmetric matching** — Many Carnatic ragas have different notes in arohanam vs
   avarohanam (vakra ragas). Without this, 237 ragas score identically for Mand. With
   asymmetric scoring, Mand jumps from #10 to #2.

3. **Voice mode** — Human voice produces gamakas (pitch oscillations) by nature in
   Carnatic music. Even when asked to sing a "plain" scale, singers add brief pitch
   transitions between notes. Voice mode uses longer minimum durations and transitional
   note filtering to handle this.

4. **No ML training required** — Works purely from the raga metadata (arohanam/avarohanam
   definitions). No labeled audio samples needed. This was a deliberate pivot after
   discovering that the available training samples were too poor for ML.

### Dependencies

```
librosa          # Pitch detection (pYIN)
numpy            # Array operations
soundfile        # Audio I/O
midiutil         # MIDI generation
sounddevice      # Microphone recording + playback (optional)
```
