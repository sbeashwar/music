"""
Raga Detection v3 — Scientific ML-based approach

Architecture:
  Stage 1: Chroma-feature ML classifier (RandomForest / SVM)
           - Extracts 12-bin chroma profiles from audio
           - Robust to multi-instrument recordings (voice, violin, mridangam, tambura)
           - Trained on labeled recordings from shared/samples/
  
  Stage 2: Rule-based re-ranking using raga metadata
           - Uses arohanam/avarohanam patterns from the 5321-raga database
           - Penalizes foreign notes, rewards coverage
           - Applies specificity penalty (raga shouldn't claim unheard notes)
  
Why chroma features?
  - Chroma captures which pitch classes (C, C#, D, ...) are present
  - This is invariant to octave, tonic, and instrument timbre
  - A raga's identity IS its pitch class usage pattern
  - Unlike raw pitch tracking, chroma works with polyphonic audio
  
Feature vector (per audio clip):
  - 12-bin chroma: mean, std, max, skew (48 dims)
  - Chroma normalized to tonic: mean, std (24 dims)  
  - Pitch class histogram (12 dims)
  - Spectral features: centroid, bandwidth, rolloff, flatness (8 dims)
  - MFCC summary statistics (26 dims)
  - Tonnetz (12 dims)
  - Total: ~130 dimensions
"""

import os
import json
import re
import numpy as np
import librosa
from collections import Counter, defaultdict
from typing import List, Tuple, Optional, Dict, Set
from dataclasses import dataclass, field
from pathlib import Path

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.svm import SVC
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.pipeline import Pipeline
    import joblib
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

class CarnaticFeatureExtractor:
    """
    Extract features optimized for Carnatic raga identification.
    
    Key insight: Carnatic music is defined by WHICH pitch classes are used
    and HOW they are used (ascending vs descending, gamakas, emphasis).
    Chroma features capture this directly.
    """
    
    def __init__(self, sr=22050, hop_length=512, n_chroma=12):
        self.sr = sr
        self.hop_length = hop_length
        self.n_chroma = n_chroma
    
    def extract(self, y: np.ndarray, sr: int = None) -> Optional[np.ndarray]:
        """Extract full feature vector from audio samples.
        
        Also stores the chroma profile in self._last_chroma_profile 
        for reuse by detect_from_audio (avoids redundant CQT).
        """
        sr = sr or self.sr
        if len(y) < sr:  # less than 1 second
            return None
        
        try:
            features = []
            
            # 1. Chroma features (core of raga identification)
            chroma = librosa.feature.chroma_cqt(
                y=y, sr=sr, hop_length=self.hop_length,
                n_chroma=self.n_chroma, bins_per_octave=24
            )
            
            # Basic chroma statistics
            chroma_mean = np.mean(chroma, axis=1)
            chroma_std = np.std(chroma, axis=1)
            features.extend(chroma_mean)                    # 12: mean per pitch class
            features.extend(chroma_std)                     # 12: variation per pitch class
            features.extend(np.max(chroma, axis=1))        # 12: peak per pitch class
            
            # Chroma skewness (asymmetry in pitch class usage over time)
            from scipy.stats import skew
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                sk = skew(chroma, axis=1)
            features.extend(np.nan_to_num(sk, nan=0.0))   # 12: skew per pitch class
            
            # 2. Tonic-normalized chroma
            # Find the dominant pitch class (likely Sa or Pa) and rotate
            tonic_bin = np.argmax(chroma_mean)
            rotated = np.roll(chroma_mean, -tonic_bin)
            features.extend(rotated)                        # 12: tonic-relative profile
            rotated_std = np.roll(chroma_std, -tonic_bin)
            features.extend(rotated_std)                    # 12: tonic-relative variation
            
            # 3. Pitch class histogram (binary presence)
            threshold = 0.3 * np.max(chroma_mean)
            pitch_present = (chroma_mean > threshold).astype(float)
            features.extend(pitch_present)                  # 12: which pitch classes active
            
            # 4. Chroma transition matrix (how notes move to each other)
            # This captures arohanam/avarohanam patterns
            chroma_binary = (chroma > np.median(chroma)).astype(float)
            # Vectorized: transitions[i, j] = sum_t binary[i,t] * binary[j,t+1]
            transitions = chroma_binary[:, :-1] @ chroma_binary[:, 1:].T
            # Normalize
            tsum = transitions.sum()
            if tsum > 0:
                transitions /= tsum
            # Ascending vs descending energy
            asc_energy = sum(transitions[i, (i+k)%12] 
                           for i in range(12) for k in range(1, 7))
            desc_energy = sum(transitions[i, (i-k)%12] 
                            for i in range(12) for k in range(1, 7))
            features.append(asc_energy)                     # 1
            features.append(desc_energy)                    # 1
            # Diagonal dominance (sustained notes)
            features.append(np.trace(transitions) / 12)     # 1
            
            # 5. Spectral features (timbre context)
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            features.append(np.mean(spectral_centroids))    # 1
            features.append(np.std(spectral_centroids))     # 1
            
            spectral_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            features.append(np.mean(spectral_bw))           # 1
            features.append(np.std(spectral_bw))            # 1
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            features.append(np.mean(spectral_rolloff))      # 1
            features.append(np.std(spectral_rolloff))       # 1
            
            spectral_flat = librosa.feature.spectral_flatness(y=y)
            features.append(np.mean(spectral_flat))         # 1
            features.append(np.std(spectral_flat))          # 1
            
            # 6. MFCCs (timbral fingerprint - helps distinguish instruments)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            features.extend(np.mean(mfcc, axis=1))          # 13
            features.extend(np.std(mfcc, axis=1))           # 13
            
            # 7. Tonnetz (harmonic relationships) - from existing chroma
            tonnetz = librosa.feature.tonnetz(chroma=chroma)
            features.extend(np.mean(tonnetz, axis=1))       # 6
            features.extend(np.std(tonnetz, axis=1))        # 6
            
            # Cache chroma profile for reuse in detection
            self._last_chroma_profile = np.mean(chroma, axis=1)
            
            return np.array(features, dtype=np.float32)
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None
    
    def extract_from_file(self, path: str, duration: float = 60.0, 
                          offset: float = 0.0) -> Optional[np.ndarray]:
        """Extract features from an audio file.
        
        Uses soundfile for WAV/FLAC (15x faster than librosa.load),
        falls back to librosa for MP3 and other formats.
        """
        try:
            y, sr = self._fast_load(path, duration, offset)
            return self.extract(y, sr)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return None
    
    def _fast_load(self, path: str, duration: float = 60.0,
                   offset: float = 0.0) -> Tuple[np.ndarray, int]:
        """Load audio, using soundfile when possible for speed."""
        import soundfile as sf
        
        ext = os.path.splitext(path)[1].lower()
        
        if ext in ('.wav', '.flac', '.ogg'):
            info = sf.info(path)
            start_frame = int(offset * info.samplerate)
            n_frames = int(duration * info.samplerate) if duration else -1
            
            y, sr = sf.read(path, start=start_frame, stop=start_frame + n_frames 
                           if n_frames > 0 else None, dtype='float32')
            
            # Convert stereo to mono
            if y.ndim > 1:
                y = np.mean(y, axis=1)
            
            # Resample if needed
            if sr != self.sr:
                y = librosa.resample(y, orig_sr=sr, target_sr=self.sr)
                sr = self.sr
            
            return y, sr
        else:
            # MP3 and others: use librosa (slower but handles all formats)
            return librosa.load(path, sr=self.sr, duration=duration, offset=offset)
            return None
    
    @property
    def feature_dim(self) -> int:
        """Approximate feature vector dimension."""
        return 133  # See counts above


# ---------------------------------------------------------------------------
# Training data preparation from filename conventions
# ---------------------------------------------------------------------------

# Known raga name mappings (filename → canonical)
RAGA_NAME_MAP = {
    'thodi': 'todi',
    'shankarabharanam': 'shankarabharanam',
    'kharaharapriya': 'kharaharapriya', 
    'harikambodhi': 'harikambhoji',
    'harikambhoji': 'harikambhoji',
    'panthuvarali': 'pantuvarali',
    'purvikalyani': 'purvakalyani',
    'purvi_kalyani': 'purvakalyani',
    'reethigowla': 'ritigaula',
    'reethigowlai': 'ritigaula',
    'shuddhadhanyasi': 'suddhadhanyasi',
    'suddhasaveri': 'suddhasaveri',
    'dwijavanthi': 'dvijavanti',
    'dwijavanti': 'dvijavanti',
    'kedaragowla': 'kedaragaula',
    'kedaragaula': 'kedaragaula',
    'natakuranji': 'nattaikurinji',
    'yadukulakambodhi': 'yadukulakambhoji',
    'yadukula_kambodhi': 'yadukulakambhoji',
    'karnatakabehag': 'karnatakabihaag',
    'salagabhairavi': 'salagabhairavi',
    'salaga_bhairavi': 'salagabhairavi',
    'senchukambodhi': 'sencukambhoji',
    'gowrimanohari': 'gaurimanohari',
    'ananda_bhairavi': 'anandabhairavi',
    'anandabhairavi': 'anandabhairavi',
    'hamsanandi': 'hamsanandi',
}

# Performance types in filenames
PERF_TYPES = {'alapana', 'tanam', 'neraval', 'swarakalpana', 'pallavi', 'varnam', 'viruttam', 'slokam'}

def parse_raga_from_filename(filename: str) -> Optional[str]:
    """
    Extract raga name from sample filename.
    
    Patterns:
      01a-mayamalavagowla-alapana-ssi-c11.mp3        → mayamalavagowla
      02b-dinamanivamsha-harikambodhi-swarakalpana... → harikambodhi
      04c-biranabrova-kalyani-neraval-ssi-c07.mp3    → kalyani
      01b-varnam-begada-swarakalpana-voleti-c01.mp3   → begada
    """
    name = os.path.splitext(filename)[0].lower()
    
    # Skip ragamalika (multi-raga) and fillers without raga identity
    if 'ragamalika' in name and 'slokam' in name:
        return None
    
    # Split on hyphens and underscores
    parts = re.split(r'[-_]', name)
    
    # Remove leading track number (e.g. "01a", "02b")
    while parts and re.match(r'^\d+[a-z]?$', parts[0]):
        parts.pop(0)
    
    # Remove FILLER prefix
    while parts and parts[0] == 'filler':
        parts.pop(0)
    
    # Find performance type keyword and take the word(s) before it as raga
    for i, part in enumerate(parts):
        if part in PERF_TYPES:
            # Raga name is the part(s) just before the performance type
            # Could be one word (kalyani) or compound (kedaragowla)
            if i > 0:
                raga = parts[i - 1]
                # Some ragas are two-word: "purvi kalyani", "yadukula kambodhi"
                if i > 1 and parts[i-2] not in PERF_TYPES:
                    two_word = parts[i-2] + parts[i-1]
                    if two_word in RAGA_NAME_MAP:
                        raga = two_word
                    else:
                        # Check if the two-parter looks like a compound raga name
                        candidate = parts[i-2] + '_' + parts[i-1]
                        if candidate in RAGA_NAME_MAP:
                            raga = RAGA_NAME_MAP[candidate]
                            return raga
                
                return RAGA_NAME_MAP.get(raga, raga)
    
    # Fallback: the second part is often the raga for alapana tracks
    if len(parts) >= 2:
        candidate = parts[0]
        if candidate not in PERF_TYPES and not candidate.startswith(('rtp', 'slokam')):
            return RAGA_NAME_MAP.get(candidate, candidate)
    
    return None


def build_training_dataset(
    samples_dir: str,
    feature_extractor: CarnaticFeatureExtractor,
    max_per_raga: int = 50,
    duration: float = 30.0,
    min_samples_per_raga: int = 3
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Build training dataset from the labeled audio samples.
    
    Scans the directory for mp3/wav files, parses raga from filename,
    extracts features, and returns X, y arrays.
    
    Data augmentation: for each file, extract features from multiple
    time offsets (0s, 15s, 30s) to get more training samples.
    
    Args:
        samples_dir: Path to directory with audio files
        feature_extractor: Feature extraction instance
        max_per_raga: Maximum samples per raga (prevents class imbalance)
        duration: Duration per clip to analyze
        min_samples_per_raga: Minimum samples needed to include a raga
        
    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Label array (raga names)
        raga_list: Sorted list of raga names
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn is required for ML training")
    
    # Phase 1: Discover and parse all files
    file_raga_map = {}
    raga_counts = Counter()
    
    for root, dirs, files in os.walk(samples_dir):
        for f in files:
            if not f.lower().endswith(('.mp3', '.wav', '.flac')):
                continue
            
            # Try to get raga from parent directory name first
            parent = os.path.basename(root)
            # Skip 'all' directory (duplicate of Songs) and 'gen' (generated)
            if parent in ('all', 'gen'):
                continue
            if parent in ('Songs', 'samples'):
                raga = parse_raga_from_filename(f)
            else:
                raga = parent.lower()  # Directory name IS the raga
            
            if raga and len(raga) > 2:
                filepath = os.path.join(root, f)
                file_raga_map[filepath] = raga
                raga_counts[raga] += 1
    
    # Filter: only ragas with enough samples
    valid_ragas = {r for r, c in raga_counts.items() if c >= min_samples_per_raga}
    print(f"Found {len(file_raga_map)} files, {len(raga_counts)} ragas, "
          f"{len(valid_ragas)} with >= {min_samples_per_raga} samples")
    print(f"Top ragas: {raga_counts.most_common(15)}")
    
    # Phase 2: Extract features with augmentation
    X_list = []
    y_list = []
    
    per_raga_count = Counter()
    
    total_valid = sum(1 for r in file_raga_map.values() if r in valid_ragas)
    processed = 0
    
    for filepath, raga in sorted(file_raga_map.items()):
        if raga not in valid_ragas:
            continue
        if per_raga_count[raga] >= max_per_raga:
            continue
        
        processed += 1
        if processed % 25 == 0 or processed == 1:
            print(f"  [{processed}/{total_valid}] Extracting features...", flush=True)
        
        # Data augmentation: extract from multiple offsets
        offsets = [0.0]
        try:
            info = librosa.get_duration(path=filepath)
            if info > duration + 15:
                offsets.append(15.0)
            if info > duration + 30:
                offsets.append(30.0)
        except:
            pass
        
        for offset in offsets:
            if per_raga_count[raga] >= max_per_raga:
                break
            
            feats = feature_extractor.extract_from_file(
                filepath, duration=duration, offset=offset
            )
            if feats is not None:
                X_list.append(feats)
                y_list.append(raga)
                per_raga_count[raga] += 1
    
    if not X_list:
        raise ValueError("No features extracted! Check audio files.")
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    # Replace NaN/Inf with 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features, "
          f"{len(set(y))} ragas")
    for raga, count in sorted(per_raga_count.items(), key=lambda x: -x[1])[:20]:
        print(f"  {raga:25s} {count:3d} samples")
    
    return X, y, sorted(set(y))


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_raga_classifier(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = 'rf',
    n_folds: int = 5
) -> Tuple[Pipeline, dict]:
    """
    Train a raga classifier.
    
    Args:
        X: Feature matrix
        y: Labels
        model_type: 'rf' (RandomForest), 'gb' (GradientBoosting), 'svm'
        n_folds: Cross-validation folds
        
    Returns:
        pipeline: Trained sklearn Pipeline (scaler + classifier)
        metrics: Dictionary with accuracy, cross-val scores, etc.
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn is required")
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Choose classifier
    if model_type == 'rf':
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=3,
            min_samples_leaf=1,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    elif model_type == 'gb':
        clf = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
    elif model_type == 'svm':
        clf = SVC(
            kernel='rbf',
            C=10.0,
            gamma='scale',
            probability=True,
            class_weight='balanced',
            random_state=42
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Build pipeline with scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', clf)
    ])
    
    # Cross-validation
    print(f"\nTraining {model_type.upper()} classifier...")
    if len(set(y)) >= n_folds:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipeline, X, y_encoded, cv=cv, scoring='accuracy')
        print(f"Cross-validation accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"  Per-fold: {[f'{s:.3f}' for s in cv_scores]}")
    else:
        cv_scores = np.array([0.0])
    
    # Train/test split for detailed metrics
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    accuracy = (y_pred == y_test).mean()
    print(f"\nTest accuracy: {accuracy:.3f}")
    print("\nClassification report (test set):")
    print(classification_report(
        y_test, y_pred, 
        target_names=le.classes_,
        zero_division=0
    ))
    
    # Retrain on full dataset for production
    pipeline.fit(X, y_encoded)
    
    metrics = {
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'n_samples': len(X),
        'n_ragas': len(le.classes_),
        'ragas': list(le.classes_),
    }
    
    return pipeline, le, metrics


def save_model(pipeline, label_encoder, metrics, feature_extractor, 
               save_dir: str):
    """Save trained model and metadata."""
    os.makedirs(save_dir, exist_ok=True)
    
    joblib.dump({
        'pipeline': pipeline,
        'label_encoder': label_encoder,
        'metrics': metrics,
        'feature_extractor': feature_extractor,
    }, os.path.join(save_dir, 'raga_classifier.pkl'))
    
    # Also save human-readable metadata
    with open(os.path.join(save_dir, 'model_info.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Model saved to {save_dir}/")


def load_model(save_dir: str):
    """Load trained model."""
    data = joblib.load(os.path.join(save_dir, 'raga_classifier.pkl'))
    return (data['pipeline'], data['label_encoder'], 
            data['metrics'], data['feature_extractor'])


# ---------------------------------------------------------------------------
# Detection (inference)
# ---------------------------------------------------------------------------

@dataclass
class DetectionResultV3:
    """Result from ML-based raga detection."""
    raga_name: str
    confidence: float
    ml_probability: float       # Raw ML probability
    rule_score: float           # Rule-based refinement score
    chroma_profile: np.ndarray  # 12-bin pitch class profile
    match_details: dict = field(default_factory=dict)


class RagaDetectorV3:
    """
    Scientific ML-based raga detector.
    
    Two-stage approach:
    1. ML classifier gives probability distribution over known ragas
    2. Rule-based refinement using raga metadata (arohanam/avarohanam)
       to re-rank and extend to ragas not in training set
    """
    
    def __init__(self, model_dir: str = None):
        self.model_dir = model_dir or os.path.join(
            os.path.dirname(__file__), 'models', 'v3'
        )
        self.pipeline = None
        self.label_encoder = None
        self.feature_extractor = CarnaticFeatureExtractor()
        self.metrics = None
        self._raga_db = None
        
        # Try to load pre-trained model
        model_path = os.path.join(self.model_dir, 'raga_classifier.pkl')
        if os.path.exists(model_path):
            self._load_model()
    
    def _load_model(self):
        """Load pre-trained model."""
        try:
            # Patch __main__.CarnaticFeatureExtractor to resolve pickle reference
            import __main__
            __main__.CarnaticFeatureExtractor = CarnaticFeatureExtractor
            
            data = joblib.load(
                os.path.join(self.model_dir, 'raga_classifier.pkl')
            )
            self.pipeline = data['pipeline']
            self.label_encoder = data['label_encoder']
            self.metrics = data.get('metrics', {})
            self.feature_extractor = data.get(
                'feature_extractor', CarnaticFeatureExtractor()
            )
            print(f"Loaded ML model: {self.metrics.get('n_ragas', '?')} ragas, "
                  f"accuracy={self.metrics.get('accuracy', '?')}")
        except Exception as e:
            print(f"Could not load model: {e}")
            import traceback; traceback.print_exc()
    
    @property
    def raga_db(self):
        """Lazy-load raga database."""
        if self._raga_db is None:
            try:
                from .raga_db import get_db
                self._raga_db = get_db()
            except:
                self._raga_db = []
        return self._raga_db
    
    @property
    def has_model(self) -> bool:
        return self.pipeline is not None
    
    def detect_from_file(self, path: str, top_n: int = 15,
                         duration: float = 60.0,
                         offset: float = 0.0) -> List[DetectionResultV3]:
        """Detect raga from audio file (uses fast loading for WAV/FLAC)."""
        y, sr = self.feature_extractor._fast_load(path, duration, offset)
        return self.detect_from_audio(y, sr, top_n)
    
    def detect_from_audio(self, y: np.ndarray, sr: int = 22050,
                          top_n: int = 15) -> List[DetectionResultV3]:
        """
        Detect raga from audio samples.
        
        If ML model is available: use ML + rule-based refinement
        If no model: fall back to pure chroma-based rule matching
        """
        features = self.feature_extractor.extract(y, sr)
        if features is None:
            return []
        
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Reuse chroma profile cached during feature extraction (no redundant CQT)
        chroma_profile = getattr(self.feature_extractor, '_last_chroma_profile', None)
        if chroma_profile is None:
            # Fallback: compute it (shouldn't normally happen)
            chroma = librosa.feature.chroma_cqt(
                y=y, sr=sr, hop_length=512, n_chroma=12, bins_per_octave=24
            )
            chroma_profile = np.mean(chroma, axis=1)
        
        if self.has_model:
            return self._detect_with_ml(features, chroma_profile, top_n)
        else:
            return self._detect_rules_only(chroma_profile, top_n)
    
    def _detect_with_ml(self, features: np.ndarray, 
                        chroma_profile: np.ndarray,
                        top_n: int) -> List[DetectionResultV3]:
        """ML-based detection with rule refinement.
        
        Also adds rule-only candidates from the full raga database
        to catch ragas not in the ML training set.
        """
        features_2d = features.reshape(1, -1)
        
        # Get ML probabilities
        probas = self.pipeline.predict_proba(features_2d)[0]
        classes = self.label_encoder.classes_
        
        # Build raga name lookup for fast access
        if not hasattr(self, '_raga_name_map'):
            self._raga_name_map = {}
            for raga in self.raga_db:
                self._raga_name_map[raga.name.lower()] = raga
        
        # Normalize chroma to find tonic
        tonic_bin = np.argmax(chroma_profile)
        rotated = np.roll(chroma_profile, -tonic_bin)
        rotated_norm = rotated / (rotated.sum() + 1e-8)
        
        # Score ML-known ragas (ML + rules combined)
        results = []
        ml_raga_names = set()
        
        for idx, (raga_name, ml_prob) in enumerate(zip(classes, probas)):
            ml_raga_names.add(raga_name.lower())
            raga = self._raga_name_map.get(raga_name.lower())
            rule_score = self._chroma_match_score(rotated_norm, raga) if raga else 0.5
            
            # Combined score: ML dominates, rules help break ties
            combined = ml_prob * 0.7 + rule_score * 0.3
            
            results.append(DetectionResultV3(
                raga_name=raga_name,
                confidence=combined,
                ml_probability=ml_prob,
                rule_score=rule_score,
                chroma_profile=chroma_profile,
                match_details={
                    'method': 'ml+rules',
                    'ml_rank': 0,  # filled below
                }
            ))
        
        # Also score ALL ragas via rules only (catches ragas outside ML training set)
        for raga in self.raga_db:
            if raga.name.lower() in ml_raga_names:
                continue  # Already scored above
            score = self._chroma_match_score(rotated_norm, raga)
            if score > 0.4:  # Only include reasonably matching ragas
                results.append(DetectionResultV3(
                    raga_name=raga.name,
                    confidence=score * 0.3,  # Rules-only = 30% weight (like rule component)
                    ml_probability=0.0,
                    rule_score=score,
                    chroma_profile=chroma_profile,
                    match_details={'method': 'rules_only_extra'}
                ))
        
        # Sort by combined confidence
        results.sort(key=lambda r: -r.confidence)
        
        # Fill in ML rank for ML-scored entries
        ml_results = [r for r in results if r.match_details.get('method') == 'ml+rules']
        ml_order = sorted(range(len(ml_results)), 
                          key=lambda i: -ml_results[i].ml_probability)
        for rank, idx in enumerate(ml_order, 1):
            ml_results[idx].match_details['ml_rank'] = rank
        
        return results[:top_n]
    
    def _detect_rules_only(self, chroma_profile: np.ndarray,
                           top_n: int) -> List[DetectionResultV3]:
        """
        Pure rule-based detection using chroma profile.
        
        Used when no ML model is trained yet.
        """
        from .raga_db import SWARA_TO_SEMITONE
        
        results = []
        
        # Normalize chroma to find tonic
        tonic_bin = np.argmax(chroma_profile)
        rotated = np.roll(chroma_profile, -tonic_bin)
        rotated_norm = rotated / (rotated.sum() + 1e-8)
        
        for raga in self.raga_db:
            score = self._chroma_match_score(rotated_norm, raga)
            if score > 0.2:
                results.append(DetectionResultV3(
                    raga_name=raga.name,
                    confidence=score,
                    ml_probability=0.0,
                    rule_score=score,
                    chroma_profile=chroma_profile,
                    match_details={
                        'method': 'rules_only',
                        'tonic_bin': int(tonic_bin),
                    }
                ))
        
        results.sort(key=lambda r: -r.confidence)
        return results[:top_n]
    
    def _compute_rule_score(self, raga_name: str, 
                            chroma_profile: np.ndarray) -> float:
        """Compute rule-based match score for a raga against chroma profile."""
        # Find this raga in our database
        matches = [r for r in self.raga_db if r.name.lower() == raga_name.lower()]
        if not matches:
            return 0.5  # Unknown raga - neutral score
        
        raga = matches[0]
        
        # Normalize chroma, rotate to best tonic alignment
        tonic_bin = np.argmax(chroma_profile)
        rotated = np.roll(chroma_profile, -tonic_bin)
        rotated_norm = rotated / (rotated.sum() + 1e-8)
        
        return self._chroma_match_score(rotated_norm, raga)
    
    def _chroma_match_score(self, chroma_norm: np.ndarray, raga) -> float:
        """
        Score how well a tonic-normalized chroma profile matches a raga.
        
        chroma_norm: 12-bin array normalized to tonic at index 0
        """
        from .raga_db import SWARA_TO_SEMITONE
        
        raga_semitones = raga.scale_semitones
        
        # Energy in raga notes vs outside
        in_raga = sum(chroma_norm[st] for st in raga_semitones if st < 12)
        total = chroma_norm.sum()
        
        if total < 1e-8:
            return 0.0
        
        match_ratio = in_raga / total
        
        # Coverage: are all raga notes represented?
        raga_notes_present = sum(
            1 for st in raga_semitones 
            if st < 12 and chroma_norm[st] > 0.02
        )
        coverage = raga_notes_present / len(raga_semitones) if raga_semitones else 0
        
        # Specificity: penalize unheard raga notes
        raga_notes_absent = sum(
            1 for st in raga_semitones
            if st < 12 and chroma_norm[st] < 0.01
        )
        specificity = 1.0 - (raga_notes_absent / max(len(raga_semitones), 1)) * 0.3
        
        return match_ratio * 0.5 + coverage * 0.3 + specificity * 0.2


# ---------------------------------------------------------------------------
# CLI: Train and evaluate
# ---------------------------------------------------------------------------

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train ML raga detector')
    parser.add_argument('--samples-dir', type=str, 
                        default=os.path.join(os.path.dirname(__file__), 
                                            '..', 'shared', 'samples', 'Songs'),
                        help='Directory with labeled audio samples')
    parser.add_argument('--model-dir', type=str,
                        default=os.path.join(os.path.dirname(__file__), 
                                            'models', 'v3'),
                        help='Directory to save trained model')
    parser.add_argument('--model-type', type=str, default='rf',
                        choices=['rf', 'gb', 'svm'],
                        help='Classifier type')
    parser.add_argument('--duration', type=float, default=15.0,
                        help='Audio clip duration for feature extraction')
    parser.add_argument('--min-samples', type=int, default=3,
                        help='Minimum samples per raga to include')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("Carnatic Raga Detector v3 — ML Training Pipeline")
    print("=" * 70)
    
    # Extract features
    extractor = CarnaticFeatureExtractor()
    X, y, raga_list = build_training_dataset(
        args.samples_dir, extractor,
        duration=args.duration,
        min_samples_per_raga=args.min_samples
    )
    
    # Train
    pipeline, le, metrics = train_raga_classifier(X, y, args.model_type)
    
    # Save
    save_model(pipeline, le, metrics, extractor, args.model_dir)
    
    print("\n" + "=" * 70)
    print("Training complete!")
    print(f"  Model: {args.model_type.upper()}")
    print(f"  Ragas: {metrics['n_ragas']}")
    print(f"  Samples: {metrics['n_samples']}")
    print(f"  CV Accuracy: {metrics['cv_mean']:.1%} ± {metrics['cv_std']:.1%}")
    print(f"  Test Accuracy: {metrics['accuracy']:.1%}")
    print("=" * 70)


if __name__ == '__main__':
    main()
