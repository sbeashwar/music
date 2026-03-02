"""
Raga Detector - Identify raga from audio

Approach:
1. Detect pitches from audio using pitch tracking
2. Find the tonic (Sa) - usually the drone or most stable pitch
3. Map all pitches to swaras relative to Sa
4. Match swara set against raga database
5. Return ranked list of matching ragas

No ML training required - uses raga grammar rules directly!
"""

import numpy as np
from typing import List, Tuple, Optional, Set
from dataclasses import dataclass
from collections import Counter

from .raga_db import RagaDB, Raga, SWARA_TO_SEMITONE, SEMITONE_TO_SWARAS, get_db

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("Warning: librosa not installed. Audio analysis will be limited.")


@dataclass
class DetectionResult:
    """Result of raga detection."""
    raga: Raga
    confidence: float
    detected_swaras: Set[str]
    match_details: dict
    
    def __repr__(self):
        return f"DetectionResult(raga='{self.raga.name}', confidence={self.confidence:.2f})"


class RagaDetector:
    """
    Detect raga from audio using pitch analysis and rule matching.
    """
    
    def __init__(self, db: Optional[RagaDB] = None):
        self.db = db or get_db()
        
        # Default tonic frequency (C4 = 261.63 Hz, but Carnatic often uses different)
        # This can be auto-detected or specified
        self.default_tonic_hz = 261.63
    
    def detect_from_file(
        self, 
        audio_path: str, 
        tonic_hz: Optional[float] = None,
        top_n: int = 5
    ) -> List[DetectionResult]:
        """
        Detect raga from an audio file.
        
        Args:
            audio_path: Path to audio file (WAV, MP3, etc.)
            tonic_hz: Tonic (Sa) frequency in Hz. If None, will auto-detect.
            top_n: Number of top matches to return
            
        Returns:
            List of DetectionResult sorted by confidence
        """
        if not HAS_LIBROSA:
            raise ImportError("librosa is required for audio analysis")
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=22050)
        
        return self.detect_from_audio(y, sr, tonic_hz, top_n)
    
    def detect_from_audio(
        self,
        y: np.ndarray,
        sr: int = 22050,
        tonic_hz: Optional[float] = None,
        top_n: int = 5
    ) -> List[DetectionResult]:
        """
        Detect raga from audio samples.
        
        Returns multiple possible matches with confidence scores. Parent (melakarta)
        and derived (janya) ragas may both appear since they share scales.
        
        Args:
            y: Audio samples
            sr: Sample rate
            tonic_hz: Tonic frequency. If None, will auto-detect.
            top_n: Number of top matches to return
        """
        # Step 1: Extract pitches
        pitches_hz = self._extract_pitches(y, sr)
        
        if len(pitches_hz) == 0:
            return []
        
        # Step 2: Get note distribution (not just presence)
        swara_distribution = self._get_swara_distribution(pitches_hz)
        
        # Step 3: Find tonic if not provided - try multiple and pick best
        if tonic_hz is None:
            tonic_hz = self._find_best_tonic(pitches_hz, top_n)
        
        # Step 4: Get swaras at this tonic
        swaras = self._pitches_to_swaras(pitches_hz, tonic_hz)
        
        # Step 5: Match against raga database - always return results
        return self._match_ragas_flexible(pitches_hz, tonic_hz, top_n)
    
    def _get_swara_distribution(self, pitches_hz: np.ndarray) -> Counter:
        """Get distribution of all detected pitches as semitones from arbitrary reference."""
        pitch_classes = Counter()
        for pitch in pitches_hz:
            if pitch > 0:
                # Convert to pitch class (0-11)
                semitone = int(round(12 * np.log2(pitch / 440.0))) % 12
                pitch_classes[semitone] += 1
        return pitch_classes
    
    def _find_best_tonic(self, pitches_hz: np.ndarray, top_n: int) -> float:
        """
        Find the tonic that gives best raga matches.
        
        When multiple tonics give same score, prefer tonics in common 
        Carnatic range (C#4-D#4, ~277-311 Hz - most common shruti).
        """
        candidates = []
        
        # Try tonics across 3 octaves (C2 to B4)
        for midi_note in range(36, 72):
            candidate_tonic = 440.0 * (2 ** ((midi_note - 69) / 12))
            results = self._match_ragas_flexible(pitches_hz, candidate_tonic, 1)
            
            if results:
                score = results[0].confidence
                # Prefer tonics around C#4-D#4 (midi 61-63) - most common shruti
                if 60 <= midi_note <= 63:
                    range_bonus = 0.002
                elif 58 <= midi_note <= 65:
                    range_bonus = 0.001
                else:
                    range_bonus = 0
                candidates.append((score + range_bonus, candidate_tonic, results))
        
        if not candidates:
            return 261.63  # Default C4
        
        # Sort by score (with range bonus), take best
        candidates.sort(reverse=True)
        return candidates[0][1]
    
    def _match_ragas_flexible(
        self,
        pitches_hz: np.ndarray,
        tonic_hz: float,
        top_n: int
    ) -> List[DetectionResult]:
        """
        Match detected notes against ragas with flexible scoring.
        
        Uses note frequency distribution, not just presence/absence.
        Returns matches even if extra notes are detected (common with gamakas).
        """
        # Get swara counts at this tonic
        swara_counts = Counter()
        for pitch in pitches_hz:
            if pitch <= 0 or tonic_hz <= 0:
                continue
            semitones = 12 * np.log2(pitch / tonic_hz)
            semitones_mod = round(semitones) % 12
            if semitones_mod in SEMITONE_TO_SWARAS:
                swara_counts[SEMITONE_TO_SWARAS[semitones_mod][0]] += 1
        
        total = sum(swara_counts.values())
        if total == 0:
            return []
        
        # Convert to percentages
        swara_pcts = {s: c / total for s, c in swara_counts.items()}
        
        # Primary swaras (>= 4% of notes) - these define the scale
        primary_swaras = {s for s, pct in swara_pcts.items() if pct >= 0.04}
        
        # All detected swaras for reference
        all_swaras = set(swara_counts.keys())
        
        # Score each raga
        results = []
        for raga in self.db:
            score = self._compute_raga_score(swara_pcts, primary_swaras, raga)
            if score > 0.3:  # Include anything with > 30% match
                result = DetectionResult(
                    raga=raga,
                    confidence=score,
                    detected_swaras=all_swaras,
                    match_details={
                        'raga_scale': raga.scale,
                        'primary_detected': primary_swaras,
                        'matching_swaras': primary_swaras & raga.scale,
                        'foreign_swaras': primary_swaras - raga.scale,
                        'missing_from_raga': raga.scale - all_swaras,
                        'swara_distribution': dict(swara_pcts),
                    }
                )
                results.append(result)
        
        # Sort by confidence
        results.sort(key=lambda x: x.confidence, reverse=True)
        return results[:top_n]
    
    def _compute_raga_score(
        self,
        swara_pcts: dict,
        primary_swaras: Set[str],
        raga
    ) -> float:
        """
        Compute match score for a raga based on swara distribution.
        
        Scoring:
        - High weight for notes that match the raga and have high frequency
        - Penalty for primary notes that don't belong to the raga
        - Partial credit for having most of the raga's notes
        - Small bonus for melakarta (parent) ragas
        """
        from .raga_db import SWARA_TO_SEMITONE
        
        raga_semitones = {SWARA_TO_SEMITONE.get(s, -1) for s in raga.scale}
        raga_semitones.discard(-1)
        
        # Weight of notes that belong to this raga
        matching_weight = 0.0
        foreign_weight = 0.0
        
        for swara, pct in swara_pcts.items():
            semitone = SWARA_TO_SEMITONE.get(swara, -1)
            if semitone == -1:
                continue
            
            if semitone in raga_semitones:
                matching_weight += pct
            else:
                # Foreign note - only penalize if it's a primary note
                if swara in primary_swaras:
                    foreign_weight += pct
        
        # Coverage: what % of raga's notes did we detect?
        detected_semitones = {SWARA_TO_SEMITONE.get(s, -1) for s in swara_pcts.keys()}
        detected_semitones.discard(-1)
        coverage = len(detected_semitones & raga_semitones) / len(raga_semitones) if raga_semitones else 0
        
        # Final score: matching - penalty + coverage bonus
        score = matching_weight - (foreign_weight * 0.5) + (coverage * 0.2)
        
        # Small bonus for melakarta ragas (more likely to be correct)
        if raga.is_melakarta:
            score += 0.01
        
        # Small bonus for well-known ragas
        well_known = {'kalyani', 'mohanam', 'shankarabharanam', 'bhairavi', 
                      'todi', 'kambhoji', 'hamsadhwani', 'mayamalavagowla',
                      'kharaharapriya', 'harikambhoji', 'arabhi', 'saveri'}
        if raga.name.lower() in well_known:
            score += 0.005
        
        return min(1.0, max(0.0, score))

    def detect_from_swaras(
        self,
        swaras: List[str],
        top_n: int = 5
    ) -> List[DetectionResult]:
        """
        Detect raga from a sequence of swara names.
        Useful for testing or when you already have note data.
        """
        swara_set = set(s.upper() for s in swaras)
        return self._match_ragas(swara_set, top_n)
    
    def _extract_pitches(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Extract pitch contour from audio."""
        # Use piptrack for pitch detection
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=80, fmax=2000)
        
        # Get the pitch with highest magnitude at each frame
        pitch_values = []
        for t in range(pitches.shape[1]):
            mag_col = magnitudes[:, t]
            if mag_col.max() > 0:
                idx = mag_col.argmax()
                pitch = pitches[idx, t]
                if pitch > 0:
                    pitch_values.append(pitch)
        
        return np.array(pitch_values)
    
    def _estimate_tonic(self, pitches_hz: np.ndarray) -> float:
        """
        Estimate the tonic (Sa) frequency.
        
        In Carnatic music, Sa is typically:
        - The most stable/sustained pitch
        - Often the drone frequency
        - Usually in the lower range of the melodic content
        """
        if len(pitches_hz) == 0:
            return self.default_tonic_hz
        
        # Quantize pitches to semitones for histogram
        # Use A4 = 440 Hz as reference
        semitones = 12 * np.log2(pitches_hz / 440.0)
        
        # Round to nearest semitone
        semitones_rounded = np.round(semitones).astype(int)
        
        # Find most common pitch classes (mod 12)
        pitch_classes = semitones_rounded % 12
        counter = Counter(pitch_classes)
        
        # The most common pitch class is likely Sa or Pa
        most_common = counter.most_common(2)
        
        if len(most_common) >= 2:
            # If two pitches are 7 semitones apart, they're likely Sa and Pa
            pc1, count1 = most_common[0]
            pc2, count2 = most_common[1]
            
            if abs((pc1 - pc2) % 12) == 7 or abs((pc2 - pc1) % 12) == 7:
                # The lower one is probably Sa
                sa_pc = min(pc1, pc2)
            else:
                sa_pc = pc1
        else:
            sa_pc = most_common[0][0] if most_common else 0
        
        # Convert pitch class back to Hz (in a reasonable octave)
        # Assume Sa is around C4-D4 range (260-294 Hz)
        sa_semitone = sa_pc + 60  # Roughly C4 range
        tonic_hz = 440.0 * (2 ** ((sa_semitone - 69) / 12))
        
        return tonic_hz
    
    def _pitches_to_swaras(
        self, 
        pitches_hz: np.ndarray, 
        tonic_hz: float,
        min_occurrence_pct: float = 0.04
    ) -> Set[str]:
        """
        Convert pitch frequencies to swara names relative to tonic.
        
        Args:
            pitches_hz: Array of pitch frequencies
            tonic_hz: Tonic (Sa) frequency
            min_occurrence_pct: Minimum percentage of occurrences to include a swara.
                               Default 4% helps filter spurious detections from gamakas
                               (e.g., D1 artifacts from D2 gamakas in Kalyani).
        """
        swara_counts = Counter()
        
        for pitch in pitches_hz:
            # Calculate semitones from tonic
            if pitch <= 0 or tonic_hz <= 0:
                continue
                
            semitones = 12 * np.log2(pitch / tonic_hz)
            semitones_mod = round(semitones) % 12
            
            # Map semitone to possible swaras
            if semitones_mod in SEMITONE_TO_SWARAS:
                # Take the first swara name for this semitone
                swara_counts[SEMITONE_TO_SWARAS[semitones_mod][0]] += 1
        
        # Filter swaras by minimum occurrence threshold
        total = sum(swara_counts.values())
        if total == 0:
            return set()
            
        swaras = set()
        for swara, count in swara_counts.items():
            pct = count / total
            if pct >= min_occurrence_pct:
                swaras.add(swara)
        
        return swaras
    
    def _match_ragas(
        self, 
        detected_swaras: Set[str], 
        top_n: int
    ) -> List[DetectionResult]:
        """Match detected swaras against raga database."""
        
        # Get candidate ragas
        candidates = self.db.find_by_scale(detected_swaras)
        
        results = []
        for raga, score in candidates[:top_n]:
            result = DetectionResult(
                raga=raga,
                confidence=score,
                detected_swaras=detected_swaras,
                match_details={
                    'raga_scale': raga.scale,
                    'matching_swaras': detected_swaras & raga.scale,
                    'foreign_swaras': detected_swaras - raga.scale,
                }
            )
            results.append(result)
        
        return results


def detect_raga(audio_path: str, tonic_hz: Optional[float] = None) -> List[DetectionResult]:
    """
    Convenience function to detect raga from an audio file.
    
    Example:
        results = detect_raga("sample.wav")
        print(f"Top match: {results[0].raga.name} ({results[0].confidence:.0%})")
    """
    detector = RagaDetector()
    return detector.detect_from_file(audio_path, tonic_hz)
