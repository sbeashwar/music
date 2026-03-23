"""
Arohanam Detector - Detect ragas from simple ascending/descending scale renditions.

Given an audio recording of someone singing/playing a simple arohanam (ascending scale)
or avarohanam (descending scale), this module:
1. Detects the tonic (Sa) frequency
2. Tracks pitch over time to get the note sequence
3. Quantizes each pitch to the nearest swara
4. Extracts the ordered swara sequence
5. Matches against the raga database using SwaraSequenceMatcher

Designed for CLEAN scale renditions (not full compositions).
"""

import numpy as np
import librosa
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path


# Swara to semitone offset from Sa (0-11)
SWARA_TO_SEMITONE = {
    'S': 0,
    'R1': 1, 'R2': 2, 'R3': 3,
    'G1': 2, 'G2': 3, 'G3': 4,
    'M1': 5, 'M2': 6,
    'P': 7,
    'D1': 8, 'D2': 9, 'D3': 10,
    'N1': 9, 'N2': 10, 'N3': 11,
}

# Semitone to canonical swara name (one-to-one, no enharmonic ambiguity)
# For semitones that map to multiple swaras (e.g., 2 -> R2/G1),
# we use the most common Carnatic convention:
SEMITONE_TO_SWARA = {
    0: 'S',
    1: 'R1',
    2: 'R2',   # also G1 - resolved by context (adjacency)
    3: 'G2',   # also R3 - resolved by context
    4: 'G3',
    5: 'M1',
    6: 'M2',
    7: 'P',
    8: 'D1',
    9: 'D2',   # also N1
    10: 'N2',  # also D3
    11: 'N3',
}

# Context-aware disambiguation for enharmonic pairs
# Key: semitone, Value: dict mapping neighbor context to specific swara
# IMPORTANT: Rules must be conservative.  For semi-9 and semi-10 the
# correct swara depends on the raga (D2 vs N1, N2 vs D3) so we only
# apply rules when the context gives very strong evidence.  Otherwise
# use the statistically more common default.
ENHARMONIC_DISAMBIGUATION = {
    2: {  # R2 vs G1
        'default': 'R2',
        'after_1': 'R2', 'before_3': 'G1', 'before_4': 'R2',
    },
    3: {  # R3 vs G2
        'default': 'G2',
        'after_2': 'R3', 'after_1': 'R3', 'before_4': 'G2', 'before_5': 'G2',
    },
    9: {  # D2 vs N1
        'default': 'D2',
        'after_8': 'D2', 'before_11': 'D2',
        # NOTE: before_10 removed — semi 9→10 could be D2→N2 or N1→D3,
        # cannot disambiguate without knowing the raga.
    },
    10: {  # D3 vs N2
        'default': 'N2',
        'after_8': 'D3', 'before_11': 'N2',
        # NOTE: after_9 removed — semi 9→10 is ambiguous (see above).
    },
}


@dataclass
class DetectedNote:
    """A single detected note from audio."""
    start_time: float    # seconds
    end_time: float      # seconds
    duration: float      # seconds
    frequency: float     # Hz
    semitone: int         # semitone offset from Sa (0-11)
    swara: str           # swara name (e.g., 'R2')
    confidence: float    # 0.0 - 1.0


@dataclass
class ArohanamResult:
    """Result of arohanam detection from audio."""
    detected_swaras: List[str]            # Ordered unique swaras detected
    detected_notes: List[DetectedNote]     # All individual notes in order
    tonic_hz: float                        # Detected tonic (Sa) frequency
    direction: str                         # 'ascending', 'descending', or 'mixed'
    semitones: List[int]                   # Semitone offsets from Sa
    raw_sequence: List[str]                # Full note sequence (may have repeats)


class ArohanamDetector:
    """
    Detects the swara sequence from a simple scale rendition.
    
    Optimized for:
    - Clean vocal recordings of arohanam/avarohanam
    - Simple instrument renditions (veena, flute, keyboard)
    - Clean, single-note sequences (no ornaments, minimal gamaka)
    """
    
    def __init__(
        self,
        sample_rate: int = 22050,
        min_note_duration: float = 0.08,   # seconds - minimum to count as a note
        pitch_tolerance: float = 0.4,       # semitones - quantization tolerance
        tonic_hz: Optional[float] = None,   # If known, provide tonic frequency
        min_confidence: float = 0.5,        # Minimum pitch confidence
        voice_mode: bool = False,           # More tolerant settings for voice
    ):
        self.sample_rate = sample_rate
        self.pitch_tolerance = pitch_tolerance
        self.tonic_hz = tonic_hz
        self.voice_mode = voice_mode
        
        if voice_mode:
            # Voice has gamakas, slides, vibrato — need slightly longer
            # minimum notes and higher confidence, but not so aggressive
            # that real swaras get dropped (e.g., brief notes in avarohanam)
            self.min_note_duration = max(min_note_duration, 0.12)
            self.min_confidence = max(min_confidence, 0.55)
            self.pitch_tolerance = max(pitch_tolerance, 0.5)
        else:
            self.min_note_duration = min_note_duration
            self.min_confidence = min_confidence
    
    def detect_from_file(self, audio_path: str) -> ArohanamResult:
        """
        Detect arohanam/avarohanam from an audio file.
        
        Args:
            audio_path: Path to audio file (WAV, MP3, FLAC, etc.)
            
        Returns:
            ArohanamResult with detected swara sequence
        """
        # Try fast loading with soundfile first
        try:
            import soundfile as sf
            y, sr = sf.read(audio_path, dtype='float32')
            if sr != self.sample_rate:
                y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
            if y.ndim > 1:
                y = np.mean(y, axis=1)
        except Exception:
            y, _ = librosa.load(audio_path, sr=self.sample_rate)
        
        return self.detect_from_audio(y)
    
    def detect_from_audio(self, audio: np.ndarray) -> ArohanamResult:
        """
        Detect arohanam/avarohanam from audio samples.
        
        Args:
            audio: Audio samples (mono, float32, at self.sample_rate)
            
        Returns:
            ArohanamResult with detected swara sequence
        """
        # Step 1: Detect pitch track
        pitches, confidences = self._extract_pitch(audio)
        
        # Step 2: Detect or use provided tonic
        tonic_hz = self.tonic_hz or self._detect_tonic(pitches, confidences)
        
        # Step 3: Convert pitches to semitone offsets from tonic
        semitone_track = self._pitches_to_semitones(pitches, tonic_hz)
        
        # Step 4: Segment into stable notes
        notes = self._segment_notes(semitone_track, confidences, tonic_hz)
        
        # Step 4b: Filter short transitional notes (voice mode)
        if self.voice_mode and len(notes) > 2:
            notes = self._filter_transitional_notes(notes)
            notes = self._consolidate_rare_neighbors(notes)
        
        # Step 4c: Reconcile borderline semitones — if a pitch sits close to
        # the boundary between two semitones and the same semitone appears
        # more clearly elsewhere, merge the borderline note.
        # Only applied in voice mode where gamaka slides cause this issue.
        if self.voice_mode and len(notes) > 2:
            notes = self._reconcile_borderline_notes(notes)
        
        # Step 5: Disambiguate enharmonic equivalents
        notes = self._disambiguate_enharmonics(notes)
        
        # Step 6: Determine direction and extract unique sequence
        direction = self._detect_direction(notes)
        
        # Step 7: Extract ordered unique swaras
        raw_sequence = [n.swara for n in notes]
        detected_swaras = self._extract_unique_ordered(raw_sequence)
        semitones = [SWARA_TO_SEMITONE[s] for s in detected_swaras]
        
        return ArohanamResult(
            detected_swaras=detected_swaras,
            detected_notes=notes,
            tonic_hz=tonic_hz,
            direction=direction,
            semitones=semitones,
            raw_sequence=raw_sequence,
        )
    
    def _extract_pitch(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract pitch track from audio using pYIN.
        
        Returns:
            Tuple of (pitches, confidences) arrays
        """
        # pYIN is excellent for monophonic pitch tracking
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=80,     # Hz - covers most vocal ranges
            fmax=800,    # Hz - upper limit for voice
            sr=self.sample_rate,
            frame_length=2048,
            hop_length=512,
        )
        
        # Replace NaN with 0
        f0 = np.nan_to_num(f0, nan=0.0)
        confidences = np.nan_to_num(voiced_probs, nan=0.0)
        
        return f0, confidences
    
    def _detect_tonic(self, pitches: np.ndarray, confidences: np.ndarray) -> float:
        """
        Detect the tonic (Sa) frequency.
        
        Strategy: The tonic is usually:
        1. The first/last note in a scale rendition
        2. The most frequently occurring pitch class in the lower range
        3. Usually the lowest sustained pitch
        """
        # Filter to confident pitches
        valid = (pitches > 0) & (confidences > self.min_confidence)
        valid_pitches = pitches[valid]
        
        if len(valid_pitches) == 0:
            return 261.63  # Default to middle C
        
        # Method 1: First stable pitch (often Sa in arohanam)
        first_pitches = valid_pitches[:min(20, len(valid_pitches))]
        median_first = np.median(first_pitches)
        
        # Method 2: Use histogram to find prominent pitch classes
        # Convert to MIDI-like continuous pitch
        midi_pitches = 12 * np.log2(valid_pitches / 440.0) + 69
        
        # Find pitch classes (mod 12)
        pitch_classes = midi_pitches % 12
        
        # Histogram of pitch classes
        hist, bin_edges = np.histogram(pitch_classes, bins=24, range=(0, 12))
        
        # The tonic is likely the pitch class of the first note
        first_pc = pitch_classes[0]
        
        # Method 3: Check last few pitches too (usually returns to Sa)
        last_pitches = valid_pitches[max(0, len(valid_pitches)-20):]
        median_last = np.median(last_pitches)
        
        # If first and last stable pitches are within a semitone, strong Sa candidate
        ratio = median_first / median_last
        if 0.95 < ratio < 1.05:
            # Very likely the tonic
            return float((median_first + median_last) / 2)
        elif ratio > 1.9 and ratio < 2.1:
            # Last pitch is an octave below first - tonic is lower
            return float(median_last)
        elif 0.48 < ratio < 0.52:
            # First pitch was lower Sa, last was upper Sa
            return float(median_first)
        
        # Default: use the lower of first/last
        return float(min(median_first, median_last))
    
    def _pitches_to_semitones(
        self, pitches: np.ndarray, tonic_hz: float
    ) -> np.ndarray:
        """Convert pitch track to semitone offsets from tonic."""
        semitones = np.full_like(pitches, fill_value=-1.0)
        valid = pitches > 0
        
        # Semitone offset: 12 * log2(f / tonic)
        semitones[valid] = 12.0 * np.log2(pitches[valid] / tonic_hz)
        
        return semitones
    
    def _filter_transitional_notes(
        self, notes: List[DetectedNote]
    ) -> List[DetectedNote]:
        """
        Remove short transitional notes that appear between two longer notes.
        
        Human voice produces brief pitch excursions during slides/gamakas
        that are not actual swaras. If a note is much shorter than its 
        neighbors AND its semitone is between them, it's likely a transition.
        """
        if len(notes) <= 2:
            return notes
        
        # Calculate median duration of all notes
        durations = [n.duration for n in notes]
        median_dur = float(np.median(durations))
        
        # A note is "short" if it's less than 40% of median duration
        short_threshold = median_dur * 0.4
        
        filtered = [notes[0]]  # Always keep first
        
        for i in range(1, len(notes) - 1):
            note = notes[i]
            prev = notes[i - 1]
            next_n = notes[i + 1]
            
            if note.duration < short_threshold:
                # Short note — check if it's transitional
                prev_semi = prev.semitone
                next_semi = next_n.semitone
                note_semi = note.semitone
                
                # It's transitional if it's between prev and next in pitch
                is_between = (min(prev_semi, next_semi) <= note_semi <= max(prev_semi, next_semi))
                # Or if it's adjacent (1 semitone) to both neighbors 
                is_adjacent = (abs(note_semi - prev_semi) <= 1 or abs(note_semi - next_semi) <= 1)
                
                if is_between or (is_adjacent and note.duration < short_threshold * 0.7):
                    continue  # Skip this transitional note
            
            filtered.append(note)
        
        filtered.append(notes[-1])  # Always keep last
        return filtered
    
    def _consolidate_rare_neighbors(
        self, notes: List[DetectedNote]
    ) -> List[DetectedNote]:
        """
        Remove rare notes that are ±1 semitone from a much more frequent note.
        
        Voice slides sometimes overshoot the target note, creating brief
        appearances of adjacent swaras (e.g., sliding from M2 to R2 briefly
        hits R1). If a semitone appears far less often than its neighbor,
        it's almost certainly a slide artifact, not an intended swara.
        """
        from collections import Counter
        
        if len(notes) <= 2:
            return notes
        
        # Count occurrences of each semitone (mod 12)
        semi_counts = Counter(n.semitone % 12 for n in notes)
        
        # Identify semitones to remove: those with count < 25% of an adjacent semitone
        remove_semitones = set()
        for semi, count in semi_counts.items():
            for neighbor in [(semi - 1) % 12, (semi + 1) % 12]:
                if neighbor in semi_counts:
                    neighbor_count = semi_counts[neighbor]
                    if count <= 1 and neighbor_count >= 3:
                        # This semitone is rare and has a very common neighbor
                        remove_semitones.add(semi)
                        break
                    elif neighbor_count > 0 and count / neighbor_count < 0.25:
                        remove_semitones.add(semi)
                        break
        
        if not remove_semitones:
            return notes
        
        return [n for n in notes if (n.semitone % 12) not in remove_semitones]
    
    def _reconcile_borderline_notes(
        self, notes: List[DetectedNote]
    ) -> List[DetectedNote]:
        """
        Fix borderline semitone assignments (voice mode only).
        
        Very conservative: only merge semitone X into X±1 when EVERY note
        assigned to X has a raw pitch that is closer to X±1 than to X.
        This catches genuine misquantisation from gamaka approach slides
        without merging legitimately different swaras.
        """
        if len(notes) < 3:
            return notes
        
        from collections import defaultdict
        
        # Find tonic frequency from S notes
        s_notes = [n for n in notes if n.semitone == 0]
        tonic_f = s_notes[0].frequency if s_notes else notes[0].frequency
        if tonic_f <= 0:
            return notes
        
        # Group notes by semitone with their raw continuous-semitone value
        semi_raw: Dict[int, List[float]] = defaultdict(list)
        for n in notes:
            if n.frequency > 0:
                raw = (12.0 * np.log2(n.frequency / tonic_f)) % 12
                semi_raw[n.semitone].append(raw)
        
        merge_map: Dict[int, int] = {}
        present = sorted(semi_raw.keys())
        
        for i in range(len(present) - 1):
            s_lo = present[i]
            s_hi = present[i + 1]
            if s_hi - s_lo != 1:
                continue
            
            boundary = s_lo + 0.5
            lo_vals = semi_raw[s_lo]
            hi_vals = semi_raw[s_hi]
            
            # Merge lo→hi only if ALL lo notes are past the boundary
            if lo_vals and all(v > boundary for v in lo_vals):
                merge_map[s_lo] = s_hi
            # Merge hi→lo only if ALL hi notes are below the boundary
            elif hi_vals and all(v < boundary for v in hi_vals):
                merge_map[s_hi] = s_lo
        
        if not merge_map:
            return notes
        
        for n in notes:
            if n.semitone in merge_map:
                new_semi = merge_map[n.semitone]
                n.semitone = new_semi
                n.swara = SEMITONE_TO_SWARA.get(new_semi, n.swara)
        
        return notes
    
    def _segment_notes(
        self, 
        semitone_track: np.ndarray, 
        confidences: np.ndarray,
        tonic_hz: float,
    ) -> List[DetectedNote]:
        """
        Segment the continuous pitch track into discrete notes.
        Groups consecutive frames with similar pitch into note segments.
        """
        hop_length = 512
        frame_duration = hop_length / self.sample_rate
        min_frames = max(1, int(self.min_note_duration / frame_duration))
        
        notes = []
        current_semi = None
        current_start = 0
        current_frames = 0
        current_pitches = []
        current_confs = []
        
        for i in range(len(semitone_track)):
            semi = semitone_track[i]
            conf = confidences[i]
            
            if semi < -0.5 or conf < self.min_confidence:
                # Silence or unvoiced
                if current_frames >= min_frames and current_semi is not None:
                    notes.append(self._create_note(
                        current_start, i, current_pitches, current_confs,
                        tonic_hz, frame_duration
                    ))
                current_semi = None
                current_frames = 0
                current_pitches = []
                current_confs = []
                continue
            
            # Round to nearest semitone
            rounded = round(semi) % 12
            
            if current_semi is None:
                # Start new note
                current_semi = rounded
                current_start = i
                current_frames = 1
                current_pitches = [semi]
                current_confs = [conf]
            elif abs(rounded - current_semi) <= 0 or abs(semi - current_semi) < self.pitch_tolerance:
                # Continue current note
                current_frames += 1
                current_pitches.append(semi)
                current_confs.append(conf)
            else:
                # New note
                if current_frames >= min_frames:
                    notes.append(self._create_note(
                        current_start, i, current_pitches, current_confs,
                        tonic_hz, frame_duration
                    ))
                
                current_semi = rounded
                current_start = i
                current_frames = 1
                current_pitches = [semi]
                current_confs = [conf]
        
        # Don't forget the last note
        if current_frames >= min_frames and current_semi is not None:
            notes.append(self._create_note(
                current_start, len(semitone_track), current_pitches, 
                current_confs, tonic_hz, frame_duration
            ))
        
        return notes
    
    def _create_note(
        self, start_frame: int, end_frame: int, 
        pitches: List[float], confs: List[float],
        tonic_hz: float, frame_duration: float,
    ) -> DetectedNote:
        """Create a DetectedNote from frame data."""
        # Trim leading/trailing 20% of frames to reduce gamaka/slide
        # artifacts that skew the semitone estimate.  The stable centre
        # of a held swara gives the best pitch reading.
        n = len(pitches)
        if n >= 5:
            trim = max(1, int(n * 0.20))
            stable_pitches = pitches[trim:-trim] if trim < n // 2 else pitches
        else:
            stable_pitches = pitches
        median_semi = np.median(stable_pitches)
        
        # Map continuous semitone to nearest discrete semitone (0-11).
        rounded = int(round(median_semi))
        
        # Special handling for the N3/upper-Sa boundary.
        # round(11.56) = 12, but 11.56 is still below the octave and
        # likely represents N3 in the singer's voice.  If the raw pitch
        # is more than 0.25 semitones (25 cents) below an octave
        # boundary, classify as N3 (semi 11) rather than upper Sa.
        if rounded != 0 and rounded % 12 == 0:
            octave = rounded  # 12, 24, ...
            if median_semi < octave - 0.25:
                semitone = 11
            else:
                semitone = 0
        elif median_semi < -0.5:
            semitone = (12 + rounded % 12) % 12
        else:
            semitone = rounded % 12
        
        swara = SEMITONE_TO_SWARA.get(semitone, f'?{semitone}')
        
        # Calculate actual frequency
        freq = tonic_hz * (2.0 ** (median_semi / 12.0))
        
        return DetectedNote(
            start_time=start_frame * frame_duration,
            end_time=end_frame * frame_duration,
            duration=(end_frame - start_frame) * frame_duration,
            frequency=float(freq),
            semitone=semitone,
            swara=swara,
            confidence=float(np.mean(confs)),
        )
    
    def _disambiguate_enharmonics(self, notes: List[DetectedNote]) -> List[DetectedNote]:
        """
        Resolve enharmonic ambiguities (R2/G1, R3/G2, D2/N1, D3/N2)
        using neighboring note context.
        """
        for i, note in enumerate(notes):
            semi = note.semitone
            if semi not in ENHARMONIC_DISAMBIGUATION:
                continue
            
            rules = ENHARMONIC_DISAMBIGUATION[semi]
            
            # Check previous note
            if i > 0:
                prev_semi = notes[i - 1].semitone
                key = f'after_{prev_semi}'
                if key in rules:
                    note.swara = rules[key]
                    continue
            
            # Check next note
            if i < len(notes) - 1:
                next_semi = notes[i + 1].semitone
                key = f'before_{next_semi}'
                if key in rules:
                    note.swara = rules[key]
                    continue
            
            # Default
            note.swara = rules['default']
        
        return notes
    
    def _detect_direction(self, notes: List[DetectedNote]) -> str:
        """Detect if the sequence is ascending, descending, or mixed."""
        if len(notes) < 2:
            return 'mixed'
        
        semitones = [n.semitone for n in notes]
        
        # Count ascending vs descending transitions
        ascending = 0
        descending = 0
        
        for i in range(1, len(semitones)):
            diff = semitones[i] - semitones[i - 1]
            if diff > 0:
                ascending += 1
            elif diff < 0:
                descending += 1
        
        total = ascending + descending
        if total == 0:
            return 'mixed'
        
        asc_ratio = ascending / total
        
        if asc_ratio > 0.7:
            return 'ascending'
        elif asc_ratio < 0.3:
            return 'descending'
        else:
            return 'mixed'
    
    def _extract_unique_ordered(self, raw_sequence: List[str]) -> List[str]:
        """
        Extract unique swaras in the order they first appear,
        removing consecutive duplicates.
        """
        if not raw_sequence:
            return []
        
        # First remove consecutive duplicates
        deduped = [raw_sequence[0]]
        for s in raw_sequence[1:]:
            if s != deduped[-1]:
                deduped.append(s)
        
        # Then extract unique swaras preserving first-appearance order
        seen = set()
        unique = []
        for s in deduped:
            if s not in seen:
                seen.add(s)
                unique.append(s)
        
        return unique


def detect_arohanam(audio_path: str, tonic_hz: Optional[float] = None) -> ArohanamResult:
    """
    Convenience function to detect arohanam from an audio file.
    
    Args:
        audio_path: Path to audio file
        tonic_hz: Known tonic frequency (if None, auto-detected)
        
    Returns:
        ArohanamResult with detected swara sequence
    """
    detector = ArohanamDetector(tonic_hz=tonic_hz)
    return detector.detect_from_file(audio_path)


if __name__ == '__main__':
    import argparse
    from raga_detection.swara_matcher import SwaraSequenceMatcher, format_match_result
    
    parser = argparse.ArgumentParser(
        description='Detect raga from arohanam/avarohanam recording'
    )
    parser.add_argument('audio_path', help='Path to audio file')
    parser.add_argument('--tonic', type=float, default=None,
                       help='Known tonic frequency in Hz')
    parser.add_argument('--top', type=int, default=10,
                       help='Number of top matches to show')
    
    args = parser.parse_args()
    
    print(f"Analyzing: {args.audio_path}")
    print()
    
    # Detect notes
    result = detect_arohanam(args.audio_path, tonic_hz=args.tonic)
    
    print(f"Tonic (Sa): {result.tonic_hz:.1f} Hz")
    print(f"Direction: {result.direction}")
    print(f"Raw sequence: {' -> '.join(result.raw_sequence)}")
    print(f"Detected swaras: {' '.join(result.detected_swaras)}")
    print(f"Semitones: {result.semitones}")
    print()
    
    # Show individual notes
    print("Detected notes:")
    for note in result.detected_notes:
        print(f"  {note.start_time:.2f}s-{note.end_time:.2f}s: "
              f"{note.swara:3s} ({note.frequency:.1f} Hz, "
              f"semi={note.semitone}, conf={note.confidence:.2f})")
    print()
    
    # Match against database
    direction = result.direction if result.direction != 'mixed' else 'ascending'
    
    matcher = SwaraSequenceMatcher()
    matches = matcher.match_swaras(
        result.detected_swaras,
        direction=direction,
        max_results=args.top,
    )
    
    if matches:
        print(f"Top {min(args.top, len(matches))} raga matches:")
        for i, m in enumerate(matches, 1):
            print(format_match_result(m, rank=i))
            print()
    else:
        print("No matching ragas found.")
        
        # Try semitone-based match as fallback
        print("Trying semitone-based match...")
        matches = matcher.match_by_semitones(
            result.semitones, max_results=args.top
        )
        if matches:
            for i, m in enumerate(matches, 1):
                print(format_match_result(m, rank=i))
                print()
