"""
Raga Detector v2 - Musical approach

Detection strategy (how a musician identifies raga):
1. Find Sa and Pa - the two fixed reference points (7 semitones apart)
2. Establish the octave range (lower S, P, higher S)
3. Map all other notes relative to Sa to identify R, G, M, D, N variants
4. Analyze note sequences to distinguish arohanam vs avarohanam patterns
5. Match against raga database

Key musical facts:
- Sa (S) and Pa (P) are fixed - never have variants
- Ma has 2 variants: M1 (shuddha) and M2 (prati)
- R, G, D, N each have 3 variants (1=shuddha/komal, 2=madhya, 3=tivra/sharp)
- Arohanam rarely uses 2 variants of same swara
- Avarohanam can use different variant than arohanam (vakra ragas)
"""

import numpy as np
from typing import List, Tuple, Optional, Set, Dict
from dataclasses import dataclass
from collections import Counter
import math

from .raga_db import RagaDB, Raga, SWARA_TO_SEMITONE, SEMITONE_TO_SWARAS, get_db

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False


@dataclass
class DetectionResult:
    """Result of raga detection."""
    raga: Raga
    confidence: float
    detected_swaras: Set[str]
    tonic_hz: float
    match_details: dict


class RagaDetectorV2:
    """
    Detect raga using musical approach - find Sa/Pa first, then map notes.
    """
    
    # Semitone intervals from Sa
    INTERVALS = {
        0: 'S',    # Sa - fixed
        1: 'R1',   # Shuddha Rishabha
        2: 'R2',   # Chatushruti Rishabha / Shuddha Gandhara
        3: 'R3',   # Shatshruti Rishabha / Sadharana Gandhara  
        4: 'G3',   # Antara Gandhara
        5: 'M1',   # Shuddha Madhyama
        6: 'M2',   # Prati Madhyama
        7: 'P',    # Panchama - fixed
        8: 'D1',   # Shuddha Dhaivata
        9: 'D2',   # Chatushruti Dhaivata / Shuddha Nishada
        10: 'D3',  # Shatshruti Dhaivata / Kaisiki Nishada
        11: 'N3',  # Kakali Nishada
    }
    
    # Note: R2=G1, R3=G2, D2=N1, D3=N2 (enharmonic equivalents)
    # We use R2, R3, D2, D3 as primary names
    
    def __init__(self, db: Optional[RagaDB] = None):
        self.db = db or get_db()
    
    def detect_from_file(
        self, 
        audio_path: str,
        top_n: int = 15
    ) -> List[DetectionResult]:
        """Detect raga from audio file."""
        if not HAS_LIBROSA:
            raise ImportError("librosa is required")
        
        y, sr = librosa.load(audio_path, sr=22050, duration=60)
        return self.detect_from_audio(y, sr, top_n)
    
    def detect_from_audio(
        self,
        y: np.ndarray,
        sr: int = 22050,
        top_n: int = 15,
        tonic_hz: Optional[float] = None
    ) -> List[DetectionResult]:
        """
        Detect raga from audio samples.
        
        Args:
            y: Audio samples
            sr: Sample rate
            top_n: Number of top matches to return
            tonic_hz: Optional manual tonic (shruti) in Hz. If None, auto-detect.
        """
        
        # Step 1: Extract pitch contour
        pitches_hz = self._extract_pitches(y, sr)
        if len(pitches_hz) < 10:
            return []
        
        # Step 2: Find Sa by detecting Sa-Pa relationship (or use manual tonic)
        manual_tonic = tonic_hz is not None
        if not manual_tonic:
            tonic_hz = self._find_tonic_by_sa_pa(pitches_hz)
        
        # Step 3: Map all pitches to swaras relative to Sa
        swara_counts, swara_sequences = self._analyze_notes(pitches_hz, tonic_hz)
        
        # Step 4: Identify which R, G, M, D, N variants are used
        # Use outlier detection to separate primary notes from artifacts
        primary_swaras, all_swaras = self._identify_swaras(swara_counts)
        
        # Step 5: Match against raga database
        if manual_tonic:
            # When user explicitly sets the tonic, trust it - don't search alternatives
            best_results = self._match_ragas(
                primary_swaras, all_swaras, swara_counts, tonic_hz, top_n,
                swara_sequences=swara_sequences
            )
        else:
            # Auto-detected tonic: try multiple tonics to find best match
            best_results = self._find_best_tonic_and_match(
                pitches_hz, swara_counts, primary_swaras, all_swaras, tonic_hz, top_n
            )
        
        return best_results
    
    def _find_best_tonic_and_match(
        self,
        pitches_hz: np.ndarray,
        initial_counts: Counter,
        initial_primary: Set[str],
        initial_all: Set[str],
        initial_tonic: float,
        top_n: int
    ) -> List[DetectionResult]:
        """
        Try multiple tonics around the initial estimate and pick best match.
        
        Uses tonic quality (how well pitches snap to semitone grid) as a
        major tiebreaker, along with confidence, note count, melakarta,
        and arohanam/avarohanam pattern bonuses.
        """
        candidates = []
        
        # Try tonics in semitone steps around initial estimate
        # First pass: fast scoring (no pattern analysis) to find best tonic
        for semitone_offset in range(-6, 7):
            tonic = initial_tonic * (2 ** (semitone_offset / 12))
            
            swara_counts, swara_sequences = self._analyze_notes(pitches_hz, tonic)
            primary_swaras, all_swaras = self._identify_swaras(swara_counts)
            
            # Skip if we get too few or too many primary notes
            if len(primary_swaras) < 4 or len(primary_swaras) > 8:
                continue
            
            # Fast match: no pattern analysis (skip swara_sequences)
            results = self._match_ragas(
                primary_swaras, all_swaras, swara_counts, tonic, top_n
            )
            
            if results:
                score = results[0].confidence
                raga = results[0].raga
                
                # Bonus for ragas with more notes (7-note > 6-note > 5-note)
                note_bonus = len(raga.scale) * 0.005
                
                # Bonus for melakarta ragas (parent ragas are more fundamental)
                mela_bonus = 0.015 if raga.is_melakarta else 0
                
                # Proximity bonus: trust the auto-detected tonic
                # When confidence scores tie (which happens often since every
                # semitone offset can perfectly match some melakarta), prefer
                # the tonic that the Sa-Pa detection identified.
                # Uses quadratic falloff so offset 0 has a clear advantage.
                proximity_bonus = (6 - abs(semitone_offset)) ** 2 * 0.0005
                
                total_score = score + note_bonus + mela_bonus + proximity_bonus
                
                candidates.append((total_score, semitone_offset, tonic, primary_swaras, all_swaras))
        
        if not candidates:
            # Fallback to initial tonic
            return self._match_ragas(
                initial_primary, initial_all, initial_counts, initial_tonic, top_n
            )
        
        # Pick the best tonic, then do a final pass WITH pattern analysis
        candidates.sort(reverse=True)
        _, _, best_tonic, _, _ = candidates[0]
        
        # Final match with pattern analysis for accurate raga ranking
        final_counts, final_seqs = self._analyze_notes(pitches_hz, best_tonic)
        final_primary, final_all = self._identify_swaras(final_counts)
        return self._match_ragas(
            final_primary, final_all, final_counts, best_tonic, top_n,
            swara_sequences=final_seqs
        )
    
    def _extract_pitches(self, y: np.ndarray, sr: int) -> np.ndarray:
        """
        Extract pitch contour from audio.
        
        Filters out octave errors by keeping only pitches in typical vocal range.
        """
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=80, fmax=1000)
        
        pitch_values = []
        for t in range(pitches.shape[1]):
            mag_col = magnitudes[:, t]
            if mag_col.max() > 0:
                idx = mag_col.argmax()
                pitch = pitches[idx, t]
                # Filter to extended singing range (80-600 Hz)
                # 80 Hz covers lower male voice and lower octave Sa
                # which is crucial for correct tonic detection
                if 80 < pitch < 600:
                    pitch_values.append(pitch)
        
        return np.array(pitch_values)
    
    def _find_tonic_by_sa_pa(self, pitches_hz: np.ndarray) -> float:
        """
        Find the tonic (Sa) by looking for Sa-Pa relationship.
        
        Sa and Pa are always 7 semitones (perfect 5th) apart.
        We look for pairs of prominent pitches with this relationship,
        then fine-tune the exact frequency.
        """
        # Quantize pitches to semitones
        semitones = 12 * np.log2(pitches_hz / 440.0)
        semitones_rounded = np.round(semitones).astype(int)
        
        # Count occurrences of each pitch class
        pitch_class_counts = Counter(semitones_rounded % 12)
        
        # Find the most common pitch classes
        common_pcs = pitch_class_counts.most_common(7)
        
        # Look for Sa-Pa pairs (7 semitones apart)
        best_sa_pc = None
        best_score = 0
        
        for pc1, count1 in common_pcs:
            pa_pc = (pc1 + 7) % 12  # Pa is 7 semitones above Sa
            count2 = pitch_class_counts.get(pa_pc, 0)
            
            # Score: weight Sa prominence more heavily than Pa
            # In Carnatic music, Sa is the most frequently visited note (anchor)
            # Pa confirms the tonic but shouldn't dominate the choice
            score = count1 * 1.5 + count2
            
            # Bonus if Pa is also in top pitches (confirms the Sa-Pa axis)
            if any(pc == pa_pc for pc, _ in common_pcs[:5]):
                score *= 1.3
            
            if score > best_score:
                best_score = score
                best_sa_pc = pc1
        
        if best_sa_pc is None:
            # Fallback: most common pitch class is Sa
            best_sa_pc = common_pcs[0][0] if common_pcs else 0
        
        # Convert pitch class to Hz
        # Collect ALL pitches matching the winning pitch class (no Hz range filter)
        sa_candidates = []
        for pitch in pitches_hz:
            semitone = round(12 * np.log2(pitch / 440.0))
            if semitone % 12 == best_sa_pc:
                sa_candidates.append(pitch)
        
        if sa_candidates:
            # Use median of all matching pitches
            raw_tonic = np.median(sa_candidates)
            
            # Octave-reduce to canonical tonic range (100-400 Hz)
            # This covers voice (C3~130 Hz) through higher instruments
            while raw_tonic > 400:
                raw_tonic /= 2.0
            while raw_tonic < 100:
                raw_tonic *= 2.0
            initial_tonic = raw_tonic
            
            # Fine-tune: find the tonic that best aligns pitches to semitone boundaries
            # This handles cases where the actual Sa is between standard notes
            best_tonic = initial_tonic
            best_alignment = 0
            
            # Search in 1 Hz increments around the initial estimate
            for offset in range(-5, 6):
                test_tonic = initial_tonic + offset
                test_semitones = 12 * np.log2(pitches_hz / test_tonic) % 12
                
                # Count how many pitches are close to integer semitones
                # (within 0.3 semitones of a note center)
                aligned = 0
                for st in test_semitones:
                    dist_to_note = min(st % 1, 1 - (st % 1))
                    if dist_to_note < 0.3:
                        aligned += 1
                
                if aligned > best_alignment:
                    best_alignment = aligned
                    best_tonic = test_tonic
            
            return best_tonic
        else:
            # Fallback: compute correct Hz from pitch class
            # pc=0 is A (440 Hz), so pc=N means N semitones above A
            # Octave-reduce to ~130-350 Hz range
            tonic = 440.0 * (2 ** (best_sa_pc / 12))
            while tonic > 350:
                tonic /= 2.0
            while tonic < 130:
                tonic *= 2.0
            return tonic
    
    def _analyze_notes(
        self, 
        pitches_hz: np.ndarray, 
        tonic_hz: float,
        min_stable_frames: int = 2
    ) -> Tuple[Counter, List[int]]:
        """
        Analyze all pitches relative to the tonic.
        
        Uses stability filtering: only counts notes that are held for
        at least min_stable_frames consecutive frames. This filters out
        transitional pitches (gamakas, slides between notes).
        
        Returns:
            swara_counts: Counter of semitone intervals from Sa
            swara_sequences: List of semitone intervals in order
        """
        if len(pitches_hz) == 0:
            return Counter(), []
        
        # First pass: convert all pitches to semitone intervals
        all_intervals = []
        for pitch in pitches_hz:
            if pitch <= 0:
                all_intervals.append(-1)  # Invalid
                continue
            semitones = 12 * np.log2(pitch / tonic_hz)
            interval = round(semitones) % 12
            all_intervals.append(interval)
        
        # Second pass: stability filtering
        # Only count notes that are held for min_stable_frames or more
        swara_counts = Counter()
        swara_sequences = []
        
        i = 0
        while i < len(all_intervals):
            if all_intervals[i] < 0:
                i += 1
                continue
            
            current_note = all_intervals[i]
            run_length = 1
            
            # Count consecutive frames with same note (within tolerance)
            j = i + 1
            while j < len(all_intervals):
                if all_intervals[j] < 0:
                    break
                # Allow for small wobble (same note or adjacent within gamaka)
                if all_intervals[j] == current_note:
                    run_length += 1
                    j += 1
                else:
                    break
            
            # Only count if held for minimum duration (stable)
            if run_length >= min_stable_frames:
                swara_counts[current_note] += run_length
                swara_sequences.extend([current_note] * run_length)
            
            i = j
        
        return swara_counts, swara_sequences
    
    def _identify_swaras(self, swara_counts: Counter) -> Tuple[Set[str], Set[str]]:
        """
        Identify which swaras are used based on semitone counts.
        
        Uses multiple strategies to separate primary notes from artifacts:
        1. Statistical outlier detection (IQR method)
        2. Gap detection - real ragas don't use all 12 notes
        3. Relative prominence - compare to strongest notes
        
        Returns:
            primary_swaras: Main notes of the raga (high occurrence)
            all_swaras: All detected notes including outliers
        """
        total = sum(swara_counts.values())
        if total == 0:
            return set(), set()
        
        # Convert to percentages with interval info
        pcts = [(interval, self.INTERVALS.get(interval, f'?{interval}'), count / total) 
                for interval, count in swara_counts.items()]
        
        # Sort by percentage descending
        pcts.sort(key=lambda x: -x[2])
        
        all_swaras = {swara for _, swara, pct in pcts if pct > 0.01}  # >1% to exist
        
        # Strategy 1: Basic threshold - top notes must have minimum presence
        values = [pct for _, _, pct in pcts if pct > 0.01]
        if len(values) < 3:
            return all_swaras, all_swaras
        
        # Strategy 2: IQR-based outlier detection
        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        lower_fence = q1 - (0.8 * iqr)  # More aggressive
        
        # Strategy 3: Relative to max - must be at least 40% of strongest note
        max_pct = max(values)
        relative_threshold = max_pct * 0.40
        
        # Strategy 4: Gap detection
        # Look for notes that are significantly weaker than their neighbors
        intervals_by_pct = {interval: pct for interval, _, pct in pcts}
        gap_filtered = set()
        
        for interval, swara, pct in pcts:
            if pct < 0.02:  # Skip very weak notes
                continue
                
            # Check if this note is in a "valley" (weaker than both neighbors)
            prev_int = (interval - 1) % 12
            next_int = (interval + 1) % 12
            prev_pct = intervals_by_pct.get(prev_int, 0)
            next_pct = intervals_by_pct.get(next_int, 0)
            
            # If both neighbors are stronger, this might be a transition note
            if prev_pct > pct * 1.3 and next_pct > pct * 1.3:
                continue  # Skip - likely transition between neighbors
            
            gap_filtered.add(swara)
        
        # Combine thresholds: must pass at least 2 of 3 criteria
        threshold = max(lower_fence, 0.04)  # At least 4%
        
        primary_swaras = set()
        for interval, swara, pct in pcts:
            passes = 0
            if pct >= threshold:
                passes += 1
            if pct >= relative_threshold:
                passes += 1
            if swara in gap_filtered:
                passes += 1
            
            if passes >= 2:
                primary_swaras.add(swara)
        
        # Always include S and P if they have significant presence
        for interval, swara, pct in pcts:
            if swara in ['S', 'P'] and pct > 0.03:
                primary_swaras.add(swara)
        
        # Limit to reasonable raga size (5-9 notes typically)
        if len(primary_swaras) > 9:
            # Keep only the strongest
            sorted_by_pct = [(swara, pct) for _, swara, pct in pcts if swara in primary_swaras]
            sorted_by_pct.sort(key=lambda x: -x[1])
            primary_swaras = {s for s, _ in sorted_by_pct[:9]}
            # Always keep S and P
            for interval, swara, pct in pcts:
                if swara in ['S', 'P'] and pct > 0.03:
                    primary_swaras.add(swara)
        
        return primary_swaras, all_swaras
    
    def _tonic_quality(self, pitches_hz: np.ndarray, tonic: float) -> float:
        """
        Measure how well pitches align to the semitone grid with this tonic.
        
        A correct tonic makes most pitches snap cleanly to integer semitone
        boundaries. A wrong tonic leaves many pitches at fractional semitones.
        Returns fraction of pitches within 0.3 semitones of a note center.
        """
        if len(pitches_hz) == 0:
            return 0.0
        semitones = 12 * np.log2(pitches_hz / tonic) % 12
        aligned = sum(1 for st in semitones if min(st % 1, 1 - (st % 1)) < 0.3)
        return aligned / len(pitches_hz)
    
    def _analyze_melodic_direction(
        self, swara_sequences: List[int]
    ) -> Tuple[Set[int], Set[int]]:
        """
        Identify which semitone intervals appear in ascending vs descending passages.
        
        This is critical for distinguishing ragas that share the same scale
        but differ in arohanam/avarohanam (e.g. Bilahari skips M1 in ascent,
        but Shankarabharanam uses it in both directions).
        
        Returns:
            ascending_semitones: Set of semitone intervals seen in ascending motion
            descending_semitones: Set of semitone intervals seen in descending motion
        """
        ascending = set()
        descending = set()
        
        for i in range(1, len(swara_sequences)):
            prev = swara_sequences[i - 1]
            curr = swara_sequences[i]
            if prev == curr:
                continue
            # Determine direction considering octave wrap
            diff = curr - prev
            if diff < -6:    # Wrapped up (e.g., 11 -> 0)
                diff += 12
            elif diff > 6:   # Wrapped down (e.g., 0 -> 11)
                diff -= 12
            if diff > 0:
                ascending.add(curr)
            elif diff < 0:
                descending.add(curr)
        
        return ascending, descending
    
    def _pattern_bonus(
        self,
        raga: 'Raga',
        ascending_semitones: Set[int],
        descending_semitones: Set[int]
    ) -> float:
        """
        Score how well ascending/descending note usage matches the raga's
        arohanam/avarohanam patterns.
        
        Key insight: janya ragas omit notes in one direction.
        If M1 (semitone 5) appears in ascending passages, it matches
        Shankarabharanam's arohanam but NOT Bilahari's (which skips M1 going up).
        
        When too many semitones appear in either direction (>8 of 12),
        it indicates a noisy/polyphonic recording where direction analysis
        is unreliable, so we return 0 to avoid unfairly penalizing janya ragas.
        """
        if not raga.arohanam or not raga.avarohanam:
            return 0.0
        
        # If ascending or descending contains too many semitones, the recording
        # is likely polyphonic (multiple instruments) or very noisy.
        # Pattern analysis becomes unreliable — skip it entirely.
        if len(ascending_semitones) > 8 or len(descending_semitones) > 8:
            return 0.0
        
        # Convert arohanam/avarohanam to semitone sets
        arohanam_st = {SWARA_TO_SEMITONE.get(s, -1) for s in raga.arohanam}
        arohanam_st.discard(-1)
        avarohanam_st = {SWARA_TO_SEMITONE.get(s, -1) for s in raga.avarohanam}
        avarohanam_st.discard(-1)
        
        full_scale_st = arohanam_st | avarohanam_st
        bonus = 0.0
        
        # Notes ascending should match the arohanam pattern
        if ascending_semitones:
            asc_in_arohanam = ascending_semitones & arohanam_st
            asc_foreign = ascending_semitones - arohanam_st
            bonus += (len(asc_in_arohanam) / len(ascending_semitones)) * 0.02
            bonus -= len(asc_foreign) * 0.005
        
        # Notes descending should match the avarohanam pattern
        if descending_semitones:
            desc_in_avarohanam = descending_semitones & avarohanam_st
            desc_foreign = descending_semitones - avarohanam_st
            bonus += (len(desc_in_avarohanam) / len(descending_semitones)) * 0.02
            bonus -= len(desc_foreign) * 0.005
        
        # Critical: penalize if notes that should be skipped in arohanam
        # actually appear in ascending passages
        arohanam_skipped = full_scale_st - arohanam_st
        if arohanam_skipped and ascending_semitones:
            skipped_in_ascent = ascending_semitones & arohanam_skipped
            if skipped_in_ascent:
                bonus -= len(skipped_in_ascent) * 0.015
        
        # Conversely: penalize if notes skipped in avarohanam appear descending
        avarohanam_skipped = full_scale_st - avarohanam_st
        if avarohanam_skipped and descending_semitones:
            skipped_in_descent = descending_semitones & avarohanam_skipped
            if skipped_in_descent:
                bonus -= len(skipped_in_descent) * 0.015
        
        return bonus
    
    def _match_ragas(
        self,
        primary_swaras: Set[str],
        all_swaras: Set[str],
        swara_counts: Counter,
        tonic_hz: float,
        top_n: int,
        swara_sequences: List[int] = None
    ) -> List[DetectionResult]:
        """
        Match detected swaras against raga database.
        
        Uses primary_swaras (after outlier removal) for matching,
        but reports all_swaras in results for transparency.
        If swara_sequences is provided, also uses ascending/descending
        pattern analysis to distinguish ragas with the same scale.
        """
        total = sum(swara_counts.values())
        if total == 0:
            return []
        
        # Convert counts to percentages
        swara_pcts = {self.INTERVALS.get(k, f'?{k}'): v/total 
                      for k, v in swara_counts.items()}
        
        # Analyze melodic direction for arohanam/avarohanam matching
        ascending_st, descending_st = set(), set()
        if swara_sequences and len(swara_sequences) > 1:
            ascending_st, descending_st = self._analyze_melodic_direction(swara_sequences)
        
        # Convert ascending/descending semitones to swara names for display
        asc_swaras = sorted(
            {self.INTERVALS.get(st, f'?{st}') for st in ascending_st},
            key=lambda s: SWARA_TO_SEMITONE.get(s, 99)
        ) if ascending_st else []
        desc_swaras = sorted(
            {self.INTERVALS.get(st, f'?{st}') for st in descending_st},
            key=lambda s: SWARA_TO_SEMITONE.get(s, 99)
        ) if descending_st else []
        
        results = []
        
        for raga in self.db:
            score = self._compute_match_score(
                primary_swaras, swara_pcts, raga,
                ascending_st, descending_st
            )
            
            if score > 0.3:  # Minimum threshold
                result = DetectionResult(
                    raga=raga,
                    confidence=score,
                    detected_swaras=primary_swaras,
                    tonic_hz=tonic_hz,
                    match_details={
                        'raga_scale': raga.scale,
                        'primary_detected': primary_swaras,
                        'all_detected': all_swaras,
                        'outliers': all_swaras - primary_swaras,
                        'matching': primary_swaras & raga.scale,
                        'foreign': primary_swaras - raga.scale - {'S', 'P'},
                        'missing': raga.scale - primary_swaras,
                        'distribution': swara_pcts,
                        'detected_ascending': asc_swaras,
                        'detected_descending': desc_swaras,
                    }
                )
                results.append(result)
        
        # Sort by confidence, with melakarta and well-known ragas first for ties
        results.sort(key=lambda r: (
            r.confidence,
            r.raga.is_melakarta,
            r.raga.name.lower() in self._well_known_ragas()
        ), reverse=True)
        
        return results[:top_n]
    
    def _compute_match_score(
        self,
        detected_swaras: Set[str],
        swara_pcts: Dict[str, float],
        raga: Raga,
        ascending_semitones: Set[int] = None,
        descending_semitones: Set[int] = None
    ) -> float:
        """
        Compute how well detected notes match a raga.
        
        Scoring:
        - Weight by how much of the audio uses raga notes (semitone-aware)
        - Penalize foreign notes (notes not in the raga)
        - Bonus for coverage (detecting all raga notes)
        - Penalty for specificity (raga claims notes we barely hear)
        - Bonus for arohanam/avarohanam pattern match
        
        Uses semitone comparison for enharmonic-aware matching:
        e.g. detected R2 matches raga G1 (both semitone 2).
        """
        # Use cached semitone set for fast comparison
        raga_semitones = raga.scale_semitones
        
        # Calculate what percentage of notes belong to this raga (by semitone)
        matching_pct = sum(
            pct for swara, pct in swara_pcts.items()
            if SWARA_TO_SEMITONE.get(swara, -1) in raga_semitones
               or swara in ['S', 'P']
        )
        
        # Penalty for foreign notes (not in raga, by semitone)
        # Use a graduated threshold: heavily penalize prominent foreign notes,
        # lightly penalize weak ones (which are likely gamaka artifacts)
        foreign_pct = 0
        for swara, pct in swara_pcts.items():
            semi = SWARA_TO_SEMITONE.get(swara, -1)
            if semi not in raga_semitones and swara not in ['S', 'P']:
                if pct > 0.05:
                    foreign_pct += pct  # Full penalty for prominent foreign notes
                elif pct > 0.03:
                    foreign_pct += pct * 0.5  # Half penalty for marginal notes
        
        # Coverage: what fraction of raga's semitones did we detect?
        # Use a weighted coverage that considers how strongly each note is present
        detected_semitone_pcts = {}
        for swara, pct in swara_pcts.items():
            semi = SWARA_TO_SEMITONE.get(swara, -1)
            if semi >= 0:
                detected_semitone_pcts[semi] = detected_semitone_pcts.get(semi, 0) + pct
        
        # Count raga notes that are well-represented (>2%) vs barely present
        raga_notes_found = 0
        for semi in raga_semitones:
            note_pct = detected_semitone_pcts.get(semi, 0)
            if note_pct > 0.02:
                raga_notes_found += 1
            elif note_pct > 0.005:
                raga_notes_found += 0.5  # Partial credit for faint notes
        coverage = raga_notes_found / len(raga_semitones) if raga_semitones else 0
        
        # Specificity: penalize ragas that claim notes we didn't detect.
        # If a raga says R2 should exist but we barely hear it,
        # that's evidence against this raga. This prevents 7-note ragas from
        # outscoring correct 6-note ragas when the extra note is noise.
        detected_semitones = {SWARA_TO_SEMITONE.get(s, -1) for s in detected_swaras}
        detected_semitones.discard(-1)
        specificity_penalty = 0
        for semi in raga_semitones - {0, 7}:  # Exclude S and P
            note_pct = detected_semitone_pcts.get(semi, 0)
            if note_pct < 0.02:
                specificity_penalty += 0.04  # Raga expects this note but we don't hear it
            elif note_pct < 0.04:
                specificity_penalty += 0.02  # Barely present
        
        # Base score
        score = matching_pct - (foreign_pct * 0.5) + (coverage * 0.20) - specificity_penalty
        
        # Arohanam/avarohanam pattern bonus
        if ascending_semitones is not None and descending_semitones is not None:
            score += self._pattern_bonus(raga, ascending_semitones, descending_semitones)
        
        # Small bonuses
        if raga.is_melakarta:
            score += 0.01
        if raga.name.lower() in self._well_known_ragas():
            score += 0.005
        
        # Don't clamp to 1.0 — let natural score differences 
        # distinguish ragas (e.g., pattern bonus, melakarta bonus)
        return max(0.0, score)
    
    def _well_known_ragas(self) -> Set[str]:
        """Common ragas that should be prioritized in ties."""
        return {
            'kalyani', 'mohanam', 'shankarabharanam', 'bhairavi',
            'todi', 'kambhoji', 'hamsadhwani', 'mayamalavagowla',
            'kharaharapriya', 'harikambhoji', 'arabhi', 'saveri',
            'kaanada', 'bilahari', 'kedaram', 'hindolam',
            'charukeshi', 'abhogi', 'madhyamavati', 'suddhasaveri',
            'natabhairavi', 'keeravani', 'simhendramadhyamam'
        }


# Convenience function
def detect_raga_v2(audio_path: str, top_n: int = 15) -> List[DetectionResult]:
    """Detect raga from audio file using musical approach."""
    detector = RagaDetectorV2()
    return detector.detect_from_file(audio_path, top_n)
