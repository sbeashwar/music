"""
Unified Raga Detector — Combines ML classification (v3) with pitch-tracking (v2).

Architecture:
  1. HPSS pre-processing: separates harmonic (melody) from percussive (mridangam)
     - Gives cleaner pitch tracking (v2 benefits from percussion removal)
     - ML classifier runs on FULL audio (trained on multi-instrument recordings)
  
  2. Dual-engine detection:
     - v2 pitch-tracking: Extracts actual swaras, identifies Sa/Pa, analyzes
       arohanam/avarohanam patterns. Precise for clean vocal recordings.
     - v3 ML classifier: RandomForest trained on 51 ragas, robust to multi-
       instrument recordings. Falls back to chroma rules for unknown ragas.
  
  3. Adaptive fusion:
     - When both agree (raga in top-K of both): high confidence (agreement bonus)
     - When ML model is confident: trust ML more (multi-instrument robustness)
     - When pitch-tracking finds clear patterns: trust v2 more (note precision)
     - Ragas only in v2: pitch-tracking only (covers all 5321 ragas)
     - Ragas only in v3: ML only (covers multi-instrument)

  4. GUI-compatible output:
     - Returns DetectionResult (same as v2) for seamless GUI integration
     - Includes both tonic_hz and detected_swaras from v2 pitch-tracking
     - match_details includes ML probability and fusion metadata
"""

import numpy as np
import time
from typing import List, Optional, Set, Dict
from dataclasses import dataclass

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

from .detector_v2 import RagaDetectorV2, DetectionResult
from .detector_v3 import RagaDetectorV3, DetectionResultV3
from .raga_db import RagaDB, Raga, get_db


class UnifiedRagaDetector:
    """
    Combines ML classification and pitch-tracking for robust raga detection.
    
    The key insight: neither approach alone works well for all recordings.
    
    - Pitch-tracking (v2): Identifies individual swaras by analyzing pitch
      contours. Works well on clean vocal recordings but struggles with
      multi-instrument audio (violin, mridangam confuse the pitch tracker).
    
    - ML classification (v3): Uses a RandomForest trained on chroma, MFCC,
      tonnetz features. Robust to multi-instrument recordings but only knows
      the 51 ragas in its training set. Its chroma-based rule fallback is
      weak because Carnatic gamakas flatten the chroma profile.
    
    By combining both:
    - Agreement between methods = high confidence
    - ML fills in when pitch-tracking fails on complex recordings
    - Pitch-tracking covers the 5270+ ragas not in ML training set
    - HPSS preprocessing helps pitch-tracking by removing percussion
    """
    
    def __init__(self, model_dir: str = None):
        self.v2 = RagaDetectorV2()
        self.v3 = RagaDetectorV3(model_dir)
        
        # Fusion parameters
        self.ml_weight = 0.55       # Weight for ML score when both available
        self.pitch_weight = 0.45    # Weight for pitch-tracking score
        self.agreement_bonus = 0.15 # Bonus when both methods agree on a raga
        self.use_hpss = False       # HPSS hurts clean vocal recordings (changes tonic)
    
    @property
    def db(self) -> RagaDB:
        """Raga database (for GUI compatibility)."""
        return self.v2.db
    
    @property
    def has_ml_model(self) -> bool:
        """Whether the ML model is loaded."""
        return self.v3.has_model
    
    def detect_from_file(self, audio_path: str, top_n: int = 15,
                         tonic_hz: float = None) -> List[DetectionResult]:
        """Detect raga from audio file."""
        if not HAS_LIBROSA:
            raise ImportError("librosa is required")
        
        # Use v3's fast loader for WAV/FLAC, librosa for MP3
        y, sr = self.v3.feature_extractor._fast_load(audio_path, duration=60.0)
        return self.detect_from_audio(y, sr, top_n, tonic_hz)
    
    def detect_from_audio(
        self, 
        y: np.ndarray, 
        sr: int = 22050, 
        top_n: int = 15, 
        tonic_hz: float = None
    ) -> List[DetectionResult]:
        """
        Detect raga from audio samples using dual-engine fusion.
        
        Args:
            y: Audio samples (mono, float)
            sr: Sample rate (default 22050)
            top_n: Number of top matches to return
            tonic_hz: Optional manual tonic frequency. If None, auto-detect.
            
        Returns:
            List of DetectionResult objects, sorted by confidence.
        """
        t0 = time.time()
        
        # --- 1. HPSS Pre-processing ---
        if self.use_hpss and len(y) > sr:
            y_harmonic, _ = librosa.effects.hpss(y)
        else:
            y_harmonic = y
        
        # --- 2. Run v2 pitch-tracking (on harmonic component) ---
        v2_results = self._run_v2(y_harmonic, sr, tonic_hz)
        
        # --- 3. Run v3 ML (on full audio — model was trained on full recordings) ---
        v3_results = self._run_v3(y, sr) if self.v3.has_model else []
        
        # --- 4. Fuse results ---
        if not v3_results:
            # No ML model — pure pitch-tracking (enhanced with HPSS)
            return v2_results[:top_n]
        
        if not v2_results:
            # Pitch-tracking failed (very short audio?) — ML only
            return self._v3_to_v2_format(v3_results[:top_n], tonic_hz)
        
        merged = self._fuse_results(v2_results, v3_results, top_n)
        
        t1 = time.time()
        # Tag the top result with timing info
        if merged:
            merged[0].match_details['unified_time'] = f"{t1-t0:.1f}s"
            merged[0].match_details['method'] = 'unified_v2+v3'
        
        return merged
    
    def _run_v2(self, y: np.ndarray, sr: int, 
                tonic_hz: float = None) -> List[DetectionResult]:
        """Run v2 pitch-tracking detector."""
        try:
            return self.v2.detect_from_audio(y, sr, top_n=50, tonic_hz=tonic_hz)
        except Exception as e:
            print(f"v2 detection error: {e}")
            return []
    
    def _run_v3(self, y: np.ndarray, sr: int) -> List[DetectionResultV3]:
        """Run v3 ML detector."""
        try:
            return self.v3.detect_from_audio(y, sr, top_n=50)
        except Exception as e:
            print(f"v3 detection error: {e}")
            return []
    
    def _fuse_results(
        self, 
        v2_results: List[DetectionResult],
        v3_results: List[DetectionResultV3],
        top_n: int
    ) -> List[DetectionResult]:
        """
        Fuse results from both detectors.
        
        Strategy:
        - Normalize v2 and v3 scores to [0, 1]
        - For ragas in both: weighted combination + agreement bonus
        - For ragas only in v2: use v2 score (covers all 5321 ragas)
        - For ragas only in v3 (ML trained): convert & add at reduced weight
        
        Returns DetectionResult objects (v2 format) for GUI compatibility.
        """
        # Build lookup maps
        v2_by_name: Dict[str, DetectionResult] = {}
        for r in v2_results:
            v2_by_name[r.raga.name.lower()] = r
        
        v3_by_name: Dict[str, DetectionResultV3] = {}
        for r in v3_results:
            v3_by_name[r.raga_name.lower()] = r
        
        # Normalize scores to [0, 1]
        v2_max = max((r.confidence for r in v2_results), default=1.0)
        v3_max = max((r.confidence for r in v3_results), default=1.0)
        
        if v2_max < 1e-8: v2_max = 1.0
        if v3_max < 1e-8: v3_max = 1.0
        
        # Sets for computing agreement
        v2_top5 = {r.raga.name.lower() for r in v2_results[:5]}
        v3_top5 = {r.raga_name.lower() for r in v3_results[:5]}
        
        # Determine audio complexity (affects weight balance)
        # If v3 ML is very confident on top match, audio is probably 
        # multi-instrument (where ML excels)
        v3_top_ml_prob = v3_results[0].ml_probability if v3_results else 0.0
        
        # Adaptive weights
        if v3_top_ml_prob > 0.3:
            # ML is very confident → likely a clean match, trust ML more
            w_ml = 0.65
            w_pitch = 0.35
        elif v3_top_ml_prob > 0.15:
            # ML has some confidence
            w_ml = self.ml_weight
            w_pitch = self.pitch_weight
        else:
            # ML is uncertain → trust pitch-tracking more
            w_ml = 0.35
            w_pitch = 0.65
        
        merged = []
        all_names = set(v2_by_name.keys()) | set(v3_by_name.keys())
        
        # Use v2's top result for GUI metadata (tonic, detected swaras, patterns)
        template_result = v2_results[0]
        
        for name in all_names:
            v2r = v2_by_name.get(name)
            v3r = v3_by_name.get(name)
            
            v2_score = (v2r.confidence / v2_max) if v2r else 0.0
            v3_score = (v3r.confidence / v3_max) if v3r else 0.0
            ml_prob = v3r.ml_probability if v3r else 0.0
            
            # Compute fused confidence
            if v2r and v3r:
                # Both methods scored this raga
                fused = w_ml * v3_score + w_pitch * v2_score
                
                # Agreement bonus: both methods' top-5 include this raga
                if name in v2_top5 and name in v3_top5:
                    fused += self.agreement_bonus
                
                method = 'both'
            elif v2r and not v3r:
                # Only pitch-tracking found it (raga not in ML training set)
                fused = v2_score * 0.9  # Slight derate since ML didn't corroborate
                method = 'v2_only'
            else:
                # Only ML found it (raga not in v2's top-50)
                # This might mean pitch-tracking missed it due to instruments
                fused = v3_score * 0.6  # More derate since no note-level evidence
                method = 'v3_only'
            
            # Build unified DetectionResult (v2 format for GUI)
            if v2r:
                # Use v2's rich result object, update confidence
                result = DetectionResult(
                    raga=v2r.raga,
                    confidence=min(fused, 1.0),
                    detected_swaras=v2r.detected_swaras,
                    tonic_hz=v2r.tonic_hz,
                    match_details={
                        **v2r.match_details,
                        'ml_probability': ml_prob,
                        'v2_confidence': v2r.confidence,
                        'v3_confidence': v3r.confidence if v3r else 0.0,
                        'fusion_method': method,
                    }
                )
            else:
                # v3-only result: need to find the Raga object
                raga_obj = self._find_raga(name)
                if not raga_obj:
                    continue  # Skip if can't find raga in DB
                
                result = DetectionResult(
                    raga=raga_obj,
                    confidence=min(fused, 1.0),
                    detected_swaras=template_result.detected_swaras,
                    tonic_hz=template_result.tonic_hz,
                    match_details={
                        **template_result.match_details,
                        'ml_probability': ml_prob,
                        'v2_confidence': 0.0,
                        'v3_confidence': v3r.confidence if v3r else 0.0,
                        'fusion_method': method,
                    }
                )
            
            merged.append(result)
        
        # Sort by fused confidence
        merged.sort(key=lambda r: -r.confidence)
        return merged[:top_n]
    
    def _find_raga(self, name: str) -> Optional[Raga]:
        """Find a Raga object by name (case-insensitive)."""
        results = self.v2.db.search(name)
        if results:
            for r in results:
                if r.name.lower() == name.lower():
                    return r
            return results[0]
        return None
    
    def _v3_to_v2_format(self, v3_results: List[DetectionResultV3],
                         tonic_hz: float = None) -> List[DetectionResult]:
        """Convert v3 results to v2 format when pitch-tracking isn't available."""
        converted = []
        for v3r in v3_results:
            raga = self._find_raga(v3r.raga_name)
            if not raga:
                continue
            converted.append(DetectionResult(
                raga=raga,
                confidence=v3r.confidence,
                detected_swaras=set(),
                tonic_hz=tonic_hz or 0.0,
                match_details={
                    'ml_probability': v3r.ml_probability,
                    'v3_confidence': v3r.confidence,
                    'fusion_method': 'v3_only_fallback',
                    'primary_detected': set(),
                    'outliers': set(),
                }
            ))
        return converted
