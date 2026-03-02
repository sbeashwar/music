"""
Swara Sequence Matcher - Match detected note sequences to ragas.

Given a sequence of detected swaras (e.g., ['S', 'R2', 'G3', 'P', 'D2', 'S']),
find matching ragas from the database by comparing arohanam/avarohanam patterns.
"""

import os
import json
from typing import List, Dict, Optional, Tuple, Set
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

# Semitone to possible swaras (reverse mapping)
SEMITONE_TO_SWARAS = {}
for swara, semi in SWARA_TO_SEMITONE.items():
    if semi not in SEMITONE_TO_SWARAS:
        SEMITONE_TO_SWARAS[semi] = []
    SEMITONE_TO_SWARAS[semi].append(swara)

# All 16 swaras (without octave variants)
ALL_SWARAS = list(SWARA_TO_SEMITONE.keys())


@dataclass
class RagaMatch:
    """Result of matching a swara sequence against a raga."""
    raga_id: str
    raga_name: str
    score: float  # 0.0 - 1.0, higher is better
    match_type: str  # 'exact', 'subset', 'superset', 'partial'
    arohanam: List[str]
    avarohanam: List[str]
    matched_swaras: List[str]  # Swaras that matched
    extra_swaras: List[str]  # Detected swaras not in raga
    missing_swaras: List[str]  # Raga swaras not detected
    is_melakarta: bool = False
    melakarta_number: int = 0
    parent_melakarta: str = ""
    details: str = ""


@dataclass
class RagaEntry:
    """A raga from the database with its scale information."""
    id: str
    name: str
    arohanam: List[str]
    avarohanam: List[str]
    is_melakarta: bool = False
    melakarta_number: int = 0
    parent_melakarta: str = ""
    swara_count: int = 0  # Number of unique swaras (excluding octave Sa)
    arohanam_set: Set[str] = field(default_factory=set)
    avarohanam_set: Set[str] = field(default_factory=set)
    all_swaras_set: Set[str] = field(default_factory=set)
    arohanam_semitones: Tuple = ()
    avarohanam_semitones: Tuple = ()


class SwaraSequenceMatcher:
    """
    Matches detected swara sequences against a database of ragas.
    
    Supports:
    - Exact arohanam/avarohanam matching
    - Set-based matching (same notes, any order)
    - Partial matching (subset/superset)
    - Semitone-based matching (handles enharmonic equivalents like R2/G1)
    - Sequence order matching (vakra ragas have non-linear order)
    """
    
    def __init__(self, metadata_dir: Optional[str] = None):
        """
        Initialize with raga metadata directory.
        
        Args:
            metadata_dir: Path to directory containing raga JSON files.
                         Defaults to shared/ragas_metadata relative to this file.
        """
        if metadata_dir is None:
            base = Path(__file__).parent.parent
            metadata_dir = str(base / "shared" / "ragas_metadata")
        
        self.metadata_dir = metadata_dir
        self.ragas: Dict[str, RagaEntry] = {}
        
        # Indices for fast lookup
        self._by_arohanam_key: Dict[str, List[str]] = {}  # frozenset key -> raga_ids
        self._by_swara_count: Dict[int, List[str]] = {}   # count -> raga_ids
        self._by_semitone_set: Dict[tuple, List[str]] = {} # sorted semitones -> raga_ids
        
        self._load_ragas()
    
    def _load_ragas(self):
        """Load all raga definitions from metadata directory."""
        if not os.path.exists(self.metadata_dir):
            raise FileNotFoundError(f"Metadata directory not found: {self.metadata_dir}")
        
        loaded = 0
        skipped = 0
        
        for filename in os.listdir(self.metadata_dir):
            if not filename.endswith('.json'):
                continue
            
            filepath = os.path.join(self.metadata_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                skipped += 1
                continue
            
            arohanam = data.get('arohanam', [])
            avarohanam = data.get('avarohanam', [])
            
            # Skip ragas without scale data
            if not arohanam or not avarohanam:
                skipped += 1
                continue
            
            # Filter to valid swaras only (remove 'S' at start/end for set operations)
            arohanam_inner = [s for s in arohanam if s in SWARA_TO_SEMITONE and s != 'S']
            avarohanam_inner = [s for s in avarohanam if s in SWARA_TO_SEMITONE and s != 'S']
            
            if not arohanam_inner:
                skipped += 1
                continue
            
            raga_id = data.get('id', filename[:-5])
            
            # Compute semitone sequences
            aro_semi = tuple(SWARA_TO_SEMITONE[s] for s in arohanam_inner)
            ava_semi = tuple(SWARA_TO_SEMITONE[s] for s in avarohanam_inner)
            
            entry = RagaEntry(
                id=raga_id,
                name=data.get('name', raga_id),
                arohanam=arohanam,
                avarohanam=avarohanam,
                is_melakarta=data.get('is_melakarta', False),
                melakarta_number=data.get('melakarta_number', 0) or 0,
                parent_melakarta=data.get('parent_melakarta', '') or '',
                swara_count=len(set(arohanam_inner) | set(avarohanam_inner)),
                arohanam_set=set(arohanam_inner),
                avarohanam_set=set(avarohanam_inner),
                all_swaras_set=set(arohanam_inner) | set(avarohanam_inner),
                arohanam_semitones=aro_semi,
                avarohanam_semitones=ava_semi,
            )
            
            self.ragas[raga_id] = entry
            loaded += 1
            
            # Build indices
            # 1. By swara count
            count = entry.swara_count
            if count not in self._by_swara_count:
                self._by_swara_count[count] = []
            self._by_swara_count[count].append(raga_id)
            
            # 2. By arohanam set (frozenset key)
            aro_key = str(sorted(entry.arohanam_set))
            if aro_key not in self._by_arohanam_key:
                self._by_arohanam_key[aro_key] = []
            self._by_arohanam_key[aro_key].append(raga_id)
            
            # 3. By semitone set
            semi_key = tuple(sorted(set(aro_semi) | set(ava_semi)))
            if semi_key not in self._by_semitone_set:
                self._by_semitone_set[semi_key] = []
            self._by_semitone_set[semi_key].append(raga_id)
        
        print(f"SwaraSequenceMatcher: Loaded {loaded} ragas ({skipped} skipped)")
    
    def _normalize_swaras(self, swaras: List[str]) -> List[str]:
        """
        Normalize a swara list:
        - Remove duplicates while preserving order
        - Remove 'S' (Sa) from middle of sequence
        - Handle common input variations
        """
        # Handle common input variations
        normalized = []
        for s in swaras:
            s = s.strip().upper()
            # Handle common abbreviations
            if s in ('SA', 'SHA'):
                s = 'S'
            elif s in ('PA',):
                s = 'P'
            elif s in ('RI1', 'RI2', 'RI3'):
                s = 'R' + s[-1]
            elif s in ('GA1', 'GA2', 'GA3'):
                s = 'G' + s[-1]
            elif s in ('MA1', 'MA2'):
                s = 'M' + s[-1]
            elif s in ('DA1', 'DA2', 'DA3'):
                s = 'D' + s[-1]
            elif s in ('NI1', 'NI2', 'NI3'):
                s = 'N' + s[-1]
            
            if s in SWARA_TO_SEMITONE:
                normalized.append(s)
        
        return normalized
    
    def match_swaras(
        self,
        detected_swaras: List[str],
        direction: str = 'ascending',
        max_results: int = 20,
        min_score: float = 0.3,
        raw_sequence: Optional[List[str]] = None,
    ) -> List[RagaMatch]:
        """
        Match a detected swara sequence against the raga database.
        
        Args:
            detected_swaras: Ordered list of swara names (e.g., ['S', 'R2', 'G3', 'P', 'D2'])
            direction: 'ascending' (arohanam), 'descending' (avarohanam), or 'both'
            max_results: Maximum number of results to return
            min_score: Minimum match score (0.0 - 1.0) to include
            raw_sequence: Full note sequence with repeats (used to split ascending 
                         vs descending swaras for asymmetric raga matching)
            
        Returns:
            List of RagaMatch objects sorted by score (highest first)
        """
        # Normalize input
        swaras = self._normalize_swaras(detected_swaras)
        if not swaras:
            return []
        
        # Remove leading/trailing Sa for set comparison
        inner_swaras = [s for s in swaras if s != 'S']
        if not inner_swaras:
            return []
        
        detected_set = set(inner_swaras)
        detected_semitones = set(SWARA_TO_SEMITONE[s] for s in inner_swaras)
        n_detected_semi = len(detected_semitones)
        
        # Split raw_sequence into ascending and descending segments
        asc_semis, desc_semis = self._split_asc_desc(raw_sequence)
        
        results = []
        
        for raga_id, raga in self.ragas.items():
            # Choose which side to compare
            if direction == 'ascending':
                raga_swaras = raga.arohanam_set
                raga_sequence = raga.arohanam
            elif direction == 'descending':
                raga_swaras = raga.avarohanam_set
                raga_sequence = raga.avarohanam
            else:  # both
                raga_swaras = raga.all_swaras_set
                raga_sequence = raga.arohanam  # primary reference
            
            n_raga = len(raga_swaras)
            if n_raga == 0:
                continue
            
            # Primary comparison at SEMITONE level (handles enharmonic equivalents)
            raga_semitones = set(SWARA_TO_SEMITONE[s] for s in raga_swaras)
            n_raga_semi = len(raga_semitones)
            
            semi_matched = detected_semitones & raga_semitones
            semi_extra = detected_semitones - raga_semitones
            semi_missing = raga_semitones - detected_semitones
            
            n_semi_matched = len(semi_matched)
            n_semi_extra = len(semi_extra)
            n_semi_missing = len(semi_missing)
            
            if n_semi_matched == 0:
                continue
            
            # Jaccard similarity at semitone level
            semi_union = len(detected_semitones | raga_semitones)
            if semi_union == 0:
                continue
            jaccard = n_semi_matched / semi_union
            
            # Classify match type
            if n_semi_matched == n_raga_semi and n_semi_extra == 0:
                match_type = 'exact'
                score = 1.0
            elif n_semi_matched == n_raga_semi and n_semi_extra > 0:
                match_type = 'superset'
                score = jaccard * 0.9
            elif n_semi_matched == n_detected_semi and n_semi_missing > 0:
                match_type = 'subset'
                score = jaccard * 0.85
            else:
                match_type = 'partial'
                score = jaccard * 0.7
            
            # Secondary: bonus for exact swara-name match (differentiates
            # enharmonic variants like R2-based vs G1-based ragas)
            swara_matched = detected_set & raga_swaras
            if len(swara_matched) > len(semi_matched):
                # More swara matches than semitone matches shouldn't happen
                pass
            elif len(swara_matched) == len(semi_matched):
                # Names match too - slight bonus
                score += 0.005
            
            # Bonus for sequence order match (important for vakra ragas)
            if direction != 'both' and n_semi_matched >= 3:
                order_bonus = self._sequence_order_score_semi(
                    [SWARA_TO_SEMITONE.get(s, -1) for s in swaras if s != 'S'],
                    [SWARA_TO_SEMITONE.get(s, -1) for s in raga_sequence 
                     if s in SWARA_TO_SEMITONE and s != 'S'],
                    raga_semitones
                )
                score = score * 0.8 + order_bonus * 0.2
            
            # Asymmetric matching bonus: when we have both ascending and
            # descending data, reward ragas whose arohanam/avarohanam structure
            # matches which swaras appear in each direction.
            if asc_semis is not None and desc_semis is not None:
                asym_bonus = self._asymmetric_score(
                    asc_semis, desc_semis, raga
                )
                # Asymmetry score can strongly differentiate, weight it heavily
                score = score * 0.65 + asym_bonus * 0.35
            
            # Bonus for popular ragas (melakartas slightly preferred when tied)
            if raga.is_melakarta:
                score += 0.01
            
            if score >= min_score:
                # Report matched/extra/missing using raga's swara names
                matched_swaras = [s for s in raga_swaras 
                                  if SWARA_TO_SEMITONE[s] in semi_matched]
                extra_semis = sorted(semi_extra)
                missing_swaras = [s for s in raga_swaras
                                  if SWARA_TO_SEMITONE[s] in semi_missing]
                
                results.append(RagaMatch(
                    raga_id=raga_id,
                    raga_name=raga.name,
                    score=score,
                    match_type=match_type,
                    arohanam=raga.arohanam,
                    avarohanam=raga.avarohanam,
                    matched_swaras=sorted(matched_swaras),
                    extra_swaras=[str(s) for s in extra_semis],
                    missing_swaras=sorted(missing_swaras),
                    is_melakarta=raga.is_melakarta,
                    melakarta_number=raga.melakarta_number,
                    parent_melakarta=raga.parent_melakarta,
                    details=f"Matched {n_semi_matched}/{n_raga_semi} semitones, "
                            f"extra: {n_semi_extra}, missing: {n_semi_missing}",
                ))
        
        # Sort by score descending, then by name
        results.sort(key=lambda m: (-m.score, m.raga_name))
        
        return results[:max_results]
    
    def _sequence_order_score(
        self,
        detected: List[str],
        raga_sequence: List[str],
        raga_swaras: Set[str],
    ) -> float:
        """
        Score how well the detected sequence order matches the raga's
        arohanam/avarohanam order. 
        
        Returns 0.0 - 1.0.
        """
        # Filter detected to only swaras that exist in the raga
        detected_in_raga = [s for s in detected if s in raga_swaras]
        raga_in_detected = [s for s in raga_sequence 
                           if s in SWARA_TO_SEMITONE and s != 'S' 
                           and s in set(detected)]
        
        if len(detected_in_raga) < 2 or len(raga_in_detected) < 2:
            return 0.5  # Not enough to judge
        
        # Count concordant pairs (pairs in same relative order)
        concordant = 0
        total_pairs = 0
        
        for i in range(len(detected_in_raga)):
            for j in range(i + 1, len(detected_in_raga)):
                s1, s2 = detected_in_raga[i], detected_in_raga[j]
                if s1 in raga_in_detected and s2 in raga_in_detected:
                    idx1_raga = raga_in_detected.index(s1)
                    idx2_raga = raga_in_detected.index(s2)
                    if idx1_raga < idx2_raga:
                        concordant += 1
                    total_pairs += 1
        
        if total_pairs == 0:
            return 0.5
        
        return concordant / total_pairs
    
    def _sequence_order_score_semi(
        self,
        detected_semitones: List[int],
        raga_semitones: List[int],
        raga_semi_set: Set[int],
    ) -> float:
        """
        Score sequence order match using semitone values.
        Avoids enharmonic naming issues by comparing raw semitone order.
        
        Returns 0.0 - 1.0.
        """
        # Filter detected to only semitones in the raga
        detected_in_raga = [s for s in detected_semitones 
                           if s >= 0 and s in raga_semi_set]
        raga_matching = [s for s in raga_semitones 
                        if s >= 0 and s in set(detected_semitones)]
        
        if len(detected_in_raga) < 2 or len(raga_matching) < 2:
            return 0.5
        
        # Count concordant pairs
        concordant = 0
        total_pairs = 0
        
        for i in range(len(detected_in_raga)):
            for j in range(i + 1, len(detected_in_raga)):
                s1, s2 = detected_in_raga[i], detected_in_raga[j]
                if s1 in raga_matching and s2 in raga_matching:
                    idx1 = raga_matching.index(s1)
                    idx2 = raga_matching.index(s2)
                    if idx1 < idx2:
                        concordant += 1
                    total_pairs += 1
        
        if total_pairs == 0:
            return 0.5
        
        return concordant / total_pairs
    
    def _split_asc_desc(
        self, raw_sequence: Optional[List[str]]
    ) -> Tuple[Optional[Set[int]], Optional[Set[int]]]:
        """
        Split a raw note sequence into ascending and descending semitone sets.
        
        Uses the turning point (where pitch starts to descend after ascending)
        to separate the arohanam from the avarohanam.
        
        Returns:
            (ascending_semitones, descending_semitones) or (None, None) if
            raw_sequence is None or too short.
        """
        if not raw_sequence or len(raw_sequence) < 4:
            return None, None
        
        # Convert to semitones
        semi_seq = []
        for s in raw_sequence:
            s_upper = s.strip().upper()
            if s_upper in SWARA_TO_SEMITONE:
                semi_seq.append((s_upper, SWARA_TO_SEMITONE[s_upper]))
        
        if len(semi_seq) < 4:
            return None, None
        
        # Find the turning point (highest semitone, or a repeated S/0 after non-zero)
        # Strategy: find where the sequence transitions from going up to going down
        max_semi = -1
        turn_idx = -1
        
        for i, (name, semi) in enumerate(semi_seq):
            if semi > max_semi:
                max_semi = semi
                turn_idx = i
            elif semi == 0 and i > 0 and semi_seq[i-1][1] > 0:
                # Upper Sa reached
                turn_idx = i
                break
        
        if turn_idx <= 0 or turn_idx >= len(semi_seq) - 1:
            # Can't find a clear turn point — try splitting in half
            turn_idx = len(semi_seq) // 2
        
        # Extract unique swaras for each half (excluding Sa)
        asc_swaras = set()
        for name, semi in semi_seq[:turn_idx + 1]:
            if semi > 0:
                asc_swaras.add(semi)
        
        desc_swaras = set()
        for name, semi in semi_seq[turn_idx:]:
            if semi > 0:
                desc_swaras.add(semi)
        
        if not asc_swaras or not desc_swaras:
            return None, None
        
        return asc_swaras, desc_swaras
    
    def _asymmetric_score(
        self,
        asc_semis: Set[int],
        desc_semis: Set[int],
        raga: RagaEntry,
    ) -> float:
        """
        Score how well the ascending/descending split matches a raga's
        arohanam/avarohanam structure.
        
        This is critical for ragas like Mand where arohanam omits swaras
        that appear in avarohanam (and vice versa for vakra ragas).
        
        Returns 0.0 - 1.0.
        """
        # Get raga's arohanam and avarohanam as semitone sets
        raga_aro_semis = set(raga.arohanam_semitones)
        raga_ava_semis = set(raga.avarohanam_semitones)
        
        if not raga_aro_semis or not raga_ava_semis:
            return 0.5
        
        # Jaccard similarity for ascending part vs raga arohanam
        aro_intersection = len(asc_semis & raga_aro_semis)
        aro_union = len(asc_semis | raga_aro_semis)
        aro_jaccard = aro_intersection / aro_union if aro_union > 0 else 0
        
        # Jaccard similarity for descending part vs raga avarohanam
        ava_intersection = len(desc_semis & raga_ava_semis)
        ava_union = len(desc_semis | raga_ava_semis)
        ava_jaccard = ava_intersection / ava_union if ava_union > 0 else 0
        
        # Average of both — both directions must match well
        return (aro_jaccard + ava_jaccard) / 2.0
    
    def match_by_semitones(
        self,
        semitones: List[int],
        max_results: int = 20,
        min_score: float = 0.3,
    ) -> List[RagaMatch]:
        """
        Match by semitone offsets from Sa (0-11).
        Useful when exact swara names are unknown (e.g., R2 vs G1 ambiguity).
        
        Args:
            semitones: List of semitone offsets from Sa (e.g., [0, 2, 4, 7, 9] for Mohanam)
            max_results: Maximum results
            min_score: Minimum score threshold
            
        Returns:
            List of RagaMatch objects
        """
        # Convert semitones to possible swaras and match
        # For each semitone, pick the most common swara name
        detected_semitone_set = set(s % 12 for s in semitones if s != 0)  # exclude Sa
        
        results = []
        
        for raga_id, raga in self.ragas.items():
            raga_semitones = set(SWARA_TO_SEMITONE[s] for s in raga.all_swaras_set)
            
            matched = detected_semitone_set & raga_semitones
            extra = detected_semitone_set - raga_semitones
            missing = raga_semitones - detected_semitone_set
            
            n_matched = len(matched)
            n_raga = len(raga_semitones)
            
            if n_matched == 0:
                continue
            
            union_size = len(detected_semitone_set | raga_semitones)
            jaccard = n_matched / union_size
            
            # Classify match
            if n_matched == n_raga and len(extra) == 0:
                match_type = 'exact'
                score = 1.0
            elif n_matched == n_raga:
                match_type = 'superset'
                score = jaccard * 0.9
            elif n_matched == len(detected_semitone_set):
                match_type = 'subset'
                score = jaccard * 0.85
            else:
                match_type = 'partial'
                score = jaccard * 0.7
            
            if raga.is_melakarta:
                score += 0.01
            
            if score >= min_score:
                # Convert matched semitones to raga's swara names
                matched_swaras = [s for s in raga.all_swaras_set 
                                 if SWARA_TO_SEMITONE[s] in matched]
                extra_swaras_semi = [str(s) for s in sorted(extra)]
                missing_swaras = [s for s in raga.all_swaras_set
                                 if SWARA_TO_SEMITONE[s] in missing]
                
                results.append(RagaMatch(
                    raga_id=raga_id,
                    raga_name=raga.name,
                    score=score,
                    match_type=match_type,
                    arohanam=raga.arohanam,
                    avarohanam=raga.avarohanam,
                    matched_swaras=sorted(matched_swaras),
                    extra_swaras=extra_swaras_semi,
                    missing_swaras=sorted(missing_swaras),
                    is_melakarta=raga.is_melakarta,
                    melakarta_number=raga.melakarta_number,
                    parent_melakarta=raga.parent_melakarta,
                    details=f"Semitone match: {n_matched}/{n_raga}",
                ))
        
        results.sort(key=lambda m: (-m.score, m.raga_name))
        return results[:max_results]
    
    def find_raga_by_name(self, name: str) -> Optional[RagaEntry]:
        """Look up a raga by name (case-insensitive, partial match)."""
        name_lower = name.lower().strip()
        
        # Exact ID match
        if name_lower in self.ragas:
            return self.ragas[name_lower]
        
        # Name match
        for raga_id, raga in self.ragas.items():
            if raga.name.lower() == name_lower:
                return raga
        
        # Partial match
        candidates = []
        for raga_id, raga in self.ragas.items():
            if name_lower in raga.name.lower() or name_lower in raga_id:
                candidates.append(raga)
        
        if len(candidates) == 1:
            return candidates[0]
        elif candidates:
            # Return shortest name (likely the base raga, not a variant)
            return min(candidates, key=lambda r: len(r.name))
        
        return None

    def get_ragas_with_swara_count(self, count: int) -> List[RagaEntry]:
        """Get all ragas with exactly N swaras."""
        raga_ids = self._by_swara_count.get(count, [])
        return [self.ragas[rid] for rid in raga_ids]
    
    @property
    def total_ragas(self) -> int:
        return len(self.ragas)
    
    def summary(self) -> str:
        """Print summary of loaded ragas."""
        by_count = {}
        for raga in self.ragas.values():
            c = raga.swara_count
            by_count[c] = by_count.get(c, 0) + 1
        
        lines = [f"Total ragas: {self.total_ragas}"]
        for count in sorted(by_count.keys()):
            lines.append(f"  {count}-swara ragas: {by_count[count]}")
        
        return "\n".join(lines)


def format_match_result(match: RagaMatch, rank: int = 0) -> str:
    """Format a RagaMatch for display."""
    prefix = f"#{rank} " if rank else ""
    lines = [
        f"{prefix}{match.raga_name} — Score: {match.score:.3f} ({match.match_type})",
        f"  Arohanam:  {' '.join(match.arohanam)}",
        f"  Avarohanam: {' '.join(match.avarohanam)}",
    ]
    if match.extra_swaras:
        lines.append(f"  Extra notes:   {', '.join(match.extra_swaras)}")
    if match.missing_swaras:
        lines.append(f"  Missing notes: {', '.join(match.missing_swaras)}")
    if match.is_melakarta:
        lines.append(f"  Melakarta #{match.melakarta_number}")
    if match.parent_melakarta:
        lines.append(f"  Parent melakarta: {match.parent_melakarta}")
    return "\n".join(lines)


# Convenience functions

def identify_raga(swaras: List[str], direction: str = 'ascending', 
                  max_results: int = 10) -> List[RagaMatch]:
    """
    Quick function to identify ragas from a list of swaras.
    
    Example:
        >>> results = identify_raga(['S', 'R2', 'G3', 'P', 'D2', 'S'])
        >>> print(results[0].raga_name)  # 'Mohanam'
    """
    matcher = SwaraSequenceMatcher()
    return matcher.match_swaras(swaras, direction=direction, max_results=max_results)


def identify_raga_from_semitones(semitones: List[int], 
                                  max_results: int = 10) -> List[RagaMatch]:
    """
    Quick function to identify ragas from semitone offsets.
    
    Example:
        >>> results = identify_raga_from_semitones([0, 2, 4, 7, 9])  # Mohanam
        >>> print(results[0].raga_name)  # 'Mohanam'
    """
    matcher = SwaraSequenceMatcher()
    return matcher.match_swaras_by_semitones(semitones, max_results=max_results)


if __name__ == '__main__':
    print("Loading raga database...")
    matcher = SwaraSequenceMatcher()
    print(matcher.summary())
    
    print("\n" + "=" * 60)
    print("Test: Identify Mohanam (S R2 G3 P D2 S)")
    print("=" * 60)
    results = matcher.match_swaras(['S', 'R2', 'G3', 'P', 'D2', 'S'], direction='ascending')
    for i, m in enumerate(results[:10], 1):
        print(format_match_result(m, rank=i))
        print()
    
    print("=" * 60)
    print("Test: Identify Kalyani (S R2 G3 M2 P D2 N3 S)")
    print("=" * 60)
    results = matcher.match_swaras(['S', 'R2', 'G3', 'M2', 'P', 'D2', 'N3', 'S'], direction='ascending')
    for i, m in enumerate(results[:10], 1):
        print(format_match_result(m, rank=i))
        print()
    
    print("=" * 60)
    print("Test: Identify Bahudari (S G3 M1 P D2 N2 S)")
    print("=" * 60)
    results = matcher.match_swaras(['S', 'G3', 'M1', 'P', 'D2', 'N2', 'S'], direction='ascending')
    for i, m in enumerate(results[:10], 1):
        print(format_match_result(m, rank=i))
        print()
    
    print("=" * 60)
    print("Test: Identify Shankarabharanam (S R2 G3 M1 P D2 N3 S)")
    print("=" * 60)
    results = matcher.match_swaras(['S', 'R2', 'G3', 'M1', 'P', 'D2', 'N3', 'S'], direction='ascending')
    for i, m in enumerate(results[:10], 1):
        print(format_match_result(m, rank=i))
        print()
    
    print("=" * 60)
    print("Test: Identify Hamsadhwani (S R2 G3 P N3 S)")
    print("=" * 60)
    results = matcher.match_swaras(['S', 'R2', 'G3', 'P', 'N3', 'S'], direction='ascending')
    for i, m in enumerate(results[:10], 1):
        print(format_match_result(m, rank=i))
        print()
    
    # Interactive mode
    print("\n" + "=" * 60)
    print("Interactive Mode - Enter swaras separated by spaces")
    print("Example: S R2 G3 P D2 S")
    print("Type 'quit' to exit")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\nEnter swaras: ").strip()
            if user_input.lower() in ('quit', 'exit', 'q'):
                break
            
            swaras = user_input.split()
            if not swaras:
                continue
            
            results = matcher.match_swaras(swaras, direction='ascending')
            if results:
                print(f"\nTop matches ({len(results)} found):")
                for i, m in enumerate(results[:10], 1):
                    print(format_match_result(m, rank=i))
                    print()
            else:
                print("No matching ragas found.")
        except (EOFError, KeyboardInterrupt):
            break
    
    print("Done.")
