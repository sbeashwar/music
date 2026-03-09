"""
Swara Sequence Matcher - Match detected note sequences to ragas.

Given a sequence of detected swaras (e.g., ['S', 'R2', 'G3', 'P', 'D2', 'S']),
find matching ragas from the database by comparing arohanam/avarohanam patterns.
"""

import os
import json
import math
import pickle
import hashlib
from collections import Counter
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


# ── Canonical 72 Melakarta ragas (Wikipedia / Govindacharya scheme) ──
# Each: (number, name, arohanam_semitones)
# All are sampurna: 7 unique swaras, same ascending & descending, include S and P.
# Semitone tuple: (R, G, M, D, N) — S=0 and P=7 are always present.
MELAKARTA_72 = {
    # Chakra 1: Indu (R1 G1 M1)
    1:  ('Kanakāngi',           (1, 2, 5, 8, 9)),
    2:  ('Ratnāngi',            (1, 2, 5, 8, 10)),
    3:  ('Gānamūrti',           (1, 2, 5, 8, 11)),
    4:  ('Vanaspati',           (1, 2, 5, 9, 10)),
    5:  ('Mānavati',            (1, 2, 5, 9, 11)),
    6:  ('Tānarūpi',           (1, 2, 5, 10, 11)),
    # Chakra 2: Netra (R1 G2 M1)
    7:  ('Senāvati',            (1, 3, 5, 8, 9)),
    8:  ('Hanumatodi',          (1, 3, 5, 8, 10)),
    9:  ('Dhenukā',             (1, 3, 5, 8, 11)),
    10: ('Nātakapriyā',         (1, 3, 5, 9, 10)),
    11: ('Kokilapriyā',         (1, 3, 5, 9, 11)),
    12: ('Rūpavati',            (1, 3, 5, 10, 11)),
    # Chakra 3: Agni (R1 G3 M1)
    13: ('Gāyakapriyā',         (1, 4, 5, 8, 9)),
    14: ('Vakulābharanam',       (1, 4, 5, 8, 10)),
    15: ('Māyāmālavagowla',     (1, 4, 5, 8, 11)),
    16: ('Chakravākam',          (1, 4, 5, 9, 10)),
    17: ('Sūryakāntam',         (1, 4, 5, 9, 11)),
    18: ('Hātakāmbari',          (1, 4, 5, 10, 11)),
    # Chakra 4: Veda (R2 G2 M1)
    19: ('Jhankāradhvani',       (2, 3, 5, 8, 9)),
    20: ('Natabhairavi',         (2, 3, 5, 8, 10)),
    21: ('Kīravāni',            (2, 3, 5, 8, 11)),
    22: ('Kharaharapriyā',       (2, 3, 5, 9, 10)),
    23: ('Gourimanohari',        (2, 3, 5, 9, 11)),
    24: ('Varunapriyā',          (2, 3, 5, 10, 11)),
    # Chakra 5: Bana (R2 G3 M1)
    25: ('Māraranjani',          (2, 4, 5, 8, 9)),
    26: ('Chārukesi',            (2, 4, 5, 8, 10)),
    27: ('Sarasāngi',            (2, 4, 5, 8, 11)),
    28: ('Harikāmbhoji',         (2, 4, 5, 9, 10)),
    29: ('Dhīrasankarābharanam', (2, 4, 5, 9, 11)),
    30: ('Nāganandini',          (2, 4, 5, 10, 11)),
    # Chakra 6: Rutu (R3 G3 M1)
    31: ('Yāgapriyā',           (3, 4, 5, 8, 9)),
    32: ('Rāgavardhini',        (3, 4, 5, 8, 10)),
    33: ('Gāngeyabhushani',     (3, 4, 5, 8, 11)),
    34: ('Vāgadhīsvari',        (3, 4, 5, 9, 10)),
    35: ('Shūlini',             (3, 4, 5, 9, 11)),
    36: ('Chalanāta',            (3, 4, 5, 10, 11)),
    # Chakra 7: Rishi (R1 G1 M2)
    37: ('Sālagam',             (1, 2, 6, 8, 9)),
    38: ('Jalārnavam',           (1, 2, 6, 8, 10)),
    39: ('Jhālavarāli',         (1, 2, 6, 8, 11)),
    40: ('Navanītam',            (1, 2, 6, 9, 10)),
    41: ('Pāvani',              (1, 2, 6, 9, 11)),
    42: ('Raghupriyā',           (1, 2, 6, 10, 11)),
    # Chakra 8: Vasu (R1 G2 M2)
    43: ('Gavāmbhodi',           (1, 3, 6, 8, 9)),
    44: ('Bhavapriyā',           (1, 3, 6, 8, 10)),
    45: ('Shubhapantuvarāli',    (1, 3, 6, 8, 11)),
    46: ('Shadvidamārgini',      (1, 3, 6, 9, 10)),
    47: ('Suvarnāngi',           (1, 3, 6, 9, 11)),
    48: ('Divyamani',            (1, 3, 6, 10, 11)),
    # Chakra 9: Brahma (R1 G3 M2)
    49: ('Dhavalāmbari',         (1, 4, 6, 8, 9)),
    50: ('Nāmanārāyani',         (1, 4, 6, 8, 10)),
    51: ('Kāmavardhini',         (1, 4, 6, 8, 11)),
    52: ('Rāmapriyā',            (1, 4, 6, 9, 10)),
    53: ('Gamanāshrama',          (1, 4, 6, 9, 11)),
    54: ('Vishvambari',           (1, 4, 6, 10, 11)),
    # Chakra 10: Disi (R2 G2 M2)
    55: ('Shāmalāngi',           (2, 3, 6, 8, 9)),
    56: ('Shanmukhapriyā',       (2, 3, 6, 8, 10)),
    57: ('Simhendramadhyamam',   (2, 3, 6, 8, 11)),
    58: ('Hemavati',             (2, 3, 6, 9, 10)),
    59: ('Dharmavati',           (2, 3, 6, 9, 11)),
    60: ('Nītimati',             (2, 3, 6, 10, 11)),
    # Chakra 11: Rudra (R2 G3 M2)
    61: ('Kāntāmani',           (2, 4, 6, 8, 9)),
    62: ('Rishabhapriyā',        (2, 4, 6, 8, 10)),
    63: ('Latāngi',              (2, 4, 6, 8, 11)),
    64: ('Vāchaspati',           (2, 4, 6, 9, 10)),
    65: ('Mechakalyāni',         (2, 4, 6, 9, 11)),
    66: ('Chitrāmbari',          (2, 4, 6, 10, 11)),
    # Chakra 12: Aditya (R3 G3 M2)
    67: ('Sucharitra',           (3, 4, 6, 8, 9)),
    68: ('Jyotisvarūpini',      (3, 4, 6, 8, 10)),
    69: ('Dhātuvardhani',       (3, 4, 6, 8, 11)),
    70: ('Nāsikābhūshani',     (3, 4, 6, 9, 10)),
    71: ('Kōsalam',             (3, 4, 6, 9, 11)),
    72: ('Rasikapriyā',         (3, 4, 6, 10, 11)),
}

# Build reverse lookup: frozenset of all 7 semitones (incl S=0, P=7) → melakarta number
_MELAKARTA_BY_SEMITONES: Dict[frozenset, int] = {}
for _num, (_name, _rgdn) in MELAKARTA_72.items():
    _full = frozenset((0, _rgdn[0], _rgdn[1], _rgdn[2], 7, _rgdn[3], _rgdn[4]))
    _MELAKARTA_BY_SEMITONES[_full] = _num


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
        self.popularity: Dict[str, int] = {}  # raga_id -> composition count
        
        # Indices for fast lookup
        self._by_arohanam_key: Dict[str, List[str]] = {}  # frozenset key -> raga_ids
        self._by_swara_count: Dict[int, List[str]] = {}   # count -> raga_ids
        self._by_semitone_set: Dict[tuple, List[str]] = {} # sorted semitones -> raga_ids
        
        self._load_ragas()
        self._load_popularity()
    
    def _load_popularity(self):
        """Load raga popularity data (composition counts from karnatik.com)."""
        pop_path = Path(__file__).parent / 'raga_popularity.json'
        if pop_path.exists():
            try:
                with open(pop_path, 'r', encoding='utf-8') as f:
                    self.popularity = json.load(f)
                print(f"SwaraSequenceMatcher: Loaded popularity for {len(self.popularity)} ragas")
            except Exception:
                self.popularity = {}
    
    def _popularity_score(self, raga_id: str) -> float:
        """Return a normalized popularity bonus (0.0 to ~0.05).
        
        Top ragas (~200+ compositions) get the full bonus.
        Unknown ragas get 0.
        """
        count = self.popularity.get(raga_id, 0)
        if count == 0:
            return 0.0
        # Log scale: log(count)/log(max_count) * 0.05
        # Rough: 200+ → 0.05, 50 → 0.04, 10 → 0.03, 1 → 0.01
        return min(0.05, math.log1p(count) / math.log1p(300) * 0.05)
    
    def _get_cache_path(self) -> Path:
        """Get path for the compiled raga cache file."""
        return Path(self.metadata_dir) / '.raga_cache.pkl'
    
    def _compute_dir_fingerprint(self) -> str:
        """Compute a fingerprint of the metadata directory for cache invalidation.
        
        Uses file count + total size + newest mtime as a fast proxy.
        """
        json_files = sorted(f for f in os.listdir(self.metadata_dir) if f.endswith('.json'))
        if not json_files:
            return 'empty'
        
        total_size = 0
        newest_mtime = 0.0
        for f in json_files:
            st = os.stat(os.path.join(self.metadata_dir, f))
            total_size += st.st_size
            newest_mtime = max(newest_mtime, st.st_mtime)
        
        key = f"{len(json_files)}:{total_size}:{newest_mtime:.6f}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def _load_ragas(self):
        """Load all raga definitions, using a cache file when available."""
        if not os.path.exists(self.metadata_dir):
            raise FileNotFoundError(f"Metadata directory not found: {self.metadata_dir}")
        
        cache_path = self._get_cache_path()
        fingerprint = self._compute_dir_fingerprint()
        
        # Try loading from cache
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    cached = pickle.load(f)
                if cached.get('fingerprint') == fingerprint:
                    self.ragas = cached['ragas']
                    self._by_arohanam_key = cached['by_arohanam_key']
                    self._by_swara_count = cached['by_swara_count']
                    self._by_semitone_set = cached['by_semitone_set']
                    print(f"SwaraSequenceMatcher: Loaded {len(self.ragas)} ragas from cache")
                    return
            except Exception:
                pass  # Cache corrupt or incompatible, rebuild
        
        # Full load from individual JSON files
        self._load_ragas_from_json()
        
        # Save cache for next time
        try:
            cached = {
                'fingerprint': fingerprint,
                'ragas': self.ragas,
                'by_arohanam_key': self._by_arohanam_key,
                'by_swara_count': self._by_swara_count,
                'by_semitone_set': self._by_semitone_set,
            }
            with open(cache_path, 'wb') as f:
                pickle.dump(cached, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"SwaraSequenceMatcher: Saved cache to {cache_path}")
        except Exception as e:
            print(f"SwaraSequenceMatcher: Could not save cache: {e}")
    
    def _load_ragas_from_json(self):
        """Load all raga definitions from individual JSON files."""
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
    
    # ── Hierarchical matching (melakarta-first) ─────────────────────
    
    def _count_semitones(
        self,
        raw_sequence: Optional[List[str]],
        inner_swaras: List[str],
    ) -> Counter:
        """Build weighted semitone occurrence counts.
        
        Uses raw_sequence (with repeats) when available, otherwise
        treats each swara in inner_swaras as count = 1.
        Only counts variable swaras (excludes Sa = 0 and Pa = 7).
        """
        counts: Counter = Counter()
        if raw_sequence:
            for s in raw_sequence:
                s_upper = s.strip().upper()
                if s_upper in SWARA_TO_SEMITONE:
                    semi = SWARA_TO_SEMITONE[s_upper]
                    if semi not in (0, 7):
                        counts[semi] += 1
        else:
            for s in inner_swaras:
                semi = SWARA_TO_SEMITONE.get(s, -1)
                if semi > 0 and semi != 7:
                    counts[semi] += 1
        return counts
    
    def _rank_melakartas(self, semi_counts: Counter) -> List[Tuple[int, float]]:
        """Score and rank the 72 canonical melakartas against detected swaras.
        
        Score = fraction of detected-swara *weight* covered by the melakarta.
        
        Before scoring, semitones that appear only once while a ±1 neighbor
        appears ≥3× more are treated as slide artifacts and down-weighted.
        
        Returns:
            List of (melakarta_number, score) sorted by score descending.
        """
        # De-noise: down-weight semitones that are likely slide artifacts
        cleaned = Counter(semi_counts)
        for semi in list(semi_counts.keys()):
            cnt = semi_counts[semi]
            if cnt > 1:
                continue
            for delta in (-1, 1):
                nbr = semi + delta
                if nbr in semi_counts and semi_counts[nbr] >= cnt * 3:
                    # Very likely a slide artifact — remove from scoring
                    del cleaned[semi]
                    break
        
        total_weight = sum(cleaned.values())
        if total_weight == 0:
            return []
        
        rankings = []
        for mnum, (mname, m_semis) in MELAKARTA_72.items():
            m_set = set(m_semis)
            covered = sum(cnt for semi, cnt in cleaned.items()
                         if semi in m_set)
            score = covered / total_weight
            rankings.append((mnum, score))
        
        rankings.sort(key=lambda x: -x[1])
        return rankings
    
    def _is_vakra(self, raga: RagaEntry) -> bool:
        """Check if a raga has vakra (zigzag) arohanam or avarohanam."""
        aro = raga.arohanam_semitones
        if len(aro) >= 2:
            for i in range(1, len(aro)):
                if aro[i] <= aro[i - 1]:
                    return True
        ava = raga.avarohanam_semitones
        if len(ava) >= 2:
            for i in range(1, len(ava)):
                if ava[i] >= ava[i - 1]:
                    return True
        return False
    
    def match_swaras_hierarchical(
        self,
        detected_swaras: List[str],
        direction: str = 'ascending',
        max_results: int = 20,
        min_score: float = 0.3,
        raw_sequence: Optional[List[str]] = None,
    ) -> List[RagaMatch]:
        """
        Hierarchical raga matching following Carnatic music theory:
        
        1. Match detected swaras to closest melakarta (weighted by occurrence)
        2. Find janya ragas of that melakarta
        3. Rank: non-vakra exact > non-vakra subset > vakra > anya-swara
        
        This avoids flat comparison of all 5,000+ ragas and instead uses the
        melakarta system to narrow candidates naturally, mirroring how a
        musician would reason about raga identification.
        
        Falls back to flat match_swaras() if no melakarta scores well.
        """
        swaras = self._normalize_swaras(detected_swaras)
        if not swaras:
            return []
        
        inner_swaras = [s for s in swaras if s != 'S']
        if not inner_swaras:
            return []
        
        detected_semitones = set(SWARA_TO_SEMITONE[s] for s in inner_swaras)
        detected_var = set(s for s in detected_semitones if s not in (0, 7))
        
        # Step 1: Build weighted semitone counts from raw sequence
        semi_counts = self._count_semitones(raw_sequence, inner_swaras)
        total_weight = sum(semi_counts.values())
        if total_weight == 0:
            return self.match_swaras(
                detected_swaras, direction, max_results, min_score, raw_sequence)
        
        # De-noise detected_var: remove semitones that appear only once
        # when a ±1 neighbor appears 3× more (slide/gamaka artifact)
        denoised_var = set(detected_var)
        for semi in list(denoised_var):
            cnt = semi_counts.get(semi, 0)
            if cnt > 1:
                continue
            for delta in (-1, 1):
                nbr = semi + delta
                if nbr in semi_counts and semi_counts[nbr] >= max(3, cnt * 3):
                    denoised_var.discard(semi)
                    break
        
        # Step 2: Rank melakartas by how well they cover detected swaras
        melakarta_rankings = self._rank_melakartas(semi_counts)
        if not melakarta_rankings:
            return self.match_swaras(
                detected_swaras, direction, max_results, min_score, raw_sequence)
        
        best_mel_score = melakarta_rankings[0][1]
        # Consider melakartas scoring at least 75% of best
        top_melakartas = [
            (m, s) for m, s in melakarta_rankings
            if s >= best_mel_score * 0.75 and s > 0.3
        ]
        
        if not top_melakartas:
            return self.match_swaras(
                detected_swaras, direction, max_results, min_score, raw_sequence)
        
        # Log top melakartas for debugging
        top3 = [(MELAKARTA_72[m][0], m, f"{s:.0%}")
                for m, s in top_melakartas[:3]]
        print(f"  Top melakartas: {top3}")
        
        # Step 3: Gather and score candidate ragas from top families
        asc_semis, desc_semis = self._split_asc_desc(raw_sequence)
        results = []
        seen: Set[str] = set()
        
        for family_rank, (mnum, mel_score) in enumerate(top_melakartas[:5]):
            mname, m_variable_semis = MELAKARTA_72[mnum]
            m_var_set = set(m_variable_semis)
            
            for raga_id, raga in self.ragas.items():
                if raga_id in seen:
                    continue
                
                # Raga's variable semitones (exclude Sa and Pa)
                raga_var_semis: Set[int] = set()
                for s in raga.all_swaras_set:
                    semi = SWARA_TO_SEMITONE[s]
                    if semi not in (0, 7):
                        raga_var_semis.add(semi)
                
                # Classify relationship to this melakarta
                anya_semis = raga_var_semis - m_var_set
                if len(anya_semis) > 1:
                    continue  # Too distant from this melakarta family
                has_anya = len(anya_semis) == 1
                
                # Score raga against denoised detected swaras
                matched = denoised_var & raga_var_semis
                extra = denoised_var - raga_var_semis
                missing = raga_var_semis - denoised_var
                
                n_matched = len(matched)
                if n_matched == 0:
                    continue
                
                # Coverage: fraction of detected weight in this raga
                matched_weight = sum(semi_counts.get(s, 0) for s in matched)
                coverage = matched_weight / total_weight
                
                # Penalty for detected swaras NOT in the raga
                extra_weight = sum(semi_counts.get(s, 0) for s in extra)
                extra_penalty = (extra_weight / total_weight) * 0.30
                
                # Penalty for raga swaras we didn't detect
                miss_penalty = len(missing) * 0.06
                
                score = coverage - extra_penalty - miss_penalty
                
                # Tier bonuses based on raga classification
                is_vakra = self._is_vakra(raga)
                is_exact_mel = (raga_var_semis == m_var_set)
                
                if raga.is_melakarta:
                    # The canonical melakarta always gets top tier + bonus,
                    # regardless of vakra phrasing in the DB avarohanam
                    score += 0.10
                elif is_exact_mel and not is_vakra:
                    score += 0.08   # Sampurna non-vakra (melakarta-equivalent)
                elif is_exact_mel and is_vakra:
                    score += 0.05   # Sampurna but vakra ordering
                elif not has_anya and not is_vakra:
                    score += 0.04   # Clean non-vakra janya
                elif not has_anya and is_vakra:
                    score += 0.02   # Vakra janya (still in family)
                elif has_anya:
                    score -= 0.03   # Anya-swara raga
                
                # Bonus for perfect match (no missing, no extra swaras)
                if not extra and not missing:
                    score += 0.015
                
                # Popularity bonus: well-known ragas preferred over obscure ones
                score += self._popularity_score(raga_id)
                
                # Small penalty for less-likely melakarta families
                score -= family_rank * 0.02
                
                # Asymmetric matching: reward ragas whose arohanam/avarohanam
                # structure matches which swaras appear in each direction
                if asc_semis is not None and desc_semis is not None:
                    asym = self._asymmetric_score(asc_semis, desc_semis, raga)
                    score = score * 0.65 + asym * 0.35
                
                if score < min_score:
                    continue
                
                seen.add(raga_id)
                
                # Classify match type
                if not extra and not missing:
                    match_type = 'exact'
                elif not extra:
                    match_type = 'superset'
                elif not missing:
                    match_type = 'subset'
                else:
                    match_type = 'partial'
                
                matched_swaras = [s for s in raga.all_swaras_set
                                  if SWARA_TO_SEMITONE[s] in matched]
                extra_swaras = [str(s) for s in sorted(extra)]
                missing_swaras = [s for s in raga.all_swaras_set
                                  if SWARA_TO_SEMITONE[s] in missing]
                
                mel_label = f"#{mnum} {mname}"
                tier = ("anya" if has_anya
                        else "vakra" if is_vakra
                        else "non-vakra")
                
                results.append(RagaMatch(
                    raga_id=raga_id,
                    raga_name=raga.name,
                    score=score,
                    match_type=match_type,
                    arohanam=raga.arohanam,
                    avarohanam=raga.avarohanam,
                    matched_swaras=sorted(matched_swaras),
                    extra_swaras=extra_swaras,
                    missing_swaras=sorted(missing_swaras),
                    is_melakarta=raga.is_melakarta,
                    melakarta_number=raga.melakarta_number,
                    parent_melakarta=raga.parent_melakarta or mel_label,
                    details=(f"Melakarta {mel_label} | {tier} "
                             f"| coverage={coverage:.0%}"
                             f"{f' | {self.popularity.get(raga_id, 0)} compositions' if self.popularity.get(raga_id, 0) else ''}"),
                ))
        
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
    
    # Use hierarchical matching for all tests
    def run_test(label, swaras, direction='ascending', raw_seq=None):
        print("\n" + "=" * 60)
        print(f"Test: {label}")
        print("=" * 60)
        results = matcher.match_swaras_hierarchical(
            swaras, direction=direction, max_results=20, raw_sequence=raw_seq)
        for i, m in enumerate(results[:5], 1):
            print(format_match_result(m, rank=i))
            print()
        # Check #1 result
        if results:
            print(f"  >>> Top match: {results[0].raga_name}")
        return results
    
    run_test("Mohanam (S R2 G3 P D2 S)",
             ['S', 'R2', 'G3', 'P', 'D2', 'S'])
    
    run_test("Kalyani (S R2 G3 M2 P D2 N3 S)",
             ['S', 'R2', 'G3', 'M2', 'P', 'D2', 'N3', 'S'])
    
    run_test("Bahudari (S G3 M1 P D2 N2 S)",
             ['S', 'G3', 'M1', 'P', 'D2', 'N2', 'S'])
    
    run_test("Shankarabharanam (S R2 G3 M1 P D2 N3 S)",
             ['S', 'R2', 'G3', 'M1', 'P', 'D2', 'N3', 'S'])
    
    run_test("Hamsadhwani (S R2 G3 P N3 S)",
             ['S', 'R2', 'G3', 'P', 'N3', 'S'])
    
    run_test("Kharaharapriya (S R2 G2 M1 P D2 N2 S)",
             ['S', 'R2', 'G2', 'M1', 'P', 'D2', 'N2', 'S'])
    
    run_test("Todi / Shubhapantuvarali (S R1 G2 M2 P D1 N3 S)",
             ['S', 'R1', 'G2', 'M2', 'P', 'D1', 'N3', 'S'])
    
    # Saraswati with spurious R1 (real test case from voice detection)
    run_test("Saraswati voice (S R2 M2 P D2 N2 + spurious R1)",
             ['S', 'R2', 'M2', 'P', 'D2', 'N2', 'R1', 'S'],
             direction='both',
             raw_seq=['S', 'R2', 'R2', 'M2', 'M2', 'P', 'P', 'D2', 'D2', 'S',
                      'S', 'N2', 'N2', 'D2', 'P', 'M2', 'M2', 'R1', 'R2', 'R2', 'S'])
    
    # Saraswati clean (no noise) — should match easily
    run_test("Saraswati clean (S R2 M2 P D2 / S N2 D2 P M2 R2)",
             ['S', 'R2', 'M2', 'P', 'D2', 'N2', 'S'],
             direction='both',
             raw_seq=['S', 'R2', 'M2', 'P', 'D2', 'S',
                      'S', 'N2', 'D2', 'P', 'M2', 'R2', 'S'])
    
    print("\nDone.")
