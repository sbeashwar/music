"""
Raga Database - Core definitions and lookups

Loads raga definitions from the shared metadata and provides
efficient lookup by scale, name, or characteristics.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from functools import lru_cache


@dataclass
class Raga:
    """Represents a single raga with all its properties."""
    name: str
    arohanam: List[str]  # Ascending scale
    avarohanam: List[str]  # Descending scale
    melakarta_number: Optional[int] = None
    is_melakarta: bool = False
    vadi: Optional[str] = None  # Dominant swara
    samvadi: Optional[str] = None  # Sub-dominant swara
    phrases: List[List[str]] = field(default_factory=list)
    alternate_names: List[str] = field(default_factory=list)
    _scale_cache: Optional[Set[str]] = field(default=None, repr=False)
    _semitone_cache: Optional[frozenset] = field(default=None, repr=False)
    
    @property
    def scale(self) -> Set[str]:
        """All unique swaras used in this raga (cached)."""
        if self._scale_cache is None:
            self._scale_cache = set(self.arohanam + self.avarohanam)
        return self._scale_cache
    
    @property
    def scale_semitones(self) -> frozenset:
        """Semitone values for the scale (cached, for fast matching)."""
        if self._semitone_cache is None:
            self._semitone_cache = frozenset(
                SWARA_TO_SEMITONE.get(s, -1) for s in self.scale
            ) - {-1}
        return self._semitone_cache
    
    @property
    def scale_signature(self) -> str:
        """Canonical string representation of the scale for matching."""
        # Sort swaras in standard order for comparison
        order = ['S', 'R1', 'R2', 'R3', 'G1', 'G2', 'G3', 'M1', 'M2', 'P', 'D1', 'D2', 'D3', 'N1', 'N2', 'N3']
        present = [s for s in order if s in self.scale]
        return '-'.join(present)


# Standard swara mappings
SWARA_TO_SEMITONE = {
    'S': 0,   # Sa - tonic
    'R1': 1,  # Shuddha Rishabha
    'R2': 2,  # Chatushruti Rishabha  
    'R3': 3,  # Shatshruti Rishabha (= G1)
    'G1': 2,  # Shuddha Gandhara (= R2)
    'G2': 3,  # Sadharana Gandhara (= R3)
    'G3': 4,  # Antara Gandhara
    'M1': 5,  # Shuddha Madhyama
    'M2': 6,  # Prati Madhyama
    'P': 7,   # Panchama
    'D1': 8,  # Shuddha Dhaivata
    'D2': 9,  # Chatushruti Dhaivata
    'D3': 10, # Shatshruti Dhaivata (= N1)
    'N1': 9,  # Shuddha Nishada (= D2)
    'N2': 10, # Kaisiki Nishada (= D3)
    'N3': 11, # Kakali Nishada
}

SEMITONE_TO_SWARAS = {}
for swara, semi in SWARA_TO_SEMITONE.items():
    if semi not in SEMITONE_TO_SWARAS:
        SEMITONE_TO_SWARAS[semi] = []
    SEMITONE_TO_SWARAS[semi].append(swara)


class RagaDB:
    """
    Database of raga definitions with efficient lookup capabilities.
    """
    
    def __init__(self, metadata_dir: Optional[str] = None):
        if metadata_dir is None:
            # Default: look in shared/ragas_metadata relative to this file
            base = Path(__file__).parent.parent
            metadata_dir = base / 'shared' / 'ragas_metadata'
        
        self.metadata_dir = Path(metadata_dir)
        self.ragas: Dict[str, Raga] = {}
        self._scale_index: Dict[str, List[str]] = {}  # scale_signature -> [raga_names]
        self._loaded = False
    
    def load(self) -> 'RagaDB':
        """Load all raga definitions from JSON files."""
        if self._loaded:
            return self
        
        if not self.metadata_dir.exists():
            print(f"Warning: Metadata directory not found: {self.metadata_dir}")
            self._loaded = True
            return self
        
        for filepath in self.metadata_dir.glob('*.json'):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                raga = self._parse_raga(data)
                if raga:
                    self.ragas[raga.name.lower()] = raga
                    
                    # Index by scale signature
                    sig = raga.scale_signature
                    if sig not in self._scale_index:
                        self._scale_index[sig] = []
                    self._scale_index[sig].append(raga.name.lower())
                    
            except Exception as e:
                # Skip malformed files silently
                pass
        
        self._loaded = True
        print(f"Loaded {len(self.ragas)} ragas from {self.metadata_dir}")
        return self
    
    def _parse_raga(self, data: dict) -> Optional[Raga]:
        """Parse a raga definition from JSON data."""
        name = data.get('name') or data.get('id', '')
        if not name:
            return None
        
        arohanam = data.get('arohanam', [])
        avarohanam = data.get('avarohanam', [])
        
        # Handle string format "S R2 G3 P D2 S"
        if isinstance(arohanam, str):
            arohanam = arohanam.split()
        if isinstance(avarohanam, str):
            avarohanam = avarohanam.split()
        
        # Skip if no scale info
        if not arohanam and not avarohanam:
            return None
        
        lakshana = data.get('raga_lakshana', {})
        classification = data.get('raga_classification', {})
        
        return Raga(
            name=name,
            arohanam=arohanam,
            avarohanam=avarohanam,
            melakarta_number=data.get('melakarta_number') or classification.get('melakarta_number'),
            is_melakarta=data.get('is_melakarta', False) or classification.get('is_melakarta', False),
            vadi=lakshana.get('vadi_swara'),
            samvadi=lakshana.get('samvadi_swara'),
            phrases=data.get('phrases', []),
            alternate_names=data.get('alternate_names', [])
        )
    
    def get(self, name: str) -> Optional[Raga]:
        """Get a raga by name."""
        self.load()
        return self.ragas.get(name.lower())
    
    def find_by_scale(self, swaras: Set[str]) -> List[Tuple[Raga, float]]:
        """
        Find ragas matching a set of swaras.
        Returns list of (raga, match_score) sorted by score descending.
        """
        self.load()
        
        results = []
        for raga in self.ragas.values():
            score = self._scale_match_score(swaras, raga.scale)
            if score > 0.5:  # At least 50% match
                results.append((raga, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def _scale_match_score(self, detected: Set[str], raga_scale: Set[str]) -> float:
        """
        Calculate how well detected swaras match a raga's scale.
        
        Score considers:
        - What % of detected notes are in the raga
        - Penalty for detected notes NOT in the raga (wrong notes)
        """
        if not detected:
            return 0.0
        
        # Normalize swaras (handle enharmonic equivalents)
        detected_semitones = {SWARA_TO_SEMITONE.get(s, -1) for s in detected}
        detected_semitones.discard(-1)
        
        raga_semitones = {SWARA_TO_SEMITONE.get(s, -1) for s in raga_scale}
        raga_semitones.discard(-1)
        
        if not detected_semitones:
            return 0.0
        
        # Intersection: notes that match
        matching = detected_semitones & raga_semitones
        
        # Notes in detected but not in raga (foreign notes)
        foreign = detected_semitones - raga_semitones
        
        # Score: matching / detected - penalty for foreign notes
        match_ratio = len(matching) / len(detected_semitones)
        foreign_penalty = len(foreign) * 0.15  # Each foreign note reduces score
        
        return max(0, match_ratio - foreign_penalty)
    
    def search(self, query: str) -> List[Raga]:
        """Search ragas by name (partial match)."""
        self.load()
        query = query.lower()
        return [r for r in self.ragas.values() 
                if query in r.name.lower() or 
                any(query in alt.lower() for alt in r.alternate_names)]
    
    @property
    def melakartas(self) -> List[Raga]:
        """Get all melakarta (parent) ragas."""
        self.load()
        return [r for r in self.ragas.values() if r.is_melakarta]
    
    def __len__(self) -> int:
        self.load()
        return len(self.ragas)
    
    def __iter__(self):
        self.load()
        return iter(self.ragas.values())


# Singleton instance for convenience
_db = None

def get_db() -> RagaDB:
    """Get the shared RagaDB instance."""
    global _db
    if _db is None:
        _db = RagaDB().load()
    return _db
