"""
Carnatic Music Patterns and Ornaments

This module defines:
- Gamakas (ornaments): kampita, jaru, nokku, odukkal, etc.
- Varisais (exercises): sarali, janta, dhatu, alankarams
- Prayogams: raga-specific characteristic phrases
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum


class GamakaType(Enum):
    """Types of gamakas (ornaments) in Carnatic music."""
    PLAIN = "plain"           # No ornament
    KAMPITA = "kampita"       # Oscillation/shake on the note
    JARU = "jaru"             # Slide from one note to another
    NOKKU = "nokku"           # Stress/push on the note
    ODUKKAL = "odukkal"       # Quick touch and release
    ORIKAI = "orikai"         # Turn/gruppetto
    RAVAI = "ravai"           # Fast shake
    SPHURITAM = "sphuritam"   # Repeated touch of lower note


@dataclass
class GamakaNote:
    """A note with gamaka information."""
    swara: str
    duration: float
    octave: int = 1
    gamaka: GamakaType = GamakaType.PLAIN
    # For slides/oscillations
    target_swara: Optional[str] = None  # For jaru (slide to)
    oscillation_range: int = 1  # Semitones for kampita
    velocity: int = 90


# Standard raga prayogams (signature phrases)
# These are the characteristic phrases that define a raga's identity
RAGA_PRAYOGAMS: Dict[str, List[List[str]]] = {
    "kalyani": [
        ["S", "R2", "G3", "R2", "S"],  # Classic opening
        ["G3", "R2", "S", "N3", "S"],  # Touching lower N3
        ["P", "M2", "G3", "M2", "P", "D2"],  # Mid-range phrase
        ["D2", "N3", "S", "N3", "D2", "P"],  # Upper phrase
        ["G3", "M2", "P", "D2", "P", "M2", "G3"],  # Ascending-descending
        ["N3", "D2", "P", "M2", "G3", "R2", "S"],  # Full descent
        ["M2", "G3", "M2", "P", "M2", "G3", "R2"],  # M2 emphasis (prati madhyama)
    ],
    "mohanam": [
        ["S", "R2", "G3", "R2", "S"],
        ["G3", "P", "G3", "R2", "S"],
        ["P", "D2", "P", "G3", "R2"],
        ["D2", "S", "D2", "P", "G3"],  # Upper Sa touch
        ["G3", "R2", "S", "D2", "S"],  # Lower D2 touch
        ["R2", "G3", "P", "D2", "S"],  # Full ascent
    ],
    "shankarabharanam": [
        ["S", "R2", "G3", "R2", "S"],
        ["G3", "M1", "P", "M1", "G3"],
        ["P", "D2", "N3", "D2", "P"],
        ["N3", "S", "N3", "D2", "P"],
        ["M1", "G3", "R2", "S", "N3", "S"],  # Characteristic descent
        ["R2", "G3", "M1", "P", "D2", "N3", "S"],
    ],
    "bhairavi": [
        ["S", "R1", "G2", "R1", "S"],
        ["G2", "M1", "P", "D1", "P"],
        ["D1", "N2", "S", "N2", "D1"],
        ["M1", "G2", "R1", "S", "N2", "S"],
        ["P", "D1", "N2", "D1", "P", "M1"],
        ["G2", "M1", "D1", "P", "M1", "G2"],  # Vakra phrase
    ],
    "kambhoji": [
        ["S", "R2", "G3", "M1", "P"],
        ["P", "D2", "P", "M1", "G3"],
        ["D2", "S", "N2", "D2", "P"],  # N2 in descent only
        ["G3", "M1", "P", "D2", "S"],
        ["M1", "G3", "R2", "S", "N2", "S"],
    ],
    "todi": [
        ["S", "R1", "G2", "R1", "S"],
        ["G2", "M1", "P", "D1", "N2", "D1"],
        ["D1", "N2", "S", "N2", "D1", "P"],
        ["M1", "G2", "R1", "S", "N2", "D1", "P"],
        ["P", "D1", "N2", "D1", "P", "M1", "G2"],
    ],
    "sankarabharanam": [  # Alternate spelling
        ["S", "R2", "G3", "R2", "S"],
        ["G3", "M1", "P", "M1", "G3"],
        ["P", "D2", "N3", "D2", "P"],
        ["N3", "S", "N3", "D2", "P"],
        ["M1", "G3", "R2", "S", "N3", "S"],
    ],
    "hamsadhwani": [
        ["S", "R2", "G3", "P", "N3", "S"],
        ["N3", "P", "G3", "R2", "S"],
        ["G3", "P", "N3", "P", "G3"],
        ["P", "N3", "S", "N3", "P"],
        ["R2", "G3", "P", "G3", "R2", "S"],
    ],
    "mayamalavagowla": [
        ["S", "R1", "G3", "R1", "S"],
        ["G3", "M1", "P", "D1", "N3", "D1", "P"],
        ["D1", "N3", "S", "N3", "D1"],
        ["M1", "G3", "R1", "S"],
        ["P", "M1", "G3", "R1", "G3", "M1", "P"],
    ],
}


def generate_sarali_pattern(scale: List[str], pattern_type: int = 1) -> List[str]:
    """
    Generate sarali varisai patterns.
    
    Types:
    1: Simple ascending/descending (S R G M P D N S | S N D P M G R S)
    2: Groups of 3 (S R G, R G M, G M P, ...)
    3: Groups of 4 (S R G M, R G M P, ...)
    """
    result = []
    n = len(scale)
    
    if pattern_type == 1:
        # Simple ascending
        result = scale[:]
    elif pattern_type == 2:
        # Groups of 3
        for i in range(n - 2):
            result.extend(scale[i:i+3])
    elif pattern_type == 3:
        # Groups of 4
        for i in range(n - 3):
            result.extend(scale[i:i+4])
    
    return result


def generate_janta_pattern(scale: List[str]) -> List[str]:
    """
    Generate janta varisai - doubled notes.
    SS RR GG MM PP DD NN SS
    """
    result = []
    for swara in scale:
        result.extend([swara, swara])
    return result


def generate_dhatu_pattern(scale: List[str], pattern_type: int = 1) -> List[str]:
    """
    Generate dhatu varisai - interlocking/zigzag patterns.
    
    Types:
    1: SR RG GM MP PD DN NS (adjacent pairs)
    2: SG RM GP MD PN DS (skip one)
    3: SGP RMP GMD PMN DNS (triplets with skip)
    """
    result = []
    n = len(scale)
    
    if pattern_type == 1:
        # Adjacent pairs: SR RG GM...
        for i in range(n - 1):
            result.extend([scale[i], scale[i+1]])
    elif pattern_type == 2:
        # Skip one: SG RM GP...
        for i in range(n - 2):
            result.extend([scale[i], scale[i+2]])
    elif pattern_type == 3:
        # Triplets with skip: SGP RMP...
        for i in range(n - 2):
            result.extend([scale[i], scale[i+1], scale[i+2]])
    
    return result


def generate_alankaram(scale: List[str], alankaram_type: int = 1) -> List[str]:
    """
    Generate alankaram patterns - structured melodic exercises.
    
    Types:
    1: Ta-ka pattern: SRS, RGR, GMG, MPM...
    2: Ta-ka-di-mi: SRGS, RGMR, GMPG...
    3: Ta-ka-ta-ki-ta: SRGRS, RGMGR, GMPGM...
    """
    result = []
    n = len(scale)
    
    if alankaram_type == 1:
        # Ta-ka: SRS, RGR, GMG...
        for i in range(n - 1):
            result.extend([scale[i], scale[i+1], scale[i]])
    elif alankaram_type == 2:
        # Ta-ka-di-mi: SRGS, RGMR...
        for i in range(n - 2):
            result.extend([scale[i], scale[i+1], scale[i+2], scale[i+1]])
    elif alankaram_type == 3:
        # Ta-ka-ta-ki-ta: SRGRS, RGMGR...
        for i in range(n - 2):
            result.extend([scale[i], scale[i+1], scale[i+2], scale[i+1], scale[i]])
    
    return result


def get_gamaka_for_swara(raga_name: str, swara: str, context: str = "plain") -> GamakaType:
    """
    Determine appropriate gamaka for a swara based on raga and context.
    
    Different ragas have different gamaka styles:
    - Todi: Heavy kampita on G2, D1
    - Bhairavi: Jaru on R1, D1
    - Kalyani: Light kampita, clean M2
    """
    raga_name = raga_name.lower()
    
    # Raga-specific gamaka rules
    gamaka_rules = {
        "todi": {
            "G2": GamakaType.KAMPITA,
            "D1": GamakaType.KAMPITA,
            "N2": GamakaType.JARU,
        },
        "bhairavi": {
            "R1": GamakaType.JARU,
            "G2": GamakaType.KAMPITA,
            "D1": GamakaType.JARU,
            "N2": GamakaType.KAMPITA,
        },
        "kalyani": {
            "G3": GamakaType.KAMPITA,
            "N3": GamakaType.KAMPITA,
            "R2": GamakaType.NOKKU,
        },
        "shankarabharanam": {
            "G3": GamakaType.KAMPITA,
            "D2": GamakaType.KAMPITA,
        },
        "mohanam": {
            "G3": GamakaType.KAMPITA,
            "D2": GamakaType.NOKKU,
        },
        "kambhoji": {
            "G3": GamakaType.KAMPITA,
            "M1": GamakaType.NOKKU,
            "D2": GamakaType.JARU,
        },
    }
    
    # Look up raga-specific rule
    if raga_name in gamaka_rules:
        if swara in gamaka_rules[raga_name]:
            return gamaka_rules[raga_name][swara]
    
    # Default: stable notes (S, P) are plain, others may have light kampita
    if swara in ("S", "P"):
        return GamakaType.PLAIN
    elif context == "long":
        return GamakaType.KAMPITA
    else:
        return GamakaType.PLAIN


def get_raga_prayogams(raga_name: str) -> List[List[str]]:
    """Get characteristic phrases for a raga."""
    raga_name = raga_name.lower()
    
    if raga_name in RAGA_PRAYOGAMS:
        return RAGA_PRAYOGAMS[raga_name]
    
    # Return empty list if no prayogams defined
    return []


def create_phrase_with_gamakas(
    swaras: List[str],
    raga_name: str,
    base_duration: float = 0.5
) -> List[GamakaNote]:
    """
    Convert a swara sequence to GamakaNotes with appropriate ornaments.
    """
    notes = []
    
    for i, swara in enumerate(swaras):
        # Determine duration - longer for phrase boundaries
        if i == 0 or i == len(swaras) - 1:
            duration = base_duration * 1.5
        else:
            duration = base_duration
        
        # Determine gamaka based on context
        context = "long" if duration > base_duration else "plain"
        gamaka = get_gamaka_for_swara(raga_name, swara, context)
        
        notes.append(GamakaNote(
            swara=swara,
            duration=duration,
            gamaka=gamaka
        ))
    
    return notes
