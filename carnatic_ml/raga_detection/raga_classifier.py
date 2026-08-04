"""
Raga structural classifier.

Implements the classical Carnatic janya classification (see
https://www.spardhaschoolofmusic.com/blog/an-introduction-to-the-carnatic-raga-classification):

  * Varja  -> swaras are *deleted* from a scale.
               - Audava  = 5 swaras in that scale
               - Shadava = 6 swaras
               - Sampurna = 7 swaras
             A raga can differ between directions, e.g. Audava-Sampurna
             (5 up, 7 down). Any scale with < 7 swaras is varja.

  * Vakra  -> a swara is *repeated* (the sanchara doubles back on a note
             against the direction of travel). At least one repeated swara
             in arohana or avarohana makes the raga vakra.  This is
             independent of varja: Kambhoji is varja but NOT vakra;
             Sahana / Ananda Bhairavi / Reethigowla ARE vakra.

NOTE ON DATA QUALITY: some entries in shared/ragas_metadata store an
*embellished sanchara* in the avarohanam field (with repeats) rather than the
clean descending scale. Such entries will be flagged vakra spuriously. The
classifier reports which scale triggered the vakra flag so callers can audit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


JATI_BY_COUNT = {1: 'svarantara', 2: 'svarantara', 3: 'svarantara',
                 4: 'svarantara', 5: 'audava', 6: 'shadava', 7: 'sampurna'}


def _inner(scale: List[str]) -> List[str]:
    """Strip the framing octave Sa from both ends so repeats are counted fairly."""
    s = list(scale)
    if s and s[0] == 'S':
        s = s[1:]
    if s and s[-1] == 'S':
        s = s[:-1]
    return s


def _unique_swara_count(scale: List[str]) -> int:
    """Number of distinct swaras in the scale, counting Sa once."""
    return len(set(_inner(scale))) + 1  # +1 for the tonic Sa


def _has_repeat(scale: List[str]) -> bool:
    """True if any swara occurs more than once in the ordered (inner) scale."""
    inner = _inner(scale)
    return len(inner) != len(set(inner))


@dataclass
class RagaClassification:
    arohana_jati: str          # audava / shadava / sampurna
    avarohana_jati: str
    arohana_count: int
    avarohana_count: int
    is_varja: bool             # any deletion (either scale < 7 swaras)
    is_vakra: bool             # any repeated swara (either scale)
    vakra_in: List[str]        # which scale(s) carry the repeat: ['arohana', ...]

    @property
    def jati_label(self) -> str:
        if self.arohana_jati == self.avarohana_jati:
            return self.arohana_jati
        return f'{self.arohana_jati}-{self.avarohana_jati}'

    @property
    def label(self) -> str:
        tags = [self.jati_label]
        if self.is_vakra:
            tags.append('vakra')
        return ' '.join(tags)


def audit_scale(scale: List[str]) -> dict:
    """
    Heuristic check for whether a scale field holds a *clean* arohana/avarohana
    or an *embellished sanchara* (which corrupts classification & matching).

    A clean scale has <= 7 unique swaras and at most a couple of vakra repeats.
    An embellished sanchara (e.g. Shankarabharanam's DB avarohanam
    'S D2 P M1 G3 R2 S N3 P D2 N3 S') has > 7 unique swaras and/or many repeats.
    """
    inner = _inner(scale)
    uniq = len(set(inner))
    repeats = len(inner) - uniq
    suspect = uniq > 7 or repeats >= 3
    return {'unique': uniq, 'repeats': repeats, 'length': len(inner),
            'suspect': suspect}


def is_clean_scale(scale: List[str]) -> bool:
    """True if the scale looks like a genuine arohana/avarohana, not a sanchara."""
    return not audit_scale(scale)['suspect']


def classify_raga(arohanam: List[str], avarohanam: List[str]) -> RagaClassification:
    """Classify a raga from its arohana/avarohana swara sequences."""
    a_cnt = _unique_swara_count(arohanam)
    v_cnt = _unique_swara_count(avarohanam)
    vakra_in = []
    if _has_repeat(arohanam):
        vakra_in.append('arohana')
    if _has_repeat(avarohanam):
        vakra_in.append('avarohana')
    return RagaClassification(
        arohana_jati=JATI_BY_COUNT.get(a_cnt, f'{a_cnt}-swara'),
        avarohana_jati=JATI_BY_COUNT.get(v_cnt, f'{v_cnt}-swara'),
        arohana_count=a_cnt,
        avarohana_count=v_cnt,
        is_varja=(a_cnt < 7 or v_cnt < 7),
        is_vakra=bool(vakra_in),
        vakra_in=vakra_in,
    )
