#!/usr/bin/env python3
"""Generate a few keep-able alapana samples and report their gamaka mix."""
import os
import random
import sys
import zlib
from collections import Counter
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from carnatic.generator import RagaGenerator          # noqa: E402
from raga_detection.swara_matcher import SwaraSequenceMatcher  # noqa: E402
from eval.gamaka_synth import synthesize              # noqa: E402

RAGAS = ['shankarabharanam', 'kalyani', 'thodi', 'mohanam',
         'hamsadhwani', 'kambhoji']
OUT = ROOT / 'eval' / 'test_audio' / 'gamaka'
OUT.mkdir(parents=True, exist_ok=True)

gen = RagaGenerator()
matcher = SwaraSequenceMatcher()

print(f'{"raga":<18}{"notes":>6}{"plain":>7}{"ornamented":>12}   gamaka breakdown')
print('-' * 78)
for name in RAGAS:
    e = matcher.find_raga_by_name(name)
    random.seed(zlib.crc32(name.encode()))
    notes = gen.generate(e.name, duration_beats=48, style='alapana')
    gc = Counter(n.gamaka.value for n in notes)
    plain = gc.get('plain', 0)
    orn = len(notes) - plain
    audio = synthesize(notes, tonic_hz=261.63, tempo=64, sample_rate=22050,
                       seed=7, drone=True)  # listening aid; NOT used for detection
    path = OUT / f'{e.id}_alapana.wav'
    sf.write(str(path), audio, 22050)
    dur = len(audio) / 22050
    brk = ', '.join(f'{k}:{v}' for k, v in gc.most_common())
    print(f'{e.name:<18}{len(notes):>6}{plain:>7}{orn:>12}   {brk}')
    print(f'    {dur:4.1f}s  {path}')
    print(f'    swaras: ' + ' '.join(n.swara for n in notes[:40]))
