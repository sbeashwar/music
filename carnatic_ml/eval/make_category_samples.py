#!/usr/bin/env python3
"""
Generate a batch of listenable samples organised by raga category.

Writes WAVs into eval/test_audio/gamaka/<category>/<raga>_<style>.wav and prints
a summary of the plain/gamaka/varisai mix so the output is auditable.

Categories (correct classical taxonomy) come from gamaka_eval.CURATED:
  sampurna / varja / vakra

Usage:
    py -3.13 eval/make_category_samples.py            # 3 styles per raga
    py -3.13 eval/make_category_samples.py --styles alapana kriti
"""
import argparse
import os
import random
import sys
import zlib
from collections import Counter
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from carnatic.generator import RagaGenerator                 # noqa: E402
from raga_detection.swara_matcher import SwaraSequenceMatcher  # noqa: E402
from raga_detection.raga_classifier import classify_raga     # noqa: E402
from eval.gamaka_synth import synthesize                     # noqa: E402
from eval.gamaka_eval import CURATED                         # noqa: E402

TONIC_HZ = 261.63  # C4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--styles', nargs='+', default=['alapana', 'kriti', 'tana'],
                    help='styles to render per raga')
    ap.add_argument('--tempo', type=int, default=68)
    ap.add_argument('--beats', type=int, default=48)
    ap.add_argument('--outroot', default=str(ROOT / 'eval' / 'test_audio' / 'gamaka'))
    args = ap.parse_args()

    gen = RagaGenerator()
    matcher = SwaraSequenceMatcher()

    total = 0
    print(f'{"category":<10}{"raga":<18}{"class":<22}{"style":<8}'
          f'{"notes":>6}{"plain%":>8}  varisai/gamaka')
    print('-' * 92)
    for category, ragas in CURATED.items():
        outdir = Path(args.outroot) / category
        outdir.mkdir(parents=True, exist_ok=True)
        for name in ragas:
            e = matcher.find_raga_by_name(name)
            if e is None:
                print(f'{category:<10}{name:<18}NOT FOUND')
                continue
            cls = classify_raga(e.arohanam, e.avarohanam).label
            for style in args.styles:
                seed = zlib.crc32(f'{name}:{style}'.encode())
                random.seed(seed)
                notes = gen.generate(e.name, duration_beats=args.beats, style=style)
                audio = synthesize(notes, tonic_hz=TONIC_HZ, tempo=args.tempo,
                                   sample_rate=22050, seed=seed,
                                   drone=True)  # listening aid; NOT used for detection
                path = outdir / f'{e.id}_{style}.wav'
                sf.write(str(path), audio, 22050)
                total += 1

                gc = Counter(n.gamaka.value for n in notes)
                plain_pct = 100.0 * gc.get('plain', 0) / max(len(notes), 1)
                orn = ', '.join(f'{k}:{v}' for k, v in gc.most_common() if k != 'plain')
                print(f'{category:<10}{e.name:<18}{cls:<22}{style:<8}'
                      f'{len(notes):>6}{plain_pct:>7.0f}%  {orn}')

    print('-' * 92)
    print(f'Wrote {total} WAVs under {args.outroot}\\<category>\\')


if __name__ == '__main__':
    main()
