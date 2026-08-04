#!/usr/bin/env python3
"""
Closed-loop gamaka eval: generate -> synthesize (with gamakas) -> detect.

For a curated set of ragas spanning the classical structural categories
(sampoorna / shadava / audava, plus vakra "crooked" ragas), this script:

  1. generates a grammar-following melody with the RagaGenerator (gamakas on),
  2. renders it to *continuous-pitch* audio with gamaka_synth (real kampita/
     jaru/nokku + legato glides between notes),
  3. runs the actual production detector (ArohanamDetector + SwaraSequenceMatcher),
  4. reports the rank of the true raga, grouped by category.

This is the honest test the DESIGN doc says is missing: the audio contains the
very gamakas that the "detect the arohanam" approach assumes away.

Usage:
    py -3.13 eval/gamaka_eval.py            # 2 samples/raga, styles alapana+kriti
    py -3.13 eval/gamaka_eval.py -v         # per-recording detail
    py -3.13 eval/gamaka_eval.py --keep     # keep the rendered WAVs
"""

import argparse
import os
import random
import sys
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from carnatic.generator import RagaGenerator                     # noqa: E402
from raga_detection.arohanam_detector import ArohanamDetector    # noqa: E402
from raga_detection.swara_matcher import SwaraSequenceMatcher    # noqa: E402
from eval.gamaka_synth import synthesize                         # noqa: E402
from eval.run_eval import raga_matches                           # noqa: E402 (reuse alias logic)
from raga_detection.raga_classifier import classify_raga         # noqa: E402

# Curated ragas grouped by the CORRECT classical taxonomy:
#   sampurna = 7 swaras both ways, no deletion, no repeat (melakartas here)
#   varja    = swaras DELETED (audava/shadava, incl. asymmetric like Kambhoji)
#   vakra    = a swara is REPEATED (sanchara doubles back), e.g. Sahana, Begada
CURATED = {
    'sampurna': ['shankarabharanam', 'kalyani', 'kharaharapriya',
                 'thodi', 'harikambhoji', 'mayamalavagowla'],
    'varja':    ['mohanam', 'hamsadhwani', 'hindolam', 'madhyamavati',
                 'abhogi', 'kambhoji', 'bilahari', 'saveri', 'bahudari'],
    'vakra':    ['sahana', 'anandabhairavi', 'ritigowla', 'begada'],
}

TONIC_HZ = 261.63  # C4 — known tonic (ideal conditions for the detector)


def structural_label(aro, ava):
    """Objective classical classification via the shared classifier."""
    return classify_raga(aro, ava).label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-v', '--verbose', action='store_true')
    ap.add_argument('--samples', type=int, default=2, help='samples per raga')
    ap.add_argument('--keep', action='store_true', help='keep rendered WAVs')
    ap.add_argument('--outdir', default=str(ROOT / 'eval' / 'test_audio' / 'gamaka'))
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print('Loading generator, detector, matcher...')
    gen = RagaGenerator()
    detector = ArohanamDetector(voice_mode=True)   # voice mode = gamaka-aware settings
    matcher = SwaraSequenceMatcher()
    print(f'Loaded {len(matcher.ragas)} ragas\n')

    styles = ['alapana', 'kriti', 'tana']
    rows = []            # per-recording dicts
    cat_stats = defaultdict(lambda: {'n': 0, 't1': 0, 't5': 0, 'rr': 0.0})

    for category, ragas in CURATED.items():
        for name in ragas:
            entry = matcher.find_raga_by_name(name)
            if entry is None:
                print(f'  [skip] {name!r} not found in DB')
                continue
            struct = structural_label(entry.arohanam, entry.avarohanam)

            for i in range(args.samples):
                seed = zlib.crc32(f'{name}:{i}'.encode())   # deterministic
                random.seed(seed)                 # generator uses global random
                style = styles[i % len(styles)]
                try:
                    notes = gen.generate(entry.name, duration_beats=40, style=style)
                except Exception as e:
                    print(f'  [skip] generate {name}: {e}')
                    continue

                audio = synthesize(notes, tonic_hz=TONIC_HZ, tempo=72,
                                   sample_rate=22050, seed=seed, drone=False)
                wav = os.path.join(args.outdir, f'{entry.id}_{i}.wav')
                sf.write(wav, audio, 22050)

                res = detector.detect_from_file(wav)
                direction = res.direction if res.direction != 'mixed' else 'ascending'
                matches = matcher.match_swaras(
                    res.detected_swaras, direction=direction,
                    raw_sequence=res.raw_sequence, max_results=20)

                rank = 0
                for r, m in enumerate(matches, 1):
                    if raga_matches(m.raga_id, entry.id, [entry.name]) or \
                       raga_matches(m.raga_name, entry.name, []):
                        rank = r
                        break
                rr = 1.0 / rank if rank else 0.0
                top = matches[0].raga_name if matches else '-'

                cs = cat_stats[category]
                cs['n'] += 1
                cs['t1'] += int(rank == 1)
                cs['t5'] += int(0 < rank <= 5)
                cs['rr'] += rr

                rows.append(dict(cat=category, raga=entry.name, struct=struct,
                                 style=style, rank=rank, top=top,
                                 det=' '.join(res.detected_swaras),
                                 tonic=res.tonic_hz))
                if args.verbose:
                    tag = 'OK ' if rank == 1 else f'#{rank or "-":>2}'
                    print(f'  {tag} {entry.name:<20} [{struct:<18}] {style:<7} '
                          f'-> {top:<20} det=({" ".join(res.detected_swaras)})')

                if not args.keep:
                    os.remove(wav)

    # ---- report -------------------------------------------------------------
    print('\n' + '=' * 74)
    print('GAMAKA CLOSED-LOOP RESULTS  (generated melody -> gamaka audio -> detect)')
    print('=' * 74)
    print(f'{"category":<12}{"n":>4}{"top-1":>10}{"top-5":>10}{"MRR":>8}')
    print('-' * 74)
    tot = {'n': 0, 't1': 0, 't5': 0, 'rr': 0.0}
    for cat in CURATED:
        cs = cat_stats[cat]
        if not cs['n']:
            continue
        for k in tot:
            tot[k] += cs[k]
        print(f'{cat:<12}{cs["n"]:>4}{cs["t1"]/cs["n"]:>9.0%}{cs["t5"]/cs["n"]:>10.0%}'
              f'{cs["rr"]/cs["n"]:>8.3f}')
    print('-' * 74)
    if tot['n']:
        print(f'{"ALL":<12}{tot["n"]:>4}{tot["t1"]/tot["n"]:>9.0%}'
              f'{tot["t5"]/tot["n"]:>10.0%}{tot["rr"]/tot["n"]:>8.3f}')

    # Per-raga misses (most informative)
    print('\nMisses (true raga not #1):')
    print(f'{"raga":<20}{"structure":<20}{"style":<7}{"rank":>5}  got -> detected')
    print('-' * 74)
    for row in rows:
        if row['rank'] != 1:
            rk = row['rank'] or '—'
            print(f'{row["raga"]:<20}{row["struct"]:<20}{row["style"]:<7}{str(rk):>5}  '
                  f'{row["top"]:<18} det=({row["det"]})')


if __name__ == '__main__':
    main()
