#!/usr/bin/env python3
"""
Analyse a real alapana recording and quantify its musical nuances, so we can
compare them against what our generator produces.

Extracts (using the same librosa/pYIN stack as the detector):
  - tonic (Sa) estimate
  - pitch-class distribution over swara positions (where time is spent)
  - note segmentation -> dwell-time stats, longest sustained notes
  - gamaka extent: per-note pitch deviation (how much notes oscillate/slide)
  - continuous-glide fraction vs steady-note fraction
  - phrasing: silence gaps -> phrase count & phrase-length distribution
  - pitch range (octaves covered)
  - transition matrix (which swara follows which) -> melodic movement
"""
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import librosa

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

SR = 22050
SEMI_NAMES = {0: 'S', 1: 'R1', 2: 'R2/G1', 3: 'G2/R3', 4: 'G3', 5: 'M1',
              6: 'M2', 7: 'P', 8: 'D1', 9: 'D2/N1', 10: 'N2/D3', 11: 'N3'}


def load_ffmpeg(path):
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    tmp = tempfile.mktemp(suffix='.wav')
    subprocess.run([ff, '-i', str(path), '-ac', '1', '-ar', str(SR),
                    '-f', 'wav', '-y', tmp], capture_output=True, check=True)
    y, _ = librosa.load(tmp, sr=SR)
    Path(tmp).unlink(missing_ok=True)
    return y


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--tonic', type=float, default=None, help='override Sa in Hz')
    args = ap.parse_args()

    path = ROOT / 'Sample_Tracks' / 'Ragam Revati _ Nagaswaram Iyermalai B Selvam.m4a'
    cache = ROOT / 'eval' / '_revati_f0.npz'
    if cache.exists():
        print(f'Loading cached pitch track {cache.name} ...')
        z = np.load(cache)
        f0, dur = z['f0'], float(z['dur'])
    else:
        print(f'Loading {path.name} ...')
        y = load_ffmpeg(path)
        dur = len(y) / SR
        print('Running pYIN pitch tracking (this takes a bit)...')
        f0, vflag, vprob = librosa.pyin(y, fmin=110, fmax=1000, sr=SR,
                                        frame_length=2048, hop_length=256)
        np.savez(cache, f0=f0, dur=dur)
    print(f'Duration: {dur:.1f}s ({dur/60:.1f} min)\n')

    hop_s = 256 / SR
    voiced = f0[~np.isnan(f0)]
    print(f'Voiced frames: {len(voiced)}/{len(f0)} ({len(voiced)/len(f0):.0%})\n')

    # ---- tonic: strongest pitch-class peak in a folded log-f histogram ----
    cents = 1200 * np.log2(voiced)
    pc = np.mod(cents, 1200)
    hist, edges = np.histogram(pc, bins=120, range=(0, 1200))
    # smooth circularly
    k = np.array([1, 2, 3, 4, 3, 2, 1], float); k /= k.sum()
    hs = np.convolve(np.concatenate([hist[-3:], hist, hist[:3]]), k, 'same')[3:-3]
    tonic_bin = int(np.argmax(hs))
    # Revathi has S and P prominent; assume the tallest peak is Sa or Pa.
    # Use median of low stable region as a sanity anchor.
    tonic_cents_in_oct = tonic_bin * 10.0
    tonic_hz = 220.0 * 2 ** (tonic_cents_in_oct / 1200.0)
    # normalise into a sensible Sa range 120-320 Hz
    while tonic_hz > 320: tonic_hz /= 2
    while tonic_hz < 120: tonic_hz *= 2
    if args.tonic:
        print(f'Auto tonic peak: {tonic_hz:.1f} Hz  ->  OVERRIDE Sa = {args.tonic:.1f} Hz')
        tonic_hz = args.tonic
    print(f'Estimated tonic (Sa): {tonic_hz:.1f} Hz\n')

    # ---- swara distribution: time spent at each semitone from Sa ----
    semis = np.mod(np.round(12 * np.log2(voiced / tonic_hz)).astype(int), 12)
    total = len(semis)
    dist = Counter(semis)
    print('Swara distribution (share of voiced time):')
    for s in range(12):
        share = dist.get(s, 0) / total
        bar = '#' * int(share * 100)
        print(f'  semi {s:2d} {SEMI_NAMES[s]:<6} {share:5.1%} {bar}')
    print()

    # ---- note segmentation on the smoothed semitone track ----
    cont_semi = 12 * np.log2(np.where(np.isnan(f0), np.nan, f0) / tonic_hz)
    seg = []  # (start_frame, end_frame, mean_semi, std_cents)
    i = 0
    n = len(cont_semi)
    while i < n:
        if np.isnan(cont_semi[i]):
            i += 1; continue
        j = i
        ref = cont_semi[i]
        vals = [cont_semi[i]]
        while j + 1 < n and not np.isnan(cont_semi[j + 1]) and abs(cont_semi[j + 1] - np.mean(vals)) < 0.7:
            j += 1; vals.append(cont_semi[j])
        seg.append((i, j, float(np.mean(vals)), float(np.std(vals) * 100)))
        i = j + 1

    durs = np.array([(b - a + 1) * hop_s for a, b, _, _ in seg])
    stds = np.array([s for _, _, _, s in seg])
    long_notes = durs[durs > 0.4]
    print(f'Segmented into {len(seg)} note-events')
    print(f'  median note dwell: {np.median(durs):.2f}s | 90th pct: {np.percentile(durs,90):.2f}s | max: {durs.max():.2f}s')
    print(f'  notes held > 0.4s: {len(long_notes)} ({len(long_notes)/len(seg):.0%})')
    print(f'  median in-note pitch deviation (gamaka extent): {np.median(stds):.0f} cents')
    print(f'  notes with heavy oscillation (>60 cents std): {(stds>60).sum()} ({(stds>60).mean():.0%})\n')

    # ---- glide vs steady: fraction of voiced frames where |d(semitone)/dt| is large ----
    d = np.abs(np.diff(cont_semi))
    d = d[~np.isnan(d)]
    glide_frac = (d > 0.15).mean()   # >0.15 semitone/frame ~ moving pitch
    print(f'Continuous-motion frames (glide/gamaka): {glide_frac:.0%} of voiced time')
    print(f'Steady-pitch frames: {1-glide_frac:.0%}\n')

    # ---- phrasing via silence gaps ----
    unvoiced_runs = []
    run = 0
    for v in np.isnan(f0):
        if v: run += 1
        elif run: unvoiced_runs.append(run); run = 0
    gaps = np.array(unvoiced_runs) * hop_s
    breath = gaps[gaps > 0.25]   # phrase breaks
    print(f'Phrase breaks (silence > 0.25s): {len(breath)}')
    if len(breath):
        # phrase durations ~ time between breaks
        print(f'  median silence gap: {np.median(breath):.2f}s | longest: {breath.max():.2f}s')
        approx_phrase = dur / (len(breath) + 1)
        print(f'  approx mean phrase length: {approx_phrase:.1f}s\n')

    # ---- pitch range ----
    lo, hi = np.nanmin(cont_semi), np.nanmax(cont_semi)
    print(f'Pitch range: {hi-lo:.1f} semitones ({(hi-lo)/12:.1f} octaves), '
          f'from {lo:+.0f} to {hi:+.0f} semitones around Sa\n')

    # ---- transition tendencies (quantised swara bigrams) ----
    qseq = [int(round(s)) % 12 for _, _, s, _ in seg]
    big = Counter(zip(qseq, qseq[1:]))
    print('Top swara transitions (movement grammar):')
    for (a, b), c in big.most_common(10):
        print(f'  {SEMI_NAMES[a]:<6} -> {SEMI_NAMES[b]:<6}  x{c}')


if __name__ == '__main__':
    main()
