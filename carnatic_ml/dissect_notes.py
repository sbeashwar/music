"""
Diagnostic v2: dissect a recording into individual notes and play each one.
- Lower stability threshold to catch brief notes
- Show gaps / silence between notes
- More generous audio padding for playback
- Show pitch density timeline
"""
import sys, time, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import librosa
import sounddevice as sd
from collections import Counter
from carnatic.detector_v2 import RagaDetectorV2

AUDIO = r'recording\voice_bahudhari2_20260216_192928.wav'

det = RagaDetectorV2()

# Load audio
y, sr = librosa.load(AUDIO, sr=22050, duration=60)
duration = len(y) / sr
print(f'Loaded {AUDIO}  ({duration:.1f}s, sr={sr})')

# --- Step 1: Extract pitches frame by frame (WIDER range) ------------------
pitches_raw, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=80, fmax=1000)
hop = 512
total_frames = pitches_raw.shape[1]

frame_data = []
for t in range(total_frames):
    mag_col = magnitudes[:, t]
    if mag_col.max() > 0:
        idx = mag_col.argmax()
        pitch = pitches_raw[idx, t]
        time_s = t * hop / sr
        if 80 < pitch < 600:  # wider than detector's 130-600
            frame_data.append((t, time_s, pitch))

print(f'Total audio frames: {total_frames}')
print(f'Pitched frames (80-600 Hz): {len(frame_data)}')
print(f'Coverage: {len(frame_data)/total_frames*100:.1f}% of frames have pitch')

# Show pitch density over time
print(f'\nPitch density per second:')
for sec in range(int(duration) + 1):
    count = sum(1 for _, t, _ in frame_data if sec <= t < sec + 1)
    bar = '█' * (count // 2) if count > 0 else '·'
    print(f'  {sec:2d}s: {count:3d} frames  {bar}')

# --- Step 2: Detect tonic --------------------------------------------------
pitches_hz = np.array([f[2] for f in frame_data])
tonic = det._find_tonic_by_sa_pa(pitches_hz)
print(f'\nDetected tonic (Sa): {tonic:.1f} Hz')

# Show raw semitone distribution
print(f'\nRaw pitch → semitone distribution:')
INTERVALS = det.INTERVALS
semi_counts = Counter()
for _, _, p in frame_data:
    semi = round(12 * np.log2(p / tonic)) % 12
    semi_counts[semi] += 1
total_pitched = sum(semi_counts.values())
for semi in sorted(semi_counts.keys()):
    swara = INTERVALS.get(semi, f'?{semi}')
    pct = semi_counts[semi] / total_pitched * 100
    bar = '█' * int(pct)
    print(f'  {semi:2d} {swara:<6} {semi_counts[semi]:4d} ({pct:5.1f}%) {bar}')

# --- Step 3: Segment into stable notes (LOWER threshold) -------------------
MIN_STABLE = 2  # lowered from 4 to catch brief notes

segments = []

all_intervals = []
for _, time_s, pitch in frame_data:
    semitones = 12 * np.log2(pitch / tonic)
    interval = round(semitones) % 12
    all_intervals.append((time_s, interval, pitch))

i = 0
while i < len(all_intervals):
    cur_time, cur_note, cur_pitch = all_intervals[i]
    start_time = cur_time
    pitches_run = [cur_pitch]
    j = i + 1
    while j < len(all_intervals):
        t2, n2, p2 = all_intervals[j]
        if n2 == cur_note:
            pitches_run.append(p2)
            j += 1
        else:
            break
    end_time = all_intervals[j - 1][0] if j > i else cur_time

    if len(pitches_run) >= MIN_STABLE:
        swara = INTERVALS.get(cur_note, f'?{cur_note}')
        segments.append((start_time, end_time, cur_note, swara, np.median(pitches_run), len(pitches_run)))

    i = j

print(f'\nFound {len(segments)} stable note segments (min {MIN_STABLE} frames)\n')

# --- Step 4: Print and play each note segment -------------------------------
ENHARMONIC = {'R2': 'R2/G1', 'R3': 'R3/G2', 'D2': 'D2/N1', 'D3': 'D3/N2'}

print(f'{"#":>3}  {"Time":>10}  {"Dur":>6}  {"Frm":>4}  {"Swara":<10}  {"Hz":>7}  {"Semi":>4}  {"Gap":>6}')
print('-' * 70)

prev_end = 0.0
for idx, (t_start, t_end, semi, swara, med_hz, nframes) in enumerate(segments, 1):
    dur = max(t_end - t_start, hop / sr)
    gap = t_start - prev_end if prev_end > 0 else 0
    gap_str = f'{gap:.1f}s' if gap > 0.3 else ''
    display_swara = ENHARMONIC.get(swara, swara)
    print(f'{idx:3d}  {t_start:7.2f}s  {dur:5.2f}s  {nframes:3d}   {display_swara:<10}  {med_hz:7.1f}  {semi:4d}  {gap_str:>6}')
    prev_end = t_end

print(f'\n--- Playing each note (0.15s padding, from original audio) ---')
print('Press Ctrl+C to stop.\n')

try:
    for idx, (t_start, t_end, semi, swara, med_hz, nframes) in enumerate(segments, 1):
        dur = max(t_end - t_start, hop / sr)
        # Generous padding so you can hear the note in context
        pad = 0.15
        s_start = max(0, int((t_start - pad) * sr))
        s_end = min(len(y), int((t_end + pad + hop / sr) * sr))
        clip = y[s_start:s_end]

        if len(clip) == 0:
            continue

        display_swara = ENHARMONIC.get(swara, swara)
        print(f'  Note {idx:2d}: {display_swara:<8} ({med_hz:.0f} Hz, {dur:.2f}s, {nframes}f) @ {t_start:.2f}s  ', end='', flush=True)
        sd.play(clip, sr)
        sd.wait()
        print('✓')
        time.sleep(0.2)

except KeyboardInterrupt:
    sd.stop()
    print('\n\nStopped.')

print('\nDone.')
