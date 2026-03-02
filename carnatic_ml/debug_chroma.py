"""Debug chroma profile for bahudhari recording."""
import sys, numpy as np
sys.path.insert(0, '.')
import soundfile as sf
import librosa
from carnatic.raga_db import get_db, SWARA_TO_SEMITONE

# Load audio
y, sr = sf.read('recording/voice_bahudhari2_20260216_192928.wav', 
                dtype='float32', stop=22050*15)
if y.ndim > 1:
    y = np.mean(y, axis=1)

# Compute chroma
chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=512, n_chroma=12, bins_per_octave=24)
chroma_mean = np.mean(chroma, axis=1)

# Find tonic
tonic_bin = np.argmax(chroma_mean)
notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
print(f"Tonic bin: {tonic_bin} ({notes[tonic_bin]})")
print(f"Chroma profile: {chroma_mean}")

# Rotate to tonic
rotated = np.roll(chroma_mean, -tonic_bin)
rotated_norm = rotated / (rotated.sum() + 1e-8)

print(f"\nTonic-normalized profile (semitones from Sa):")
swara_names = ['S', 'R1', 'R2/G1', 'R3/G2', 'G3', 'M1', 'M2', 'P', 'D1', 'D2/N1', 'D3/N2', 'N3']
for i in range(12):
    bar = '#' * int(rotated_norm[i] * 100)
    print(f"  {i:2d} {swara_names[i]:8s} {rotated_norm[i]:.3f} {bar}")

# Bahudhari expected semitones: {0, 4, 5, 7, 9, 10}
db = get_db()
bahudari = [r for r in db if r.name == 'bahudAri'][0]
print(f"\nBahudari scale semitones: {bahudari.scale_semitones}")
print(f"Bahudari scale: {bahudari.scale}")

# Score
in_raga = sum(rotated_norm[st] for st in bahudari.scale_semitones if st < 12)
total = rotated_norm.sum()
match_ratio = in_raga / total
print(f"\nMatch ratio: {match_ratio:.3f} ({in_raga:.3f}/{total:.3f})")

# Coverage
raga_notes_present = sum(1 for st in bahudari.scale_semitones if st < 12 and rotated_norm[st] > 0.02)
coverage = raga_notes_present / len(bahudari.scale_semitones)
print(f"Coverage: {coverage:.3f} ({raga_notes_present}/{len(bahudari.scale_semitones)})")

# Specificity
absent = sum(1 for st in bahudari.scale_semitones if st < 12 and rotated_norm[st] < 0.01)
specificity = 1.0 - (absent / max(len(bahudari.scale_semitones), 1)) * 0.3
print(f"Specificity: {specificity:.3f}")

score = match_ratio * 0.5 + coverage * 0.3 + specificity * 0.2
print(f"Total score: {score:.3f}")

# Try all 12 tonic rotations to see which gives best bahudari match
print("\n--- All tonic rotations ---")
best_score = 0
best_bin = 0
for tbin in range(12):
    rot = np.roll(chroma_mean, -tbin)
    rnorm = rot / (rot.sum() + 1e-8)
    in_r = sum(rnorm[st] for st in bahudari.scale_semitones if st < 12)
    mr = in_r / rnorm.sum()
    rp = sum(1 for st in bahudari.scale_semitones if st < 12 and rnorm[st] > 0.02)
    cov = rp / len(bahudari.scale_semitones)
    ab = sum(1 for st in bahudari.scale_semitones if st < 12 and rnorm[st] < 0.01)
    sp = 1.0 - (ab / max(len(bahudari.scale_semitones), 1)) * 0.3
    sc = mr * 0.5 + cov * 0.3 + sp * 0.2
    marker = " <<<" if sc > best_score else ""
    if sc > best_score:
        best_score = sc
        best_bin = tbin
    print(f"  tonic={tbin:2d} ({notes[tbin]:2s}) match={mr:.3f} cov={cov:.3f} spec={sp:.3f} => {sc:.3f}{marker}")

print(f"\nBest tonic for bahudari: {best_bin} ({notes[best_bin]}), score={best_score:.3f}")
