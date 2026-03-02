"""Quick A/B: v2 on raw audio vs v2 on HPSS harmonic component."""
import librosa, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from carnatic.detector_v2 import RagaDetectorV2

det = RagaDetectorV2()

tests = [
    ("recording/kalyani_recording_20260208_224545.wav", "kalyani"),
    ("recording/voice_bahudhari2_20260216_192928.wav", "bahudari"),
    ("recording/voice_bahudhari_20260216_192928.wav", "bahudari"),
]

for fname, expected in tests:
    if not os.path.exists(fname):
        continue
    y, sr = librosa.load(fname, sr=22050, duration=30)
    y_h = librosa.effects.hpss(y)[0]

    basename = os.path.basename(fname)
    print(f"\n{'='*60}")
    print(f"{basename} (expected: {expected})")
    print(f"{'='*60}")

    # v2 on original
    r1 = det.detect_from_audio(y, sr, top_n=5)
    print(f"ORIGINAL: {r1[0].raga.name} ({r1[0].confidence:.3f}) tonic={r1[0].tonic_hz:.1f}")
    print(f"  swaras: {sorted(r1[0].detected_swaras)}")
    for i, r in enumerate(r1[:5], 1):
        marker = " <<<" if expected in r.raga.name.lower() else ""
        print(f"  {i}. {r.raga.name:25s} {r.confidence:.3f}{marker}")

    # v2 on harmonic
    r2 = det.detect_from_audio(y_h, sr, top_n=5)
    print(f"HARMONIC: {r2[0].raga.name} ({r2[0].confidence:.3f}) tonic={r2[0].tonic_hz:.1f}")
    print(f"  swaras: {sorted(r2[0].detected_swaras)}")
    for i, r in enumerate(r2[:5], 1):
        marker = " <<<" if expected in r.raga.name.lower() else ""
        print(f"  {i}. {r.raga.name:25s} {r.confidence:.3f}{marker}")
