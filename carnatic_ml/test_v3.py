"""Quick test of v3 model loading and inference."""
import sys
sys.path.insert(0, '.')
from carnatic.detector_v3 import RagaDetectorV3
import librosa
import time

det = RagaDetectorV3()
print('Model loaded:', det.has_model)
if det.metrics:
    print(f"Ragas: {det.metrics['n_ragas']}, CV: {det.metrics['cv_mean']:.1%}")

recordings = [
    ("Voice Bahudhari 2", "recording/voice_bahudhari2_20260216_192928.wav", "bahudari"),
    ("Voice Bahudhari 1", "recording/voice_bahudhari_20260216_192928.wav", "bahudari"),
    ("Voice Shankarabharanam", "recording/voice_recording_20260201_223502.wav", "shankarabharanam"),
    ("Voice Kalyani", "recording/kalyani_recording_20260208_224545.wav", "kalyani"),
    ("Piano", "recording/piano_recording_20260201_222255.wav", "shankarabharanam"),
    ("Pro Bahudhari MP3", r"Z:\Music\Carnatic\Renditions\Thiruppugaz\033.Karipuraari-Bahudaari-Aadi-Tisram.mp3", "bahudari"),
]

for name, path, expected in recordings:
    print(f"\n{'='*60}")
    print(f"TEST: {name} (expected: {expected})")
    try:
        t0 = time.time()
        results = det.detect_from_file(path, top_n=10, duration=15.0)
        elapsed = time.time() - t0
        
        for i, r in enumerate(results[:5], 1):
            marker = " <<<" if expected in r.raga_name.lower() else ""
            print(f"  {i}. {r.raga_name:25s} conf={r.confidence:.3f}  "
                  f"ml={r.ml_probability:.3f}  rule={r.rule_score:.3f}{marker}")
        
        # Search deeper
        found = False
        for i, r in enumerate(results, 1):
            if expected in r.raga_name.lower():
                if i > 5:
                    print(f"  ... {expected} at #{i}: conf={r.confidence:.3f}")
                found = True
                break
        if not found:
            print(f"  *** {expected} NOT in top {len(results)} ***")
        
        print(f"  ({elapsed:.1f}s)")
    except Exception as e:
        print(f"  ERROR: {e}")
