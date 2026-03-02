"""Quick single-file test of v3."""
import sys, time
sys.path.insert(0, '.')
print("Importing...", flush=True)
from carnatic.detector_v3 import RagaDetectorV3

det = RagaDetectorV3()
print(f"Model: {det.has_model}", flush=True)

print("Loading audio...", flush=True)
t0 = time.time()
import soundfile as sf
import numpy as np
y, sr = sf.read('recording/voice_bahudhari2_20260216_192928.wav', dtype='float32',
                stop=22050*15)
if y.ndim > 1:
    y = np.mean(y, axis=1)
print(f"Loaded in {time.time()-t0:.1f}s: {len(y)} samples, sr={sr}", flush=True)

print("Extracting features...", flush=True)
t1 = time.time()
features = det.feature_extractor.extract(y, sr)
print(f"Features in {time.time()-t1:.1f}s: shape={features.shape if features is not None else None}", flush=True)

print("Running ML inference...", flush=True)
t2 = time.time()
results = det.detect_from_audio(y, sr, top_n=10)
print(f"Detection in {time.time()-t2:.1f}s", flush=True)

for i, r in enumerate(results[:10], 1):
    marker = " <<<" if "bahud" in r.raga_name.lower() else ""
    print(f"  {i}. {r.raga_name:25s} conf={r.confidence:.3f} ml={r.ml_probability:.3f} rule={r.rule_score:.3f} [{r.match_details.get('method','')}]{marker}")

# Search deeper
found = False
for i, r in enumerate(results, 1):
    if "bahud" in r.raga_name.lower():
        print(f"  ... bahudari at #{i}: conf={r.confidence:.3f}" )
        found = True
if not found:
    print(f"  *** bahudari NOT in results ***")

print(f"\nTotal: {time.time()-t0:.1f}s")
