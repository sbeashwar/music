"""Quick test: v3 ML detector on pro kalyani."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import librosa
from carnatic.detector_v3 import RagaDetectorV3

det = RagaDetectorV3()
y, sr = librosa.load("shared/samples/Songs/01b-varnam-kalyani-ata-swarakalpana-mdr-c06.mp3", sr=22050, duration=30)
results = det.detect_from_audio(y, sr, top_n=15)

print("v3 results for pro kalyani recording:")
for i, r in enumerate(results[:15], 1):
    method = r.match_details.get("method", "?")
    marker = " <<<" if "kalya" in r.raga_name.lower() else ""
    print(f"{i:2}. {r.raga_name:25s} conf={r.confidence:.3f}  ml={r.ml_probability:.3f}  rule={r.rule_score:.3f}  [{method}]{marker}")
