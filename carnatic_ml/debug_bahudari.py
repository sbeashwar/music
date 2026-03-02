"""Diagnose bahudari detection: generated vs voice recording."""
from raga_detection.arohanam_detector import ArohanamDetector
from raga_detection.swara_matcher import SwaraSequenceMatcher

m = SwaraSequenceMatcher()

print("=== Generated bahudari (clean mode) ===")
d_clean = ArohanamDetector(voice_mode=False)
r1 = d_clean.detect_from_file("output/bahudari_scale.wav")
print(f"Tonic: {r1.tonic_hz:.1f} Hz")
print(f"Direction: {r1.direction}")
print(f"Raw: {' -> '.join(r1.raw_sequence)}")
print(f"Swaras: {r1.detected_swaras}")
matches1 = m.match_swaras(r1.detected_swaras, direction=r1.direction, max_results=5, raw_sequence=r1.raw_sequence)
for i, x in enumerate(matches1[:5], 1):
    tag = " <<<" if "bahud" in x.raga_id.lower() else ""
    print(f"  {i}. {x.raga_name:30s} {x.score:.4f}{tag}")
print()

print("=== Voice recording (voice mode) ===")
d_voice = ArohanamDetector(voice_mode=True)
r2 = d_voice.detect_from_file("recording/sing_bahudari_20260301_165336.wav")
print(f"Tonic: {r2.tonic_hz:.1f} Hz")
print(f"Direction: {r2.direction}")
print(f"Raw: {' -> '.join(r2.raw_sequence)}")
print(f"Swaras: {r2.detected_swaras}")
print(f"Notes ({len(r2.detected_notes)}):")
for n in r2.detected_notes:
    print(f"  {n.swara:4s} semi={n.semitone:2d} freq={n.frequency:.1f}Hz conf={n.confidence:.2f} dur={n.duration:.2f}s")
print()
matches2 = m.match_swaras(r2.detected_swaras, direction=r2.direction, max_results=10, raw_sequence=r2.raw_sequence)
for i, x in enumerate(matches2[:10], 1):
    tag = " <<<" if "bahud" in x.raga_id.lower() or "bahud" in x.raga_name.lower() else ""
    print(f"  {i}. {x.raga_name:30s} {x.score:.4f} ({x.match_type}){tag}")

print()
print("=== Voice recording (clean mode - no voice filtering) ===")
r3 = d_clean.detect_from_file("recording/sing_bahudari_20260301_165336.wav")
print(f"Raw: {' -> '.join(r3.raw_sequence)}")
print(f"Swaras: {r3.detected_swaras}")
print(f"Notes: {len(r3.detected_notes)} vs voice mode: {len(r2.detected_notes)}")
