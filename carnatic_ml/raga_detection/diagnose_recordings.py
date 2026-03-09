"""Diagnostic script: run full detection pipeline on saved recordings."""

import os
from raga_detection.arohanam_detector import ArohanamDetector
from raga_detection.swara_matcher import SwaraSequenceMatcher

matcher = SwaraSequenceMatcher()

tests = [
    ('kalyani_recording_20260208_224545.wav', 'Kalyani', True),
    ('kalyani_recording_20260208_224545.wav', 'Kalyani', False),
    ('sing_bahudari_20260301_165336.wav', 'Bahudari', True),
    ('sing_bahudari_20260301_165336.wav', 'Bahudari', False),
    ('voice_saraswarti_recording_20260308_185005.wav', 'Saraswati', True),
    ('voice_saraswarti_recording_20260308_185005.wav', 'Saraswati', False),
    ('piano_recording_20260201_222255.wav', 'Shankarabharanam', True),
    ('piano_recording_20260201_222255.wav', 'Shankarabharanam', False),
    ('voice_recording_20260201_223502.wav', 'Shankarabharanam', True),
    ('voice_recording_20260201_223502.wav', 'Shankarabharanam', False),
]

for fname, expected, voice in tests:
    mode = 'voice' if voice else 'clean'
    det = ArohanamDetector(voice_mode=voice)
    try:
        result = det.detect_from_file(os.path.join('recording', fname))
        swaras = ' '.join(result.detected_swaras)
        seq = ' '.join(result.raw_sequence)
        matches = matcher.match_swaras_hierarchical(
            result.detected_swaras, direction=result.direction,
            max_results=5, raw_sequence=result.raw_sequence)
        if matches:
            top = matches[0].raga_name
            score = matches[0].score
            ok = 'OK' if expected.lower() in top.lower() else 'MISS'
            print(f'[{ok}] {expected} ({mode}): swaras=[{swaras}] -> #1: {top} ({score:.3f})')
            if ok == 'MISS':
                # Show more detail
                print(f'       Tonic: {result.tonic_hz:.1f} Hz  Seq: {seq}')
                for i, m in enumerate(matches[:5], 1):
                    print(f'       #{i}: {m.raga_name} ({m.score:.3f})')
        else:
            print(f'[MISS] {expected} ({mode}): swaras=[{swaras}] -> NO MATCHES')
    except Exception as e:
        print(f'[ERR] {expected} ({mode}): {e}')

