"""Test the unified detector on known recordings."""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

import librosa
from carnatic.detector_unified import UnifiedRagaDetector

# Test recordings with expected ragas
TEST_CASES = [
    # Pro recording — kalyani is in ML training set
    ("shared/samples/Songs/01b-varnam-kalyani-ata-swarakalpana-mdr-c06.mp3", "kalyani"),
    # Clean voice recordings
    ("recording/kalyani_recording_20260208_224545.wav", "kalyani"),
    ("recording/voice_bahudhari_20260216_192928.wav", "bahudari"),
    ("recording/voice_bahudhari2_20260216_192928.wav", "bahudari"),
]

def main():
    print("=" * 70)
    print("Unified Raga Detector — Test Suite")
    print("=" * 70)
    
    t0 = time.time()
    detector = UnifiedRagaDetector()
    t_load = time.time() - t0
    print(f"Loaded in {t_load:.1f}s  |  ML model: {detector.has_ml_model}")
    print()
    
    total = 0
    correct = 0
    
    for path, expected in TEST_CASES:
        if not os.path.exists(path):
            print(f"SKIP: {path} (not found)")
            continue
        
        print(f"\n{'='*60}")
        print(f"File: {os.path.basename(path)}")
        print(f"Expected raga: {expected}")
        print(f"{'='*60}")
        
        t1 = time.time()
        y, sr = librosa.load(path, sr=22050, duration=30)
        t_load_audio = time.time() - t1
        
        t2 = time.time()
        results = detector.detect_from_audio(y, sr, top_n=15)
        t_detect = time.time() - t2
        
        print(f"  Load: {t_load_audio:.1f}s  |  Detect: {t_detect:.1f}s")
        
        total += 1
        
        if not results:
            print("  No results!")
            continue
        
        top = results[0]
        top_name = top.raga.name.lower()
        is_correct = expected.lower() in top_name or top_name in expected.lower()
        
        if is_correct:
            correct += 1
            status = "CORRECT"
        else:
            status = "WRONG"
            # Find where expected raga actually is
            for i, r in enumerate(results, 1):
                if expected.lower() in r.raga.name.lower():
                    status += f" (expected at #{i})"
                    break
            else:
                status += " (expected not in top 15)"
        
        print(f"  >> {status}")
        print(f"  Tonic: {top.tonic_hz:.1f} Hz")
        print(f"  Detected swaras: {sorted(top.detected_swaras) if top.detected_swaras else '(none)'}")
        
        # Show top matches
        print(f"\n  Top 10 matches:")
        for i, r in enumerate(results[:10], 1):
            method = r.match_details.get('fusion_method', '?')
            ml_prob = r.match_details.get('ml_probability', 0)
            v2_conf = r.match_details.get('v2_confidence', r.confidence)
            v3_conf = r.match_details.get('v3_confidence', 0)
            mela = " [M]" if r.raga.is_melakarta else ""
            
            marker = " <<<" if expected.lower() in r.raga.name.lower() else ""
            print(f"  {i:2}. {r.raga.name:<25} conf={r.confidence:.3f}  "
                  f"v2={v2_conf:.3f} v3={v3_conf:.3f} ml={ml_prob:.3f} "
                  f"[{method}]{mela}{marker}")
    
    print(f"\n{'='*60}")
    print(f"Results: {correct}/{total} correct")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
