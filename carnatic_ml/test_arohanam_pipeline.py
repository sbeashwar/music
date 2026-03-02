"""
End-to-end test: Generate raga scale WAVs, then detect them.

Tests the full pipeline:
1. Generate clean scale audio files using raga_player
2. Run arohanam_detector on each
3. Feed detected swaras to swara_matcher
4. Check if correct raga is in top results
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from raga_detection.raga_player import play_raga
from raga_detection.arohanam_detector import ArohanamDetector
from raga_detection.swara_matcher import SwaraSequenceMatcher, format_match_result

# Test ragas
TEST_RAGAS = [
    'mohanam',
    'kalyani',
    'bahudari',
    'shankarabharanam',
    'hamsadhwani',
    'kharaharapriya',
    'todi',
]


def main():
    output_dir = 'output/test_scales'
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize components once
    matcher = SwaraSequenceMatcher()
    detector = ArohanamDetector()
    
    correct = 0
    total = 0
    
    for raga_name in TEST_RAGAS:
        total += 1
        wav_path = os.path.join(output_dir, f'{raga_name}_scale.wav')
        
        # Step 1: Generate WAV
        try:
            path, info = play_raga(raga_name, output_path=wav_path, format='wav')
        except ValueError as e:
            print(f'SKIP {raga_name}: {e}')
            continue
        
        aro_str = ' '.join(info['arohanam'])
        ava_str = ' '.join(info['avarohanam'])
        
        print(f'=== {info["name"]} ===')
        print(f'  Arohanam:  {aro_str}')
        print(f'  Avarohanam: {ava_str}')
        
        # Step 2: Detect from audio
        result = detector.detect_from_file(wav_path)
        
        print(f'  Tonic detected: {result.tonic_hz:.1f} Hz')
        print(f'  Direction: {result.direction}')
        print(f'  Raw sequence: {" -> ".join(result.raw_sequence)}')
        print(f'  Detected swaras: {" ".join(result.detected_swaras)}')
        
        # Step 3: Match
        direction = result.direction if result.direction != 'mixed' else 'ascending'
        matches = matcher.match_swaras(
            result.detected_swaras, 
            direction=direction, 
            max_results=10,
            raw_sequence=result.raw_sequence,
        )
        
        # Check if correct raga is in top results
        found = False
        found_rank = -1
        raga_lower = raga_name.lower()
        
        for i, m in enumerate(matches, 1):
            if raga_lower in m.raga_id.lower() or raga_lower in m.raga_name.lower():
                found = True
                found_rank = i
                break
        
        if found:
            correct += 1
            status = f'PASS (rank #{found_rank})'
        else:
            status = 'FAIL'
        
        print(f'  Result: {status}')
        
        # Show top 5 matches
        print(f'  Top 5 matches:')
        for i, m in enumerate(matches[:5], 1):
            marker = ' <<<' if raga_lower in m.raga_id.lower() or raga_lower in m.raga_name.lower() else ''
            print(f'    {i}. {m.raga_name:30s} score={m.score:.3f} ({m.match_type}){marker}')
        print()
    
    print('=' * 60)
    print(f'Results: {correct}/{total} correct')
    print('=' * 60)


if __name__ == '__main__':
    main()
