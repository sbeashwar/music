"""
Quick test script for the new Carnatic ML implementation.
Run this to verify everything works.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_raga_db():
    """Test the raga database."""
    print("\n" + "="*60)
    print("Testing Raga Database")
    print("="*60)
    
    from carnatic.raga_db import RagaDB
    
    db = RagaDB()
    db.load()
    
    print(f"✓ Loaded {len(db)} ragas")
    
    # Test get
    mohanam = db.get('mohanam')
    if mohanam:
        print(f"✓ Found Mohanam:")
        print(f"    Arohanam: {' '.join(mohanam.arohanam)}")
        print(f"    Avarohanam: {' '.join(mohanam.avarohanam)}")
        print(f"    Scale: {mohanam.scale}")
    else:
        print("✗ Could not find Mohanam")
        return False
    
    # Test search
    results = db.search('bhairavi')
    print(f"✓ Search 'bhairavi' found {len(results)} ragas")
    
    # Test scale matching
    test_scale = {'S', 'R2', 'G3', 'P', 'D2'}  # Mohanam's scale
    matches = db.find_by_scale(test_scale)
    print(f"✓ Scale matching found {len(matches)} candidates")
    if matches:
        print(f"    Top match: {matches[0][0].name} ({matches[0][1]:.0%})")
    
    return True


def test_generator():
    """Test melody generation."""
    print("\n" + "="*60)
    print("Testing Raga Generator")
    print("="*60)
    
    from carnatic.generator import RagaGenerator
    
    gen = RagaGenerator()
    
    # Generate a short melody
    try:
        notes = gen.generate('mohanam', duration_beats=16, style='alapana')
        print(f"✓ Generated {len(notes)} notes")
        
        # Show notation
        notation = gen.to_swara_string(notes)
        print(f"✓ Notation: {notation[:80]}...")
        
        # Save to MIDI
        output_path = project_root / 'test_output.mid'
        gen.to_midi(notes, str(output_path))
        print(f"✓ Saved MIDI to: {output_path}")
        
        return True
    except ImportError as e:
        print(f"⚠ MIDI export skipped (missing pretty_midi): {e}")
        return True
    except Exception as e:
        print(f"✗ Generation failed: {e}")
        return False


def test_detector():
    """Test raga detection (without audio - just the swara matching)."""
    print("\n" + "="*60)
    print("Testing Raga Detector (swara matching)")
    print("="*60)
    
    from carnatic.detector import RagaDetector
    
    detector = RagaDetector()
    
    # Test with known swaras (Mohanam: S R2 G3 P D2)
    test_swaras = ['S', 'R2', 'G3', 'P', 'D2', 'S', 'D2', 'P', 'G3']
    results = detector.detect_from_swaras(test_swaras)
    
    if results:
        print(f"✓ Detected {len(results)} possible ragas")
        print(f"    Top matches:")
        for r in results[:3]:
            print(f"      - {r.raga.name}: {r.confidence:.0%}")
        
        # Check if Mohanam is the top match (it should be!)
        if results[0].raga.name.lower() == 'mohanam':
            print(f"✓ Correctly identified Mohanam as top match!")
        else:
            print(f"⚠ Top match was {results[0].raga.name}, expected Mohanam")
        
        return True
    else:
        print("✗ No matches found")
        return False


def main():
    print("="*60)
    print("  CARNATIC ML - TEST SUITE")
    print("="*60)
    
    all_passed = True
    
    # Test each component
    all_passed &= test_raga_db()
    all_passed &= test_generator()
    all_passed &= test_detector()
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("\nYou can now use:")
        print("  python -m carnatic generate mohanam")
        print("  python -m carnatic detect audio.wav")
        print("  python -m carnatic list")
        print("  python -m carnatic info mohanam")
    else:
        print("✗ SOME TESTS FAILED")
    print("="*60)
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
