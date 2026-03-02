"""
Carnatic ML - Command Line Interface

Usage:
    python -m carnatic detect audio.wav          # Detect raga from audio
    python -m carnatic generate mohanam          # Generate melody in raga
    python -m carnatic list                      # List available ragas
    python -m carnatic info mohanam              # Show raga details
    python -m carnatic gui                       # Launch GUI detector
"""

import argparse
import sys
from pathlib import Path


def cmd_gui(args):
    """Launch the GUI detector."""
    from .gui import main
    main()
    return 0


def cmd_detect(args):
    """Detect raga from audio file."""
    from .detector_v2 import RagaDetectorV2
    
    detector = RagaDetectorV2()
    
    print(f"\n🎵 Analyzing: {args.audio}")
    print("-" * 50)
    
    try:
        # Get more results to count total matches
        results = detector.detect_from_file(
            args.audio,
            top_n=max(args.top, 100)
        )
        
        if not results:
            print("Could not detect raga. Check audio quality.")
            return 1
        
        # Count how many have same top score
        top_score = results[0].confidence
        same_score = sum(1 for r in results if r.confidence >= top_score - 0.01)
        
        details = results[0].match_details
        primary = sorted(details.get('primary_detected', set()))
        outliers = sorted(details.get('outliers', set()))
        
        print(f"\nTonic (Sa): {results[0].tonic_hz:.1f} Hz")
        print(f"Detected scale: {', '.join(primary)}")
        if outliers:
            print(f"Outliers removed: {', '.join(outliers)}")
        
        if same_score > 10:
            print(f"\n⚠️  {same_score}+ ragas share this scale (parent + janya ragas)")
            print("   Use prayogams (characteristic phrases) to distinguish them.")
        
        print(f"\nTop {min(args.top, len(results))} possible matches:\n")
        
        for i, result in enumerate(results[:args.top], 1):
            conf_bar = "█" * int(result.confidence * 20)
            mela = " [M]" if result.raga.is_melakarta else ""
            print(f"  {i}. {result.raga.name:<25} {result.confidence:>6.1%} {conf_bar}{mela}")
            if args.verbose:
                if result.raga.arohanam:
                    print(f"     Arohanam: {' '.join(result.raga.arohanam)}")
                if result.raga.avarohanam:
                    print(f"     Avarohanam: {' '.join(result.raga.avarohanam)}")
                print()
        
        return 0
        
    except FileNotFoundError:
        print(f"Error: File not found: {args.audio}")
        return 1
    except ImportError as e:
        print(f"Error: {e}")
        print("Install dependencies: pip install librosa")
        return 1


def cmd_generate(args):
    """Generate melody in specified raga."""
    from .generator import RagaGenerator
    
    generator = RagaGenerator()
    
    print(f"\n🎹 Generating {args.style} in raga: {args.raga}")
    print(f"   Duration: ~{args.duration} beats")
    print("-" * 50)
    
    try:
        notes = generator.generate(
            args.raga,
            duration_beats=args.duration,
            style=args.style
        )
        
        # Show swara notation
        notation = generator.to_swara_string(notes)
        print(f"\nGenerated melody ({len(notes)} notes):")
        print(f"  {notation[:100]}{'...' if len(notation) > 100 else ''}")
        
        # Export to MIDI
        output = args.output or f"{args.raga}_{args.style}.mid"
        midi_path = generator.to_midi(notes, output, tempo=args.tempo)
        
        print(f"\n✅ Saved to: {midi_path}")
        
        # Show stats
        total_beats = sum(n.duration for n in notes)
        print(f"   Total duration: {total_beats:.1f} beats ({total_beats * 60 / args.tempo:.1f} seconds)")
        
        return 0
        
    except ValueError as e:
        print(f"Error: {e}")
        return 1
    except ImportError as e:
        print(f"Error: {e}")
        print("Install dependencies: pip install pretty_midi")
        return 1


def cmd_list(args):
    """List available ragas."""
    from .raga_db import get_db
    
    db = get_db()
    
    if args.search:
        ragas = db.search(args.search)
        print(f"\n🔍 Ragas matching '{args.search}': {len(ragas)}\n")
    else:
        ragas = list(db)
        print(f"\n📚 Total ragas: {len(ragas)}\n")
    
    if args.melakartas:
        ragas = [r for r in ragas if r.is_melakarta]
        print(f"   (Showing {len(ragas)} melakartas)\n")
    
    # Sort and display
    ragas = sorted(ragas, key=lambda r: (not r.is_melakarta, r.melakarta_number or 999, r.name))
    
    for raga in ragas[:args.limit]:
        mela = f"M{raga.melakarta_number}" if raga.melakarta_number else "  "
        scale = ' '.join(raga.arohanam) if raga.arohanam else ''
        print(f"  {mela:>4} {raga.name:<25} {scale}")
    
    if len(ragas) > args.limit:
        print(f"\n  ... and {len(ragas) - args.limit} more. Use --limit to show more.")
    
    return 0


def cmd_info(args):
    """Show detailed raga information."""
    from .raga_db import get_db
    
    db = get_db()
    raga = db.get(args.raga)
    
    if not raga:
        # Try search
        matches = db.search(args.raga)
        if matches:
            print(f"\nRaga '{args.raga}' not found. Did you mean:")
            for m in matches[:5]:
                print(f"  - {m.name}")
        else:
            print(f"\nRaga '{args.raga}' not found.")
        return 1
    
    print(f"\n{'=' * 50}")
    print(f"  {raga.name.upper()}")
    print(f"{'=' * 50}")
    
    if raga.is_melakarta:
        print(f"  Melakarta #{raga.melakarta_number}")
    
    print(f"\n  Arohanam:   {' '.join(raga.arohanam)}")
    print(f"  Avarohanam: {' '.join(raga.avarohanam)}")
    print(f"  Scale:      {' '.join(sorted(raga.scale))}")
    
    if raga.vadi:
        print(f"\n  Vadi:       {raga.vadi}")
    if raga.samvadi:
        print(f"  Samvadi:    {raga.samvadi}")
    
    if raga.alternate_names:
        print(f"\n  Also known as: {', '.join(raga.alternate_names)}")
    
    if raga.phrases:
        print(f"\n  Characteristic phrases:")
        for phrase in raga.phrases[:3]:
            print(f"    {' '.join(phrase)}")
    
    print()
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog='carnatic',
        description='Carnatic Music ML - Raga Detection and Generation'
    )
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # GUI command
    p_gui = subparsers.add_parser('gui', help='Launch GUI detector')
    
    # Detect command
    p_detect = subparsers.add_parser('detect', help='Detect raga from audio')
    p_detect.add_argument('audio', help='Path to audio file')
    p_detect.add_argument('--tonic', type=float, help='Tonic (Sa) frequency in Hz')
    p_detect.add_argument('--top', type=int, default=5, help='Number of matches')
    p_detect.add_argument('-v', '--verbose', action='store_true')
    
    # Generate command
    p_gen = subparsers.add_parser('generate', help='Generate melody')
    p_gen.add_argument('raga', help='Raga name')
    p_gen.add_argument('-d', '--duration', type=int, default=32, help='Duration in beats')
    p_gen.add_argument('-s', '--style', choices=['alapana', 'kriti', 'tana'], default='alapana')
    p_gen.add_argument('-o', '--output', help='Output MIDI path')
    p_gen.add_argument('--tempo', type=int, default=80, help='Tempo (BPM)')
    
    # List command
    p_list = subparsers.add_parser('list', help='List available ragas')
    p_list.add_argument('--search', help='Search by name')
    p_list.add_argument('--melakartas', action='store_true', help='Show only melakartas')
    p_list.add_argument('--limit', type=int, default=50, help='Max ragas to show')
    
    # Info command
    p_info = subparsers.add_parser('info', help='Show raga details')
    p_info.add_argument('raga', help='Raga name')
    
    args = parser.parse_args()
    
    if args.command == 'gui':
        return cmd_gui(args)
    elif args.command == 'detect':
        return cmd_detect(args)
    elif args.command == 'generate':
        return cmd_generate(args)
    elif args.command == 'list':
        return cmd_list(args)
    elif args.command == 'info':
        return cmd_info(args)
    else:
        parser.print_help()
        return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
