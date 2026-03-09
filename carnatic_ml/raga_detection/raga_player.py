"""
Raga Player - Generate MIDI and audio for raga scales.

Given a raga name, this module:
1. Looks up the arohanam/avarohanam from the database
2. Generates a MIDI file with the scale
3. Optionally synthesizes audio using a simple sine wave

Usage:
    python raga_player.py mohanam
    python raga_player.py kalyani --tonic C4 --tempo 120 --output kalyani.mid
"""

import os
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

try:
    from midiutil import MIDIFile
    HAS_MIDIUTIL = True
except ImportError:
    HAS_MIDIUTIL = False


# Swara to semitone offset from Sa
SWARA_TO_SEMITONE = {
    'S': 0,
    'R1': 1, 'R2': 2, 'R3': 3,
    'G1': 2, 'G2': 3, 'G3': 4,
    'M1': 5, 'M2': 6,
    'P': 7,
    'D1': 8, 'D2': 9, 'D3': 10,
    'N1': 9, 'N2': 10, 'N3': 11,
}

# Tonic name to MIDI note number (octave 4)
TONIC_TO_MIDI = {
    'C': 60, 'C#': 61, 'Db': 61,
    'D': 62, 'D#': 63, 'Eb': 63,
    'E': 64, 'F': 65,
    'F#': 66, 'Gb': 66,
    'G': 67, 'G#': 68, 'Ab': 68,
    'A': 69, 'A#': 70, 'Bb': 70,
    'B': 71,
}

# Also support octave notation: C3, C4, C5, etc.
def parse_tonic(tonic_str: str) -> int:
    """Parse a tonic string like 'C4', 'G#3', 'A' to MIDI note number."""
    tonic_str = tonic_str.strip()
    
    # Check for octave number at end
    if tonic_str[-1].isdigit():
        octave = int(tonic_str[-1])
        note_name = tonic_str[:-1]
    else:
        octave = 4  # Default octave
        note_name = tonic_str
    
    base_midi = TONIC_TO_MIDI.get(note_name)
    if base_midi is None:
        raise ValueError(f"Unknown tonic: {note_name}")
    
    # Adjust for octave (MIDI C4 = 60)
    return base_midi + (octave - 4) * 12


def swaras_to_midi_notes(
    swaras: List[str], 
    tonic_midi: int = 60,
    handle_upper_sa: bool = True,
) -> List[int]:
    """
    Convert a list of swaras to MIDI note numbers.
    
    Args:
        swaras: List of swara names (e.g., ['S', 'R2', 'G3', 'P', 'D2', 'S'])
        tonic_midi: MIDI note number for Sa (default: 60 = C4)
        handle_upper_sa: If True, the last 'S' in arohanam is treated as upper octave
        
    Returns:
        List of MIDI note numbers
    """
    midi_notes = []
    prev_midi = tonic_midi - 1  # Track last note for octave handling
    
    for i, swara in enumerate(swaras):
        if swara not in SWARA_TO_SEMITONE:
            continue
        
        semi = SWARA_TO_SEMITONE[swara]
        midi = tonic_midi + semi
        
        # Handle octave wrapping: if this note would be lower than the previous,
        # it should be in the next octave (for ascending sequences)
        if handle_upper_sa and midi <= prev_midi and i > 0:
            midi += 12
        
        midi_notes.append(midi)
        prev_midi = midi
    
    return midi_notes


def generate_midi(
    arohanam: List[str],
    avarohanam: List[str],
    output_path: str,
    tonic: str = 'C4',
    tempo: int = 80,
    note_duration: float = 0.5,   # beats per note
    pause_between: float = 2.0,   # beats pause between arohanam and avarohanam
    instrument: int = 73,         # GM instrument (73 = flute, 0 = piano, 40 = violin)
    volume: int = 100,
) -> str:
    """
    Generate a MIDI file for a raga's arohanam and avarohanam.
    
    Args:
        arohanam: List of swara names for ascending scale
        avarohanam: List of swara names for descending scale
        output_path: Path to save the MIDI file
        tonic: Tonic note (e.g., 'C4', 'G3', 'A')
        tempo: Beats per minute
        note_duration: Duration of each note in beats
        pause_between: Pause between arohanam and avarohanam in beats
        instrument: GM MIDI instrument number
        volume: MIDI velocity (0-127)
        
    Returns:
        Path to the saved MIDI file
    """
    if not HAS_MIDIUTIL:
        raise ImportError("midiutil is required. Install with: pip install midiutil")
    
    tonic_midi = parse_tonic(tonic)
    
    # Create MIDI
    midi = MIDIFile(1)  # One track
    track = 0
    channel = 0
    time = 0.0  # Start time in beats
    
    midi.addTempo(track, 0, tempo)
    midi.addProgramChange(track, channel, 0, instrument)
    
    # Add arohanam notes
    aro_midi = swaras_to_midi_notes(arohanam, tonic_midi, handle_upper_sa=True)
    for note in aro_midi:
        midi.addNote(track, channel, note, time, note_duration, volume)
        time += note_duration
    
    # Pause
    time += pause_between
    
    # Add avarohanam notes (descending)
    ava_midi = swaras_to_midi_notes(avarohanam, tonic_midi, handle_upper_sa=False)
    # For descending, start from upper octave
    if ava_midi and aro_midi:
        # Ensure first note of avarohanam matches last note of arohanam
        offset = aro_midi[-1] - ava_midi[0]
        if offset > 0:
            ava_midi = [n + offset for n in ava_midi]
    
    for note in ava_midi:
        midi.addNote(track, channel, note, time, note_duration, volume)
        time += note_duration
    
    # Save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'wb') as f:
        midi.writeFile(f)
    
    return output_path


def generate_audio_wave(
    arohanam: List[str],
    avarohanam: List[str],
    output_path: str,
    tonic_hz: float = 261.63,   # C4
    sample_rate: int = 22050,
    note_duration: float = 0.5,  # seconds per note
    pause_between: float = 1.0,  # seconds between arohanam and avarohanam
) -> str:
    """
    Generate a simple WAV file with sine waves for the raga scale.
    No MIDI dependencies needed.
    
    Args:
        arohanam: Ascending scale swaras
        avarohanam: Descending scale swaras
        output_path: Path to save WAV file
        tonic_hz: Frequency of Sa in Hz
        sample_rate: Audio sample rate
        note_duration: Duration of each note in seconds
        pause_between: Silence between arohanam and avarohanam
        
    Returns:
        Path to saved WAV file
    """
    import soundfile as sf
    
    def swara_to_freq(swara: str, base_hz: float, octave_up: bool = False) -> float:
        semi = SWARA_TO_SEMITONE.get(swara, 0)
        freq = base_hz * (2.0 ** (semi / 12.0))
        if octave_up:
            freq *= 2.0
        return freq
    
    def generate_note(freq: float, duration: float, sr: int) -> np.ndarray:
        """Generate a sine wave with gentle attack/release."""
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        
        # Sine wave with harmonics for richer sound
        wave = (0.6 * np.sin(2 * np.pi * freq * t) +
                0.25 * np.sin(2 * np.pi * 2 * freq * t) +
                0.1 * np.sin(2 * np.pi * 3 * freq * t) +
                0.05 * np.sin(2 * np.pi * 4 * freq * t))
        
        # ADSR envelope
        attack = int(0.05 * sr)
        release = int(0.05 * sr)
        
        envelope = np.ones_like(wave)
        if attack > 0:
            envelope[:attack] = np.linspace(0, 1, attack)
        if release > 0:
            envelope[-release:] = np.linspace(1, 0, release)
        
        return wave * envelope * 0.5  # Scale amplitude
    
    audio_parts = []
    
    # Arohanam (ascending)
    prev_semi = -1
    for i, swara in enumerate(arohanam):
        semi = SWARA_TO_SEMITONE.get(swara, 0)
        # Upper Sa: last note of arohanam if it's 'S' should be octave up
        is_upper_sa = (swara == 'S' and i == len(arohanam) - 1 and i > 0)
        octave_up = is_upper_sa or (semi <= prev_semi and prev_semi >= 0 and swara != arohanam[0])
        freq = swara_to_freq(swara, tonic_hz, octave_up=octave_up)
        audio_parts.append(generate_note(freq, note_duration, sample_rate))
        prev_semi = semi if not octave_up else semi + 12
    
    # Pause
    silence = np.zeros(int(sample_rate * pause_between))
    audio_parts.append(silence)
    
    # Avarohanam (descending) - start from upper Sa
    upper_hz = tonic_hz * 2  # Upper octave
    for i, swara in enumerate(avarohanam):
        semi = SWARA_TO_SEMITONE.get(swara, 0)
        # First note is upper Sa, then descend
        if i == 0 and swara == 'S':
            freq = upper_hz
        else:
            freq = tonic_hz * (2.0 ** (semi / 12.0))
        audio_parts.append(generate_note(freq, note_duration, sample_rate))
    
    # Concatenate
    audio = np.concatenate(audio_parts)
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    # Save
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    sf.write(output_path, audio, sample_rate)
    
    return output_path


def play_raga(
    raga_name: str,
    output_path: Optional[str] = None,
    format: str = 'midi',
    tonic: str = 'C4',
    tempo: int = 80,
    instrument: int = 73,
) -> Tuple[str, dict]:
    """
    Look up a raga and generate its scale as MIDI or WAV.
    
    Args:
        raga_name: Name of the raga (e.g., 'mohanam', 'kalyani')
        output_path: Output file path (auto-generated if None)
        format: 'midi' or 'wav'
        tonic: Tonic note (e.g., 'C4')
        tempo: BPM for MIDI
        instrument: GM instrument for MIDI
        
    Returns:
        Tuple of (output_path, raga_info_dict)
    """
    from raga_detection.swara_matcher import SwaraSequenceMatcher
    
    matcher = SwaraSequenceMatcher()
    raga = matcher.find_raga_by_name(raga_name)
    
    if raga is None:
        raise ValueError(f"Raga not found: {raga_name}")
    
    # Generate output path if not provided
    if output_path is None:
        safe_name = raga.id.replace(' ', '_').lower()
        ext = '.mid' if format == 'midi' else '.wav'
        output_path = f"output/{safe_name}_scale{ext}"
    
    raga_info = {
        'name': raga.name,
        'id': raga.id,
        'arohanam': raga.arohanam,
        'avarohanam': raga.avarohanam,
        'swara_count': raga.swara_count,
        'is_melakarta': raga.is_melakarta,
    }
    
    if format == 'midi':
        generate_midi(
            raga.arohanam, raga.avarohanam, output_path,
            tonic=tonic, tempo=tempo, instrument=instrument
        )
    else:
        tonic_midi = parse_tonic(tonic)
        tonic_hz = 440.0 * (2.0 ** ((tonic_midi - 69) / 12.0))
        generate_audio_wave(
            raga.arohanam, raga.avarohanam, output_path,
            tonic_hz=tonic_hz
        )
    
    return output_path, raga_info


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate audio for a raga scale')
    parser.add_argument('raga', help='Raga name (e.g., mohanam, kalyani)')
    parser.add_argument('--output', '-o', help='Output file path')
    parser.add_argument('--format', '-f', choices=['midi', 'wav'], default='midi',
                       help='Output format (default: midi)')
    parser.add_argument('--tonic', '-t', default='C4',
                       help='Tonic note (default: C4)')
    parser.add_argument('--tempo', type=int, default=80,
                       help='Tempo in BPM (default: 80)')
    parser.add_argument('--instrument', type=int, default=73,
                       help='MIDI instrument number (default: 73=flute)')
    parser.add_argument('--list-instruments', action='store_true',
                       help='List common GM instrument numbers')
    
    args = parser.parse_args()
    
    if args.list_instruments:
        instruments = {
            0: 'Acoustic Grand Piano',
            24: 'Nylon Guitar',
            40: 'Violin',
            42: 'Cello',
            68: 'Oboe',
            71: 'Clarinet',
            73: 'Flute',
            74: 'Recorder',
            104: 'Sitar',
            105: 'Banjo',
            109: 'Bagpipe',
            110: 'Fiddle',
            111: 'Shanai',
        }
        print("Common GM Instruments:")
        for num, name in sorted(instruments.items()):
            print(f"  {num:3d}: {name}")
        exit(0)
    
    try:
        path, info = play_raga(
            args.raga,
            output_path=args.output,
            format=args.format,
            tonic=args.tonic,
            tempo=args.tempo,
            instrument=args.instrument,
        )
        
        aro_str = ' '.join(info['arohanam'])
        ava_str = ' '.join(info['avarohanam'])
        
        print(f"Raga: {info['name']}")
        print(f"Arohanam:  {aro_str}")
        print(f"Avarohanam: {ava_str}")
        print(f"Swaras: {info['swara_count']} notes")
        if info['is_melakarta']:
            print(f"Melakarta raga")
        print(f"Tonic: {args.tonic}")
        print(f"Output: {path}")
        
    except ValueError as e:
        print(f"Error: {e}")
    except ImportError as e:
        print(f"Error: {e}")
