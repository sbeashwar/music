"""
Raga Generator - Generate authentic Carnatic melodies

Features:
1. **Stepwise motion** along arohanam/avarohanam
2. **Gamakas** (ornaments) - kampita, jaru, nokku
3. **Varisai patterns** - sarali, janta, dhatu, alankarams
4. **Prayogams** - raga-specific signature phrases
5. MIDI output with pitch bends for gamakas

Key principles:
- Melodies move step-by-step along the scale
- Gamakas give life and expression to notes
- Characteristic phrases define raga identity
- Patterns like janta/dhatu add rhythmic interest
"""

import os
import random
import json
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

from .raga_db import RagaDB, Raga, SWARA_TO_SEMITONE, get_db
from .patterns import (
    GamakaType, GamakaNote, RAGA_PRAYOGAMS,
    generate_sarali_pattern, generate_janta_pattern,
    generate_dhatu_pattern, generate_alankaram,
    get_gamaka_for_swara, get_raga_prayogams
)

try:
    import pretty_midi
    import numpy as np
    HAS_MIDI = True
except ImportError:
    HAS_MIDI = False
    print("Warning: pretty_midi not installed. MIDI export disabled.")


@dataclass
class Note:
    """A single note in the melody."""
    swara: str
    octave: int  # 0 = mandra (low), 1 = madhya (middle), 2 = tara (high)
    duration: float  # in beats
    velocity: int = 100
    gamaka: GamakaType = GamakaType.PLAIN
    # For pitch bend effects
    pitch_bend_cents: List[Tuple[float, int]] = field(default_factory=list)


class RagaGenerator:
    """
    Generate authentic Carnatic melodies with gamakas and patterns.
    
    Features:
    - Stepwise motion along arohanam/avarohanam
    - Raga-specific prayogams (signature phrases)
    - Varisai patterns (janta, dhatu, alankarams)
    - Gamakas (oscillations, slides)
    """
    
    def __init__(self, db: Optional[RagaDB] = None):
        self.db = db or get_db()
        
        # MIDI settings
        self.tonic_midi = 60  # C4 as Sa
        self.tempo = 80  # BPM
        
        # Generation parameters
        self.phrase_length_range = (6, 16)
        
        # Feature toggles
        self.use_prayogams = True
        self.use_patterns = True
        self.use_gamakas = True
    
    def generate(
        self,
        raga_name: str,
        duration_beats: int = 32,
        style: str = 'alapana'
    ) -> List[Note]:
        """
        Generate a melody in the specified raga.
        
        Args:
            raga_name: Name of the raga
            duration_beats: Approximate length in beats
            style: 'alapana' (slow, free), 'kriti' (composed), 'tana' (fast)
            
        Returns:
            List of Note objects
        """
        raga = self.db.get(raga_name)
        if not raga:
            raise ValueError(f"Unknown raga: {raga_name}")
        
        # Build the melodic paths for this raga
        ascending_path = self._build_path(raga.arohanam)
        descending_path = self._build_path(raga.avarohanam)
        
        # Get raga-specific prayogams
        prayogams = get_raga_prayogams(raga_name)
        
        # Style affects durations and patterns
        if style == 'alapana':
            duration_range = (0.5, 2.0)
            phrase_lengths = (8, 20)
            pattern_probability = 0.15  # Less patterns in alapana
            prayogam_probability = 0.4  # More prayogams
        elif style == 'tana':
            duration_range = (0.15, 0.3)
            phrase_lengths = (16, 32)
            pattern_probability = 0.5  # Heavy patterns in tana
            prayogam_probability = 0.2
        else:  # kriti
            duration_range = (0.4, 1.0)
            phrase_lengths = (6, 12)
            pattern_probability = 0.3
            prayogam_probability = 0.3
        
        notes = []
        current_beat = 0
        
        # Start at Sa in middle octave
        current_pos = 0
        current_octave = 1
        direction = 1
        
        while current_beat < duration_beats:
            phrase_notes = []
            
            # Decide what to generate: prayogam, pattern, or stepwise phrase
            roll = random.random()
            
            if self.use_prayogams and prayogams and roll < prayogam_probability:
                # Insert a characteristic prayogam
                phrase_notes = self._generate_prayogam(
                    prayogams, raga_name, current_octave, duration_range
                )
            elif self.use_patterns and roll < prayogam_probability + pattern_probability:
                # Insert a pattern (janta, dhatu, or alankaram)
                phrase_notes = self._generate_pattern(
                    ascending_path, descending_path,
                    current_pos, current_octave, direction,
                    duration_range, style, raga_name
                )
            else:
                # Standard stepwise phrase
                phrase_len = random.randint(*phrase_lengths)
                phrase_notes = self._generate_phrase_stepwise(
                    ascending_path, descending_path,
                    current_pos, current_octave, direction,
                    phrase_len, duration_range, style, raga_name
                )
            
            for note in phrase_notes:
                notes.append(note)
                current_beat += note.duration
            
            # Update state from last note
            if phrase_notes:
                last_note = phrase_notes[-1]
                current_octave = last_note.octave
                try:
                    current_pos = ascending_path.index(last_note.swara)
                except ValueError:
                    current_pos = 0
                direction = -direction if random.random() < 0.5 else direction
        
        return notes
    
    def _build_path(self, scale_sequence: List[str]) -> List[str]:
        """Build a clean path from arohanam/avarohanam."""
        path = []
        for s in scale_sequence:
            if s not in path:
                path.append(s)
        return path
    
    def _generate_prayogam(
        self,
        prayogams: List[List[str]],
        raga_name: str,
        octave: int,
        duration_range: Tuple[float, float]
    ) -> List[Note]:
        """Generate a characteristic phrase (prayogam) for the raga."""
        phrase = random.choice(prayogams)
        notes = []
        
        for i, swara in enumerate(phrase):
            # Vary duration within phrase
            if i == 0 or i == len(phrase) - 1:
                dur = random.uniform(duration_range[1] * 0.7, duration_range[1])
            else:
                dur = random.uniform(duration_range[0], duration_range[0] * 1.5)
            
            dur = round(dur * 4) / 4
            dur = max(0.25, dur)
            
            # Apply gamaka
            gamaka = get_gamaka_for_swara(raga_name, swara, "long" if dur > 0.5 else "plain")
            
            notes.append(Note(
                swara=swara, octave=octave, duration=dur,
                velocity=95 if i == 0 else 85,
                gamaka=gamaka
            ))
        
        return notes
    
    def _generate_pattern(
        self,
        asc_path: List[str],
        desc_path: List[str],
        start_pos: int,
        octave: int,
        direction: int,
        duration_range: Tuple[float, float],
        style: str,
        raga_name: str
    ) -> List[Note]:
        """Generate a varisai pattern (janta, dhatu, or alankaram)."""
        
        # Choose pattern type
        pattern_type = random.choice(['janta', 'dhatu', 'alankaram'])
        
        # Get a segment of the scale to work with
        path = asc_path if direction > 0 else list(reversed(desc_path))
        segment_start = max(0, start_pos)
        segment = path[segment_start:segment_start + 4]
        
        if len(segment) < 3:
            segment = path[:4]  # Fallback to beginning
        
        # Generate the pattern
        if pattern_type == 'janta':
            # Doubled notes: SS RR GG MM
            swaras = generate_janta_pattern(segment)
            base_dur = duration_range[0] * 0.7  # Faster for janta
        elif pattern_type == 'dhatu':
            # Interlocking: SR RG GM MP
            swaras = generate_dhatu_pattern(segment, pattern_type=1)
            base_dur = duration_range[0] * 0.8
        else:
            # Alankaram: SRS RGR GMG
            swaras = generate_alankaram(segment, alankaram_type=1)
            base_dur = duration_range[0] * 0.6
        
        notes = []
        for i, swara in enumerate(swaras):
            # Slight variation in duration
            dur = base_dur * random.uniform(0.9, 1.1)
            dur = round(dur * 8) / 8  # Quantize to 32nd notes
            dur = max(0.125, dur)
            
            # Lighter gamakas in fast patterns
            gamaka = GamakaType.PLAIN if style == 'tana' else get_gamaka_for_swara(raga_name, swara, "plain")
            
            notes.append(Note(
                swara=swara, octave=octave, duration=dur,
                velocity=80 + random.randint(0, 15),
                gamaka=gamaka
            ))
        
        return notes
    
    def _generate_phrase_stepwise(
        self,
        asc_path: List[str],
        desc_path: List[str],
        start_pos: int,
        start_octave: int,
        initial_direction: int,
        length: int,
        duration_range: Tuple[float, float],
        style: str,
        raga_name: str
    ) -> List[Note]:
        """Generate a phrase using stepwise motion with gamakas."""
        notes = []
        pos = start_pos
        octave = start_octave
        direction = initial_direction
        
        path = asc_path
        path_len = len(path)
        recent_positions = []
        
        for i in range(length):
            swara = path[pos % path_len]
            
            # Determine duration
            if style == 'alapana':
                if swara in ('S', 'P') or i == length - 1:
                    dur = random.uniform(duration_range[1] * 0.8, duration_range[1])
                else:
                    dur = random.uniform(*duration_range)
            else:
                dur = random.uniform(*duration_range)
            
            dur = round(dur * 4) / 4
            dur = max(0.25, dur)
            
            # Apply gamaka based on note and context
            context = "long" if dur > 0.75 else "plain"
            gamaka = get_gamaka_for_swara(raga_name, swara, context) if self.use_gamakas else GamakaType.PLAIN
            
            notes.append(Note(
                swara=swara, octave=octave, duration=dur,
                velocity=90, gamaka=gamaka
            ))
            recent_positions.append(pos)
            
            # Movement logic
            if random.random() < 0.85:
                new_pos = pos + direction
            else:
                direction = -direction
                new_pos = pos + direction
            
            # Octave boundaries
            if new_pos >= path_len:
                if octave < 2:
                    octave += 1
                    new_pos = 1
                else:
                    direction = -1
                    new_pos = pos - 1
            elif new_pos < 0:
                if octave > 0:
                    octave -= 1
                    new_pos = path_len - 2
                else:
                    direction = 1
                    new_pos = 1
            
            # Avoid repetition
            if len(recent_positions) >= 4 and len(set(recent_positions[-4:])) <= 2:
                new_pos = pos + direction * 2
                new_pos = max(0, min(path_len - 1, new_pos))
            
            pos = new_pos
            
            # Natural phrase ending
            if i >= length - 3 and swara != 'S' and pos > 0:
                direction = -1
        
        return notes
    
    def to_midi(
        self,
        notes: List[Note],
        output_path: str,
        tempo: Optional[int] = None
    ) -> str:
        """
        Convert generated notes to a MIDI file with gamaka pitch bends.
        
        Args:
            notes: List of Note objects
            output_path: Path for output MIDI file
            tempo: BPM (default: self.tempo)
            
        Returns:
            Path to created MIDI file
        """
        if not HAS_MIDI:
            raise ImportError("pretty_midi is required for MIDI export")
        
        tempo = tempo or self.tempo
        
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        instrument = pretty_midi.Instrument(program=0)  # Piano
        
        current_time = 0.0
        seconds_per_beat = 60.0 / tempo
        
        for note in notes:
            # Calculate MIDI pitch
            semitones = SWARA_TO_SEMITONE.get(note.swara, 0)
            midi_pitch = self.tonic_midi + semitones + (note.octave - 1) * 12
            midi_pitch = max(0, min(127, midi_pitch))
            
            # Calculate timing
            duration_seconds = note.duration * seconds_per_beat
            
            midi_note = pretty_midi.Note(
                velocity=note.velocity,
                pitch=midi_pitch,
                start=current_time,
                end=current_time + duration_seconds
            )
            instrument.notes.append(midi_note)
            
            # Add pitch bend for gamakas
            if hasattr(note, 'gamaka') and note.gamaka != GamakaType.PLAIN:
                self._add_gamaka_pitch_bend(
                    instrument, note.gamaka,
                    current_time, duration_seconds
                )
            
            current_time += duration_seconds
        
        midi.instruments.append(instrument)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        midi.write(output_path)
        
        return output_path
    
    def _add_gamaka_pitch_bend(
        self,
        instrument: 'pretty_midi.Instrument',
        gamaka: GamakaType,
        start_time: float,
        duration: float
    ):
        """Add pitch bend events to simulate gamakas."""
        if not HAS_MIDI:
            return
        
        # Pitch bend range: -8192 to 8191 (typically ±2 semitones)
        # 4096 = 1 semitone
        
        if gamaka == GamakaType.KAMPITA:
            # Oscillation - sine wave pattern
            num_oscillations = max(2, int(duration * 4))  # ~4 oscillations per second
            for i in range(num_oscillations * 4):
                t = start_time + (duration * i / (num_oscillations * 4))
                # Oscillate ±50 cents (±quarter tone)
                import math
                bend = int(2048 * math.sin(2 * math.pi * i / 4))
                instrument.pitch_bends.append(
                    pretty_midi.PitchBend(pitch=bend, time=t)
                )
            # Return to center at end
            instrument.pitch_bends.append(
                pretty_midi.PitchBend(pitch=0, time=start_time + duration)
            )
        
        elif gamaka == GamakaType.JARU:
            # Slide - start below, slide up
            instrument.pitch_bends.append(
                pretty_midi.PitchBend(pitch=-2048, time=start_time)
            )
            instrument.pitch_bends.append(
                pretty_midi.PitchBend(pitch=0, time=start_time + duration * 0.3)
            )
        
        elif gamaka == GamakaType.NOKKU:
            # Stress - slight bend up then back
            instrument.pitch_bends.append(
                pretty_midi.PitchBend(pitch=1024, time=start_time)
            )
            instrument.pitch_bends.append(
                pretty_midi.PitchBend(pitch=0, time=start_time + duration * 0.2)
            )
    
    def to_swara_string(self, notes: List[Note]) -> str:
        """Convert notes to swara notation string."""
        parts = []
        for note in notes:
            s = note.swara
            if note.octave == 0:
                s = s.lower()  # Mandra sthayi in lowercase
            elif note.octave == 2:
                s = s + "'"  # Tara sthayi with apostrophe
            parts.append(s)
        return ' '.join(parts)


def generate_raga(
    raga_name: str,
    output_path: Optional[str] = None,
    duration_beats: int = 32,
    style: str = 'alapana'
) -> str:
    """
    Convenience function to generate a raga melody.
    
    Example:
        midi_path = generate_raga("mohanam", "output.mid", duration_beats=64)
    """
    generator = RagaGenerator()
    notes = generator.generate(raga_name, duration_beats, style)
    
    if output_path is None:
        output_path = f"{raga_name}_generated.mid"
    
    return generator.to_midi(notes, output_path)
