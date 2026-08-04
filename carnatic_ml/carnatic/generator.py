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
    get_gamaka_for_swara, get_raga_prayogams, get_jiva_swaras
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
                # Insert a varisai (janta, sarali, dhatu, alankaram) or a
                # raga-unique motif — all built from the arohana/avarohana
                # ladders so they respect varja & vakra grammar.
                phrase_notes = self._generate_pattern(
                    raga.arohanam, raga.avarohanam,
                    current_octave, direction,
                    duration_range, style, raga_name
                )
            else:
                # Standard stepwise phrase — faithful to arohanam (ascent) and
                # avarohanam (descent), preserving varja & vakra structure.
                phrase_len = random.randint(*phrase_lengths)
                phrase_notes = self._generate_phrase_faithful(
                    raga.arohanam, raga.avarohanam,
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
        arohanam: List[str],
        avarohanam: List[str],
        octave: int,
        direction: int,
        duration_range: Tuple[float, float],
        style: str,
        raga_name: str
    ) -> List[Note]:
        """
        Generate a varisai (sarali / janta / dhatu / alankaram) or a raga-unique
        motif, built from the arohana/avarohana ladders so it stays in-grammar
        for varja (direction-specific swaras) and vakra (repeated swaras) ragas.

        Each pattern swara carries its correct octave (derived from the ladder
        semitone), so patterns that touch the upper/lower Sa wrap properly.
        """
        A, B = self._raga_ladders(arohanam, avarohanam)
        ladder = A if direction > 0 else B          # ascend=arohanam, descend=avarohanam
        if len(ladder) < 3:
            ladder = A if len(A) >= 3 else B

        # Pick a window of the ladder to work over (preserves literal order,
        # including any vakra repeats inside the window).
        win = min(random.choice([4, 5]), len(ladder))
        max_start = max(0, len(ladder) - win)
        start = random.randint(0, max_start)
        segment = ladder[start:start + win]         # list of (swara, semitone)
        semi_of = {s: semi for s, semi in segment}  # swara -> in-octave semitone

        seg_swaras = [s for s, _ in segment]

        # ── choose a pattern; sarali & a raga-unique motif are now included ──
        ptype = random.choice(['sarali', 'sarali', 'janta', 'dhatu',
                               'alankaram', 'unique'])

        if ptype == 'sarali':
            swaras = generate_sarali_pattern(seg_swaras, pattern_type=random.choice([2, 3]))
            base_dur = duration_range[0] * 0.9
        elif ptype == 'janta':
            swaras = generate_janta_pattern(seg_swaras)
            base_dur = duration_range[0] * 0.7
        elif ptype == 'dhatu':
            swaras = generate_dhatu_pattern(seg_swaras, pattern_type=random.choice([1, 3]))
            base_dur = duration_range[0] * 0.8
        elif ptype == 'alankaram':
            swaras = generate_alankaram(seg_swaras, alankaram_type=random.choice([1, 2, 3]))
            base_dur = duration_range[0] * 0.6
        else:  # 'unique' — a raga-signature motif (prayogam), else a scale chunk
            prayogams = get_raga_prayogams(raga_name)
            if prayogams:
                swaras = random.choice(prayogams)[:]
            else:
                swaras = seg_swaras[:]              # in-grammar fallback
            base_dur = duration_range[0] * 1.0
            semi_of = None                          # prayogam swaras: recompute octave

        notes = []
        jiva = get_jiva_swaras(raga_name)
        for swara in swaras:
            # Octave for this swara: from the ladder window when available,
            # otherwise recompute from its base semitone.
            if semi_of is not None and swara in semi_of:
                semi = semi_of[swara]
            else:
                semi = SWARA_TO_SEMITONE.get(swara, 0)
            note_octave = max(0, min(2, octave + semi // 12))

            dur = base_dur * random.uniform(0.9, 1.1)
            dur = round(dur * 8) / 8
            dur = max(0.125, dur)

            gamaka = (GamakaType.PLAIN if style == 'tana'
                      else get_gamaka_for_swara(raga_name, swara, "plain"))
            if swara in jiva and style != 'tana' and gamaka == GamakaType.PLAIN:
                gamaka = GamakaType.KAMPITA

            notes.append(Note(
                swara=swara, octave=note_octave, duration=dur,
                velocity=80 + random.randint(0, 15),
                gamaka=gamaka
            ))

        return notes
    
    def _raga_ladders(self, arohanam: List[str], avarohanam: List[str]):
        """
        Build absolute-within-octave semitone ladders for both scales.

        Each ladder is a list of (swara, semitone) where semitone is 0..12
        (0 = lower Sa, 12 = upper Sa of the same sthayi). The literal
        arohanam/avarohanam order is preserved, so vakra repeats and
        varja (direction-specific) swaras are kept intact.
        """
        def build(seq, is_aro):
            n = len(seq)
            out = []
            for i, s in enumerate(seq):
                if s == 'S':
                    # Frame the octave: aro ends on upper Sa, ava starts on it.
                    if is_aro:
                        semi = 12 if i == n - 1 else 0
                    else:
                        semi = 0 if i == n - 1 else 12
                else:
                    semi = SWARA_TO_SEMITONE.get(s, 0)
                out.append((s, semi))
            return out
        return build(arohanam, True), build(avarohanam, False)

    @staticmethod
    def _nearest_idx(ladder, target_semi: int) -> int:
        """Index of the ladder entry whose semitone is closest to target."""
        best, best_d = 0, 99
        for i, (_, semi) in enumerate(ladder):
            d = abs(semi - target_semi)
            if d < best_d:
                best_d, best = d, i
        return best

    def _faithful_line(self, arohanam: List[str], avarohanam: List[str],
                       length: int) -> List[Tuple[str, int]]:
        """
        Produce an ordered (swara, octave) line that ascends along the
        arohanam and descends along the avarohanam.

        - Varja: descent uses the avarohanam's own swaras (e.g. Kambhoji's N2).
        - Vakra: repeated swaras in either scale are reproduced (e.g. Sahana's
          P M1 D2, Begada's G3 R2 G3), because we walk the literal sequence.
        """
        A, B = self._raga_ladders(arohanam, avarohanam)
        if len(A) < 2 or len(B) < 2:
            return [('S', 1)]

        out: List[Tuple[str, int]] = []
        oct_ = 1          # 1 = madhya sthayi
        cur_abs = 0       # absolute semitone from madhya Sa
        up = True
        max_oct, min_oct = 2, 0
        guard = 0

        while len(out) < length and guard < length * 6:
            guard += 1
            ladder = A if up else B
            csemi = cur_abs - (oct_ - 1) * 12           # semitone within octave
            i = self._nearest_idx(ladder, csemi)
            run = random.randint(3, len(ladder))
            end = min(i + run, len(ladder))
            if end <= i:
                end = i + 1

            for k in range(i, end):
                s, semi = ladder[k]
                ab = (oct_ - 1) * 12 + semi
                octave = max(0, min(2, 1 + ab // 12))
                cur_abs = ab
                # Collapse adjacent duplicates from run seams (e.g. 'S S' at a
                # turn). Non-adjacent vakra repeats (S G3 R2 G3) are untouched.
                if out and out[-1] == (s, octave):
                    continue
                out.append((s, octave))

            reached_end = (end == len(ladder))
            if up:
                if reached_end:
                    if oct_ < max_oct and random.random() < 0.35:
                        oct_ += 1          # keep ascending into the next sthayi
                    else:
                        up = False         # turn: descend along avarohanam
                else:
                    up = False             # mid-phrase reversal -> descend
            else:
                if reached_end:
                    if oct_ > min_oct and random.random() < 0.35:
                        oct_ -= 1          # keep descending into the lower sthayi
                    else:
                        up = True          # turn: ascend along arohanam
                else:
                    up = True              # mid-phrase reversal -> ascend

        return out[:length]

    def _generate_phrase_faithful(
        self,
        arohanam: List[str],
        avarohanam: List[str],
        length: int,
        duration_range: Tuple[float, float],
        style: str,
        raga_name: str
    ) -> List[Note]:
        """Wrap a faithful arohana/avarohana line with durations + gamakas."""
        line = self._faithful_line(arohanam, avarohanam, length)
        jiva = get_jiva_swaras(raga_name)
        notes = []
        n = len(line)

        for i, (swara, octave) in enumerate(line):
            # Duration: dwell on Sa/Pa and phrase endings in alapana.
            if style == 'alapana':
                if swara in ('S', 'P') or i == n - 1:
                    dur = random.uniform(duration_range[1] * 0.8, duration_range[1])
                else:
                    dur = random.uniform(*duration_range)
            else:
                dur = random.uniform(*duration_range)
            dur = round(dur * 4) / 4
            dur = max(0.25, dur)

            context = "long" if dur > 0.75 else "plain"
            gamaka = get_gamaka_for_swara(raga_name, swara, context) \
                if self.use_gamakas else GamakaType.PLAIN

            # Emphasise jeeva swaras: longer dwell + ensure a gamaka.
            if swara in jiva:
                dur = round(min(dur * 1.5, duration_range[1] * 1.5) * 4) / 4
                if self.use_gamakas and gamaka == GamakaType.PLAIN:
                    gamaka = GamakaType.KAMPITA

            notes.append(Note(swara=swara, octave=octave, duration=dur,
                              velocity=90, gamaka=gamaka))

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
        jiva = get_jiva_swaras(raga_name)

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

            # Emphasise jeeva (life) swaras: dwell longer and ensure a gamaka.
            # e.g. Kalyani's M2 (prati madhyama) should be held and oscillated,
            # not passed through as a plain note.
            if swara in jiva:
                dur = round(min(dur * 1.5, duration_range[1] * 1.5) * 4) / 4
                if self.use_gamakas and gamaka == GamakaType.PLAIN:
                    gamaka = GamakaType.KAMPITA

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
                    new_pos = 0  # land on the octave Sa, don't skip it
                else:
                    direction = -1
                    new_pos = pos - 1
            elif new_pos < 0:
                if octave > 0:
                    octave -= 1
                    # the note just below Sa is the last swara of the scale
                    # (e.g. N for Kalyani) — must not be skipped in descent
                    new_pos = path_len - 1
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
