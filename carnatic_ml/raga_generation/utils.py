import json
import numpy as np
import pretty_midi
import os
import random
from tqdm import tqdm


class RagaGenerator:
    def __init__(self, sample_rate=22050, bpm=80):
        self.sample_rate = sample_rate
        self.bpm = bpm
        self.raga_definitions = self._load_raga_definitions()

    def _load_raga_definitions(self):
        """Load structured raga definitions from shared/ragas_metadata.

        The metadata JSONs contain:
          - name
          - arohanam: list of swaras (ascending)
          - avarohanam: list of swaras (descending)
          - raga_lakshana: detailed constraints and features
          - raga_classification: characteristics
        """
        raga_defs = {}
        raga_dir = os.path.join(os.path.dirname(__file__), '..', 'shared', 'ragas_metadata')

        if not os.path.exists(raga_dir):
            # Fallback to older raga_definitions directory
            raga_dir = os.path.join(os.path.dirname(__file__), '..', 'shared', 'raga_definitions')
            if not os.path.exists(raga_dir):
                os.makedirs(raga_dir, exist_ok=True)
                return raga_defs

        for file in os.listdir(raga_dir):
            if file.endswith('.json'):
                path = os.path.join(raga_dir, file)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        key = data.get('name', os.path.splitext(file)[0]).lower()
                        raga_defs[key] = data
                except Exception as e:
                    print(f"Error loading {file}: {str(e)}")
                    continue

        return raga_defs

    def get_raga_definition(self, raga_name):
        return self.raga_definitions.get(raga_name.lower())

    def get_raga_constraints(self, raga_name):
        """Get comprehensive constraints for the raga from metadata.

        Returns:
            Dictionary containing:
            - allowed_swaras: List of allowed swaras
            - scale: Primary ascending scale
            - rules: Generation rules based on raga characteristics
            - emphasis: Important notes (vadi, samvadi, etc.)
        """
        rd = self.get_raga_definition(raga_name)
        if rd:
            # Get primary scale
            scale = rd.get('arohanam', [])
            
            # Get allowed swaras by removing varjya (excluded) swaras
            all_swaras = {'S', 'R1', 'R2', 'R3', 'G1', 'G2', 'G3', 'M1', 'M2', 
                         'P', 'D1', 'D2', 'D3', 'N1', 'N2', 'N3'}
            varjya = set(rd.get('raga_lakshana', {}).get('varjya_swaras', []))
            allowed_swaras = list(all_swaras - varjya)

            # Get emphasis notes
            lakshana = rd.get('raga_lakshana', {})
            emphasis = {
                'vadi': lakshana.get('vadi_swara', ''),
                'samvadi': lakshana.get('samvadi_swara', ''),
                'jeeva': lakshana.get('jeeva_swaras', []),
                'nyasa': lakshana.get('nyasa_swaras', [])
            }

            # Define melody generation rules
            rules = {
                'vakra': lakshana.get('vakra_swaras', []),  # zigzag patterns
                'varjya': list(varjya),  # prohibited notes
                'ascending': rd.get('arohanam', []),  # ascending pattern
                'descending': rd.get('avarohanam', []),  # descending pattern
            }

            return {
                'allowed_swaras': allowed_swaras,
                'scale': scale,
                'rules': rules,
                'emphasis': emphasis
            }

        # Fallback to basic scales if metadata not found
        raga_scales = {
            'mohanam': {
                'allowed_swaras': ['S', 'R2', 'G3', 'P', 'D2'],
                'scale': ['S', 'R2', 'G3', 'P', 'D2'],
                'rules': {'ascending': [], 'descending': [], 'vakra': [], 'varjya': []},
                'emphasis': {'vadi': '', 'samvadi': '', 'jeeva': [], 'nyasa': []}
            },
            'shankarabharanam': {
                'allowed_swaras': ['S', 'R2', 'G3', 'M1', 'P', 'D2', 'N3'],
                'scale': ['S', 'R2', 'G3', 'M1', 'P', 'D2', 'N3'],
                'rules': {'ascending': [], 'descending': [], 'vakra': [], 'varjya': []},
                'emphasis': {'vadi': '', 'samvadi': '', 'jeeva': [], 'nyasa': []}
            }
        }
        return raga_scales.get(raga_name.lower(), raga_scales['mohanam'])

    def get_raga_scale(self, raga_name):
        """Get the canonical scale for the raga (arohanam if available)."""
        constraints = self.get_raga_constraints(raga_name)
        return constraints['scale']

    def generate_alapana(self, raga_name, duration_seconds=60):
        """Generate an alapana in the given raga using metadata-based constraints.

        The generator will create melodic phrases that:
        1. Use only allowed swaras
        2. Follow ascending/descending patterns
        3. Include special movement patterns (vakra)
        4. Emphasize important notes (vadi, samvadi, etc.)
        5. End phrases on nyasa swaras
        """
        from . import helper_functions

        constraints = self.get_raga_constraints(raga_name)
        
        # Extract components from constraints
        allowed_swaras = constraints['allowed_swaras']
        scale = constraints['scale']
        rules = constraints['rules']
        emphasis = constraints['emphasis']

        # Build phrase templates using rules
        phrases = []
        
        # Add ascending and descending patterns
        if rules['ascending']:
            phrases.append(rules['ascending'])
        if rules['descending']:
            phrases.append(rules['descending'])

        # Add variations using emphasis notes
        vadi = emphasis['vadi']
        samvadi = emphasis['samvadi']
        jeeva = emphasis['jeeva']
        nyasa = emphasis['nyasa']

        if vadi and samvadi:
            # Create phrases emphasizing vadi-samvadi relationship
            phrases.append(helper_functions.create_emphasis_phrase([vadi, samvadi], allowed_swaras))

        if jeeva:
            # Create phrases around jeeva swaras
            phrases.append(helper_functions.create_emphasis_phrase(jeeva, allowed_swaras))

        # Ensure we have at least basic scale movement if no other phrases
        if not phrases:
            phrases = [scale, list(reversed(scale[:-1]))]
            
        # Generate the alapana
        notes = []  # list of (swara, octave, duration)
        current_octave = 4
        time = 0.0
        
        while time < duration_seconds:
            # Choose phrase and adjust for nyasa swaras
            phrase = random.choice(phrases)
            if nyasa and random.random() < 0.7:  # 70% chance to end on nyasa
                phrase = helper_functions.adjust_phrase_ending(phrase, nyasa)

            # Add dynamics through duration variation
            for swara in phrase:
                if swara in jeeva:
                    # Emphasize jeeva swaras with longer duration
                    duration = random.choice([0.5, 0.75, 1.0])
                elif swara in emphasis['vadi']:
                    # Emphasize vadi with slightly longer duration
                    duration = random.choice([0.5, 0.75])
                else:
                    duration = 0.5

                notes.append((swara, current_octave, duration))
                time += duration
                if time >= duration_seconds:
                    break

        # Convert to MIDI events
        midi_notes = []
        time = 0.0
        for swara, octave, duration in notes:
            midi_note = self._swara_to_midi(swara, octave)
            midi_notes.append({'pitch': midi_note, 'start': time, 'end': time + duration, 'velocity': 100})
            time += duration

        return midi_notes

    def _swara_to_midi(self, swara, octave):
        # keep for backward compatibility
        return self.swara_to_midi(swara, octave, tonic_midi=60)

    def swara_to_midi(self, swara, octave=4, tonic_midi=60):
        """Convert a swara token to a MIDI pitch number using a selectable tonic.

        Args:
            swara (str): swara token like 'S','R2','G3'
            octave (int): octave relative to middle reference (4)
            tonic_midi (int): MIDI note number for Sa at octave reference (default C4=60)

        Returns:
            int: MIDI pitch number
        """
        swara_map = {
            'S': 0,
            'R1': 1, 'R2': 2, 'R3': 3,
            'G1': 1, 'G2': 3, 'G3': 4,
            'M1': 5, 'M2': 6,
            'P': 7,
            'D1': 8, 'D2': 9, 'D3': 10,
            'N1': 8, 'N2': 10, 'N3': 11,
        }

        base = int(tonic_midi)
        key = str(swara).upper()
        if key in swara_map:
            offset = swara_map[key]
        else:
            # try to extract alphabetic part
            import re
            m = re.match(r'([A-Za-z]+)(\d*)', key)
            if m:
                k = m.group(1)
                offset = swara_map.get(k, 0)
            else:
                offset = 0

        # octave adjustment relative to reference octave 4
        return base + offset + (octave - 4) * 12

    def save_to_midi(self, notes, output_path):
        midi = pretty_midi.PrettyMIDI()
        piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
        piano = pretty_midi.Instrument(program=piano_program)

        for note in notes:
            midi_note = pretty_midi.Note(velocity=note['velocity'], pitch=note['pitch'], start=note['start'], end=note['end'])
            piano.notes.append(midi_note)

        midi.instruments.append(piano)
        midi.write(output_path)
        return output_path
