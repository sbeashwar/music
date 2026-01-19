import os
import argparse
import numpy as np
from utils import RagaGenerator

def generate_raga_music(raga_name, output_dir, duration_seconds=60, format='midi'):
    """
    Generate music in the specified raga and save it to a file.
    
    Args:
        raga_name (str): Name of the raga to generate
        output_dir (str): Directory to save the output file
        duration_seconds (int): Duration of the generated music in seconds
        format (str): Output format ('midi' or 'wav')
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the raga generator
    generator = RagaGenerator()
    
    # Generate the alapana
    print(f"Generating {raga_name} alapana...")
    notes = generator.generate_alapana(raga_name, duration_seconds)
    
    # Save the output
    if format.lower() == 'midi':
        output_path = os.path.join(output_dir, f"{raga_name}_alapana.mid")
        generator.save_to_midi(notes, output_path)
        print(f"MIDI file saved to {output_path}")
    else:
        # In a real implementation, you would synthesize the audio
        # For now, we'll just save as MIDI
        output_path = os.path.join(output_dir, f"{raga_name}_alapana.mid")
        generator.save_to_midi(notes, output_path)
        print(f"Audio generation for format '{format}' not implemented yet. Saved as MIDI: {output_path}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Generate music in a specific raga')
    parser.add_argument('raga', type=str, help='Name of the raga to generate')
    parser.add_argument('--output_dir', type=str, default='output',
                       help='Directory to save the output file')
    parser.add_argument('--duration', type=int, default=60,
                       help='Duration of the generated music in seconds')
    parser.add_argument('--format', type=str, default='midi',
                       choices=['midi', 'wav'],
                       help='Output format (midi or wav)')
    
    args = parser.parse_args()
    
    generate_raga_music(
        args.raga,
        args.output_dir,
        args.duration,
        args.format
    )

if __name__ == "__main__":
    main()
