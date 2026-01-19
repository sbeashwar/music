import os
import argparse
import time

# Ensure dataset and models are created/importable
import raga_generation.prepare_dataset  # when imported this writes sequence files
from raga_generation import RagaGenerator

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')
MODEL_FILE = os.path.join(MODEL_DIR, 'small_raga_gen.h5')
TOKENIZER_FILE = os.path.join(MODEL_DIR, 'tokenizer.json')


def ensure_trained():
    if os.path.exists(MODEL_FILE) and os.path.exists(TOKENIZER_FILE):
        print('Model and tokenizer already present.')
        return
    print('Model not found - training a small model (this may take a little while)...')
    # import and run the train function
    from raga_generation import train_small
    train_small.train()


def generate_for_raga(raga_name, duration_seconds, out_path, temperature=0.8):
    # Ensure model exists
    ensure_trained()

    # Use the sampler to get a generated swara sequence
    try:
        from raga_generation.sample import sample
    except Exception as e:
        raise RuntimeError('Sampler unavailable: ' + str(e))

    # Prepare seed phrase from raga definitions
    rg = RagaGenerator()
    rd = rg.get_raga_definition(raga_name)
    seed = rd['phrases'][0] if rd and rd.get('phrases') else ['S','R2','G3']

    from raga_generation.tokenizer import SwaraTokenizer
    tok = SwaraTokenizer()
    seed_ids = tok.encode(seed)

    print(f'Sampling for raga="{raga_name}" using seed {seed}...')
    tokens = sample(seed_ids, length=max(8, int(duration_seconds/0.5)), temperature=temperature)

    # tokens is a list of swara strings
    # convert to MIDI notes using RagaGenerator mapping
    notes = []
    time_cursor = 0.0
    dur = 0.5
    # map Sa to C3 (MIDI 48) so Ableton displays notes in expected octave
    tonic_midi = 48
    for swara in tokens:
        pitch = rg.swara_to_midi(swara, octave=4, tonic_midi=tonic_midi)
        notes.append({'pitch': pitch, 'start': time_cursor, 'end': time_cursor + dur, 'velocity': 100})
        time_cursor += dur

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rg.save_to_midi(notes, out_path)
    print('Saved MIDI to', out_path)
    return out_path


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('raga', help='Raga name to generate (e.g. mohanam)')
    p.add_argument('--duration', type=int, default=8, help='Desired duration in seconds (approx)')
    p.add_argument('--out', default=None, help='Output MIDI path')
    p.add_argument('--temperature', type=float, default=0.8)
    args = p.parse_args()

    out = args.out or os.path.join(os.path.dirname(__file__), f"{args.raga}_demo_{int(time.time())}.mid")
    path = generate_for_raga(args.raga, args.duration, out, temperature=args.temperature)
    print('Done:', path)
