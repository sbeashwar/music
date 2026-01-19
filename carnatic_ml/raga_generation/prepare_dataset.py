import json
import os

# Simple preparer that converts raga definition phrases into training sequences
# Each sequence is a list of swaras; output is saved as a .json list of sequences

RAGA_DIR = os.path.join(os.path.dirname(__file__), '..', 'shared', 'raga_definitions')
OUT_DIR = os.path.join(os.path.dirname(__file__), 'data')

os.makedirs(OUT_DIR, exist_ok=True)

for file in os.listdir(RAGA_DIR):
    if not file.endswith('.json'):
        continue
    path = os.path.join(RAGA_DIR, file)
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    raga_name = data.get('name', os.path.splitext(file)[0]).lower()
    phrases = data.get('phrases', [])
    # expand phrases into sequences of tokens
    sequences = []
    for p in phrases:
        # create a few variations: as-is, repeated, and with simple ornamentation
        sequences.append(p)
        sequences.append(p + list(reversed(p[:-1])))
    out_path = os.path.join(OUT_DIR, f"{raga_name}_sequences.json")
    with open(out_path, 'w', encoding='utf-8') as out:
        json.dump(sequences, out, ensure_ascii=False, indent=2)
    print(f"Wrote {out_path} with {len(sequences)} sequences")
