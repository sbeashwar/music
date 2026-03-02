import json, os

target = {'R2', 'G3', 'M1', 'P', 'D2', 'N3'}
matches = []
d = 'shared/ragas_metadata'

for f in os.listdir(d):
    if not f.endswith('.json'):
        continue
    with open(os.path.join(d, f), encoding='utf-8') as fh:
        j = json.load(fh)
    aro = j.get('arohanam', [])
    ava = j.get('avarohanam', [])
    all_swaras = set(s for s in aro + ava if s != 'S')
    if all_swaras == target:
        matches.append((j.get('name', '?'), j.get('id', '?'), j.get('is_melakarta', False), aro))

matches.sort(key=lambda x: x[0])
print(f"Ragas with exact same swara set as Mand ({len(matches)} total):")
for name, rid, mela, aro in matches:
    tag = " [MELAKARTA]" if mela else ""
    print(f"  {name:30s}  aro: {' '.join(aro)}{tag}")
