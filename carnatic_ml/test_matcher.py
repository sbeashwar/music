"""Quick test of the swara sequence matcher."""
from raga_detection.swara_matcher import SwaraSequenceMatcher, format_match_result

matcher = SwaraSequenceMatcher()

tests = [
    ('Mohanam', ['S', 'R2', 'G3', 'P', 'D2', 'S']),
    ('Kalyani', ['S', 'R2', 'G3', 'M2', 'P', 'D2', 'N3', 'S']),
    ('Bahudari', ['S', 'G3', 'M1', 'P', 'D2', 'N2', 'S']),
    ('Shankarabharanam', ['S', 'R2', 'G3', 'M1', 'P', 'D2', 'N3', 'S']),
    ('Hamsadhwani', ['S', 'R2', 'G3', 'P', 'N3', 'S']),
    ('Kharaharapriya', ['S', 'R2', 'G2', 'M1', 'P', 'D2', 'N2', 'S']),
    ('Todi', ['S', 'R1', 'G2', 'M2', 'P', 'D1', 'N2', 'S']),
]

for name, swaras in tests:
    swaras_str = ' '.join(swaras)
    print(f'=== {name} ({swaras_str}) ===')
    results = matcher.match_swaras(swaras, direction='ascending', max_results=5)
    for i, m in enumerate(results[:5], 1):
        marker = ' <<<' if name.lower() in m.raga_name.lower() or name.lower() in m.raga_id else ''
        print(f'  {i}. {m.raga_name:30s} score={m.score:.3f} ({m.match_type}){marker}')
    print()

# Also test with noisy input (extra notes from gamaka detection)
print('=== Noisy Bahudari (S R2 G3 M1 P D2 D3 N2 S) - extra R2, D3 ===')
results = matcher.match_swaras(['S', 'R2', 'G3', 'M1', 'P', 'D2', 'D3', 'N2', 'S'], direction='ascending', max_results=5)
for i, m in enumerate(results[:5], 1):
    marker = ' <<<' if 'bahud' in m.raga_id else ''
    print(f'  {i}. {m.raga_name:30s} score={m.score:.3f} ({m.match_type}){marker}')
print()

# Test lookup by name
print('=== Lookup: Mohanam ===')
raga = matcher.find_raga_by_name('mohanam')
if raga:
    print(f'  Found: {raga.name}')
    print(f'  Arohanam: {" ".join(raga.arohanam)}')
    print(f'  Avarohanam: {" ".join(raga.avarohanam)}')
    print(f'  Swara count: {raga.swara_count}')
