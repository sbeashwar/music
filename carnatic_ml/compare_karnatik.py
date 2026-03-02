"""Compare our raga metadata with karnatik.com data for B-ragas."""
import json, os

# Karnatik.com data: name -> (melakarta_number, arohanam_str, avarohanam_str)
karnatik = {
    'balacandrika': (22, 'S G2 M1 P D2 N2 S', 'S N2 D2 M1 G2 R2 S'),
    'balahamsa': (28, 'S R2 M1 P D2 S', 'S N2 D2 P M1 R2 M1 G3 S'),
    'balamurali': (27, 'S R2 G3 P D1 S', 'S D1 P G3 R2 S R2 G2 S'),
    'bahudari': (28, 'S G3 M1 P D2 N2 S', 'S N2 P M1 G3 S'),
    'bandhuvarali': (45, 'S M2 S N3 D1 P M2', 'D1 M2 G2 R1 S'),
    'begada': (29, 'S G3 R2 G3 M1 P D2 N2 D2 P', 'S N3 D2 P M1 G3 R2 S'),
    'behag': (29, 'S G3 M1 P N3 D2 N3 S', 'S N3 D2 P M1 G3 R2 S'),
    'bhageshri': (22, 'S G2 M1 D2 N2 S', 'S N2 D2 M1 P D2 G2 M1 R2 S'),
    'bhagyashabari': (10, 'S R1 G2 M1 D2 N2 S', 'S N2 D2 M1 G2 R1 S'),
    'bhanucandrika': (8, 'S M1 D1 N2 S', 'S N2 D1 M1 G1 S'),
    'bhanudhanyasi': (45, 'S R1 G2 M2 N3 D1 N3', 'D1 P M2 G2 R1 S N3 S'),
    'bhanukiravani': (45, 'S R1 G2 M2 P D1 N3 S', 'S N3 D1 M2 G2 R1 S'),
    'bhanumanjari': (34, 'S R3 G3 M1 P N2 S', 'S N2 P M1 R3 G3 R3 S'),
    'bhanupriya': (21, 'S R2 G2 D1 N3 S', 'S N3 D1 G2 R2 S'),
    'bharati': (19, 'S R2 G2 M1 P S', 'S P M1 G2 R2 S'),
    'bhashini': (56, 'S G2 R2 G2 M2 P D1 N2 S', 'S N2 D1 P M2 G2 R1 S'),
    'bhavapriya': (44, 'S R1 G2 M2 P D1 N2 S', 'S N2 D1 P M2 G2 R1 S'),
    'bhavini': (15, 'S G3 M1 P D1 N3 S', 'S N3 D1 P M1 G3 S'),
    'bhairavam': (17, 'S R1 G3 M1 P D2 N3 S', 'S D2 P M1 G3 R1 S'),
    'bhairavi': (20, 'S R2 G2 M1 P D2 N2 S', 'S N2 D1 P M1 G2 R2 S'),
    'bhaktapriya': (16, 'S G3 M1 P D2 N2 S', 'S N2 D2 P M1 R1 M1 G3 S'),
    'bhimplas': (22, 'N2 S G2 M1 P N2 S', 'S N2 D2 P M1 G2 R2 S'),
    'bhinnapancamam': (3, 'S R1 G1 M1 P D1 N3 S', 'S N3 D1 P M1 G1 R1 S'),
    'bhogasaveri': (37, 'S R1 M2 D1 N1', 'D1 P M2 G1 R1 S'),
    'bhogavasanta': (51, 'S R1 G3 M2 D1 N3 S', 'S N3 D1 M2 G3 R1 S'),
    'bhogishwari': (64, 'S R2 G3 P D2 N2 D2 S', 'S N2 D2 P M2 G3 R2 S'),
    'bhogi': (7, 'S G2 M1 P D1 N1 D1 S', 'S N1 D1 P M1 G2 S'),
    'bhupalam': (8, 'S R1 G2 P D1 S', 'S D1 P G2 R1 S'),
    'bhupali': (28, 'S R2 G3 P D2 S', 'S D2 P G3 R2 S'),
    'bhupkalyani': (65, 'S R2 G3 P D2 S', 'S N3 D2 P M2 G3 R2 S'),
    'bhushavali': (64, 'S R2 G3 M2 P D2 S', 'S N2 D2 P M2 G3 R2 S'),
    'bhujangini': (16, 'S R1 G3 M1 D2 N2 S', 'S N2 D2 M1 G3 R1 S'),
    'bhuvanagandhari': (20, 'S R2 M1 P N2 S', 'S N2 D1 P M1 G2 S'),
    'bibhas': (15, 'S R1 G3 P D1 S', 'S D1 P M1 R1 S'),
    'bilahari': (29, 'S R2 G3 P D2 S', 'S N3 D2 P M1 G3 R2 S'),
    'bindumalini': (16, 'S G3 R1 G3 M1 P N2 S', 'S N2 S D2 P G3 R1 S'),
    'bowli': (15, 'S R1 G3 P D1 S', 'S N3 D1 P G3 R1 S'),
    'brindavanasaranga': (22, 'S R2 M1 P N3 S', 'S N2 P M1 R2 G2 R2 S'),
    'brindavani': (22, 'S R2 M1 P N3 S', 'S N2 P M1 R2 S'),
    'budhamanohari': (29, 'S R2 G3 M1 S P S', 'S P M1 G3 R2 S'),
    'budharanjani': (29, 'S R2 G3 M1 P S', 'S N3 P M1 G3 M1 R2 S'),
    'bhadratodi': (8, 'S R1 G2 M1 D1 S', 'S N2 D1 P G2 S'),
    'bhagavatapriya': (22, 'S R2 G2 M1 P N2 S', 'S N2 D2 P M1 G2 R2 S'),
    'bagavataranjana': (64, 'S R2 M2 P D2 N2 S', 'S N2 D2 P M2 G3 R2 S'),
}

mdir = r'C:\git\music\carnatic_ml\shared\ragas_metadata'
conflicts = []
matched = 0
missing_names = []

for kname, (kmela, karo_s, kava_s) in sorted(karnatik.items()):
    karo = karo_s.split()
    kava = kava_s.split()

    fpath = os.path.join(mdir, kname + '.json')
    if not os.path.exists(fpath):
        missing_names.append(kname)
        continue

    with open(fpath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    our_aro = data.get('arohanam', [])
    our_ava = data.get('avarohanam', [])
    our_mela = str(data.get('melakarta_number', '')).strip()

    issues = []
    if our_aro != karo:
        issues.append('  ARO ours:     %s' % ' '.join(our_aro))
        issues.append('  ARO karnatik: %s' % karo_s)
    if our_ava != kava:
        issues.append('  AVA ours:     %s' % ' '.join(our_ava))
        issues.append('  AVA karnatik: %s' % kava_s)
    if our_mela not in ['', 'None'] and our_mela != str(kmela):
        issues.append('  MELA ours: %s  karnatik: %s' % (our_mela, kmela))

    if issues:
        conflicts.append((kname, issues))
    else:
        matched += 1

print('Matched (no conflicts): %d' % matched)
print('Missing from our DB: %d' % len(missing_names))
if missing_names:
    print('  %s' % ', '.join(missing_names))
print()
print('=== CONFLICTS (%d) ===' % len(conflicts))
for kname, issues in conflicts:
    print()
    print('[%s]' % kname)
    for iss in issues:
        print(iss)
