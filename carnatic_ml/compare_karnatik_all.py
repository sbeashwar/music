"""
Compare ALL raga metadata against karnatik.com (A-Z).
Fetches each index page, parses arohanam/avarohanam, and compares with our JSON files.
"""
import json, os, re, sys, time
from urllib.request import urlopen, Request
from urllib.error import URLError

METADATA_DIR = os.path.join(os.path.dirname(__file__), 'shared', 'ragas_metadata')

# All karnatik.com raga index page suffixes (no F, Q, W, X, Z pages)
LETTERS = list('abcdeghijklmnoprstuvy')

# ── Name normalisation ──────────────────────────────────────────────
# karnatik.com uses transliteration with capitals for retroflex etc.
# Our file names are all-lowercase, no diacritics, no spaces.
def normalise_name(raw: str) -> str:
    """Turn a karnatik.com raga name into our filename (without .json)."""
    name = raw.strip()
    # Remove parenthetical aliases  e.g. "dhAtuvardani (dhowta pancamam)"
    name = re.sub(r'\s*\(.*?\)\s*', '', name)
    # Remove anything after a comma  e.g. "dEsh, dEshi, dEsi"
    if ',' in name:
        name = name.split(',')[0].strip()
    # Special characters -> ascii
    name = name.replace('ā', 'a').replace('ī', 'i').replace('ū', 'u')
    name = name.replace('ṇ', 'n').replace('ṅ', 'n').replace('ñ', 'n')
    name = name.replace('ḍ', 'd').replace('ṭ', 't').replace('ś', 'sh').replace('ṣ', 'sh')
    name = name.lower()
    # Remove spaces, hyphens
    name = name.replace(' ', '').replace('-', '')
    return name

# ── Swara token normalisation ────────────────────────────────────────
VALID_SWARAS = {
    'S', 'R1', 'R2', 'R3', 'G1', 'G2', 'G3',
    'M1', 'M2', 'P', 'D1', 'D2', 'D3', 'N1', 'N2', 'N3',
    'D', 'N'  # sometimes karnatik uses bare D, N for dharmavati etc.
}

def parse_swaras(text: str) -> list:
    """Parse a swara string like 'S R2 G3 M1 P D2 N3 S' into a list."""
    tokens = text.strip().split()
    result = []
    for t in tokens:
        t = t.strip(',;.')
        if t in VALID_SWARAS:
            result.append(t)
        elif t.upper() in VALID_SWARAS:
            result.append(t.upper())
    return result

# ── Page fetcher ─────────────────────────────────────────────────────
def fetch_page(letter: str) -> str:
    """Fetch karnatik.com raga index page for a given letter."""
    url = f'https://www.karnatik.com/ragas{letter}.shtml'
    req = Request(url, headers={'User-Agent': 'Mozilla/5.0 (raga-compare)'})
    try:
        with urlopen(req, timeout=30) as resp:
            return resp.read().decode('utf-8', errors='replace')
    except URLError as e:
        print(f'  WARNING: Failed to fetch {url}: {e}')
        return ''

# ── Raga parser ──────────────────────────────────────────────────────
def parse_ragas_from_html(html: str) -> dict:
    """
    Extract raga definitions from a karnatik.com page.
    Returns dict: normalised_name -> {
        'raw_name': str,
        'mela': int or None,
        'arohanam': list[str],
        'avarohanam': list[str],
        'is_mela': bool,
    }
    """
    # Strip all HTML tags to get plain text
    text = re.sub(r'<[^>]+>', ' ', html)
    # Collapse whitespace but keep newlines
    text = re.sub(r'[ \t]+', ' ', text)
    # Normalise various A:/Aa: patterns
    # The pattern is: raga_name \n mela_num mela_name janya/mela \n A: ... \n Av: ...

    ragas = {}

    # Pattern to match raga entries.
    # Groups: 1=raga_name, 2=mela_number, 3=janya/mela, 4=arohanam, 5=avarohanam
    # karnatik format examples:
    #   "  AbhEri\n 22 karaharapriya janyaA: S G2 M1 P N2 SAv: S N2 D2 P M1 G2 R2 S"
    #   "  hanumatODi\n 8 hanumatODi melaAa: S R1 G2 M1 P D1 N2 SAv: ..."
    # The "Aa:" or "A:" marks arohanam, "Av:" marks avarohanam

    # Split by lines and process
    lines = text.split('\n')
    full_text = ' '.join(lines)

    # Find all arohanam/avarohanam patterns
    # Pattern: melakarta_num name (janya|mela) ... A[a]: swaras Av: swaras
    # We look for the pattern: NUMBER WORD (janya|mela) ... A/Aa: SWARAS ... Av: SWARAS
    
    # More robust approach: find each "Aa:" or "A:" followed by swaras, then "Av:" followed by swaras
    # and look backward for the raga name and melakarta number

    # Let's use a different strategy: split the text at each raga boundary
    # A raga entry typically starts with the raga name on its own "line" 
    # followed by mela number

    # Actually, let's use a regex approach on the full text
    # Pattern: (raga_name) \s+ (number) \s+ (mela_name) \s+ (janya|mela)[^\n]* Aa?: (swaras) Av: (swaras)
    
    # First, let me try a simpler approach: find all Aa?:/Av: pairs
    # and look backward for context

    # The most reliable pattern seems to be:
    # <number> <word> (janya|mela|mEla) ... A[a]: <swaras> ... Av: <swaras>
    
    # Let's find all occurrences of "A:" or "Aa:" followed by swaras
    # The raga name comes before the melakarta number

    # Regex pattern for the whole entry
    # Note: raga names can have capitals, spaces, etc.
    pattern = re.compile(
        r'(\d{1,2})\s+'          # melakarta number
        r'(\S+)\s+'              # melakarta name  
        r'(janya|mela|mEla)\s*'  # janya or mela
        r'A[a]?:\s*'             # arohanam marker
        r'([SRGMDNP0-9\s]+?)\s*'  # arohanam swaras
        r'Av:\s*'                # avarohanam marker
        r'([SRGMDNP0-9\s]+?)\s*'  # avarohanam swaras
        r'(?:Songs|Film|type|Alternate|Other|Called|Hindustani|This|A |There|Very|Not|Where|scope|close|name|Description|Mentioned)',  # end marker
        re.DOTALL
    )

    for m in pattern.finditer(full_text):
        mela_num = int(m.group(1))
        mela_name = m.group(2)
        is_mela = m.group(3).lower() in ('mela', 'mēla', 'mela')
        aro_text = m.group(4)
        ava_text = m.group(5)

        aro = parse_swaras(aro_text)
        ava = parse_swaras(ava_text)

        if not aro or not ava:
            continue

        # Find the raga name: look backward from the match start
        # The raga name is typically the last "word chunk" before the mela number
        before = full_text[:m.start()].rstrip()
        # Get last non-empty segment
        # The raga name is usually right before the number
        # Look for the last substantial word(s) before the number
        name_match = re.search(r'([\w\s]+?)\s*$', before)
        if not name_match:
            continue
        raw_name = name_match.group(1).strip()
        # Clean up: sometimes we get trailing junk
        # The name should not contain common non-name words
        raw_name = raw_name.strip()
        # Remove leading numbers or common prefixes
        raw_name = re.sub(r'^\d+\s*', '', raw_name)
        # Take only the last "word" if it got concatenated
        # Actually let's be more careful - get the last line before the match
        before_lines = before.split('  ')  # double-space often separates entries
        if before_lines:
            raw_name = before_lines[-1].strip()
            raw_name = re.sub(r'^\d+\s*', '', raw_name)
            raw_name = raw_name.strip()

        if not raw_name or len(raw_name) < 2:
            continue

        norm_name = normalise_name(raw_name)
        if not norm_name or len(norm_name) < 2:
            continue

        ragas[norm_name] = {
            'raw_name': raw_name,
            'mela': mela_num,
            'arohanam': aro,
            'avarohanam': ava,
            'is_mela': is_mela,
        }

    return ragas


# ── Main comparison logic ────────────────────────────────────────────
def load_our_metadata():
    """Load all our raga JSON files."""
    our_ragas = {}
    for fname in os.listdir(METADATA_DIR):
        if not fname.endswith('.json'):
            continue
        fpath = os.path.join(METADATA_DIR, fname)
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            name = fname[:-5]  # remove .json
            our_ragas[name] = data
        except Exception:
            pass
    return our_ragas


def compare():
    our_ragas = load_our_metadata()
    print(f'Loaded {len(our_ragas)} ragas from our metadata.')
    print()

    all_karnatik = {}
    for letter in LETTERS:
        sys.stdout.write(f'Fetching ragas{letter}.shtml ... ')
        sys.stdout.flush()
        html = fetch_page(letter)
        if html:
            ragas = parse_ragas_from_html(html)
            print(f'{len(ragas)} ragas parsed')
            all_karnatik.update(ragas)
        else:
            print('FAILED')
        time.sleep(0.5)  # be polite

    print(f'\nTotal karnatik.com ragas parsed: {len(all_karnatik)}')
    print()

    matched = 0
    conflicts = []
    missing = []
    empty_ours = []

    for kname, kdata in sorted(all_karnatik.items()):
        if kname not in our_ragas:
            missing.append((kname, kdata['raw_name']))
            continue

        ours = our_ragas[kname]
        our_aro = ours.get('arohanam', [])
        our_ava = ours.get('avarohanam', [])

        # Skip if our data is empty (nothing to compare)
        if not our_aro and not our_ava:
            empty_ours.append((kname, kdata))
            continue

        issues = []
        if our_aro != kdata['arohanam']:
            issues.append(f"  ARO ours:     {' '.join(our_aro)}")
            issues.append(f"  ARO karnatik: {' '.join(kdata['arohanam'])}")
        if our_ava != kdata['avarohanam']:
            issues.append(f"  AVA ours:     {' '.join(our_ava)}")
            issues.append(f"  AVA karnatik: {' '.join(kdata['avarohanam'])}")

        if issues:
            conflicts.append((kname, kdata, issues))
        else:
            matched += 1

    print(f'=== RESULTS ===')
    print(f'Matched (no conflicts): {matched}')
    print(f'Conflicts: {len(conflicts)}')
    print(f'Empty in our DB (karnatik has data): {len(empty_ours)}')
    print(f'Not in our DB: {len(missing)}')
    print()

    if conflicts:
        print(f'=== CONFLICTS ({len(conflicts)}) ===')
        for kname, kdata, issues in conflicts:
            print(f'\n[{kname}] (karnatik: {kdata["raw_name"]}, mela {kdata["mela"]})')
            for iss in issues:
                print(iss)

    if empty_ours:
        print(f'\n=== EMPTY IN OUR DB ({len(empty_ours)}) ===')
        for kname, kdata in empty_ours:
            print(f'  {kname}: A={" ".join(kdata["arohanam"])} | Av={" ".join(kdata["avarohanam"])}')

    if missing:
        print(f'\n=== MISSING FROM OUR DB ({len(missing)}) ===')
        for kname, raw in missing[:50]:
            print(f'  {kname} ({raw})')
        if len(missing) > 50:
            print(f'  ... and {len(missing)-50} more')

    # Also output a JSON summary for further processing
    summary = {
        'matched': matched,
        'conflicts': [(kname, kdata['arohanam'], kdata['avarohanam'],
                        our_ragas[kname].get('arohanam', []),
                        our_ragas[kname].get('avarohanam', []))
                       for kname, kdata, _ in conflicts],
        'empty_ours': [(kname, kdata['arohanam'], kdata['avarohanam'])
                        for kname, kdata in empty_ours],
    }
    summary_path = os.path.join(os.path.dirname(__file__), 'karnatik_comparison.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f'\nDetailed comparison saved to {summary_path}')


if __name__ == '__main__':
    compare()
