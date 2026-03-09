"""
Build a raga popularity map from the karnatik.com compositions PDF.

Parses compositions2.pdf, extracts raga names, fuzzy-matches them to
our database raga IDs, and produces a JSON popularity map:
  { raga_id: composition_count, ... }

Usage:
    py -3.11 -m raga_detection.build_popularity
"""

import os
import re
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import PyPDF2


# ── Paths ──────────────────────────────────────────────────────────
BASE = Path(__file__).parent.parent
PDF_PATH = BASE / "compositions2.pdf"
META_DIR = BASE / "shared" / "ragas_metadata"
OUTPUT_PATH = BASE / "raga_detection" / "raga_popularity.json"


# ── Step 1: Extract lines from PDF ────────────────────────────────

def extract_pdf_lines(pdf_path: Path) -> list[str]:
    """Extract all text lines from the compositions PDF."""
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        lines = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                for line in text.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('FILE'):
                        lines.append(line)
    return lines


# ── Step 2: Load DB raga names ────────────────────────────────────

def normalize_raga_name(name: str) -> str:
    """Normalize a raga name for fuzzy matching.
    
    Strips diacritics, converts to lowercase, removes spaces/hyphens,
    and normalizes common Carnatic transliteration variations.
    """
    # Unicode normalize (decompose diacritics)
    name = unicodedata.normalize('NFD', name)
    # Remove combining marks (diacritics)
    name = ''.join(c for c in name if unicodedata.category(c) != 'Mn')
    
    # Lowercase
    name = name.lower().strip()
    
    # Remove spaces, hyphens, underscores
    name = re.sub(r'[\s\-_]+', '', name)
    
    # Common Carnatic transliteration normalizations
    # Double vowels → single
    name = re.sub(r'aa', 'a', name)
    name = re.sub(r'ee', 'i', name)
    name = re.sub(r'oo', 'u', name)
    
    # Common consonant variations
    name = name.replace('sh', 's')
    name = name.replace('th', 't')
    name = name.replace('dh', 'd')
    name = name.replace('bh', 'b')
    name = name.replace('kh', 'k')
    name = name.replace('gh', 'g')
    name = name.replace('ph', 'p')
    name = name.replace('ch', 'c')
    
    # Remove trailing vowels that vary
    # e.g., 'kalyani' vs 'kalyaani' vs 'kalyAnI'
    
    return name


def build_db_raga_index(meta_dir: Path) -> dict:
    """Build multiple lookup indices for DB raga names.
    
    Returns:
        {
            'by_normalized': { normalized_name: [raga_id, ...] },
            'by_id': { raga_id: raga_name },
            'all_names': { normalized_variant: raga_id }  # includes id-based names
        }
    """
    by_normalized = defaultdict(list)
    by_id = {}
    all_names = {}
    
    for fn in os.listdir(meta_dir):
        if not fn.endswith('.json'):
            continue
        
        filepath = meta_dir / fn
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue
        
        raga_id = data.get('id', fn[:-5])
        raga_name = data.get('name', raga_id)
        
        by_id[raga_id] = raga_name
        
        # Add multiple normalized forms
        for variant in [raga_name, raga_id]:
            norm = normalize_raga_name(variant)
            if norm:
                by_normalized[norm].append(raga_id)
                if norm not in all_names:
                    all_names[norm] = raga_id
    
    return {
        'by_normalized': dict(by_normalized),
        'by_id': by_id,
        'all_names': all_names,
    }


# ── Step 3: Parse PDF lines to extract raga names ─────────────────

def extract_raga_from_line(line: str) -> str | None:
    """Extract the raga name from a PDF line.
    
    Format: c####.shtml <song_name> <raga> <composer>
    
    Strategy: The raga is typically a single word (sometimes two) 
    between the song name and the composer name. Known composers
    are used as right-boundary markers.
    """
    # Remove the URL prefix
    m = re.match(r'c\d+\.shtml\s+', line)
    if not m:
        return None
    
    rest = line[m.end():]
    
    # The rest is: <song_name> <raga> <composer>
    # We need to find the raga. Strategy: work backwards from known composers,
    # or use the DB to find the raga token.
    return rest  # Return the full rest for multi-pass matching


# Words that commonly appear in composer/song names and should NOT be matched
# as raga names even though they may coincidentally exist in the DB.
COMPOSER_STOPWORDS = {
    normalize_raga_name(w) for w in [
        'Bhaarati', 'Bhaaratiyaar', 'Krishna', 'Shivan', 'Aiyyar',
        'Daasar', 'Dikshitar', 'Aiyyangaar', 'Bhaagavatar', 'Raamadaas',
        'Tooran', 'TaaNDavar', 'Bhaaratiyar', 'Bhaarati', 'Bruhmendrar',
        'Suddhaananda', 'Tyaagaraaja', 'DEshikar', 'Teertar',
        'Shankaraacaarya', 'Annamaacaarya', 'Arunagirinaatar',
        'VenkaTasubbaiyyar', 'Subramanya', 'PaTnam', 'Walajapet',
        'Padma', 'Veeraraghavan', 'Subri', 'Sundaram', 'Kovai',
        'Baala', 'Muttiah', 'Periyasaami', 'Neelakaanta', 'KOTeeshwara',
        'Ambujam', 'Narasimhachar', 'Chidambaram', 'Ramaraj',
        'varNa', 'geetam', 'bhajan', 'stOtra', 'tappa', 'note',
        'keertanai', 'Kabir', 'Surdas', 'Meera', 'GOpaalakrishna',
        'Muttuswaamee', 'Paapanaasam', 'Badraacala', 'Maisuur',
        'VaasudEvacaarya', 'BaalamuraLi', 'OotukkaaDu',
        'Purandara', 'Sadaashiva', 'Gopalakrishna',
    ]
}

# Known major composers (for finding the right boundary)
KNOWN_COMPOSERS = [
    'Tyaagaraaja', 'Muttuswaamee Dikshitar', 'Shyaamaa Shaastri',
    'Purandara Daasar', 'Paapanaasam Shivan', 'Suddhaananda Bhaarati',
    'OotukkaaDu VenkaTasubbaiyyar', 'Subramanya Bhaaratiyaar',
    'Annamaacaarya', 'H.N. Muttiah Bhaagavatar', 'BaalamuraLi Krishna',
    'Ambujam Krishna', 'Sadaashiva Bruhmendrar', 'GOpaalakrishna Bhaaratiyaar',
    'Muttu TaaNDavar', 'NaaraayaNa Teertar', 'Swaamee Dayaananda Saraswati',
    'KOTeeshwara Aiyyar', 'PaTnam Subramanya Aiyyar', 'Periyasaami Tooran',
    'Badraacala Raamadaas', 'Maisuur VaasudEvacaarya', 'Neelakaanta Shivan',
    'K. Ramaraj', 'K Ramaraj', 'M.M. DanDapaaNi DEshikar',
    'Swaati TirunaaL', 'KaNNan Aiyyangaar', 'Arunagirinaatar',
    'Harikesanallur Muttiah Bhaagavatar', 'T. N. Baala',
    'Gopalakrishna Bharati', 'Walajapet Venkatramana Bhaagavatar',
    'Padma Veeraraghavan', 'Kovai Subri', 'B.M.Sundaram',
    'Kabir', 'Surdas', 'Meera', 'Raamadaas',
]


def match_raga_in_line(
    rest: str, 
    db_index: dict,
    known_composer_patterns: list[re.Pattern],
) -> str | None:
    """Try to find a DB raga match in a composition line.
    
    Uses a multi-strategy approach:
    1. Try matching 1-word and 2-word raga tokens against DB
    2. Use composer as right boundary if possible
    3. Longest match wins
    """
    all_names = db_index['all_names']
    
    # Try to find the composer boundary (rightmost known composer)
    composer_start = len(rest)
    for pattern in known_composer_patterns:
        m = pattern.search(rest)
        if m and m.start() < composer_start:
            composer_start = m.start()
    
    if composer_start == len(rest):
        # No known composer found; assume last 1-2 words are composer
        words = rest.split()
        if len(words) >= 3:
            # Try last word, last 2 words as composer
            composer_start = rest.rfind(words[-1])
            # Back up one more word if we can
            if len(words) >= 4:
                composer_start = rest.rfind(words[-2])
    
    # Song + raga portion (before composer)
    song_raga = rest[:composer_start].strip()
    
    if not song_raga:
        return None
    
    words = song_raga.split()
    if not words:
        return None
    
    # Try from the end of song_raga: 1-word, 2-word, 3-word raga
    best_match = None
    best_match_len = 0
    
    for n_words in range(1, min(4, len(words)) + 1):
        candidate = ' '.join(words[-n_words:])
        norm = normalize_raga_name(candidate)
        if norm in all_names:
            if n_words > best_match_len:
                best_match = all_names[norm]
                best_match_len = n_words
    
    return best_match


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    
    return prev_row[-1]


def fuzzy_match_raga(
    candidate: str, 
    db_index: dict,
    max_distance: int = 2,
) -> str | None:
    """Try fuzzy matching a candidate raga name against the DB.
    
    Uses normalized Levenshtein distance with a threshold.
    """
    norm_candidate = normalize_raga_name(candidate)
    if not norm_candidate or len(norm_candidate) < 3:
        return None
    
    best_id = None
    best_dist = max_distance + 1
    
    for norm_name, raga_id in db_index['all_names'].items():
        if abs(len(norm_name) - len(norm_candidate)) > max_distance:
            continue
        dist = levenshtein_distance(norm_candidate, norm_name)
        if dist < best_dist:
            best_dist = dist
            best_id = raga_id
    
    return best_id if best_dist <= max_distance else None


def extract_raga_brute_force(
    rest: str,
    db_index: dict,
) -> str | None:
    """Brute-force: try every contiguous 1-3 word window as raga candidate."""
    all_names = db_index['all_names']
    words = rest.split()
    
    best_match = None
    best_pos = -1  # prefer matches further right (closer to composer)
    
    for n_words in range(1, min(4, len(words)) + 1):
        for start in range(len(words) - n_words + 1):
            candidate = ' '.join(words[start:start + n_words])
            norm = normalize_raga_name(candidate)
            # Skip words that are common in composer/song context
            if norm in COMPOSER_STOPWORDS:
                continue
            if norm in all_names:
                # Prefer rightward matches and longer name matches
                pos_score = start * 10 + n_words
                if pos_score > best_pos:
                    best_pos = pos_score
                    best_match = all_names[norm]
    
    return best_match


# ── Step 4: Main pipeline ─────────────────────────────────────────

def build_popularity_map():
    """Full pipeline: PDF → raga counts → JSON."""
    
    print("Step 1: Extracting text from PDF...")
    lines = extract_pdf_lines(PDF_PATH)
    print(f"  Extracted {len(lines)} composition lines")
    
    print("Step 2: Loading raga database...")
    db_index = build_db_raga_index(META_DIR)
    print(f"  {len(db_index['by_id'])} ragas, {len(db_index['all_names'])} name variants")
    
    print("Step 3: Matching ragas...")
    # Pre-compile composer patterns
    composer_patterns = []
    for c in sorted(KNOWN_COMPOSERS, key=len, reverse=True):
        pattern = re.compile(re.escape(c), re.IGNORECASE)
        composer_patterns.append(pattern)
    
    raga_counts = Counter()
    unmatched_lines = []
    matched = 0
    
    for line in lines:
        # Extract the part after the URL
        m = re.match(r'c\d+\.shtml\s+', line)
        if not m:
            continue
        rest = line[m.end():]
        
        # Strategy 1: structured match with composer boundary
        raga_id = match_raga_in_line(rest, db_index, composer_patterns)
        
        # Strategy 2: brute-force window scan
        if raga_id is None:
            raga_id = extract_raga_brute_force(rest, db_index)
        
        if raga_id:
            raga_counts[raga_id] += 1
            matched += 1
        else:
            unmatched_lines.append(rest)
    
    print(f"  Matched: {matched}/{len(lines)} ({matched/len(lines)*100:.1f}%)")
    print(f"  Unmatched: {len(unmatched_lines)}")
    
    # Step 3b: Quick fuzzy matching using prefix buckets (much faster than full Levenshtein)
    print("Step 3b: Fast fuzzy matching unmatched lines...")
    
    # Build prefix buckets for fast lookup (first 3 chars of normalized name)
    prefix_buckets: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for norm_name, raga_id in db_index['all_names'].items():
        if len(norm_name) >= 3:
            prefix_buckets[norm_name[:3]].append((norm_name, raga_id))
    
    still_unmatched = []
    fuzzy_matched = 0
    
    for rest in unmatched_lines:
        words = rest.split()
        found = False
        # Try 1-2 word windows from right side
        for n_words in range(1, min(3, len(words))):
            for start in range(len(words) - 1, max(-1, len(words) - 5 - n_words), -1):
                candidate = ' '.join(words[start:start + n_words])
                norm_cand = normalize_raga_name(candidate)
                if not norm_cand or len(norm_cand) < 3:
                    continue
                # Skip composer/stopwords
                if norm_cand in COMPOSER_STOPWORDS:
                    continue
                # Adaptive distance threshold: short names need exact/near-exact match
                max_dist = 1 if len(norm_cand) <= 6 else 2
                # Only check names sharing the same 3-char prefix
                prefix = norm_cand[:3]
                for norm_name, raga_id in prefix_buckets.get(prefix, []):
                    if abs(len(norm_name) - len(norm_cand)) > max_dist:
                        continue
                    dist = levenshtein_distance(norm_cand, norm_name)
                    if dist <= max_dist:
                        raga_counts[raga_id] += 1
                        fuzzy_matched += 1
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if not found:
            still_unmatched.append(rest)
    
    print(f"  Fuzzy matched: {fuzzy_matched} more")
    print(f"  Still unmatched: {len(still_unmatched)}")
    
    # Show some unmatched for debugging
    if still_unmatched:
        print("  Sample unmatched:")
        for line in still_unmatched[:20]:
            print(f"    {line[:100]}")
    
    # Step 4: Save results
    # Convert to {raga_id: count} sorted by count descending
    popularity = dict(raga_counts.most_common())
    
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(popularity, f, indent=2, ensure_ascii=False)
    
    print(f"\nStep 4: Saved popularity map to {OUTPUT_PATH}")
    print(f"  {len(popularity)} ragas with compositions")
    
    # Print top ragas
    print("\nTop 40 most popular ragas:")
    for i, (raga_id, count) in enumerate(raga_counts.most_common(40), 1):
        name = db_index['by_id'].get(raga_id, raga_id)
        print(f"  {i:3d}. {name:30s} ({raga_id:30s}) — {count} compositions")
    
    # Print distribution summary
    counts = list(raga_counts.values())
    print(f"\nDistribution:")
    print(f"  Total ragas with compositions: {len(counts)}")
    print(f"  Total compositions matched: {sum(counts)}")
    print(f"  Max compositions (single raga): {max(counts)}")
    brackets = [(1, 1), (2, 5), (6, 10), (11, 50), (51, 100), (101, 500), (501, 10000)]
    for lo, hi in brackets:
        n = sum(1 for c in counts if lo <= c <= hi)
        if n:
            print(f"  Ragas with {lo}-{hi} compositions: {n}")
    
    return popularity


if __name__ == '__main__':
    build_popularity_map()
