#!/usr/bin/env python3
"""
Repair master raga data in shared/ragas_metadata/.

1. MELAKARTAS (authoritative): every melakarta is sampurna krama by definition.
   Regenerate arohanam = [S,R,G,M,P,D,N,S] and avarohanam = strict reverse from
   the canonical 72-melakarta (Govindacharya) semitone table, which is
   cross-checked with Wikipedia. Fix classification (krama/sampurna) and clear
   spurious vakra flags. This deterministically fixes embellished-sanchara
   entries such as shankarAbharaNam #29.

2. Specific corroborated janya fixes (e.g. Sahana) whose stored scale was an
   embellished sanchara rather than a clean arohana/avarohana.

Idempotent: only writes files that actually change; prints every diff.

Run:  py -3.13 eval/repair_raga_data.py            # dry run (report only)
      py -3.13 eval/repair_raga_data.py --apply    # write changes
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from raga_detection.swara_matcher import MELAKARTA_72   # noqa: E402

META = ROOT / 'shared' / 'ragas_metadata'

# Canonical swara name for a semitone at a given scale position (R/G/M/D/N).
POS_NAME = {
    'R': {1: 'R1', 2: 'R2', 3: 'R3'},
    'G': {2: 'G1', 3: 'G2', 4: 'G3'},
    'M': {5: 'M1', 6: 'M2'},
    'D': {8: 'D1', 9: 'D2', 10: 'D3'},
    'N': {9: 'N1', 10: 'N2', 11: 'N3'},
}
ALL_SWARAS = ['S', 'R1', 'R2', 'R3', 'G1', 'G2', 'G3', 'M1', 'M2',
              'P', 'D1', 'D2', 'D3', 'N1', 'N2', 'N3']


def mela_scale(num: int):
    """Return canonical arohanam (with framing upper S) for a melakarta number."""
    r, g, m, d, n = MELAKARTA_72[num][1]
    swaras = ['S', POS_NAME['R'][r], POS_NAME['G'][g], POS_NAME['M'][m],
              'P', POS_NAME['D'][d], POS_NAME['N'][n]]
    return swaras + ['S']


# Corroborated janya fixes: {raga_id: (arohanam, avarohanam, note)}
JANYA_FIXES = {
    'sahana': (
        ['S', 'R2', 'G3', 'M1', 'P', 'M1', 'D2', 'N2', 'S'],
        ['S', 'N2', 'D2', 'P', 'M1', 'G3', 'M1', 'R2', 'S'],
        'Ubhaya-vakra sampurna janya of Harikambhoji (Wikipedia); '
        'was an over-embellished sanchara',
    ),
}


def load(fp):
    with open(fp, 'r', encoding='utf-8') as f:
        return json.load(f)


def save(fp, data):
    with open(fp, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write('\n')


def fix_classification(data, arohanam, avarohanam, is_mela):
    scale = set(s for s in arohanam + avarohanam if s != 'S')
    varjya = [s for s in ALL_SWARAS if s not in scale and s != 'S' and s != 'P']
    rc = data.setdefault('raga_classification', {})
    rl = data.setdefault('raga_lakshana', {})
    if is_mela:
        rc['type'] = 'krama'
        rc['arohana_type'] = 'sampurna'
        rc['avarohana_type'] = 'sampurna'
        rc['swara_count'] = 7
        rl['vakra_swaras'] = []
    rl['varjya_swaras'] = varjya


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true', help='write changes')
    args = ap.parse_args()

    changed = 0

    # ---- 1. melakartas ----
    for fp in sorted(META.glob('*.json')):
        try:
            data = load(fp)
        except Exception:
            continue
        if not data.get('is_melakarta'):
            continue
        num = data.get('melakarta_number')
        try:
            num = int(num)
        except (TypeError, ValueError):
            continue
        if num not in MELAKARTA_72:
            continue
        # Normalise the stored number to an int while we're here.
        data['melakarta_number'] = num
        canon_aro = mela_scale(num)
        canon_ava = canon_aro[::-1]
        if data.get('arohanam') == canon_aro and data.get('avarohanam') == canon_ava:
            continue  # already clean
        print(f'[MELA #{num:<2}] {data.get("name","?")}')
        print(f'    aro: {data.get("arohanam")}')
        print(f'      -> {canon_aro}')
        print(f'    ava: {data.get("avarohanam")}')
        print(f'      -> {canon_ava}')
        data['arohanam'] = canon_aro
        data['avarohanam'] = canon_ava
        fix_classification(data, canon_aro, canon_ava, is_mela=True)
        changed += 1
        if args.apply:
            save(fp, data)

    # ---- 2. corroborated janya fixes ----
    for rid, (aro, ava, note) in JANYA_FIXES.items():
        fp = META / f'{rid}.json'
        if not fp.exists():
            print(f'[JANYA] {rid}: file missing, skipped')
            continue
        data = load(fp)
        if data.get('arohanam') == aro and data.get('avarohanam') == ava:
            continue
        print(f'[JANYA] {data.get("name", rid)} — {note}')
        print(f'    aro: {data.get("arohanam")}  ->  {aro}')
        print(f'    ava: {data.get("avarohanam")}  ->  {ava}')
        data['arohanam'] = aro
        data['avarohanam'] = ava
        fix_classification(data, aro, ava, is_mela=False)
        changed += 1
        if args.apply:
            save(fp, data)

    print(f'\n{"APPLIED" if args.apply else "DRY RUN"}: {changed} file(s) '
          f'{"written" if args.apply else "would change"}.')
    if args.apply:
        cache = META / '.raga_cache.pkl'
        if cache.exists():
            cache.unlink()
            print('Invalidated .raga_cache.pkl (will rebuild on next load).')


if __name__ == '__main__':
    main()
