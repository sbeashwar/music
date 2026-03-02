"""
Apply fixes from karnatik.com comparison to our raga metadata JSON files.
Reads karnatik_comparison.json (produced by compare_karnatik_all.py) and updates
arohanam/avarohanam in our shared/ragas_metadata/*.json files.
"""
import json, os

METADATA_DIR = os.path.join(os.path.dirname(__file__), 'shared', 'ragas_metadata')
COMPARISON = os.path.join(os.path.dirname(__file__), 'karnatik_comparison.json')


def apply_fixes():
    with open(COMPARISON, 'r', encoding='utf-8') as f:
        data = json.load(f)

    conflicts = data['conflicts']   # list of [name, karnatik_aro, karnatik_ava, our_aro, our_ava]
    empty_ours = data['empty_ours']  # list of [name, karnatik_aro, karnatik_ava]

    fixed = 0
    filled = 0
    errors = []

    # Fix conflicts
    for entry in conflicts:
        name, k_aro, k_ava, our_aro, our_ava = entry
        fpath = os.path.join(METADATA_DIR, f'{name}.json')
        if not os.path.exists(fpath):
            errors.append(f'{name}: file not found')
            continue

        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                raga = json.load(f)
        except Exception as e:
            errors.append(f'{name}: read error: {e}')
            continue

        changed = False
        if our_aro != k_aro:
            raga['arohanam'] = k_aro
            changed = True
        if our_ava != k_ava:
            raga['avarohanam'] = k_ava
            changed = True

        if changed:
            with open(fpath, 'w', encoding='utf-8') as f:
                json.dump(raga, f, indent=2, ensure_ascii=False)
            fixed += 1

    # Fill empty entries
    for entry in empty_ours:
        name, k_aro, k_ava = entry
        fpath = os.path.join(METADATA_DIR, f'{name}.json')
        if not os.path.exists(fpath):
            errors.append(f'{name}: file not found (empty fill)')
            continue

        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                raga = json.load(f)
        except Exception as e:
            errors.append(f'{name}: read error: {e}')
            continue

        raga['arohanam'] = k_aro
        raga['avarohanam'] = k_ava
        with open(fpath, 'w', encoding='utf-8') as f:
            json.dump(raga, f, indent=2, ensure_ascii=False)
        filled += 1

    print(f'Fixed {fixed} conflicting ragas')
    print(f'Filled {filled} empty ragas')
    if errors:
        print(f'\nErrors ({len(errors)}):')
        for e in errors:
            print(f'  {e}')

    return fixed, filled


if __name__ == '__main__':
    fixed, filled = apply_fixes()
    print(f'\nTotal files updated: {fixed + filled}')
