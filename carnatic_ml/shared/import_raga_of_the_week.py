"""
Import and reconcile raga data from joereynolds/raga-of-the-week GitHub repo.

This script:
1. Downloads swara and raga definitions from the repo
2. Maps swara_ids to notation (S, R1, R2, etc.)
3. Reconciles naming differences between the two data sources
4. Merges useful data (similar ragas, western equivalents, melakarta relationships)
5. Outputs unified raga definitions compatible with this project

Usage:
    python import_raga_of_the_week.py [--output-dir OUTPUT_DIR]
"""

import os
import json
import re
import argparse
import requests
from typing import Dict, List, Optional, Tuple
from pathlib import Path

# GitHub raw content base URL
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/joereynolds/raga-of-the-week/main/database/seeders/data"


def normalize_raga_name(name: str) -> str:
    """
    Normalize raga name for matching:
    - Remove diacritics
    - Convert to lowercase
    - Remove common suffixes/variations
    """
    # Unicode normalization - remove diacritics
    import unicodedata
    normalized = unicodedata.normalize('NFD', name)
    normalized = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')
    
    # Lowercase and strip
    normalized = normalized.lower().strip()
    
    # Common spelling variations
    replacements = [
        ('sh', 's'),  # shankarabharanam -> sankarabharanam
        ('kh', 'k'),  # kharaharapriya -> karaharapriya
        ('th', 't'),
        ('dh', 'd'),
        ('bh', 'b'),
        ('ph', 'p'),
        ('gh', 'g'),
        ('ch', 'c'),
    ]
    
    result = normalized
    # Don't apply replacements for now - just return normalized
    return result


def fetch_swaras() -> Dict[int, Dict]:
    """Fetch swara definitions from the repo."""
    url = f"{GITHUB_RAW_BASE}/swaras/swaras.json"
    print(f"Fetching swaras from {url}...")
    
    response = requests.get(url)
    response.raise_for_status()
    
    swaras_list = response.json()
    
    # Create lookup by ID (1-indexed in original)
    swaras = {}
    for i, swara in enumerate(swaras_list, start=1):
        swaras[i] = {
            'id': i,
            'notation': swara['notation'],
            'display_notation': swara['display_notation'],
            'name_short': swara['name_short'],
            'name_full': swara.get('name_full', ''),
            'note': swara['note'],  # Western note (C, Db, D, etc.)
            'interval': swara['interval'],  # Interval formula (1, b2, 2, etc.)
            'scientific_pitch': swara['scientific_pitch']
        }
    
    print(f"  Loaded {len(swaras)} swaras")
    return swaras


def fetch_raga_file(filename: str) -> Optional[Dict]:
    """Fetch a single raga JSON file from the repo."""
    url = f"{GITHUB_RAW_BASE}/ragas/{filename}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"  Warning: Could not fetch {filename}: {e}")
        return None


def list_raga_files() -> List[str]:
    """List all raga JSON files in the repo (melakartas + janyas)."""
    api_url = "https://api.github.com/repos/joereynolds/raga-of-the-week/contents/database/seeders/data/ragas"
    
    print(f"Listing raga files...")
    response = requests.get(api_url)
    response.raise_for_status()
    
    files = []
    for item in response.json():
        if item['type'] == 'file' and item['name'].endswith('.json'):
            files.append(item['name'])
        elif item['type'] == 'dir' and item['name'] == 'janyas':
            # Also get janyas
            janyas_url = f"{api_url}/janyas"
            janyas_response = requests.get(janyas_url)
            if janyas_response.ok:
                for janya in janyas_response.json():
                    if janya['name'].endswith('.json'):
                        files.append(f"janyas/{janya['name']}")
    
    print(f"  Found {len(files)} raga files")
    return files


def convert_swara_ids_to_notation(swara_entries: List[Dict], swaras: Dict[int, Dict]) -> List[str]:
    """Convert swara_id references to notation strings."""
    # Sort by order to ensure correct sequence
    sorted_entries = sorted(swara_entries, key=lambda x: x.get('order', 0))
    
    notations = []
    for entry in sorted_entries:
        swara_id = entry.get('swara_id')
        if swara_id and swara_id in swaras:
            notation = swaras[swara_id]['notation']
            # Handle upper Sa (displayed as Ṡ, stored as $)
            if notation == '$':
                notation = 'S'  # We'll mark octave separately if needed
            notations.append(notation)
    
    return notations


def load_existing_metadata(metadata_dir: str) -> Dict[str, Dict]:
    """Load existing raga metadata from the project."""
    existing = {}
    
    if not os.path.exists(metadata_dir):
        return existing
    
    for filename in os.listdir(metadata_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(metadata_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    key = data.get('id', data.get('name', filename[:-5])).lower()
                    existing[key] = data
            except Exception as e:
                print(f"  Warning: Could not load {filename}: {e}")
    
    print(f"  Loaded {len(existing)} existing raga definitions")
    return existing


def match_raga_names(rotw_name: str, existing_names: List[str]) -> Optional[str]:
    """
    Try to match a raga-of-the-week name to existing metadata names.
    Returns the matching key if found, None otherwise.
    """
    rotw_normalized = normalize_raga_name(rotw_name)
    
    for existing_name in existing_names:
        if normalize_raga_name(existing_name) == rotw_normalized:
            return existing_name
    
    # Try partial matching for common variations
    for existing_name in existing_names:
        existing_norm = normalize_raga_name(existing_name)
        # Check if one is contained in the other (handles prefix/suffix variations)
        if rotw_normalized in existing_norm or existing_norm in rotw_normalized:
            if abs(len(rotw_normalized) - len(existing_norm)) <= 3:
                return existing_name
    
    return None


def create_unified_raga(
    rotw_data: Dict,
    swaras: Dict[int, Dict],
    existing_data: Optional[Dict] = None,
    is_melakarta: bool = False,
    melakarta_number: int = 0,
    parent_melakarta_id: Optional[int] = None
) -> Dict:
    """
    Create a unified raga definition merging raga-of-the-week data
    with existing metadata.
    """
    raga_info = rotw_data.get('ragas', {})
    arohanas = rotw_data.get('arohanas', [])
    avarohanas = rotw_data.get('avarohanas', [])
    
    # Get name (may have Unicode diacritics)
    name = raga_info.get('name', '')
    name_normalized = normalize_raga_name(name)
    
    # Convert swara IDs to notation
    arohanam = convert_swara_ids_to_notation(arohanas, swaras)
    avarohanam = convert_swara_ids_to_notation(avarohanas, swaras)
    
    # Start with existing data or create new
    if existing_data:
        unified = existing_data.copy()
        # Update with new data but preserve existing fields
        unified['alternate_names'] = list(set(
            unified.get('alternate_names', []) + [name]
        ))
    else:
        unified = {
            'id': name_normalized,
            'name': name,
            'alternate_names': [],
        }
    
    # Update core scale info from raga-of-the-week (authoritative for melakartas)
    unified['arohanam'] = arohanam
    unified['avarohanam'] = avarohanam
    
    # Melakarta info
    unified['is_melakarta'] = is_melakarta
    if is_melakarta:
        unified['melakarta_number'] = melakarta_number
    
    if parent_melakarta_id:
        unified['parent_melakarta_id'] = parent_melakarta_id
    
    # Add raga-of-the-week specific ID for reference
    unified['rotw_id'] = raga_info.get('id')
    
    # Ensure raga_lakshana exists
    if 'raga_lakshana' not in unified:
        unified['raga_lakshana'] = {
            'vadi_swara': '',
            'samvadi_swara': '',
            'jeeva_swaras': [],
            'nyasa_swaras': [],
            'graha_swara': '',
            'amsa_swara': '',
            'varjya_swaras': [],
            'vakra_swaras': []
        }
    
    # Compute varjya swaras (notes NOT in the scale)
    all_swaras = {'S', 'R1', 'R2', 'R3', 'G1', 'G2', 'G3', 'M1', 'M2', 'P', 'D1', 'D2', 'D3', 'N1', 'N2', 'N3'}
    used_swaras = set(arohanam + avarohanam)
    varjya = list(all_swaras - used_swaras)
    unified['raga_lakshana']['varjya_swaras'] = sorted(varjya)
    
    # Add source info
    unified['metadata'] = unified.get('metadata', {})
    unified['metadata']['rotw_source'] = True
    unified['metadata']['rotw_id'] = raga_info.get('id')
    
    return unified


def import_all_ragas(output_dir: str, existing_metadata_dir: Optional[str] = None):
    """
    Main import function - fetches all ragas from raga-of-the-week
    and creates unified definitions.
    """
    print("=" * 60)
    print("Importing ragas from joereynolds/raga-of-the-week")
    print("=" * 60)
    
    # 1. Fetch swara definitions
    swaras = fetch_swaras()
    
    # 2. Load existing metadata if available
    existing = {}
    if existing_metadata_dir and os.path.exists(existing_metadata_dir):
        print(f"\nLoading existing metadata from {existing_metadata_dir}...")
        existing = load_existing_metadata(existing_metadata_dir)
    
    # 3. List and fetch all raga files
    raga_files = list_raga_files()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Track statistics
    stats = {
        'melakartas': 0,
        'janyas': 0,
        'matched': 0,
        'new': 0,
        'errors': 0
    }
    
    # Track melakarta mapping for janya parent references
    melakarta_id_map = {}  # rotw_id -> name
    
    print(f"\nProcessing {len(raga_files)} raga files...")
    
    for filename in raga_files:
        rotw_data = fetch_raga_file(filename)
        if not rotw_data:
            stats['errors'] += 1
            continue
        
        raga_info = rotw_data.get('ragas', {})
        rotw_id = raga_info.get('id', 0)
        rotw_name = raga_info.get('name', filename)
        
        # Determine if melakarta (IDs 1-72 are melakartas)
        is_melakarta = rotw_id <= 72 and 'janyas' not in filename
        is_janya = 'janyas' in filename
        
        # Get parent melakarta for janyas
        parent_id = None
        if is_janya and 'melakarta_janya_links' in rotw_data:
            parent_id = rotw_data['melakarta_janya_links'].get('raga_id')
        
        # Try to match with existing metadata
        existing_match = match_raga_names(rotw_name, list(existing.keys()))
        existing_data = existing.get(existing_match) if existing_match else None
        
        if existing_match:
            stats['matched'] += 1
        else:
            stats['new'] += 1
        
        # Create unified raga definition
        unified = create_unified_raga(
            rotw_data,
            swaras,
            existing_data,
            is_melakarta=is_melakarta,
            melakarta_number=rotw_id if is_melakarta else 0,
            parent_melakarta_id=parent_id
        )
        
        if is_melakarta:
            stats['melakartas'] += 1
            melakarta_id_map[rotw_id] = unified['id']
        else:
            stats['janyas'] += 1
        
        # Save to output directory
        safe_name = re.sub(r'[^\w\-]', '_', unified['id'].lower())
        output_file = os.path.join(output_dir, f"{safe_name}.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(unified, f, ensure_ascii=False, indent=2)
    
    # Save swara reference
    swaras_file = os.path.join(output_dir, '_swaras_reference.json')
    with open(swaras_file, 'w', encoding='utf-8') as f:
        json.dump(list(swaras.values()), f, ensure_ascii=False, indent=2)
    
    # Save melakarta mapping
    mapping_file = os.path.join(output_dir, '_melakarta_mapping.json')
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(melakarta_id_map, f, ensure_ascii=False, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("Import Summary")
    print("=" * 60)
    print(f"  Melakartas imported: {stats['melakartas']}")
    print(f"  Janyas imported: {stats['janyas']}")
    print(f"  Matched with existing: {stats['matched']}")
    print(f"  New ragas: {stats['new']}")
    print(f"  Errors: {stats['errors']}")
    print(f"\nOutput saved to: {output_dir}")
    print(f"  - Individual raga files: *.json")
    print(f"  - Swara reference: _swaras_reference.json")
    print(f"  - Melakarta mapping: _melakarta_mapping.json")


def main():
    parser = argparse.ArgumentParser(
        description='Import raga data from joereynolds/raga-of-the-week'
    )
    parser.add_argument(
        '--output-dir',
        default='ragas_metadata_merged',
        help='Output directory for unified raga definitions'
    )
    parser.add_argument(
        '--existing-dir',
        default=None,
        help='Directory containing existing raga metadata to merge with'
    )
    
    args = parser.parse_args()
    
    # Resolve paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    output_dir = args.output_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(script_dir, output_dir)
    
    existing_dir = args.existing_dir
    if existing_dir and not os.path.isabs(existing_dir):
        existing_dir = os.path.join(script_dir, existing_dir)
    elif existing_dir is None:
        # Default to ragas_metadata in same directory
        existing_dir = os.path.join(script_dir, 'ragas_metadata')
    
    import_all_ragas(output_dir, existing_dir)


if __name__ == '__main__':
    main()
