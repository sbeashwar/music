#!/usr/bin/env python3
"""
Evaluation script for raga detection.

Runs the detector on a labeled test set and computes metrics:
- Top-1 accuracy (exact match)
- Top-5 accuracy (correct raga in top 5)
- MRR (Mean Reciprocal Rank)
- Per-raga breakdown
- Failure categorization

Usage:
    python eval/run_eval.py eval/test_sets/generated_scales.json
    python eval/run_eval.py eval/test_sets/*.json --verbose
    python eval/run_eval.py eval/test_sets/*.json --diagnostic  # save detailed logs
"""

import argparse
import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import traceback

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from raga_detection.arohanam_detector import ArohanamDetector
from raga_detection.swara_matcher import SwaraSequenceMatcher


@dataclass
class DetectionResult:
    """Result of running detection on one recording."""
    recording_id: str
    ground_truth_raga: str
    detected_tonic_hz: float
    detected_swaras: List[str]
    detected_semitones: List[int]
    raw_sequence: List[str]
    direction: str
    top_matches: List[Tuple[str, float]]  # (raga_id, score) pairs
    rank_of_correct: int  # 0 if not found in top-N
    reciprocal_rank: float
    is_top1: bool
    is_top5: bool
    failure_category: str  # 'correct' | 'tonic_wrong' | 'quantization' | 'enharmonic' | 'low_rank' | 'not_found' | 'error'
    error_message: str = ""
    notes: str = ""


@dataclass
class EvalMetrics:
    """Aggregate metrics for a test set."""
    test_set_name: str
    total_recordings: int
    successful_detections: int
    top1_correct: int
    top5_correct: int
    top1_accuracy: float
    top5_accuracy: float
    mrr: float  # Mean Reciprocal Rank
    failure_breakdown: Dict[str, int] = field(default_factory=dict)
    per_raga_accuracy: Dict[str, Dict] = field(default_factory=dict)


def normalize_raga_name(name: str) -> str:
    """Normalize raga name for comparison."""
    return name.lower().strip().replace(' ', '').replace('_', '').replace('-', '')


def raga_matches(detected: str, ground_truth: str, aliases: List[str] = None) -> bool:
    """Check if detected raga matches ground truth (considering aliases)."""
    d = normalize_raga_name(detected)
    gt = normalize_raga_name(ground_truth)

    if d == gt:
        return True

    # Check aliases
    if aliases:
        for alias in aliases:
            if d == normalize_raga_name(alias):
                return True

    # Known equivalences
    equivalences = [
        ('shankarabharanam', 'dheerasankarabharanam', 'sankarabharanam'),
        ('mohanam', 'mohana'),
        ('hamsadhwani', 'hamsadhvani'),
        ('thodi', 'todi', 'hanumatodi'),
        ('kalyani', 'mechakalyani'),
        ('bhairavi', 'sindubhairavi'),
        ('kambhoji', 'kambodhi', 'harikambhoji'),
        ('mayamalavagowla', 'maayamaalavagowla', 'mayamalavagaula'),
    ]

    for group in equivalences:
        normalized_group = [normalize_raga_name(r) for r in group]
        if d in normalized_group and gt in normalized_group:
            return True

    return False


def categorize_failure(
    result: DetectionResult,
    ground_truth_tonic: Optional[float],
    matcher: SwaraSequenceMatcher,
) -> str:
    """Categorize the type of failure for analysis."""
    if result.is_top1:
        return 'correct'

    if result.error_message:
        return 'error'

    # Check if tonic was way off (if we know ground truth tonic)
    if ground_truth_tonic and result.detected_tonic_hz:
        ratio = result.detected_tonic_hz / ground_truth_tonic
        # Allow octave difference, but flag if pitch class is wrong
        cents_off = abs(1200 * (ratio - 1))  # Simplified
        if cents_off > 100 and cents_off < 1100:  # More than 1 semitone, not octave
            return 'tonic_wrong'

    # Check if correct raga is in top 20 but just ranked low
    if result.rank_of_correct > 0 and result.rank_of_correct <= 20:
        return 'low_rank'

    # Check if the detected swaras match the expected (could be enharmonic issue)
    if result.ground_truth_raga in matcher.ragas:
        gt_raga = matcher.ragas[result.ground_truth_raga]
        gt_semis = set(gt_raga.arohanam_semitones) | set(gt_raga.avarohanam_semitones)
        detected_semis = set(result.detected_semitones)

        overlap = len(gt_semis & detected_semis) / max(len(gt_semis), 1)
        if overlap < 0.5:
            return 'quantization'  # Detected notes don't match expected

    if result.rank_of_correct == 0:
        return 'not_found'

    return 'low_rank'


def run_detection(
    recording: dict,
    detector: ArohanamDetector,
    matcher: SwaraSequenceMatcher,
    top_n: int = 20,
    verbose: bool = False,
) -> DetectionResult:
    """Run detection on a single recording."""
    rec_id = recording['id']
    gt_raga = recording['raga']
    gt_aliases = recording.get('raga_aliases', [])
    gt_tonic = recording.get('tonic_hz')
    audio_path = recording['path']

    try:
        # Run detection
        if verbose:
            print(f"  Detecting {rec_id}...", end=' ', flush=True)

        arohanam_result = detector.detect_from_file(audio_path)

        # Match against database
        direction = arohanam_result.direction if arohanam_result.direction != 'mixed' else 'ascending'
        matches = matcher.match_swaras(
            arohanam_result.detected_swaras,
            direction=direction,
            raw_sequence=arohanam_result.raw_sequence,
            max_results=top_n,
        )

        # Extract top matches
        top_matches = [(m.raga_id, m.score) for m in matches]

        # Find rank of correct raga
        rank = 0
        for i, (raga_id, score) in enumerate(top_matches, 1):
            if raga_matches(raga_id, gt_raga, gt_aliases):
                rank = i
                break

        is_top1 = rank == 1
        is_top5 = 0 < rank <= 5
        rr = 1.0 / rank if rank > 0 else 0.0

        result = DetectionResult(
            recording_id=rec_id,
            ground_truth_raga=gt_raga,
            detected_tonic_hz=arohanam_result.tonic_hz,
            detected_swaras=arohanam_result.detected_swaras,
            detected_semitones=arohanam_result.semitones,
            raw_sequence=arohanam_result.raw_sequence,
            direction=arohanam_result.direction,
            top_matches=top_matches,
            rank_of_correct=rank,
            reciprocal_rank=rr,
            is_top1=is_top1,
            is_top5=is_top5,
            failure_category='correct' if is_top1 else 'pending',
        )

        # Categorize failure
        if not is_top1:
            result.failure_category = categorize_failure(result, gt_tonic, matcher)

        if verbose:
            status = "OK" if is_top1 else f"MISS (rank={rank}, got {top_matches[0][0] if top_matches else 'none'})"
            print(status)

        return result

    except Exception as e:
        if verbose:
            print(f"ERROR: {e}")
        return DetectionResult(
            recording_id=rec_id,
            ground_truth_raga=gt_raga,
            detected_tonic_hz=0,
            detected_swaras=[],
            detected_semitones=[],
            raw_sequence=[],
            direction='unknown',
            top_matches=[],
            rank_of_correct=0,
            reciprocal_rank=0,
            is_top1=False,
            is_top5=False,
            failure_category='error',
            error_message=str(e),
        )


def evaluate_test_set(
    test_set_path: str,
    detector: ArohanamDetector,
    matcher: SwaraSequenceMatcher,
    verbose: bool = False,
    diagnostic: bool = False,
) -> Tuple[EvalMetrics, List[DetectionResult]]:
    """Evaluate detector on a test set."""

    with open(test_set_path, 'r') as f:
        test_set = json.load(f)

    name = test_set.get('name', Path(test_set_path).stem)
    recordings = test_set.get('recordings', [])

    if verbose:
        print(f"\n{'='*60}")
        print(f"Test Set: {name}")
        print(f"Recordings: {len(recordings)}")
        print('='*60)

    results = []

    for rec in recordings:
        # Resolve path relative to carnatic_ml root
        audio_path = rec['path']
        if not os.path.isabs(audio_path):
            audio_path = str(Path(__file__).parent.parent / audio_path)
        rec_copy = rec.copy()
        rec_copy['path'] = audio_path

        result = run_detection(rec_copy, detector, matcher, verbose=verbose)
        results.append(result)

    # Compute metrics
    total = len(results)
    successful = sum(1 for r in results if r.failure_category != 'error')
    top1 = sum(1 for r in results if r.is_top1)
    top5 = sum(1 for r in results if r.is_top5)
    mrr = sum(r.reciprocal_rank for r in results) / total if total > 0 else 0

    # Failure breakdown
    failures = defaultdict(int)
    for r in results:
        failures[r.failure_category] += 1

    # Per-raga breakdown
    per_raga = defaultdict(lambda: {'total': 0, 'top1': 0, 'top5': 0})
    for r in results:
        raga = r.ground_truth_raga
        per_raga[raga]['total'] += 1
        if r.is_top1:
            per_raga[raga]['top1'] += 1
        if r.is_top5:
            per_raga[raga]['top5'] += 1

    metrics = EvalMetrics(
        test_set_name=name,
        total_recordings=total,
        successful_detections=successful,
        top1_correct=top1,
        top5_correct=top5,
        top1_accuracy=top1 / total if total > 0 else 0,
        top5_accuracy=top5 / total if total > 0 else 0,
        mrr=mrr,
        failure_breakdown=dict(failures),
        per_raga_accuracy=dict(per_raga),
    )

    return metrics, results


def print_metrics(metrics: EvalMetrics):
    """Print formatted metrics."""
    print(f"\n{'='*60}")
    print(f"Results: {metrics.test_set_name}")
    print('='*60)
    print(f"Total recordings:     {metrics.total_recordings}")
    print(f"Successful runs:      {metrics.successful_detections}")
    print(f"Top-1 accuracy:       {metrics.top1_accuracy:.1%} ({metrics.top1_correct}/{metrics.total_recordings})")
    print(f"Top-5 accuracy:       {metrics.top5_accuracy:.1%} ({metrics.top5_correct}/{metrics.total_recordings})")
    print(f"MRR:                  {metrics.mrr:.3f}")
    print()
    print("Failure breakdown:")
    for cat, count in sorted(metrics.failure_breakdown.items()):
        print(f"  {cat}: {count}")


def save_diagnostic(results: List[DetectionResult], output_path: str):
    """Save detailed diagnostic results to JSON."""
    data = [asdict(r) for r in results]
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Diagnostic results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate raga detection on test sets')
    parser.add_argument('test_sets', nargs='+', help='Path(s) to test set JSON files')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show per-recording results')
    parser.add_argument('--diagnostic', '-d', action='store_true', help='Save detailed diagnostic output')
    parser.add_argument('--voice-mode', action='store_true', help='Use voice mode (more tolerant for singing)')
    parser.add_argument('--output-dir', '-o', default='eval/results', help='Directory for diagnostic output')

    args = parser.parse_args()

    # Initialize detector and matcher
    print("Loading detector and matcher...")
    detector = ArohanamDetector(voice_mode=args.voice_mode)
    matcher = SwaraSequenceMatcher()
    print(f"Loaded {len(matcher.ragas)} ragas")

    # Ensure output dir exists
    if args.diagnostic:
        os.makedirs(args.output_dir, exist_ok=True)

    # Run evaluation on each test set
    all_metrics = []

    for test_set_path in args.test_sets:
        if not os.path.exists(test_set_path):
            print(f"Warning: Test set not found: {test_set_path}")
            continue

        metrics, results = evaluate_test_set(
            test_set_path,
            detector,
            matcher,
            verbose=args.verbose,
            diagnostic=args.diagnostic,
        )

        print_metrics(metrics)
        all_metrics.append(metrics)

        if args.diagnostic:
            diag_path = os.path.join(
                args.output_dir,
                f"{metrics.test_set_name}_diagnostic.json"
            )
            save_diagnostic(results, diag_path)

    # Print aggregate if multiple test sets
    if len(all_metrics) > 1:
        total = sum(m.total_recordings for m in all_metrics)
        top1 = sum(m.top1_correct for m in all_metrics)
        top5 = sum(m.top5_correct for m in all_metrics)
        mrr = sum(m.mrr * m.total_recordings for m in all_metrics) / total if total > 0 else 0

        print(f"\n{'='*60}")
        print("AGGREGATE RESULTS")
        print('='*60)
        print(f"Total recordings:     {total}")
        print(f"Top-1 accuracy:       {top1/total:.1%} ({top1}/{total})")
        print(f"Top-5 accuracy:       {top5/total:.1%} ({top5}/{total})")
        print(f"MRR:                  {mrr:.3f}")


if __name__ == '__main__':
    main()
