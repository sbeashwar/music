# Test Sets for Raga Detection Evaluation

This directory contains labeled test sets for evaluating raga detection accuracy.

## Structure

Each test set is a JSON file with this schema:

```json
{
  "name": "test_set_name",
  "description": "What this test set covers",
  "created": "2026-06-17",
  "recordings": [
    {
      "id": "unique_id",
      "path": "relative/path/to/audio.wav",
      "raga": "mohanam",
      "raga_aliases": ["bhupali"],
      "tonic_hz": 261.63,
      "source": "generated | user_recording | saraga | songs_library",
      "notes": "optional notes about this recording"
    }
  ]
}
```

## Test Sets

- `generated_scales.json` — Programmatically generated clean scales (sanity check)
- `songs_library.json` — Excerpts from shared/samples/Songs/ with raga labels from filenames
- `saraga_excerpts.json` — Clips from Saraga Carnatic dataset with verified ground truth
- `user_recordings.json` — User voice recordings (add as you test)

## Usage

```bash
python eval/run_eval.py eval/test_sets/generated_scales.json
python eval/run_eval.py eval/test_sets/*.json  # all sets
```
