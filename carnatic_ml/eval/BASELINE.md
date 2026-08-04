# Baseline Metrics — Phase 0

**Date:** 2026-06-17
**Detector version:** v3 Arohanam (current HEAD)
**Test harness:** eval/run_eval.py

---

## Test Sets

| Test Set | Recordings | Description |
|----------|------------|-------------|
| `generated_scales` | 20 | Programmatically generated WAV files (clean sine-wave scales at C4) |
| `songs_library` | 30 | Real concert recordings from `shared/samples/Songs/` (alapana/swarakalpana excerpts) |

---

## Results: Generated Scales

These are **ideal conditions** ��� clean synthetic audio with known tonic. This should be near 100%.

| Metric | Value |
|--------|-------|
| **Top-1 Accuracy** | 55.0% (11/20) |
| **Top-5 Accuracy** | 75.0% (15/20) |
| **MRR** | 0.618 |

### Failure Breakdown
- `correct`: 11
- `low_rank`: 5 (correct raga in top-20, but not #1)
- `not_found`: 4 (correct raga not in top-20)

### Per-Recording Detail

| Raga | Result | Notes |
|------|--------|-------|
| mohanam | OK | |
| kalyani | rank=2 | got `mecakalyaani` (alternate spelling) |
| shankarabharanam | OK | |
| bhairavi | OK | |
| thodi | OK | |
| kharaharapriya | OK | |
| hamsadhwani | OK | |
| abheri | rank=0 | got `manjari` — vakra raga, asymmetric detection failing |
| hindolam | OK | |
| madhyamavati | OK | |
| bilahari | rank=0 | got `dheerasankarabharanam` — swara set overlap issue |
| kambhoji | OK | |
| saveri | rank=14 | asymmetric raga, similar swara sets |
| anandabhairavi | rank=3 | got `kharaharapriya` |
| begada | OK | |
| mayamalavagowla | OK | |
| sahana | rank=4 | got `harikambhoji` |
| arabhi | rank=5 | got `amari` |
| dhanyasi | rank=0 | got `hanumatodi` |
| atana | rank=0 | got `devashraya` |

### Analysis

The 55% top-1 on **generated clean scales** is concerning — this should be near 100%. Likely causes:

1. **Asymmetric/vakra ragas** (abheri, saveri, sahana, arabhi) have different arohanam vs avarohanam; the matcher isn't fully leveraging direction info
2. **Alias/spelling variants** (kalyani → mecakalyaani) — need better equivalence handling
3. **Swara set collisions** — many janya ragas share the same swara set as their parent melakarta

---

## Results: Songs Library

Real concert recordings — alapana and swarakalpana clips with voice + accompaniment.

| Metric | Value |
|--------|-------|
| **Top-1 Accuracy** | ~0% |
| **Top-5 Accuracy** | ~0% |
| **MRR** | ~0 |

### Why?

Example: `01a-chakravakam-alapana-ssi-c05.mp3`
- Ground truth: **chakravakam** (melakarta #16: S R1 G3 M1 P D2 N2)
- Detected swaras: **S N2 G2** (only 3 of 7!)
- Tonic detected: 160.9 Hz (reasonable — D#3)
- Top match: `indirai` (0.572) — completely wrong

**Root cause:** An alapana doesn't spell out the scale linearly. It's phrases, gamakas, exploration. The current detector expects arohanam-avarohanam structure and can't handle free melodic movement.

**Implication:** The current architecture is fundamentally unsuited for real recordings. Phase 1–3 improvements won't help here. Need either:
1. Restrict use case to clean scale input only, OR
2. Implement phrase-based detection (Phase 4) for real recording support

---

## Saraga Dataset

Downloaded and verified:
- **Path:** `Z:\backup\saraga\saraga1.5_carnatic.zip`
- **Size:** 14.4GB
- **MD5:** `e4fcd380b4f6d025964cd16aee00273d` (verified)
- **Status:** Ready to extract and build test set

---

## Next Steps (Phase 1)

Based on the baseline:

1. **Fix raga name aliasing** — kalyani/mecakalyaani etc. should be treated as equivalent
2. **Improve asymmetric scoring** — the 35% weight on aro/ava direction matching isn't enough for vakra ragas
3. **Add tonic verification** — for generated scales, tonic is known (261.63 Hz); detector should be hitting it exactly
4. **Extract Saraga test set** — real recordings with verified tonics will expose more failure modes

---

## Reproducing

```bash
cd C:\git\music\carnatic_ml

# Generate test audio
py -3.13 -c "from eval.run_eval import *; ..."  # see test set creation code

# Run evaluation
py -3.13 eval/run_eval.py eval/test_sets/generated_scales.json -v

# With diagnostic output
py -3.13 eval/run_eval.py eval/test_sets/generated_scales.json --diagnostic -o eval/results
```
