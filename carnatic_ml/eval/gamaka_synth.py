"""
Gamaka-aware additive synthesizer.

Renders a generator ``List[Note]`` (from ``carnatic.generator.RagaGenerator``)
into a *continuous-pitch* WAV that actually contains gamakas:

- kampita  -> sinusoidal oscillation around the note (a defining Carnatic ornament)
- jaru     -> slide into the note from below
- nokku    -> quick grace from above, falling onto the note
- plain    -> steady note with light vibrato

Crucially, consecutive notes are joined with a short *portamento* glide so the
pitch track is continuous (as a human voice is). This is deliberately the hard
case for a detector that quantises to discrete swaras early: there are no clean
note boundaries and pitch energy is smeared across neighbouring swaras.

This is a synthesis of what the DESIGN doc calls the missing piece — audio that
looks like Carnatic music instead of a sequence of clean sine beeps.
"""

from __future__ import annotations

import numpy as np

# Swara -> semitone offset from Sa (matches the rest of the codebase)
SWARA_TO_SEMITONE = {
    'S': 0,
    'R1': 1, 'R2': 2, 'R3': 3,
    'G1': 2, 'G2': 3, 'G3': 4,
    'M1': 5, 'M2': 6,
    'P': 7,
    'D1': 8, 'D2': 9, 'D3': 10,
    'N1': 9, 'N2': 10, 'N3': 11,
}


def _note_base_semitone(note) -> float:
    """Absolute semitone offset from the madhya-sthayi Sa for a generator Note."""
    semi = SWARA_TO_SEMITONE.get(note.swara, 0)
    # octave: 0 = mandra (low), 1 = madhya (middle), 2 = tara (high)
    return semi + 12 * (note.octave - 1)


def _gamaka_contour(gamaka_name: str, n: int, dur_s: float,
                    rng: np.random.Generator) -> np.ndarray:
    """
    Pitch offset contour (in *semitones*, relative to the note's target pitch)
    for a single note of ``n`` samples lasting ``dur_s`` seconds.
    """
    t = np.linspace(0.0, dur_s, n, endpoint=False)
    g = (gamaka_name or 'plain').lower()

    if g == 'kampita':
        # Oscillation between the note and (roughly) its lower neighbour.
        # Rate ~5 Hz, depth ~0.8 semitone — the swing that smears a "clean"
        # swara across two pitch classes.
        rate = rng.uniform(4.5, 6.0)
        depth = rng.uniform(0.6, 0.9)
        env = np.clip(t / max(dur_s * 0.15, 1e-3), 0, 1)  # ease in the shake
        return -0.5 * depth + 0.5 * depth * np.cos(2 * np.pi * rate * t) * env

    if g == 'jaru':
        # Slide up into the note from ~2.5 semitones below over the first ~35%.
        start = -rng.uniform(2.0, 3.0)
        frac = np.clip(t / (dur_s * 0.35 + 1e-6), 0, 1)
        # smootherstep for a natural glide
        s = frac * frac * (3 - 2 * frac)
        return start * (1 - s)

    if g == 'nokku':
        # Grace from above: start ~+2 semitones and fall onto the note fast.
        start = rng.uniform(1.5, 2.5)
        frac = np.clip(t / (dur_s * 0.18 + 1e-6), 0, 1)
        s = frac * frac * (3 - 2 * frac)
        return start * (1 - s)

    # plain (and any unhandled ornament): light human vibrato
    rate = rng.uniform(4.5, 6.0)
    depth = rng.uniform(0.04, 0.10)
    env = np.clip(t / max(dur_s * 0.3, 1e-3), 0, 1)
    return depth * np.sin(2 * np.pi * rate * t) * env


def make_tambura(duration_s: float, tonic_hz: float, sample_rate: int = 22050,
                 madhyama: bool = False, seed: int = 0) -> np.ndarray:
    """
    Synthesize a tambura-style drone bed that continuously sounds the tonic.

    Standard 4-string tuning (pancama sruti): Pa(below) – Sa – Sa – Sa(lower
    octave). For madhyama ragas the first string is Ma instead of Pa. Each
    string is plucked in a repeating cycle with a long, harmonic-rich decay
    (approximating the jivari buzz) so Sa is always audible under the melody.
    """
    rng = np.random.default_rng(seed)
    n = int(duration_s * sample_rate)
    out = np.zeros(n, dtype=np.float64)

    sa = tonic_hz
    sa_low = tonic_hz / 2.0
    drone1 = tonic_hz * 2 ** (5 / 12.0) / 2.0 if madhyama else \
        tonic_hz * 2 ** (7 / 12.0) / 2.0        # Ma or Pa just below Sa
    strings = [drone1, sa_low, sa, sa]           # cycle order

    def pluck(freq, length):
        t = np.linspace(0, length, int(length * sample_rate), endpoint=False)
        # rich harmonic stack (tambura is bright) with slow decay
        w = np.zeros_like(t)
        for h, amp in enumerate([1.0, 0.7, 0.5, 0.35, 0.22, 0.14, 0.08], start=1):
            w += amp * np.sin(2 * np.pi * freq * h * t)
        env = np.exp(-t * 2.2)                   # long ring
        env[: int(0.005 * sample_rate)] *= np.linspace(0, 1, int(0.005 * sample_rate))
        return w * env

    pluck_gap = 0.62                             # seconds between plucks
    length = 2.6                                 # each pluck rings this long
    pos = 0.0
    i = 0
    while pos < duration_s:
        f = strings[i % len(strings)] * rng.uniform(0.999, 1.001)
        p = pluck(f, length)
        start = int(pos * sample_rate)
        end = min(start + len(p), n)
        out[start:end] += p[: end - start]
        pos += pluck_gap
        i += 1

    peak = np.max(np.abs(out)) or 1.0
    return (out / peak).astype(np.float64)


def synthesize(notes, tonic_hz: float = 261.63, tempo: int = 80,
               sample_rate: int = 22050, seed: int = 0,
               glide_s: float = 0.045, drone: bool = True,
               drone_level: float = 0.28) -> np.ndarray:
    """
    Render a list of generator Notes to a mono float32 waveform.

    Args:
        notes: list of ``carnatic.generator.Note``
        tonic_hz: Sa frequency (C4 by default)
        tempo: BPM (note.duration is in beats)
        sample_rate: audio sample rate
        seed: RNG seed for reproducible gamaka jitter
        glide_s: portamento time between consecutive notes (seconds)
        drone: mix a tambura Sa/Pa drone under the melody so the reference
               shruti (Sa) is always audible (as in real Carnatic listening)
        drone_level: drone amplitude relative to the melody (0..1)

    Returns:
        np.ndarray float32 in [-1, 1]
    """
    rng = np.random.default_rng(seed)
    spb = 60.0 / tempo  # seconds per beat

    # 1) Build a per-note semitone contour, then stitch with glides so the
    #    whole melody has ONE continuous pitch track.
    seg_semitones = []   # list of np arrays (semitone offset from Sa)
    seg_amp = []         # matching amplitude envelopes
    prev_end_semi = None

    for note in notes:
        dur_s = max(note.duration * spb, 0.06)
        n = max(int(round(dur_s * sample_rate)), 2)
        target = _note_base_semitone(note)

        gname = getattr(getattr(note, 'gamaka', None), 'value', 'plain')
        contour = target + _gamaka_contour(gname, n, dur_s, rng)

        # Portamento: blend the start of this note from the previous note's
        # ending pitch, so there is no hard pitch jump (legato voice).
        if prev_end_semi is not None:
            gn = min(int(glide_s * sample_rate), n // 2)
            if gn > 1:
                ramp = np.linspace(0.0, 1.0, gn)
                s = ramp * ramp * (3 - 2 * ramp)  # smootherstep
                contour[:gn] = prev_end_semi * (1 - s) + contour[:gn] * s
        prev_end_semi = contour[-1]

        # Amplitude: gentle attack/release but a raised floor between notes
        # (legato) so pitch continuity is preserved and the detector cannot
        # rely on silence to segment notes.
        env = np.ones(n)
        a = min(int(0.02 * sample_rate), n // 4)
        r = min(int(0.03 * sample_rate), n // 4)
        if a > 0:
            env[:a] = np.linspace(0.55, 1.0, a)
        if r > 0:
            env[-r:] = np.linspace(1.0, 0.55, r)
        vel = getattr(note, 'velocity', 90) / 110.0

        seg_semitones.append(contour)
        seg_amp.append(env * vel)

    if not seg_semitones:
        return np.zeros(1, dtype=np.float32)

    semis = np.concatenate(seg_semitones)
    amps = np.concatenate(seg_amp)

    # 2) Semitone contour -> instantaneous frequency
    freq = tonic_hz * np.power(2.0, semis / 12.0)

    # 3) Phase-accumulation synthesis (correct for time-varying pitch) with a
    #    few harmonics for a reed/voice-like timbre.
    phase = 2 * np.pi * np.cumsum(freq) / sample_rate
    sig = (0.6 * np.sin(phase)
           + 0.25 * np.sin(2 * phase)
           + 0.12 * np.sin(3 * phase)
           + 0.05 * np.sin(4 * phase))
    sig *= amps

    # Overall fade to avoid clicks
    fade = min(int(0.05 * sample_rate), len(sig) // 4)
    if fade > 0:
        sig[:fade] *= np.linspace(0, 1, fade)
        sig[-fade:] *= np.linspace(1, 0, fade)

    peak = np.max(np.abs(sig)) or 1.0
    melody = sig / peak * 0.85

    if drone:
        # Detect madhyama ragas (no Pa in the line) to tune the drone string.
        madhyama = 7 not in set(int(round(s)) % 12 for s in semis)
        bed = make_tambura(len(melody) / sample_rate, tonic_hz,
                           sample_rate, madhyama=madhyama, seed=seed)
        bed = bed[:len(melody)]
        if len(bed) < len(melody):
            bed = np.pad(bed, (0, len(melody) - len(bed)))
        mix = melody + drone_level * bed
        mix /= (np.max(np.abs(mix)) or 1.0)
        return (mix * 0.9).astype(np.float32)

    return melody.astype(np.float32)
