"""
Real-time raga detection from continuous microphone input or audio file.

Accumulates swara evidence over time using windowed analysis —
each new chunk of audio adds to a running semitone histogram.
The raga is matched against the accumulated evidence, not just
one snapshot, so it progressively improves as more music is heard.
"""

import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import librosa

try:
    import sounddevice as sd
except ImportError:
    sd = None


# Semitone ↔ swara name mappings (same as in arohanam_detector)
SEMITONE_TO_SWARA = {
    0: 'S', 1: 'R1', 2: 'R2', 3: 'G2', 4: 'G3',
    5: 'M1', 6: 'M2', 7: 'P', 8: 'D1', 9: 'D2',
    10: 'N2', 11: 'N3',
}


@dataclass
class LiveStatus:
    """Snapshot of the real-time detector state, pushed to the GUI."""
    elapsed_seconds: float = 0.0
    tonic_hz: float = 0.0
    current_swaras: List[str] = field(default_factory=list)
    swara_counts: Dict[str, int] = field(default_factory=dict)
    top_match: str = ""
    top_score: float = 0.0
    top5: List[Tuple[str, float]] = field(default_factory=list)
    confidence: str = "listening..."  # low / medium / high / detected!
    raw_sequence: List[str] = field(default_factory=list)
    done: bool = False
    error: str = ""
    window_count: int = 0


class RealtimeRagaDetector:
    """
    Streams mic audio (or reads a file), analyses in overlapping windows,
    and accumulates a semitone histogram.  Periodically matches the
    accumulated evidence against 5300+ ragas.

    Usage (microphone):
        det = RealtimeRagaDetector(matcher, on_update=callback)
        det.start()       # begins listening
        det.stop()        # manual stop

    Usage (file):
        det = RealtimeRagaDetector(matcher, on_update=callback)
        det.start_file("song.mp3")  # streams the file in chunks
    """

    SAMPLE_RATE = 22050
    # Window size for each pYIN analysis chunk
    WINDOW_SECONDS = 5.0
    # Overlap between consecutive windows (seconds)
    WINDOW_OVERLAP = 1.0
    # How often to publish a matcher update (seconds)
    MATCH_INTERVAL = 3.0
    # Minimum accumulated evidence before first match
    MIN_NOTES_FOR_MATCH = 5
    # Score threshold to declare "detected"
    HIGH_CONFIDENCE_SCORE = 0.95
    # Score must beat runner-up by this margin to auto-stop
    MIN_LEAD = 0.03
    # Maximum listening time before giving best guess
    MAX_LISTEN_SECONDS = 180
    # Number of consecutive "stable" rounds to auto-stop
    STABLE_ROUNDS_NEEDED = 3
    # Minimum elapsed time before auto-stop
    MIN_TIME_FOR_AUTODETECT = 10.0

    def __init__(
        self,
        matcher,  # SwaraSequenceMatcher
        on_update: Optional[Callable[[LiveStatus], None]] = None,
    ):
        self.matcher = matcher
        self.on_update = on_update

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._audio_buffer: List[np.ndarray] = []
        self._lock = threading.Lock()
        self._analysed_up_to: int = 0  # samples already analysed

        # Accumulated evidence: pitch class histogram (0=C, 1=C#, ..., 11=B)
        self._pitch_class_counts: Counter = Counter()
        self._total_note_weight: float = 0.0
        self._raw_pitch_classes: List[int] = []  # sequence of pitch classes
        self._best_tonic_pc: Optional[int] = None  # best tonic pitch class
        self._tonic_hz: float = 0.0
        self._window_count: int = 0
        self._stable_winner: str = ""
        self._stable_count: int = 0

    # ------------------------------------------------------------------ public
    def start(self):
        """Begin listening from the default microphone."""
        if sd is None:
            if self.on_update:
                self.on_update(LiveStatus(
                    error="sounddevice not installed", done=True))
            return
        if self._running:
            return

        self._reset()
        self._running = True
        self._thread = threading.Thread(
            target=self._run_mic, daemon=True)
        self._thread.start()

    def start_file(self, audio_path: str):
        """Begin analysing an audio file in streaming chunks."""
        if self._running:
            return
        self._reset()
        self._running = True
        self._thread = threading.Thread(
            target=self._run_file, args=(audio_path,), daemon=True)
        self._thread.start()

    def stop(self):
        """Stop listening / processing."""
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------ reset
    def _reset(self):
        self._audio_buffer = []
        self._analysed_up_to = 0
        self._pitch_class_counts = Counter()
        self._total_note_weight = 0.0
        self._raw_pitch_classes = []
        self._best_tonic_pc = None
        self._tonic_hz = 0.0
        self._window_count = 0
        self._stable_winner = ""
        self._stable_count = 0

    # ------------------------------------------------------------------ mic loop
    def _run_mic(self):
        """Record from mic + periodically analyse new windows."""
        start_time = time.time()
        last_match = 0.0

        def audio_callback(indata, frames, time_info, status):
            with self._lock:
                self._audio_buffer.append(indata[:, 0].copy())

        try:
            with sd.InputStream(samplerate=self.SAMPLE_RATE, channels=1,
                                dtype='float32', callback=audio_callback,
                                blocksize=2048):
                while self._running:
                    elapsed = time.time() - start_time
                    total_samples = self._total_buffered_samples()

                    # Enough new audio for a window?
                    new_samples = total_samples - self._analysed_up_to
                    window_samples = int(self.WINDOW_SECONDS * self.SAMPLE_RATE)
                    overlap_samples = int(self.WINDOW_OVERLAP * self.SAMPLE_RATE)

                    if new_samples >= window_samples:
                        self._analyse_new_window(window_samples, overlap_samples)

                        # Match periodically
                        now = time.time()
                        if (now - last_match >= self.MATCH_INTERVAL
                                and self._total_note_weight >= self.MIN_NOTES_FOR_MATCH):
                            last_match = now
                            should_stop = self._run_match(elapsed)
                            if should_stop and elapsed >= self.MIN_TIME_FOR_AUTODETECT:
                                self._finalise(elapsed, reason="high confidence")
                                return

                    # Timeout
                    if elapsed >= self.MAX_LISTEN_SECONDS:
                        self._finalise(elapsed, reason="max time reached")
                        return

                    # Heartbeat
                    if self._window_count == 0 and self.on_update:
                        self.on_update(LiveStatus(
                            elapsed_seconds=elapsed,
                            confidence="listening...",
                        ))

                    time.sleep(0.2)

        except Exception as e:
            if self.on_update:
                self.on_update(LiveStatus(error=str(e), done=True))
        finally:
            self._running = False

    # ------------------------------------------------------------------ file loop
    def _run_file(self, audio_path: str):
        """Detect raga from an audio file.

        Uses windowed pitch-class accumulation for progressive GUI updates,
        then a multi-tonic final match with tonic-prominence preference.
        """
        start_time = time.time()

        try:
            # Notify: loading
            if self.on_update:
                self.on_update(LiveStatus(
                    confidence="loading file...",
                    elapsed_seconds=0,
                ))

            # Load the full file
            try:
                y, _ = librosa.load(audio_path, sr=self.SAMPLE_RATE)
            except Exception:
                y = self._load_with_ffmpeg(audio_path)

            if len(y) < self.SAMPLE_RATE:
                raise RuntimeError(
                    f"Could not load audio from {audio_path} "
                    "(file too short or format unsupported)")

            # Progressive pitch-class accumulation with periodic matching
            window_samples = int(self.WINDOW_SECONDS * self.SAMPLE_RATE)
            step = int((self.WINDOW_SECONDS - self.WINDOW_OVERLAP) * self.SAMPLE_RATE)
            pos = 0
            match_every = 3
            windows_since_match = 0

            while pos + window_samples <= len(y) and self._running:
                self._process_chunk(y[pos:pos + window_samples])
                simulated_elapsed = (pos + window_samples) / self.SAMPLE_RATE
                windows_since_match += 1

                if (windows_since_match >= match_every
                        and self._total_note_weight >= self.MIN_NOTES_FOR_MATCH):
                    windows_since_match = 0
                    should_stop = self._run_match(simulated_elapsed)
                    if (should_stop
                            and simulated_elapsed >= self.MIN_TIME_FOR_AUTODETECT):
                        self._finalise(simulated_elapsed, reason="high confidence")
                        return

                pos += step

            # Process remaining tail
            if pos < len(y) and self._running:
                tail = y[pos:]
                if len(tail) >= self.SAMPLE_RATE:
                    self._process_chunk(tail)

            # Final match
            total_dur = len(y) / self.SAMPLE_RATE
            self._finalise(total_dur, reason="end of file")

        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            if self.on_update:
                self.on_update(LiveStatus(error=f"{e}\n{tb}", done=True))
        finally:
            self._running = False

    # ------------------------------------------------------------------ analysis
    @staticmethod
    def _load_with_ffmpeg(audio_path: str) -> np.ndarray:
        """Decode audio file via ffmpeg (handles m4a/mp4/aac/etc.)."""
        import subprocess, tempfile, os
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            ffmpeg = 'ffmpeg'

        tmp = tempfile.mktemp(suffix='.wav')
        try:
            subprocess.run(
                [ffmpeg, '-i', audio_path, '-ac', '1', '-ar', '22050',
                 '-f', 'wav', '-y', tmp],
                capture_output=True, check=True,
            )
            y, _ = librosa.load(tmp, sr=22050)
            return y
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def _total_buffered_samples(self) -> int:
        with self._lock:
            return sum(len(b) for b in self._audio_buffer)

    def _get_audio_range(self, start: int, end: int) -> np.ndarray:
        """Extract a sample range from the buffer."""
        with self._lock:
            full = np.concatenate(self._audio_buffer) if self._audio_buffer else np.array([])
        return full[start:end]

    def _analyse_new_window(self, window_samples: int, overlap_samples: int):
        """Extract & process the next window of audio from the mic buffer."""
        start = max(0, self._analysed_up_to - overlap_samples)
        end = self._analysed_up_to + window_samples - overlap_samples
        chunk = self._get_audio_range(start, end)

        if len(chunk) < self.SAMPLE_RATE:
            return

        self._process_chunk(chunk)
        self._analysed_up_to = end

    def _process_chunk(self, chunk: np.ndarray):
        """Run pYIN on a single audio chunk, accumulate pitch class evidence."""
        # --- pYIN pitch tracking ---
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                chunk,
                fmin=80, fmax=800,
                sr=self.SAMPLE_RATE,
                frame_length=2048,
                hop_length=1024,
            )
        except Exception:
            return

        f0 = np.nan_to_num(f0, nan=0.0)
        confs = np.nan_to_num(voiced_probs, nan=0.0)

        # --- Convert to pitch classes and accumulate ---
        valid = (f0 > 0) & (confs > 0.4)
        if not np.any(valid):
            self._window_count += 1
            return

        freqs = f0[valid]
        conf_vals = confs[valid]

        # MIDI pitch → pitch class (0=C, 1=C#, ..., 11=B)
        midi = 12.0 * np.log2(freqs / 440.0) + 69.0
        pitch_classes = np.round(midi).astype(int) % 12

        for pc, conf in zip(pitch_classes, conf_vals):
            pc = int(pc)
            self._pitch_class_counts[pc] += conf
            self._total_note_weight += conf

            # Track pitch class sequence (dedup consecutive)
            if not self._raw_pitch_classes or self._raw_pitch_classes[-1] != pc:
                self._raw_pitch_classes.append(pc)

        if len(self._raw_pitch_classes) > 200:
            self._raw_pitch_classes = self._raw_pitch_classes[-150:]

        self._window_count += 1

    # ------------------------------------------------------------------ matching
    def _build_swaras_from_counts(
        self, tonic_pc: int = 0
    ) -> Tuple[List[str], Dict[str, int]]:
        """
        Build an ordered swara list from the accumulated pitch class histogram,
        relative to a given tonic pitch class.
        Returns (unique_swaras, swara_name_counts).
        """
        if not self._pitch_class_counts:
            return [], {}

        total = sum(self._pitch_class_counts.values())
        if total <= 0:
            return [], {}

        # Convert pitch class counts → semitone-from-tonic counts
        semitone_counts: Counter = Counter()
        for pc, weight in self._pitch_class_counts.items():
            semitone = (pc - tonic_pc) % 12
            semitone_counts[semitone] += weight

        # Threshold: include semitones with > 3% of total weight.
        # Skip Sa (0) and Pa (7) from threshold check — always include.
        threshold = total * 0.03
        included = set()
        for semi, weight in semitone_counts.items():
            if semi in (0, 7) or weight >= threshold:
                included.add(semi)

        # Build ordered swaras (ascending by semitone)
        swaras = []
        swara_counts = {}
        for semi in sorted(included):
            name = SEMITONE_TO_SWARA.get(semi, f'?{semi}')
            swaras.append(name)
            swara_counts[name] = int(semitone_counts[semi])

        return swaras, swara_counts

    def _pc_sequence_to_swaras(self, tonic_pc: int) -> List[str]:
        """Convert raw pitch class sequence to swara names."""
        result: List[str] = []
        for pc in self._raw_pitch_classes:
            semi = (pc - tonic_pc) % 12
            name = SEMITONE_TO_SWARA.get(semi, '?')
            if not result or result[-1] != name:
                result.append(name)
        return result

    def _candidate_tonics(self, max_candidates: int = 4) -> List[int]:
        """Return pitch classes eligible as tonic candidates.

        Only pitch classes that actually appear in the accumulated counts
        (with weight > 1% of total) can be Sa.  Ranked by weight.
        """
        total = sum(self._pitch_class_counts.values())
        if total <= 0:
            return []
        threshold = total * 0.01
        eligible = [
            pc for pc, w in self._pitch_class_counts.items()
            if w >= threshold
        ]
        eligible.sort(
            key=lambda k: self._pitch_class_counts[k], reverse=True)
        return eligible[:max_candidates]

    def _adjusted_score(
        self, raw_score: float, n_swaras: int, candidate_pc: int,
    ) -> float:
        """Adjust a raw match score for tonic selection.

        Combines:
        - Penalty for > 7 detected swaras (wrong tonic -> spurious notes)
        - Small bonus for high tonic-pitch-class weight (Sa prominence)
        """
        extra = max(0, n_swaras - 7)
        penalty = extra * 0.06

        # Mild tonic-prominence tiebreaker (max +0.02)
        total = sum(self._pitch_class_counts.values())
        if total > 0:
            tonic_frac = self._pitch_class_counts.get(candidate_pc, 0) / total
            bonus = min(0.02, tonic_frac * 0.08)
        else:
            bonus = 0.0

        return raw_score - penalty + bonus

    def _run_match(self, elapsed: float) -> bool:
        """
        Try candidate tonics (detected pitch classes by count), match each
        against the raga database, and pick the tonic that gives the
        best match score.  Returns True if auto-stop criteria met.
        """
        if not self._pitch_class_counts:
            return False

        candidates = self._candidate_tonics()
        if not candidates:
            return False

        # Always include the currently winning tonic (if any)
        if self._best_tonic_pc is not None and self._best_tonic_pc not in candidates:
            candidates.append(self._best_tonic_pc)

        best_adj_score = -1.0
        best_tonic_pc = candidates[0]
        best_matches = []
        best_swaras: List[str] = []
        best_counts: Dict[str, int] = {}

        for candidate_pc in candidates:
            swaras, counts = self._build_swaras_from_counts(tonic_pc=candidate_pc)
            if not swaras:
                continue

            raw_seq = self._pc_sequence_to_swaras(candidate_pc)

            matches = self.matcher.match_swaras_hierarchical(
                swaras,
                direction='mixed',
                max_results=10,
                raw_sequence=raw_seq,
            )

            if not matches:
                continue

            adj_score = self._adjusted_score(
                matches[0].score, len(swaras), candidate_pc)

            if adj_score > best_adj_score:
                best_adj_score = adj_score
                best_tonic_pc = candidate_pc
                best_matches = matches
                best_swaras = swaras
                best_counts = counts

        if not best_matches:
            return False

        self._best_tonic_pc = best_tonic_pc
        # Convert pitch class to Hz in the C3-B3 octave (typical Sa range)
        self._tonic_hz = 440.0 * (2.0 ** ((best_tonic_pc + 48 - 69) / 12.0))

        top_name = best_matches[0].raga_name
        top_score = best_matches[0].score
        runner_up = best_matches[1].score if len(best_matches) > 1 else 0.0
        lead = top_score - runner_up
        top5 = [(m.raga_name, m.score) for m in best_matches[:5]]

        # Confidence
        if top_score >= self.HIGH_CONFIDENCE_SCORE and lead >= self.MIN_LEAD:
            conf = "high"
        elif top_score >= 0.85:
            conf = "medium"
        else:
            conf = "low"

        # Stability tracking
        if top_name == self._stable_winner:
            self._stable_count += 1
        else:
            self._stable_winner = top_name
            self._stable_count = 1

        # Publish
        raw_seq = self._pc_sequence_to_swaras(best_tonic_pc)
        if self.on_update:
            self.on_update(LiveStatus(
                elapsed_seconds=elapsed,
                tonic_hz=self._tonic_hz,
                current_swaras=best_swaras,
                swara_counts=best_counts,
                top_match=top_name,
                top_score=top_score,
                top5=top5,
                confidence=conf,
                raw_sequence=raw_seq,
                window_count=self._window_count,
            ))

        # Auto-stop?
        return (conf == "high"
                and self._stable_count >= self.STABLE_ROUNDS_NEEDED)

    # ------------------------------------------------------------------ finalise
    def _finalise(self, elapsed: float, reason: str = ""):
        """Send final result and stop."""
        self._running = False

        # Two-tier tonic selection: prefer the top-2 most prominent pitch
        # classes (Sa is almost always the most-played note).  Only widen
        # to more candidates if the top-2 result is too weak.
        candidates_narrow = self._candidate_tonics(max_candidates=2)
        candidates_wide = self._candidate_tonics(max_candidates=6)

        def _best_for(candidates):
            best_adj = -1.0
            best_pc = candidates[0] if candidates else 0
            best_m = []
            best_s: List[str] = []
            best_c: Dict[str, int] = {}
            for cpc in candidates:
                s, c = self._build_swaras_from_counts(tonic_pc=cpc)
                if not s:
                    continue
                rs = self._pc_sequence_to_swaras(cpc)
                ms = self.matcher.match_swaras_hierarchical(
                    s, direction='mixed', max_results=10,
                    raw_sequence=rs,
                )
                if not ms:
                    continue
                adj = self._adjusted_score(ms[0].score, len(s), cpc)
                if adj > best_adj:
                    best_adj = adj
                    best_pc = cpc
                    best_m = ms
                    best_s = s
                    best_c = c
            return best_adj, best_pc, best_m, best_s, best_c

        adj_n, pc_n, m_n, s_n, c_n = _best_for(candidates_narrow)
        adj_w, pc_w, m_w, s_w, c_w = _best_for(candidates_wide)

        # Use the narrow (top-2) result unless the wider search gives a
        # substantially better score (0.10+ lead), indicating the narrow
        # result is clearly wrong.
        if adj_w > adj_n + 0.10:
            best_tonic_pc, best_matches = pc_w, m_w
            best_swaras, best_counts = s_w, c_w
        else:
            best_tonic_pc, best_matches = pc_n, m_n
            best_swaras, best_counts = s_n, c_n

        self._best_tonic_pc = best_tonic_pc
        self._tonic_hz = 440.0 * (2.0 ** ((best_tonic_pc + 48 - 69) / 12.0))

        top_name = best_matches[0].raga_name if best_matches else ""
        top_score = best_matches[0].score if best_matches else 0.0
        top5 = [(m.raga_name, m.score) for m in best_matches[:5]]

        conf = "detected!" if top_score >= 0.85 else "best guess"
        raw_seq = self._pc_sequence_to_swaras(best_tonic_pc)

        if self.on_update:
            self.on_update(LiveStatus(
                elapsed_seconds=elapsed,
                tonic_hz=self._tonic_hz,
                current_swaras=best_swaras,
                swara_counts=best_counts,
                top_match=top_name,
                top_score=top_score,
                top5=top5,
                confidence=conf,
                raw_sequence=raw_seq,
                window_count=self._window_count,
                done=True,
            ))

        # Auto-save mic recording
        self._save_recording(top_name)

    def _save_recording(self, raga_name: str):
        """Save mic audio (if any) for later reference."""
        try:
            import os
            import soundfile as sf
            from datetime import datetime

            with self._lock:
                if not self._audio_buffer:
                    return
                audio = np.concatenate(self._audio_buffer)

            rec_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 'recording')
            os.makedirs(rec_dir, exist_ok=True)

            safe = raga_name.encode('ascii', 'ignore').decode().lower().replace(' ', '_')[:20] if raga_name else 'unknown'
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(rec_dir, f"live_{safe}_{ts}.wav")
            sf.write(path, audio, self.SAMPLE_RATE)
        except Exception:
            pass
