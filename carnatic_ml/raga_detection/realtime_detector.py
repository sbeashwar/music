"""
Real-time raga detection from continuous microphone input.

Designed for krithi singing — accumulates swara evidence over time
until the raga can be identified with high confidence.
"""

import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import sounddevice as sd
except ImportError:
    sd = None


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


class RealtimeRagaDetector:
    """
    Streams mic audio, accumulates swara evidence, and progressively
    matches against the raga database until confident.

    Usage:
        det = RealtimeRagaDetector(matcher, on_update=callback)
        det.start()      # begins listening
        det.stop()        # manual stop (also stops on auto-detect)
    """

    SAMPLE_RATE = 22050
    # How often (seconds) to analyse the accumulated audio
    ANALYSIS_INTERVAL = 3.0
    # Minimum audio (seconds) before first analysis
    MIN_AUDIO_FOR_ANALYSIS = 4.0
    # Score threshold to declare "detected"
    HIGH_CONFIDENCE_SCORE = 1.00
    # Score must beat runner-up by this margin to auto-stop
    MIN_LEAD = 0.04
    # Maximum listening time before giving best guess
    MAX_LISTEN_SECONDS = 120
    # Number of consecutive "stable" rounds to auto-stop
    STABLE_ROUNDS_NEEDED = 2

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

        # Accumulated evidence across analysis windows
        self._semitone_weights: Counter = Counter()  # semitone -> weighted count
        self._raw_swaras: List[str] = []             # full sequence across windows
        self._tonic_hz: float = 0.0
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

        self._running = True
        self._audio_buffer = []
        self._semitone_weights = Counter()
        self._raw_swaras = []
        self._tonic_hz = 0.0
        self._stable_winner = ""
        self._stable_count = 0

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop listening."""
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------ internal
    def _run(self):
        """Main loop: record + periodic analysis."""
        from raga_detection.arohanam_detector import ArohanamDetector

        detector = ArohanamDetector(voice_mode=True)
        start_time = time.time()
        last_analysis = 0.0

        def audio_callback(indata, frames, time_info, status):
            with self._lock:
                self._audio_buffer.append(indata[:, 0].copy())

        try:
            with sd.InputStream(samplerate=self.SAMPLE_RATE, channels=1,
                                dtype='float32', callback=audio_callback,
                                blocksize=2048):
                while self._running:
                    elapsed = time.time() - start_time
                    now = time.time()

                    # Periodic analysis
                    if (now - last_analysis >= self.ANALYSIS_INTERVAL
                            and elapsed >= self.MIN_AUDIO_FOR_ANALYSIS):
                        last_analysis = now
                        self._analyse_window(detector, elapsed)

                    # Auto-timeout
                    if elapsed >= self.MAX_LISTEN_SECONDS:
                        self._finalise(elapsed, reason="max time reached")
                        break

                    # Publish listening heartbeat
                    if last_analysis == 0.0 and self.on_update:
                        self.on_update(LiveStatus(
                            elapsed_seconds=elapsed,
                            confidence="listening...",
                        ))

                    time.sleep(0.25)

        except Exception as e:
            if self.on_update:
                self.on_update(LiveStatus(error=str(e), done=True))
        finally:
            self._running = False

    def _get_audio(self) -> np.ndarray:
        """Return all accumulated audio as a single array."""
        with self._lock:
            if not self._audio_buffer:
                return np.array([], dtype=np.float32)
            return np.concatenate(self._audio_buffer)

    def _analyse_window(self, detector, elapsed: float):
        """Run detection on the full accumulated audio so far."""
        from raga_detection.arohanam_detector import (
            ArohanamDetector, SWARA_TO_SEMITONE, SEMITONE_TO_SWARA,
        )

        audio = self._get_audio()
        if len(audio) < self.SAMPLE_RATE * 2:
            return

        try:
            result = detector.detect_from_audio(audio)
        except Exception:
            return

        if result.tonic_hz > 0:
            self._tonic_hz = result.tonic_hz

        # Build swara counts from the raw sequence
        swara_counts: Dict[str, int] = Counter()
        for s in result.raw_sequence:
            swara_counts[s] += 1

        # Match against database
        matches = self.matcher.match_swaras_hierarchical(
            result.detected_swaras,
            direction=result.direction,
            max_results=10,
            raw_sequence=result.raw_sequence,
        )

        top_name = matches[0].raga_name if matches else ""
        top_score = matches[0].score if matches else 0.0
        runner_up = matches[1].score if len(matches) > 1 else 0.0
        lead = top_score - runner_up

        top5 = [(m.raga_name, m.score) for m in matches[:5]]

        # Determine confidence level
        if top_score >= self.HIGH_CONFIDENCE_SCORE and lead >= self.MIN_LEAD:
            conf = "high"
        elif top_score >= 0.90:
            conf = "medium"
        else:
            conf = "low"

        # Track stability — same winner across rounds?
        if top_name == self._stable_winner:
            self._stable_count += 1
        else:
            self._stable_winner = top_name
            self._stable_count = 1

        # Publish update
        status = LiveStatus(
            elapsed_seconds=elapsed,
            tonic_hz=self._tonic_hz,
            current_swaras=result.detected_swaras,
            swara_counts=dict(swara_counts),
            top_match=top_name,
            top_score=top_score,
            top5=top5,
            confidence=conf,
            raw_sequence=result.raw_sequence,
        )

        if self.on_update:
            self.on_update(status)

        # Auto-stop: high confidence AND stable for N consecutive rounds
        if (conf == "high"
                and self._stable_count >= self.STABLE_ROUNDS_NEEDED
                and elapsed >= 8.0):
            self._finalise(elapsed, reason="high confidence")

    def _finalise(self, elapsed: float, reason: str = ""):
        """Send final result and stop."""
        self._running = False

        from raga_detection.arohanam_detector import ArohanamDetector
        detector = ArohanamDetector(voice_mode=True)

        audio = self._get_audio()
        if len(audio) < self.SAMPLE_RATE:
            if self.on_update:
                self.on_update(LiveStatus(
                    elapsed_seconds=elapsed,
                    confidence="insufficient audio",
                    done=True,
                ))
            return

        # Final analysis on complete audio
        try:
            result = detector.detect_from_audio(audio)
            matches = self.matcher.match_swaras_hierarchical(
                result.detected_swaras,
                direction=result.direction,
                max_results=10,
                raw_sequence=result.raw_sequence,
            )
        except Exception as e:
            if self.on_update:
                self.on_update(LiveStatus(
                    error=str(e), done=True, elapsed_seconds=elapsed))
            return

        top_name = matches[0].raga_name if matches else ""
        top_score = matches[0].score if matches else 0.0
        top5 = [(m.raga_name, m.score) for m in matches[:5]]

        swara_counts: Dict[str, int] = Counter(result.raw_sequence)

        status = LiveStatus(
            elapsed_seconds=elapsed,
            tonic_hz=result.tonic_hz,
            current_swaras=result.detected_swaras,
            swara_counts=dict(swara_counts),
            top_match=top_name,
            top_score=top_score,
            top5=top5,
            confidence="detected!" if top_score >= 0.85 else "best guess",
            raw_sequence=result.raw_sequence,
            done=True,
        )

        if self.on_update:
            self.on_update(status)

        # Auto-save recording
        self._save_recording(audio, top_name)

    def _save_recording(self, audio: np.ndarray, raga_name: str):
        """Save the recorded audio for later reference."""
        try:
            import os
            import soundfile as sf
            from datetime import datetime

            rec_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), 'recording')
            os.makedirs(rec_dir, exist_ok=True)

            safe = raga_name.lower().replace(' ', '_')[:20] if raga_name else 'unknown'
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(rec_dir, f"live_{safe}_{ts}.wav")
            sf.write(path, audio, self.SAMPLE_RATE)
        except Exception:
            pass  # non-critical
