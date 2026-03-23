"""
GUI for Carnatic Raga Detection and Playback

Two main features:
1. Detect raga from audio (mic recording or file) using arohanam-based detection
2. Play/generate any raga scale by name (search 5,300+ ragas)
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import numpy as np
import os

# Check for sounddevice (microphone recording)
try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False

# Check for audio playback
try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False


class RagaGUI:
    """Main GUI application with Detect and Play tabs."""

    def __init__(self, root):
        self.root = root
        self.root.title("Carnatic Raga Tool")
        self.root.geometry("800x700")
        self.root.resizable(True, True)

        # Shared state
        self.matcher = None
        self.detector = None
        self.audio_path = None
        self.recorded_audio = None
        self.is_recording = False
        self.record_seconds = 30
        self.last_generated_wav = None
        self.live_detector = None

        self._create_widgets()
        self._load_backend_async()

    # ------------------------------------------------------------------ UI
    def _create_widgets(self):
        # Configure grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Notebook (tabs)
        self.notebook = ttk.Notebook(self.root)
        self.notebook.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)

        # --- Tab 1: Detect ---
        self.detect_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.detect_tab, text="  Detect Raga  ")
        self._create_detect_tab()

        # --- Tab 2: Play ---
        self.play_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.play_tab, text="  Play Raga  ")
        self._create_play_tab()

        # --- Tab 3: Live Detect ---
        self.live_tab = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(self.live_tab, text="  Live Detect  ")
        self._create_live_tab()

        # --- Status bar (shared) ---
        status_frame = ttk.Frame(self.root)
        status_frame.grid(row=1, column=0, sticky="ew", padx=5, pady=(0, 5))
        status_frame.columnconfigure(0, weight=1)

        self.status_var = tk.StringVar(value="Loading raga database...")
        ttk.Label(
            status_frame, textvariable=self.status_var, foreground="gray"
        ).grid(row=0, column=0, sticky="w")

        self.progress = ttk.Progressbar(status_frame, mode="indeterminate", length=200)
        self.progress.grid(row=1, column=0, sticky="ew", pady=(2, 0))
        self.progress.start()

    # ========================== DETECT TAB ==========================
    def _create_detect_tab(self):
        tab = self.detect_tab
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(6, weight=1)   # results row

        # Title
        ttk.Label(
            tab, text="Detect Raga from Audio",
            font=("Segoe UI", 14, "bold")
        ).grid(row=0, column=0, pady=(0, 10))

        # -- Recording frame --
        rec_frame = ttk.LabelFrame(tab, text="Record from Microphone", padding=8)
        rec_frame.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        rec_frame.columnconfigure(2, weight=1)

        self.record_btn = ttk.Button(
            rec_frame, text="Record 30s", command=self._toggle_recording, width=14
        )
        self.record_btn.grid(row=0, column=0, padx=(0, 5))

        self.save_btn = ttk.Button(
            rec_frame, text="Save", command=self._save_recording, width=8, state="disabled"
        )
        self.save_btn.grid(row=0, column=1, padx=(0, 8))

        self.record_progress = ttk.Progressbar(rec_frame, mode="determinate", maximum=self.record_seconds)
        self.record_progress.grid(row=0, column=2, sticky="ew", padx=(0, 8))

        self.record_status = ttk.Label(rec_frame, text="Ready", foreground="gray")
        self.record_status.grid(row=0, column=3)

        if not HAS_SOUNDDEVICE:
            self.record_btn.configure(state="disabled")
            self.record_status.configure(text="pip install sounddevice")

        # -- File selection frame --
        file_frame = ttk.LabelFrame(tab, text="Or Load Audio File", padding=8)
        file_frame.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        file_frame.columnconfigure(0, weight=1)

        self.file_label = ttk.Label(file_frame, text="No file selected", foreground="gray")
        self.file_label.grid(row=0, column=0, sticky="w", padx=(0, 10))

        ttk.Button(file_frame, text="Browse...", command=self._browse_file).grid(row=0, column=1)

        # -- Source mode --
        mode_frame = ttk.LabelFrame(tab, text="Audio Source", padding=8)
        mode_frame.grid(row=3, column=0, sticky="ew", pady=(0, 8))

        self.source_var = tk.StringVar(value="voice")
        ttk.Radiobutton(mode_frame, text="Voice / Singing", variable=self.source_var, value="voice").grid(row=0, column=0, padx=(0, 15))
        ttk.Radiobutton(mode_frame, text="Clean / Generated", variable=self.source_var, value="clean").grid(row=0, column=1)
        ttk.Label(mode_frame, text="(Voice mode filters gamakas & slides)", foreground="gray", font=("Segoe UI", 8)).grid(row=0, column=2, padx=(15, 0))

        # -- Detect button --
        self.detect_btn = ttk.Button(
            tab, text="Detect Raga", command=self._run_detection, state="disabled"
        )
        self.detect_btn.grid(row=5, column=0, pady=8)

        # -- Results --
        results_frame = ttk.LabelFrame(tab, text="Results", padding=8)
        results_frame.grid(row=6, column=0, sticky="nsew")
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)

        self.detect_results = tk.Text(
            results_frame, wrap="word", font=("Consolas", 10),
            state="disabled", bg="#f5f5f5"
        )
        self.detect_results.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(results_frame, orient="vertical", command=self.detect_results.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.detect_results.configure(yscrollcommand=sb.set)

    # ========================== PLAY TAB ==========================
    def _create_play_tab(self):
        tab = self.play_tab
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(4, weight=1)   # info row

        # Title
        ttk.Label(
            tab, text="Play Raga Scale",
            font=("Segoe UI", 14, "bold")
        ).grid(row=0, column=0, pady=(0, 10))

        # -- Search frame --
        search_frame = ttk.LabelFrame(tab, text="Search Raga", padding=8)
        search_frame.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        search_frame.columnconfigure(1, weight=1)

        ttk.Label(search_frame, text="Raga name:").grid(row=0, column=0, padx=(0, 5))

        self.raga_search_var = tk.StringVar()
        self.raga_search_entry = ttk.Entry(
            search_frame, textvariable=self.raga_search_var, font=("Segoe UI", 11)
        )
        self.raga_search_entry.grid(row=0, column=1, sticky="ew", padx=(0, 5))
        self.raga_search_entry.bind("<Return>", lambda e: self._search_raga())

        self.search_btn = ttk.Button(
            search_frame, text="Search", command=self._search_raga, width=10, state="disabled"
        )
        self.search_btn.grid(row=0, column=2)

        # -- Search results listbox --
        list_frame = ttk.Frame(tab)
        list_frame.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        list_frame.columnconfigure(0, weight=1)

        self.raga_listbox = tk.Listbox(
            list_frame, height=6, font=("Consolas", 10), selectmode=tk.SINGLE
        )
        self.raga_listbox.grid(row=0, column=0, sticky="ew")
        lb_sb = ttk.Scrollbar(list_frame, orient="vertical", command=self.raga_listbox.yview)
        lb_sb.grid(row=0, column=1, sticky="ns")
        self.raga_listbox.configure(yscrollcommand=lb_sb.set)
        self.raga_listbox.bind("<<ListboxSelect>>", self._on_raga_select)

        # -- Playback options --
        opt_frame = ttk.LabelFrame(tab, text="Playback Options", padding=8)
        opt_frame.grid(row=3, column=0, sticky="ew", pady=(0, 8))

        ttk.Label(opt_frame, text="Tonic:").grid(row=0, column=0, padx=(0, 5))
        self.tonic_var = tk.StringVar(value="C4")
        tonic_combo = ttk.Combobox(
            opt_frame, textvariable=self.tonic_var, state="readonly", width=6,
            values=["C3", "C#3", "D3", "D#3", "E3", "F3",
                    "C4", "C#4", "D4", "D#4", "E4", "F4",
                    "C5"]
        )
        tonic_combo.grid(row=0, column=1, padx=(0, 15))

        ttk.Label(opt_frame, text="Format:").grid(row=0, column=2, padx=(0, 5))
        self.format_var = tk.StringVar(value="wav")
        ttk.Radiobutton(opt_frame, text="WAV", variable=self.format_var, value="wav").grid(row=0, column=3)
        ttk.Radiobutton(opt_frame, text="MIDI", variable=self.format_var, value="midi").grid(row=0, column=4, padx=(5, 15))

        self.play_btn = ttk.Button(
            opt_frame, text="Generate & Play", command=self._generate_and_play, state="disabled", width=16
        )
        self.play_btn.grid(row=0, column=5, padx=(10, 0))

        self.open_folder_btn = ttk.Button(
            opt_frame, text="Open Folder", command=self._open_output_folder, width=12
        )
        self.open_folder_btn.grid(row=0, column=6, padx=(5, 0))

        # -- Raga info --
        info_frame = ttk.LabelFrame(tab, text="Raga Info", padding=8)
        info_frame.grid(row=4, column=0, sticky="nsew")
        info_frame.columnconfigure(0, weight=1)
        info_frame.rowconfigure(0, weight=1)

        self.play_info = tk.Text(
            info_frame, wrap="word", font=("Consolas", 10),
            state="disabled", bg="#f5f5f5"
        )
        self.play_info.grid(row=0, column=0, sticky="nsew")
        sb2 = ttk.Scrollbar(info_frame, orient="vertical", command=self.play_info.yview)
        sb2.grid(row=0, column=1, sticky="ns")
        self.play_info.configure(yscrollcommand=sb2.set)

    # ------------------------------------------------------------------ BACKEND LOADING
    def _load_backend_async(self):
        """Load matcher & detector in background."""
        def load():
            try:
                from raga_detection.swara_matcher import SwaraSequenceMatcher
                from raga_detection.arohanam_detector import ArohanamDetector
                m = SwaraSequenceMatcher()
                d_voice = ArohanamDetector(voice_mode=True)
                d_clean = ArohanamDetector(voice_mode=False)
                self.root.after(0, lambda: self._on_backend_loaded(m, d_voice, d_clean))
            except Exception as e:
                import traceback
                err = f"{e}\n\n{traceback.format_exc()}"
                self.root.after(0, lambda: self._on_load_error(err))

        threading.Thread(target=load, daemon=True).start()

    def _on_backend_loaded(self, matcher, detector_voice, detector_clean):
        self.matcher = matcher
        self.detector = detector_voice
        self.detector_clean = detector_clean
        self.progress.stop()
        self.progress.grid_remove()
        self.status_var.set(f"Ready  |  {matcher.total_ragas:,} ragas loaded")
        self.search_btn.configure(state="normal")
        self._update_detect_button()
        self._set_detect_text(
            "How to use:\n\n"
            "1. Record from microphone or load an audio file\n"
            "   - Sing the arohanam (ascending) and avarohanam (descending)\n"
            "   - Hold each note for ~1 second, keep it clean\n"
            "   - Avoid gamakas / ornaments for best results\n\n"
            "2. Click 'Detect Raga' to identify the raga\n\n"
            "Supported formats: WAV, MP3, FLAC, OGG"
        )
        self._set_play_text(
            "Type a raga name and press Search (or Enter).\n"
            "Select from results, then click 'Generate & Play'.\n\n"
            f"Database: {matcher.total_ragas:,} ragas available.\n\n"
            "Examples: mohanam, kalyani, bahudari, todi,\n"
            "          shankarabharanam, hamsadhwani, kharaharapriya"
        )

    def _on_load_error(self, error):
        self.progress.stop()
        self.progress.grid_remove()
        self.status_var.set("Error loading backend")
        self._set_detect_text(f"Failed to load:\n\n{error}")
        self._set_play_text(f"Failed to load:\n\n{error}")

    # ------------------------------------------------------------------ DETECT: Recording
    def _toggle_recording(self):
        if self.is_recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self):
        if not HAS_SOUNDDEVICE:
            return
        self.is_recording = True
        self.recorded_audio = None
        self.audio_path = None
        self.file_label.configure(text="No file selected", foreground="gray")
        self.record_btn.configure(text="Stop")
        self.record_status.configure(text="Recording...", foreground="red")
        self.record_progress['value'] = 0
        self.detect_btn.configure(state="disabled")
        self.save_btn.configure(state="disabled")

        self._set_detect_text(
            "Recording... Sing into your microphone!\n\n"
            "Tips:\n"
            "- Start with Sa\n"
            "- Sing arohanam then avarohanam\n"
            "- Hold each note for ~1 second\n"
            "- Pa helps detect tonic"
        )

        def record():
            try:
                sr = 22050
                audio_buffer = []

                def callback(indata, frames, time, status):
                    audio_buffer.append(indata.copy())

                with sd.InputStream(samplerate=sr, channels=1, dtype='float32',
                                    callback=callback, blocksize=1024):
                    for i in range(self.record_seconds):
                        if not self.is_recording:
                            break
                        sd.sleep(1000)
                        self.root.after(0, lambda v=i+1: self._update_rec_progress(v))

                if audio_buffer:
                    self.recorded_audio = np.concatenate(audio_buffer).flatten()
                    self.root.after(0, self._on_rec_done)
                else:
                    self.root.after(0, self._on_rec_cancelled)
            except Exception as e:
                self.root.after(0, lambda: self._on_rec_error(str(e)))

        threading.Thread(target=record, daemon=True).start()

    def _update_rec_progress(self, secs):
        self.record_progress['value'] = secs
        remaining = self.record_seconds - secs
        self.record_status.configure(text=f"Recording... {remaining}s", foreground="red")

    def _stop_recording(self):
        self.is_recording = False

    def _on_rec_done(self):
        self.is_recording = False
        self.record_btn.configure(text="Record 30s")
        self.record_progress['value'] = self.record_seconds
        self.save_btn.configure(state="normal")
        self._update_detect_button()
        dur = len(self.recorded_audio) / 22050

        # Auto-save the recording
        saved_path = self._auto_save_recording()
        save_msg = f"\nAuto-saved to: {saved_path}" if saved_path else ""
        self.record_status.configure(text=f"Saved!", foreground="green")

        self._set_detect_text(
            f"Recording complete! ({dur:.1f}s){save_msg}\n\n"
            "Click 'Detect Raga' to analyze."
        )

    def _on_rec_cancelled(self):
        self.is_recording = False
        self.record_btn.configure(text="Record 30s")
        self.record_status.configure(text="Cancelled", foreground="gray")
        self._update_detect_button()

    def _on_rec_error(self, error):
        self.is_recording = False
        self.record_btn.configure(text="Record 30s")
        self.record_status.configure(text="Error", foreground="red")
        self._set_detect_text(f"Recording error:\n\n{error}")

    def _auto_save_recording(self) -> str:
        """Auto-save recording to recording/ directory. Returns saved path or empty string."""
        if self.recorded_audio is None:
            return ""
        try:
            from datetime import datetime
            rec_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'recording')
            os.makedirs(rec_dir, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(rec_dir, f"recording_{ts}.wav")
            import soundfile as sf
            sf.write(path, self.recorded_audio, 22050)
            self.audio_path = path
            return path
        except Exception:
            return ""

    def _save_recording(self):
        if self.recorded_audio is None:
            return
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = filedialog.asksaveasfilename(
            title="Save Recording", defaultextension=".wav",
            initialfile=f"recording_{ts}.wav",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        if path:
            try:
                import soundfile as sf
                sf.write(path, self.recorded_audio, 22050)
                self.record_status.configure(text="Saved!", foreground="green")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save: {e}")

    # ------------------------------------------------------------------ DETECT: File
    def _browse_file(self):
        path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[
                ("Audio files", "*.mp3 *.wav *.flac *.ogg *.m4a"),
                ("All files", "*.*")
            ]
        )
        if path:
            self.audio_path = path
            self.recorded_audio = None
            self.record_status.configure(text="Ready", foreground="gray")
            self.record_progress['value'] = 0
            name = Path(path).name
            if len(name) > 55:
                name = name[:52] + "..."
            self.file_label.configure(text=name, foreground="black")
            self._update_detect_button()

    def _update_detect_button(self):
        if self.detector and (self.audio_path or self.recorded_audio is not None):
            self.detect_btn.configure(state="normal")
        else:
            self.detect_btn.configure(state="disabled")

    # ------------------------------------------------------------------ DETECT: Run
    def _run_detection(self):
        if not self.detector or not self.matcher:
            return
        if self.recorded_audio is None and not self.audio_path:
            return

        self.detect_btn.configure(state="disabled")
        self.status_var.set("Analyzing audio...")
        self._set_detect_text("Detecting raga... please wait.")

        audio_path = self.audio_path
        recorded = self.recorded_audio
        # Pick detector based on source mode
        source = self.source_var.get()
        detector = self.detector if source == "voice" else self.detector_clean
        matcher = self.matcher

        def detect():
            try:
                if recorded is not None:
                    result = detector.detect_from_audio(recorded)
                else:
                    result = detector.detect_from_file(audio_path)

                matches = matcher.match_swaras_hierarchical(
                    result.detected_swaras,
                    direction=result.direction,
                    max_results=20,
                    raw_sequence=result.raw_sequence,
                )
                self.root.after(0, lambda: self._show_detection(result, matches))
            except Exception as e:
                import traceback
                err = f"{e}\n\n{traceback.format_exc()}"
                self.root.after(0, lambda: self._show_detect_error(err))

        threading.Thread(target=detect, daemon=True).start()

    def _show_detection(self, result, matches):
        self.detect_btn.configure(state="normal")
        self._update_detect_button()

        lines = []
        lines.append("=" * 52)
        lines.append("  RAGA DETECTION RESULTS")
        lines.append("=" * 52)
        lines.append("")
        lines.append(f"  Tonic (Sa):  {result.tonic_hz:.1f} Hz")
        lines.append(f"  Direction:   {result.direction}")
        lines.append(f"  Swaras:      {' '.join(result.detected_swaras)}")
        lines.append(f"  Semitones:   {result.semitones}")
        lines.append(f"  Sequence:    {' -> '.join(result.raw_sequence)}")
        lines.append("")

        if not matches:
            lines.append("  No matching ragas found.")
            lines.append("")
            lines.append("  Tips:")
            lines.append("  - Sing more clearly with distinct notes")
            lines.append("  - Hold each note for 1-2 seconds")
            lines.append("  - Reduce background noise")
        else:
            lines.append("-" * 52)
            lines.append(f"  Top {min(len(matches), 15)} Matches")
            lines.append("-" * 52)

            for i, m in enumerate(matches[:15], 1):
                pct = min(m.score * 100, 100.0)
                bar = "█" * int(pct / 5)
                mela = " [M]" if m.is_melakarta else ""
                lines.append("")
                lines.append(f"  {i:2}. {m.raga_name:<26} {pct:5.1f}% {bar}{mela}")
                lines.append(f"      ↑ {' '.join(m.arohanam)}")
                lines.append(f"      ↓ {' '.join(m.avarohanam)}")
                lines.append(f"      {m.details}")

        lines.append("")
        lines.append("-" * 52)

        self._set_detect_text("\n".join(lines))
        if matches:
            self.status_var.set(
                f"Detection complete  |  Top match: {matches[0].raga_name} "
                f"({min(matches[0].score * 100, 100):.1f}%)"
            )
        else:
            self.status_var.set("Detection complete  |  No matches found")

    def _show_detect_error(self, error):
        self.detect_btn.configure(state="normal")
        self._update_detect_button()
        self.status_var.set("Detection failed")
        self._set_detect_text(f"Error during detection:\n\n{error}")

    # ------------------------------------------------------------------ PLAY: Search
    def _search_raga(self):
        if not self.matcher:
            return
        query = self.raga_search_var.get().strip()
        if not query:
            return

        self.raga_listbox.delete(0, tk.END)
        query_lower = query.lower()

        # Collect matches: exact first, then partial
        results = []
        seen = set()

        # Exact id / name match
        for raga_id, raga in self.matcher.ragas.items():
            if raga_id == query_lower or raga.name.lower() == query_lower:
                if raga_id not in seen:
                    results.append(raga)
                    seen.add(raga_id)

        # Partial matches
        for raga_id, raga in self.matcher.ragas.items():
            if raga_id not in seen:
                if query_lower in raga_id or query_lower in raga.name.lower():
                    results.append(raga)
                    seen.add(raga_id)

        # Sort by name length (shorter = more likely the base raga)
        results.sort(key=lambda r: (len(r.name), r.name))

        if not results:
            self.raga_listbox.insert(tk.END, "(no results)")
            self._set_play_text(f"No ragas found matching '{query}'.\n\nTry a different spelling.")
            self.play_btn.configure(state="disabled")
            return

        # Populate listbox (max 100)
        for raga in results[:100]:
            mela = " [M]" if raga.is_melakarta else ""
            swaras = ' '.join(raga.arohanam) if raga.arohanam else '?'
            self.raga_listbox.insert(
                tk.END, f"{raga.name}{mela}  |  {swaras}"
            )

        # Store reference
        self._search_results = results[:100]
        self.status_var.set(f"Found {len(results)} ragas matching '{query}'")

        # Auto-select first
        if results:
            self.raga_listbox.selection_set(0)
            self._on_raga_select(None)

    def _on_raga_select(self, event):
        sel = self.raga_listbox.curselection()
        if not sel or not hasattr(self, '_search_results'):
            return
        idx = sel[0]
        if idx >= len(self._search_results):
            return

        raga = self._search_results[idx]
        self.play_btn.configure(state="normal")

        lines = []
        lines.append(f"  Raga: {raga.name}")
        lines.append(f"  ID:   {raga.id}")
        lines.append("")
        lines.append(f"  Arohanam:    {' '.join(raga.arohanam)}")
        lines.append(f"  Avarohanam:  {' '.join(raga.avarohanam)}")
        lines.append(f"  Swara count: {raga.swara_count}")
        lines.append("")
        if raga.is_melakarta:
            lines.append(f"  Melakarta #: {raga.melakarta_number}")
        elif raga.parent_melakarta:
            lines.append(f"  Parent Melakarta: {raga.parent_melakarta}")
        lines.append("")
        lines.append("  Click 'Generate & Play' to hear this raga.")

        self._set_play_text("\n".join(lines))

    # ------------------------------------------------------------------ PLAY: Generate
    def _generate_and_play(self):
        sel = self.raga_listbox.curselection()
        if not sel or not hasattr(self, '_search_results'):
            return
        idx = sel[0]
        if idx >= len(self._search_results):
            return

        raga = self._search_results[idx]
        fmt = self.format_var.get()
        tonic = self.tonic_var.get()

        self.play_btn.configure(state="disabled")
        self.status_var.set(f"Generating {raga.name} scale...")

        def generate():
            try:
                from raga_detection.raga_player import generate_midi, generate_audio_wave, parse_tonic

                # Ensure output directory
                os.makedirs("output", exist_ok=True)
                safe_name = raga.id.replace(' ', '_').lower()
                ext = '.mid' if fmt == 'midi' else '.wav'
                out_path = os.path.join("output", f"{safe_name}_scale{ext}")

                if fmt == 'midi':
                    generate_midi(
                        raga.arohanam, raga.avarohanam, out_path,
                        tonic=tonic, tempo=80, instrument=73
                    )
                else:
                    tonic_midi = parse_tonic(tonic)
                    tonic_hz = 440.0 * (2.0 ** ((tonic_midi - 69) / 12.0))
                    generate_audio_wave(
                        raga.arohanam, raga.avarohanam, out_path,
                        tonic_hz=tonic_hz
                    )

                self.root.after(0, lambda: self._on_generated(raga, out_path, fmt))
            except Exception as e:
                import traceback
                err = f"{e}\n\n{traceback.format_exc()}"
                self.root.after(0, lambda: self._on_generate_error(err))

        threading.Thread(target=generate, daemon=True).start()

    def _on_generated(self, raga, out_path, fmt):
        self.play_btn.configure(state="normal")
        abs_path = os.path.abspath(out_path)
        self.last_generated_wav = abs_path if fmt == 'wav' else None

        lines = []
        lines.append(f"  Generated: {raga.name}")
        lines.append(f"  File: {abs_path}")
        lines.append(f"  Format: {fmt.upper()}")
        lines.append("")
        lines.append(f"  Arohanam:    {' '.join(raga.arohanam)}")
        lines.append(f"  Avarohanam:  {' '.join(raga.avarohanam)}")
        lines.append("")

        # Try to play WAV if possible
        played = False
        if fmt == 'wav' and HAS_SOUNDDEVICE and HAS_SOUNDFILE:
            try:
                data, sr = sf.read(abs_path, dtype='float32')
                sd.play(data, sr)
                played = True
                lines.append("  Playing audio...")
            except Exception as e:
                lines.append(f"  (Could not play: {e})")

        if not played:
            if fmt == 'midi':
                lines.append("  Open the .mid file in your MIDI player.")
            else:
                lines.append("  Open the .wav file in your audio player,")
                lines.append("  or install sounddevice + soundfile for playback.")

        self._set_play_text("\n".join(lines))
        self.status_var.set(f"Generated {raga.name} scale  |  {abs_path}")

    def _on_generate_error(self, error):
        self.play_btn.configure(state="normal")
        self.status_var.set("Generation failed")
        self._set_play_text(f"Error generating raga:\n\n{error}")

    def _open_output_folder(self):
        out_dir = os.path.abspath("output")
        os.makedirs(out_dir, exist_ok=True)
        os.startfile(out_dir)

    # ========================== LIVE DETECT TAB ==========================
    def _create_live_tab(self):
        tab = self.live_tab
        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(4, weight=1)  # results row

        # Title
        ttk.Label(
            tab, text="Live Raga Detection",
            font=("Segoe UI", 14, "bold")
        ).grid(row=0, column=0, pady=(0, 5))

        ttk.Label(
            tab,
            text="Sing a krithi / raga phrases — detection runs continuously until the raga is identified.",
            foreground="gray", font=("Segoe UI", 9),
            wraplength=700,
        ).grid(row=1, column=0, pady=(0, 8))

        # -- Control frame --
        ctrl_frame = ttk.Frame(tab)
        ctrl_frame.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        ctrl_frame.columnconfigure(2, weight=1)

        self.live_btn = ttk.Button(
            ctrl_frame, text="Start Listening",
            command=self._toggle_live, width=18,
        )
        self.live_btn.grid(row=0, column=0, padx=(0, 10))
        if not HAS_SOUNDDEVICE:
            self.live_btn.configure(state="disabled")

        self.live_elapsed_var = tk.StringVar(value="")
        ttk.Label(
            ctrl_frame, textvariable=self.live_elapsed_var,
            font=("Segoe UI", 10),
        ).grid(row=0, column=1, padx=(0, 10))

        self.live_confidence_var = tk.StringVar(value="")
        self.live_conf_label = ttk.Label(
            ctrl_frame, textvariable=self.live_confidence_var,
            font=("Segoe UI", 10, "bold"),
        )
        self.live_conf_label.grid(row=0, column=2, sticky="w")

        # -- Live swara display --
        swara_frame = ttk.LabelFrame(tab, text="Detected Swaras", padding=8)
        swara_frame.grid(row=3, column=0, sticky="ew", pady=(0, 8))
        swara_frame.columnconfigure(0, weight=1)

        self.live_swara_var = tk.StringVar(value="—")
        ttk.Label(
            swara_frame, textvariable=self.live_swara_var,
            font=("Consolas", 14), foreground="#2060a0",
        ).grid(row=0, column=0, sticky="w")

        self.live_seq_var = tk.StringVar(value="")
        ttk.Label(
            swara_frame, textvariable=self.live_seq_var,
            font=("Consolas", 9), foreground="gray",
            wraplength=700, justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(4, 0))

        # -- Results --
        results_frame = ttk.LabelFrame(tab, text="Top Matches (updating live)", padding=8)
        results_frame.grid(row=4, column=0, sticky="nsew")
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)

        self.live_results = tk.Text(
            results_frame, wrap="word", font=("Consolas", 10),
            state="disabled", bg="#f5f5f5",
        )
        self.live_results.grid(row=0, column=0, sticky="nsew")
        sb = ttk.Scrollbar(results_frame, orient="vertical",
                           command=self.live_results.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.live_results.configure(yscrollcommand=sb.set)

    # ------------------------------------------------------------------ LIVE: Controls
    def _toggle_live(self):
        if self.live_detector and self.live_detector.is_running:
            self._stop_live()
        else:
            self._start_live()

    def _start_live(self):
        if not self.matcher:
            return

        from raga_detection.realtime_detector import RealtimeRagaDetector

        self.live_detector = RealtimeRagaDetector(
            matcher=self.matcher,
            on_update=lambda s: self.root.after(0, lambda: self._on_live_update(s)),
        )

        self.live_btn.configure(text="Stop Listening")
        self.live_elapsed_var.set("0s")
        self.live_confidence_var.set("listening...")
        self.live_conf_label.configure(foreground="gray")
        self.live_swara_var.set("—")
        self.live_seq_var.set("")
        self._set_live_text(
            "Listening... sing raga phrases into your microphone.\n\n"
            "  - Sing naturally (krithis, phrases, or scales)\n"
            "  - Detection updates every ~3 seconds\n"
            "  - Will auto-stop when raga is identified with high confidence\n"
            "  - Or click 'Stop Listening' for the best guess so far\n"
        )
        self.status_var.set("Live detection active — listening...")
        self.live_detector.start()

    def _stop_live(self):
        if self.live_detector:
            self.live_detector.stop()
            # The finalise callback will fire and update the UI
            self.live_btn.configure(text="Start Listening")

    # ------------------------------------------------------------------ LIVE: Updates
    def _on_live_update(self, status):
        from raga_detection.realtime_detector import LiveStatus

        # Update elapsed time
        secs = int(status.elapsed_seconds)
        self.live_elapsed_var.set(f"{secs}s")

        # Update confidence indicator
        conf = status.confidence
        color_map = {
            "listening...": "gray",
            "low": "#c06000",
            "medium": "#a0a000",
            "high": "#00a000",
            "detected!": "#008000",
            "best guess": "#806000",
            "insufficient audio": "red",
        }
        self.live_confidence_var.set(conf)
        self.live_conf_label.configure(
            foreground=color_map.get(conf, "gray"))

        # Update detected swaras
        if status.current_swaras:
            self.live_swara_var.set(" ".join(status.current_swaras))
        if status.raw_sequence:
            # Show last 30 notes of the sequence
            seq = status.raw_sequence[-30:]
            self.live_seq_var.set(" → ".join(seq))

        # Update results
        if status.top5:
            lines = []
            if status.done:
                lines.append("=" * 52)
                if status.confidence == "detected!":
                    lines.append("  ★ RAGA DETECTED ★")
                else:
                    lines.append(f"  BEST GUESS ({status.confidence})")
                lines.append("=" * 52)
            else:
                lines.append(f"  updating... ({secs}s elapsed)")
                lines.append("-" * 52)

            lines.append("")
            if status.tonic_hz > 0:
                lines.append(f"  Tonic (Sa): {status.tonic_hz:.1f} Hz")
            if status.current_swaras:
                lines.append(f"  Swaras:     {' '.join(status.current_swaras)}")
            lines.append("")

            for i, (name, score) in enumerate(status.top5, 1):
                pct = min(score * 100, 100.0)
                bar = "█" * int(pct / 5)
                marker = " ◄" if i == 1 else ""
                lines.append(f"  {i}. {name:<26} {pct:5.1f}% {bar}{marker}")

            if status.swara_counts:
                lines.append("")
                lines.append("  Swara frequency:")
                sorted_counts = sorted(
                    status.swara_counts.items(),
                    key=lambda x: -x[1])
                counts_str = "  " + "  ".join(
                    f"{s}:{c}" for s, c in sorted_counts if s != 'S')
                lines.append(counts_str)

            if status.done:
                lines.append("")
                lines.append(f"  Listened for {secs} seconds")
                lines.append("-" * 52)

            self._set_live_text("\n".join(lines))

        # Update status bar
        if status.top_match:
            self.status_var.set(
                f"Live: {status.top_match} ({min(status.top_score * 100, 100):.0f}%) "
                f"| {conf} | {secs}s")

        # Handle errors
        if status.error:
            self._set_live_text(f"Error: {status.error}")
            self.status_var.set(f"Live detection error: {status.error}")

        # Handle completion
        if status.done:
            self.live_btn.configure(text="Start Listening")
            if status.top_match:
                self.status_var.set(
                    f"Live detection complete: {status.top_match} "
                    f"({min(status.top_score * 100, 100):.0f}%)")

    # ------------------------------------------------------------------ HELPERS
    def _set_detect_text(self, text):
        self.detect_results.configure(state="normal")
        self.detect_results.delete(1.0, tk.END)
        self.detect_results.insert(1.0, text)
        self.detect_results.configure(state="disabled")

    def _set_play_text(self, text):
        self.play_info.configure(state="normal")
        self.play_info.delete(1.0, tk.END)
        self.play_info.insert(1.0, text)
        self.play_info.configure(state="disabled")

    def _set_live_text(self, text):
        self.live_results.configure(state="normal")
        self.live_results.delete(1.0, tk.END)
        self.live_results.insert(1.0, text)
        self.live_results.configure(state="disabled")


def main():
    root = tk.Tk()
    style = ttk.Style()
    if "clam" in style.theme_names():
        style.theme_use("clam")
    app = RagaGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
