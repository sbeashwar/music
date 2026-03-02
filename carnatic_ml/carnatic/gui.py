"""
GUI for Raga Detection with Microphone Recording

Record from microphone or load audio files to detect the raga.
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path so 'carnatic' package is importable
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import numpy as np

# Check for sounddevice
try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False


class RagaDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Carnatic Raga Detector")
        self.root.geometry("700x620")
        self.root.resizable(True, True)
        
        # State
        self.detector = None
        self.audio_path = None
        self.recorded_audio = None
        self.is_recording = False
        self.record_seconds = 30
        
        self._create_widgets()
        self._load_detector_async()
    
    def _create_widgets(self):
        # Main frame with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Configure grid weights for resizing
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(5, weight=1)  # Results frame row
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="🎵 Carnatic Raga Detector",
            font=("Segoe UI", 16, "bold")
        )
        title_label.grid(row=0, column=0, pady=(0, 15))
        
        # === Recording Frame ===
        record_frame = ttk.LabelFrame(main_frame, text="🎤 Record from Microphone", padding="10")
        record_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        record_frame.columnconfigure(1, weight=1)
        
        self.record_btn = ttk.Button(
            record_frame,
            text="🔴 Record 30 sec",
            command=self._toggle_recording,
            width=18
        )
        self.record_btn.grid(row=0, column=0, padx=(0, 5))
        
        self.save_btn = ttk.Button(
            record_frame,
            text="💾 Save",
            command=self._save_recording,
            width=8,
            state="disabled"
        )
        self.save_btn.grid(row=0, column=1, padx=(0, 10))
        
        # Recording progress bar
        self.record_progress = ttk.Progressbar(
            record_frame,
            mode="determinate",
            maximum=self.record_seconds
        )
        self.record_progress.grid(row=0, column=2, sticky="ew", padx=(0, 10))
        
        self.record_status = ttk.Label(record_frame, text="Ready to record", foreground="gray")
        self.record_status.grid(row=0, column=3)
        
        if not HAS_SOUNDDEVICE:
            self.record_btn.configure(state="disabled")
            self.record_status.configure(text="Install sounddevice: pip install sounddevice")
        
        # === File Selection Frame ===
        file_frame = ttk.LabelFrame(main_frame, text="📁 Or Load Audio File", padding="10")
        file_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        file_frame.columnconfigure(0, weight=1)
        
        self.file_label = ttk.Label(
            file_frame, 
            text="No file selected",
            foreground="gray"
        )
        self.file_label.grid(row=0, column=0, sticky="w", padx=(0, 10))
        
        self.browse_btn = ttk.Button(
            file_frame, 
            text="Browse...",
            command=self._browse_file
        )
        self.browse_btn.grid(row=0, column=1)
        
        # === Shruti (Tonic) Selection Frame ===
        shruti_frame = ttk.LabelFrame(main_frame, text="🎹 Shruti (Tonic) Setting", padding="10")
        shruti_frame.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        shruti_frame.columnconfigure(1, weight=1)
        
        ttk.Label(shruti_frame, text="Sa = ").grid(row=0, column=0)
        
        # Shruti dropdown
        self.shruti_var = tk.StringVar(value="Auto-detect")
        self.shruti_notes = [
            "Auto-detect",
            "C (130.8 Hz)", "C# (138.6 Hz)", "D (146.8 Hz)", "D# (155.6 Hz)",
            "E (164.8 Hz)", "F (174.6 Hz)", "F# (185.0 Hz)", "G (196.0 Hz)",
            "G# (207.7 Hz)", "A (220.0 Hz)", "A# (233.1 Hz)", "B (246.9 Hz)"
        ]
        self.shruti_combo = ttk.Combobox(
            shruti_frame,
            textvariable=self.shruti_var,
            values=self.shruti_notes,
            state="readonly",
            width=18
        )
        self.shruti_combo.grid(row=0, column=1, padx=5, sticky="w")
        
        # Octave selector
        ttk.Label(shruti_frame, text="Octave:").grid(row=0, column=2, padx=(10, 0))
        self.octave_var = tk.StringVar(value="3")
        self.octave_combo = ttk.Combobox(
            shruti_frame,
            textvariable=self.octave_var,
            values=["2 (Low)", "3 (Mid)", "4 (High)"],
            state="readonly",
            width=10
        )
        self.octave_combo.grid(row=0, column=3, padx=5)
        
        shruti_hint = ttk.Label(
            shruti_frame,
            text="Set your Sa if auto-detection is wrong",
            foreground="gray",
            font=("Segoe UI", 8)
        )
        shruti_hint.grid(row=1, column=0, columnspan=4, sticky="w", pady=(5, 0))
        
        # === Detect Button ===
        self.detect_btn = ttk.Button(
            main_frame,
            text="🔍 Detect Raga",
            command=self._detect_raga,
            state="disabled"
        )
        self.detect_btn.grid(row=4, column=0, pady=10)
        
        # === Results Frame ===
        results_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        results_frame.grid(row=5, column=0, sticky="nsew", pady=(0, 10))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Results text with scrollbar
        self.results_text = tk.Text(
            results_frame,
            wrap="word",
            font=("Consolas", 10),
            state="disabled",
            bg="#f5f5f5"
        )
        self.results_text.grid(row=0, column=0, sticky="nsew")
        
        scrollbar = ttk.Scrollbar(
            results_frame, 
            orient="vertical",
            command=self.results_text.yview
        )
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.results_text.configure(yscrollcommand=scrollbar.set)
        
        # === Status Bar ===
        self.status_var = tk.StringVar(value="Loading raga database...")
        self.status_label = ttk.Label(
            main_frame,
            textvariable=self.status_var,
            foreground="gray"
        )
        self.status_label.grid(row=6, column=0, sticky="w")
        
        # Loading progress bar
        self.progress = ttk.Progressbar(
            main_frame,
            mode="indeterminate",
            length=200
        )
        self.progress.grid(row=7, column=0, sticky="ew", pady=(5, 0))
        self.progress.start()
    
    def _load_detector_async(self):
        """Load the detector in a background thread."""
        def load():
            try:
                from carnatic.detector_v2 import RagaDetectorV2
                self.detector = RagaDetectorV2()
                raga_count = len(self.detector.db.ragas)
                self.root.after(0, lambda: self._on_detector_loaded(raga_count))
            except Exception as e:
                err_msg = str(e)
                self.root.after(0, lambda: self._on_load_error(err_msg))
        
        thread = threading.Thread(target=load, daemon=True)
        thread.start()
    
    def _on_detector_loaded(self, raga_count):
        """Called when detector is loaded."""
        self.progress.stop()
        self.progress.grid_remove()
        self.status_var.set(f"Ready • {raga_count:,} ragas loaded")
        self._update_detect_button()
        self._set_results(
            "Choose one option:\n\n"
            "1. 🎤 Click 'Record 30 sec' and sing into your microphone\n"
            "   - Sing Sa, Pa, and other notes clearly\n"
            "   - Hold notes for better detection\n\n"
            "2. 📁 Click 'Browse' to load an audio file\n"
            "   - Supports MP3, WAV, FLAC, OGG\n\n"
            "Then click 'Detect Raga' to analyze."
        )
    
    def _on_load_error(self, error):
        """Called when detector fails to load."""
        self.progress.stop()
        self.progress.grid_remove()
        self.status_var.set("Error loading detector")
        self._set_results(f"Failed to load raga detector:\n\n{error}\n\n"
                         "Make sure dependencies are installed:\n"
                         "  pip install librosa sounddevice")
    
    def _toggle_recording(self):
        """Start or stop recording."""
        if self.is_recording:
            self._stop_recording()
        else:
            self._start_recording()
    
    def _start_recording(self):
        """Start recording from microphone."""
        if not HAS_SOUNDDEVICE:
            messagebox.showerror("Error", "sounddevice not installed.\n\npip install sounddevice")
            return
        
        self.is_recording = True
        self.recorded_audio = None
        self.audio_path = None  # Clear any loaded file
        self.file_label.configure(text="No file selected", foreground="gray")
        
        self.record_btn.configure(text="⏹ Stop Recording")
        self.record_status.configure(text="Recording...", foreground="red")
        self.record_progress['value'] = 0
        
        self.browse_btn.configure(state="disabled")
        self.detect_btn.configure(state="disabled")
        self.save_btn.configure(state="disabled")
        
        self._set_results("🎤 Recording... Sing into your microphone!\n\n"
                         "Tips:\n"
                         "- Start with Sa (the tonic)\n"
                         "- Sing the arohanam and avarohanam\n"
                         "- Hold notes for 1-2 seconds each\n"
                         "- Include Pa if possible (helps detect tonic)")
        
        # Start continuous recording in background
        def record():
            try:
                sr = 22050
                duration = self.record_seconds
                total_samples = int(sr * duration)
                
                # Use continuous stream recording (no gaps!)
                audio_buffer = []
                samples_recorded = 0
                
                def audio_callback(indata, frames, time, status):
                    """Called for each audio block."""
                    if status:
                        print(f"Audio status: {status}")
                    audio_buffer.append(indata.copy())
                
                # Start the input stream
                with sd.InputStream(samplerate=sr, channels=1, dtype='float32',
                                   callback=audio_callback, blocksize=1024):
                    
                    # Update progress every second
                    for i in range(duration):
                        if not self.is_recording:
                            break
                        sd.sleep(1000)  # Sleep 1 second
                        self.root.after(0, lambda v=i+1: self._update_record_progress(v))
                
                # Combine all audio chunks
                if audio_buffer:
                    self.recorded_audio = np.concatenate(audio_buffer).flatten()
                    self.root.after(0, self._on_recording_complete)
                else:
                    self.root.after(0, self._on_recording_cancelled)
                    
            except Exception as e:
                self.root.after(0, lambda: self._on_recording_error(str(e)))
        
        thread = threading.Thread(target=record, daemon=True)
        thread.start()
    
    def _update_record_progress(self, seconds):
        """Update recording progress bar."""
        self.record_progress['value'] = seconds
        remaining = self.record_seconds - seconds
        self.record_status.configure(text=f"Recording... {remaining}s left", foreground="red")
    
    def _stop_recording(self):
        """Stop recording early."""
        self.is_recording = False
    
    def _on_recording_complete(self):
        """Called when recording finishes."""
        self.is_recording = False
        self.record_btn.configure(text="🔴 Record 30 sec")
        self.record_status.configure(text="Recording complete!", foreground="green")
        self.record_progress['value'] = self.record_seconds
        
        self.browse_btn.configure(state="normal")
        self.save_btn.configure(state="normal")
        self._update_detect_button()
        
        duration = len(self.recorded_audio) / 22050
        self._set_results(f"✅ Recording complete!\n\n"
                         f"Duration: {duration:.1f} seconds\n"
                         f"Samples: {len(self.recorded_audio):,}\n\n"
                         f"Click 'Detect Raga' to analyze.\n"
                         f"Click '💾 Save' to save the recording.")
    
    def _on_recording_cancelled(self):
        """Called when recording is cancelled."""
        self.is_recording = False
        self.record_btn.configure(text="🔴 Record 30 sec")
        self.record_status.configure(text="Recording cancelled", foreground="gray")
        self.browse_btn.configure(state="normal")
        self._update_detect_button()
    
    def _on_recording_error(self, error):
        """Called when recording fails."""
        self.is_recording = False
        self.record_btn.configure(text="🔴 Record 30 sec")
        self.record_status.configure(text="Recording failed", foreground="red")
        self.browse_btn.configure(state="normal")
        self._set_results(f"Recording error:\n\n{error}\n\n"
                         "Make sure your microphone is connected and working.")
    
    def _save_recording(self):
        """Save the recorded audio to a WAV file."""
        if self.recorded_audio is None:
            return
        
        from datetime import datetime
        import scipy.io.wavfile as wav
        
        # Generate default filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"recording_{timestamp}.wav"
        
        path = filedialog.asksaveasfilename(
            title="Save Recording",
            defaultextension=".wav",
            initialfile=default_name,
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        
        if path:
            try:
                # Convert float32 [-1, 1] to int16
                audio_int16 = (self.recorded_audio * 32767).astype(np.int16)
                wav.write(path, 22050, audio_int16)
                self.record_status.configure(text=f"Saved!", foreground="green")
                self._set_results(f"✅ Recording saved to:\n{path}\n\n"
                                 f"You can reload this file later for testing.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save: {e}")
    
    def _browse_file(self):
        """Open file browser to select audio."""
        filetypes = [
            ("Audio files", "*.mp3 *.wav *.flac *.ogg *.m4a"),
            ("MP3 files", "*.mp3"),
            ("WAV files", "*.wav"),
            ("All files", "*.*")
        ]
        
        path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=filetypes
        )
        
        if path:
            self.audio_path = path
            self.recorded_audio = None  # Clear any recording
            self.record_status.configure(text="Ready to record", foreground="gray")
            self.record_progress['value'] = 0
            
            filename = Path(path).name
            if len(filename) > 50:
                filename = filename[:47] + "..."
            self.file_label.configure(text=filename, foreground="black")
            self._update_detect_button()
    
    def _update_detect_button(self):
        """Enable/disable detect button based on state."""
        if self.detector and (self.audio_path or self.recorded_audio is not None):
            self.detect_btn.configure(state="normal")
        else:
            self.detect_btn.configure(state="disabled")
    
    def _get_tonic_hz(self) -> float:
        """Get tonic frequency from shruti selector, or None for auto-detect."""
        shruti = self.shruti_var.get()
        if shruti == "Auto-detect":
            return None
        
        # Base frequencies for octave 3 (C3 to B3)
        base_freqs = {
            "C": 130.81, "C#": 138.59, "D": 146.83, "D#": 155.56,
            "E": 164.81, "F": 174.61, "F#": 185.00, "G": 196.00,
            "G#": 207.65, "A": 220.00, "A#": 233.08, "B": 246.94
        }
        
        # Extract note name (e.g., "C#" from "C# (138.6 Hz)")
        note = shruti.split(" ")[0]
        base_hz = base_freqs.get(note, 220.0)
        
        # Adjust for octave
        octave_str = self.octave_var.get()
        octave = int(octave_str.split(" ")[0]) if octave_str else 3
        
        # Octave 3 is base, octave 2 is half, octave 4 is double
        if octave == 2:
            return base_hz / 2
        elif octave == 4:
            return base_hz * 2
        else:
            return base_hz
    
    def _detect_raga(self):
        """Run raga detection."""
        if not self.detector:
            return
        
        if self.recorded_audio is None and not self.audio_path:
            return
        
        # Disable UI during detection
        self.detect_btn.configure(state="disabled")
        self.browse_btn.configure(state="disabled")
        self.record_btn.configure(state="disabled")
        self.status_var.set("Analyzing audio...")
        self.progress.grid()
        self.progress.start()
        self._set_results("🔍 Detecting raga...\n\nThis may take a few seconds.")
        
        # Store references for thread
        audio_path = self.audio_path
        recorded_audio = self.recorded_audio
        detector = self.detector
        tonic_hz = self._get_tonic_hz()  # Get manual tonic or None
        
        def detect():
            try:
                if recorded_audio is not None:
                    # Use recorded audio directly
                    results = detector.detect_from_audio(recorded_audio, sr=22050, top_n=15, tonic_hz=tonic_hz)
                else:
                    # Load from file
                    import librosa
                    y, sr = librosa.load(audio_path, sr=22050, duration=60)
                    results = detector.detect_from_audio(y, sr, top_n=15, tonic_hz=tonic_hz)
                
                results_copy = list(results) if results else []
                self.root.after(0, lambda r=results_copy: self._show_results(r))
            except Exception as e:
                import traceback
                error_msg = f"{str(e)}\n\n{traceback.format_exc()}"
                self.root.after(0, lambda err=error_msg: self._show_error(err))
        
        thread = threading.Thread(target=detect, daemon=True)
        thread.start()
    
    def _show_results(self, results):
        """Display detection results."""
        self.progress.stop()
        self.progress.grid_remove()
        self.detect_btn.configure(state="normal")
        self.browse_btn.configure(state="normal")
        self.record_btn.configure(state="normal")
        
        if not results:
            self._set_results("No raga detected.\n\n"
                            "Tips:\n"
                            "- Sing more clearly with distinct notes\n"
                            "- Try holding Sa and Pa longer\n"
                            "- Reduce background noise")
            self.status_var.set("Detection complete - no matches")
            return
        
        # Format results
        lines = []
        lines.append("=" * 50)
        lines.append("  RAGA DETECTION RESULTS")
        lines.append("=" * 50)
        lines.append("")
        
        # Tonic and detected swaras
        details = results[0].match_details
        tonic = results[0].tonic_hz
        primary = sorted(details.get('primary_detected', set()))
        outliers = sorted(details.get('outliers', set()))
        
        lines.append(f"Tonic (Sa): {tonic:.1f} Hz")
        lines.append(f"Detected Scale: {' '.join(primary)}")
        if outliers:
            lines.append(f"Outliers removed: {' '.join(outliers)}")
        
        # Show detected ascending/descending patterns
        asc = details.get('detected_ascending', [])
        desc = details.get('detected_descending', [])
        if asc:
            lines.append(f"Ascending notes:  S {' '.join(asc)} S")
        if desc:
            lines.append(f"Descending notes: S {' '.join(desc)} S")
        lines.append("")
        
        # Count same-score matches
        top_score = results[0].confidence
        same_score = sum(1 for r in results if r.confidence >= top_score - 0.01)
        if same_score > 10:
            lines.append(f"⚠️ {same_score}+ ragas share this scale")
            lines.append("")
        
        lines.append("-" * 50)
        lines.append("  Top Matches")
        lines.append("-" * 50)
        
        for i, r in enumerate(results[:10], 1):
            confidence = min(r.confidence * 100, 100.0)
            bar = "█" * int(confidence / 5)
            mela = " [M]" if r.raga.is_melakarta else ""
            lines.append("")
            lines.append(f"{i:2}. {r.raga.name:<22} {confidence:5.1f}% {bar}{mela}")
            if r.raga.arohanam:
                lines.append(f"    ↑ {' '.join(r.raga.arohanam)}")
            if r.raga.avarohanam:
                lines.append(f"    ↓ {' '.join(r.raga.avarohanam)}")
        
        lines.append("")
        lines.append("-" * 50)
        
        self._set_results("\n".join(lines))
        top = results[0]
        display_conf = min(top.confidence, 1.0)
        self.status_var.set(f"Detection complete • Top match: {top.raga.name} ({display_conf:.0%})")
    
    def _show_error(self, error):
        """Display error message."""
        self.progress.stop()
        self.progress.grid_remove()
        self.detect_btn.configure(state="normal")
        self.browse_btn.configure(state="normal")
        self.record_btn.configure(state="normal")
        self.status_var.set("Detection failed")
        self._set_results(f"Error during detection:\n\n{error}")
    
    def _set_results(self, text):
        """Set the results text."""
        self.results_text.configure(state="normal")
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, text)
        self.results_text.configure(state="disabled")


def main():
    """Launch the GUI."""
    root = tk.Tk()
    
    # Apply a theme
    style = ttk.Style()
    if "clam" in style.theme_names():
        style.theme_use("clam")
    
    app = RagaDetectorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
