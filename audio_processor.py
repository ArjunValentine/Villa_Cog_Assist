#!/usr/bin/env python3
"""
Layered Audio Processing Pipeline — Ontological Resonance Matrix

Three-tier progressive audio feature extraction with edge filtering:
  Tier 1 (Core):       FFT consonance, spectral features, Silero VAD — always active
  Tier 2 (Enhanced):   MFCCs, chroma, onset, tempo, pitch — when CPU allows
  Tier 3 (Canary):     Behavioral audio analysis, prosody, anomaly detection
  Edge Filter:         YAMNet classification + Whisper.cpp STT — speech→text at the edge

Lean Workflow:
  1. Sensory Intake → 2. Edge Filtering (VAD/YAMNet/Whisper) → 3. Local DB Query →
  4. Traffic Cop Triage (Phi-3 mini) → 5. Escalate to Claude only when needed

Design principle: Extract → Vectorize → Score → Discard
Raw audio samples are NEVER stored. Only compact feature vectors + transcribed text persist.

Author: Peter J Villa / ArjunValentine
License: MIT
"""

import numpy as np
import threading
import time
import json
import sqlite3
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Tuple

# Tier 1: always available
import pyaudio

# Tier 2: optional — enhanced spectral features
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

# Edge Filter: Silero VAD (PyTorch or ONNX)
SILERO_AVAILABLE = False
try:
    import torch
    SILERO_AVAILABLE = True
except ImportError:
    try:
        import onnxruntime
        SILERO_AVAILABLE = True  # will use ONNX path
    except ImportError:
        pass

# Edge Filter: YAMNet audio classification (TFLite)
YAMNET_AVAILABLE = False
try:
    import tensorflow as tf
    YAMNET_AVAILABLE = True
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
        YAMNET_AVAILABLE = True
    except ImportError:
        pass

# Keep legacy alias
TFLITE_AVAILABLE = YAMNET_AVAILABLE

# Edge Filter: Whisper STT (local speech-to-text)
WHISPER_AVAILABLE = False
try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class AudioFeatureVector:
    """
    Complete audio feature vector for a single analysis window.
    This is what gets stored — never raw audio.
    """
    timestamp: float = 0.0
    tier: int = 1  # highest tier that produced features

    # --- Tier 1: Core (always present) ---
    consonance_score: float = 0.0        # 0.0–1.0 frequency ratio harmony
    spectral_centroid: float = 0.0       # Hz — brightness of sound
    spectral_bandwidth: float = 0.0      # Hz — spread of frequencies
    spectral_flatness: float = 0.0       # 0.0–1.0 noise vs tonal
    spectral_rolloff: float = 0.0        # Hz — frequency below which 85% energy
    rms_energy: float = 0.0              # normalized loudness
    zero_crossing_rate: float = 0.0      # rate — texture indicator
    vad_state: bool = False              # voice activity detected
    vad_confidence: float = 0.0          # Silero VAD speech probability
    peak_frequencies: List[float] = field(default_factory=list)

    # --- Tier 2: Enhanced (when librosa available) ---
    mfcc: Optional[List[float]] = None          # 13 coefficients — audio fingerprint
    chroma: Optional[List[float]] = None        # 12 bins — harmonic profile
    onset_rate: float = 0.0                     # events/sec
    tempo_estimate: float = 0.0                 # BPM
    pitch_mean: float = 0.0                     # Hz fundamental frequency
    pitch_variance: float = 0.0                 # Hz² — voice stability indicator

    # --- Edge Filter: Classification + STT ---
    audio_event: str = ""                       # YAMNet or heuristic event class
    event_confidence: float = 0.0               # classification confidence
    transcribed_text: str = ""                  # Whisper.cpp STT output (speech only)
    yamnet_top3: List[str] = field(default_factory=list)  # top 3 YAMNet classes

    # --- Tier 3: Canary (behavioral analysis) ---
    prosody_valence: float = 0.5                # 0=distressed, 1=positive
    speech_rate: float = 0.0                    # estimated syllables/sec
    repetition_score: float = 0.0               # 0=novel, 1=highly repetitive
    turn_taking_score: float = 0.0              # conversation quality
    response_latency: float = 0.0               # seconds between speech segments
    anomaly_flag: bool = False                  # canary trigger
    baseline_deviation: float = 0.0             # z-score vs rolling history

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dictionary."""
        d = asdict(self)
        # numpy arrays may sneak in — ensure pure python types
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
            elif isinstance(v, (np.floating, np.integer)):
                d[k] = float(v)
        return d

    def to_compact_vector(self) -> List[float]:
        """Flatten to a numeric vector for embedding/comparison."""
        vec = [
            self.consonance_score, self.spectral_centroid, self.spectral_bandwidth,
            self.spectral_flatness, self.spectral_rolloff, self.rms_energy,
            self.zero_crossing_rate, float(self.vad_state),
        ]
        if self.mfcc:
            vec.extend(self.mfcc)
        if self.chroma:
            vec.extend(self.chroma)
        vec.extend([
            self.onset_rate, self.tempo_estimate, self.pitch_mean, self.pitch_variance,
            self.prosody_valence, self.speech_rate, self.repetition_score,
            self.baseline_deviation,
        ])
        return vec


# ---------------------------------------------------------------------------
# Tier 1: Core — PyAudio + numpy FFT (always active)
# ---------------------------------------------------------------------------

class Tier1Core:
    """
    Core audio features extracted from raw FFT.
    Zero ML dependencies. ~2% CPU.

    Scores frequency ratios against known musical consonance intervals.
    Extracts spectral shape features for environment characterization.
    Simple energy-based voice activity detection.
    """

    # Consonant frequency ratios and their harmony scores
    CONSONANT_RATIOS: Dict[Tuple[int, int], float] = {
        (1, 1): 1.00,   # unison
        (2, 1): 0.95,   # octave
        (3, 2): 0.90,   # perfect fifth
        (4, 3): 0.85,   # perfect fourth
        (5, 4): 0.80,   # major third
        (6, 5): 0.75,   # minor third
        (5, 3): 0.70,   # major sixth
        (8, 5): 0.65,   # minor sixth
        (9, 8): 0.55,   # major second
        (16, 15): 0.40, # minor second (dissonant)
    }
    RATIO_TOLERANCE = 0.04  # ±4% tolerance for ratio matching

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

    def extract(self, samples: np.ndarray) -> dict:
        """Extract Tier 1 features from raw audio samples."""
        n = len(samples)
        if n < 64:
            return self._silent_features()

        # Apply Hanning window to reduce spectral leakage
        windowed = samples * np.hanning(n)

        # FFT
        fft = np.fft.rfft(windowed)
        magnitudes = np.abs(fft)
        freqs = np.fft.rfftfreq(n, 1.0 / self.sample_rate)

        # Total spectral energy
        total_energy = np.sum(magnitudes ** 2)
        if total_energy < 1e-10:
            return self._silent_features()

        # --- Spectral Features ---
        mag_sum = np.sum(magnitudes)

        # Centroid: center of mass of the spectrum (brightness)
        centroid = np.sum(freqs * magnitudes) / mag_sum

        # Bandwidth: spread around the centroid
        bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * magnitudes) / mag_sum)

        # Flatness: geometric mean / arithmetic mean (1.0 = white noise, 0.0 = pure tone)
        mag_nonzero = magnitudes[magnitudes > 0]
        if len(mag_nonzero) > 0:
            log_mean = np.mean(np.log(mag_nonzero + 1e-10))
            geometric_mean = np.exp(log_mean)
            arithmetic_mean = np.mean(mag_nonzero)
            flatness = float(geometric_mean / (arithmetic_mean + 1e-10))
        else:
            flatness = 0.0

        # Rolloff: frequency below which 85% of energy is concentrated
        cumulative = np.cumsum(magnitudes ** 2)
        rolloff_idx = np.searchsorted(cumulative, 0.85 * total_energy)
        rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])

        # RMS energy (loudness)
        rms = float(np.sqrt(np.mean(samples ** 2)))

        # Zero-crossing rate (texture: speech ~0.05, noise ~0.2+)
        zcr = float(np.sum(np.abs(np.diff(np.sign(samples)))) / (2.0 * n))

        # --- Peak Frequency Extraction ---
        # Find local maxima in the magnitude spectrum
        peak_indices = self._find_spectral_peaks(magnitudes, freqs)
        peak_freqs = freqs[peak_indices]
        peak_freqs = peak_freqs[peak_freqs > 20]  # filter sub-audible
        peak_freqs = peak_freqs[:8]  # top 8

        # --- Voice Activity Detection ---
        # Heuristic: sufficient energy + centroid in speech range (85–4000 Hz)
        vad = rms > 0.008 and 85 < centroid < 4000

        # --- Consonance Scoring ---
        consonance = self._score_consonance(peak_freqs)

        return {
            'consonance_score': consonance,
            'spectral_centroid': float(centroid),
            'spectral_bandwidth': float(bandwidth),
            'spectral_flatness': flatness,
            'spectral_rolloff': rolloff,
            'rms_energy': rms,
            'zero_crossing_rate': zcr,
            'vad_state': bool(vad),
            'peak_frequencies': peak_freqs.tolist(),
        }

    def _find_spectral_peaks(self, magnitudes: np.ndarray, freqs: np.ndarray, min_prominence: float = 0.1) -> np.ndarray:
        """Find peaks in magnitude spectrum using simple local-maximum detection."""
        if len(magnitudes) < 3:
            return np.array([], dtype=int)

        # Normalize magnitudes
        mag_norm = magnitudes / (np.max(magnitudes) + 1e-10)

        # Local maxima: higher than both neighbors and above prominence threshold
        peaks = []
        for i in range(1, len(mag_norm) - 1):
            if mag_norm[i] > mag_norm[i - 1] and mag_norm[i] > mag_norm[i + 1]:
                if mag_norm[i] > min_prominence:
                    peaks.append(i)

        # Sort by magnitude (descending)
        peaks.sort(key=lambda i: magnitudes[i], reverse=True)
        return np.array(peaks[:8], dtype=int)

    def _score_consonance(self, peak_freqs: np.ndarray) -> float:
        """
        Score pairwise frequency ratios against known consonant intervals.

        Musical consonance is determined by the simplicity of frequency ratios.
        A perfect fifth (3:2) sounds harmonious; a tritone (√2:1) sounds tense.
        """
        if len(peak_freqs) < 2:
            return 0.5  # neutral for single tone or silence

        scores = []
        for i in range(len(peak_freqs)):
            for j in range(i + 1, len(peak_freqs)):
                if peak_freqs[j] < 20:
                    continue

                ratio = peak_freqs[i] / peak_freqs[j]
                if ratio < 1.0:
                    ratio = 1.0 / ratio

                # Find best matching consonant interval
                best_match = 0.0
                for (num, den), interval_score in self.CONSONANT_RATIOS.items():
                    target = num / den
                    if abs(ratio - target) < self.RATIO_TOLERANCE * target:
                        best_match = max(best_match, interval_score)

                scores.append(best_match)

        if not scores:
            return 0.3  # no clear intervals = mildly dissonant

        return float(np.clip(np.mean(scores), 0.0, 1.0))

    def _silent_features(self) -> dict:
        """Return feature vector for silence / no signal."""
        return {
            'consonance_score': 0.0,
            'spectral_centroid': 0.0,
            'spectral_bandwidth': 0.0,
            'spectral_flatness': 0.0,
            'spectral_rolloff': 0.0,
            'rms_energy': 0.0,
            'zero_crossing_rate': 0.0,
            'vad_state': False,
            'peak_frequencies': [],
        }


# ---------------------------------------------------------------------------
# Tier 2: Enhanced — librosa features (activated when CPU allows)
# ---------------------------------------------------------------------------

class Tier2Enhanced:
    """
    Enhanced audio features using librosa.
    MFCCs, chroma, onset detection, tempo, pitch analysis.
    ~8% CPU. ~60-dim feature vector.

    MFCCs are the audio equivalent of facial landmarks — a compact fingerprint
    of the spectral shape that enables pattern matching over time.
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        if not LIBROSA_AVAILABLE:
            raise ImportError("librosa required for Tier 2 — install with: pip install librosa")

    def extract(self, samples: np.ndarray) -> dict:
        """Extract Tier 2 features from raw audio samples."""
        y = samples.astype(np.float32)
        sr = self.sample_rate

        # MFCCs: 13 Mel-frequency cepstral coefficients (audio fingerprint)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1)

        # Chroma: 12 pitch classes (harmonic content profile)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        # Onset detection: rhythmic events per second
        onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
        duration = len(y) / sr
        onset_rate = len(onset_frames) / duration if duration > 0 else 0.0

        # Tempo estimation (BPM)
        tempo_result = librosa.beat.beat_track(y=y, sr=sr)
        if isinstance(tempo_result[0], np.ndarray):
            tempo = float(tempo_result[0][0]) if len(tempo_result[0]) > 0 else 0.0
        else:
            tempo = float(tempo_result[0])

        # Pitch tracking (fundamental frequency)
        pitches, pitch_mags = librosa.core.piptrack(y=y, sr=sr)
        pitch_values = []
        for t in range(pitch_mags.shape[1]):
            idx = pitch_mags[:, t].argmax()
            if pitch_mags[idx, t] > 0.1:
                pitch_values.append(pitches[idx, t])

        pitch_mean = float(np.mean(pitch_values)) if pitch_values else 0.0
        pitch_var = float(np.var(pitch_values)) if len(pitch_values) > 1 else 0.0

        return {
            'mfcc': mfcc_mean.tolist(),
            'chroma': chroma_mean.tolist(),
            'onset_rate': float(onset_rate),
            'tempo_estimate': tempo,
            'pitch_mean': pitch_mean,
            'pitch_variance': pitch_var,
        }


# ---------------------------------------------------------------------------
# Edge Filter: Silero VAD — replaces heuristic voice activity detection
# ---------------------------------------------------------------------------

class SileroVAD:
    """
    Silero Voice Activity Detection.

    Replaces the energy-based heuristic VAD with a trained neural model.
    ~1MB model, processes 512-sample chunks (32ms at 16kHz).
    Much more accurate at distinguishing speech from ambient noise.

    Supports PyTorch (preferred) or ONNX runtime (lighter).
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.model = None
        self._use_onnx = False
        self._state = None  # model hidden state (for streaming)
        self._load_model()

    def _load_model(self):
        """Load Silero VAD model (PyTorch or ONNX)."""
        try:
            import torch
            self.model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                trust_repo=True,
            )
            self.model.eval()
            self._torch = torch
            print("[SileroVAD] Loaded PyTorch model")
        except Exception as e:
            print(f"[SileroVAD] PyTorch load failed ({e}), trying ONNX...")
            try:
                import onnxruntime
                # Expect silero_vad.onnx in working directory or models/
                import os
                for path in ['silero_vad.onnx', 'models/silero_vad.onnx']:
                    if os.path.exists(path):
                        self.model = onnxruntime.InferenceSession(path)
                        self._use_onnx = True
                        print(f"[SileroVAD] Loaded ONNX model from {path}")
                        return
                print("[SileroVAD] ONNX model file not found")
            except Exception as e2:
                print(f"[SileroVAD] ONNX load also failed: {e2}")

    def detect(self, samples: np.ndarray) -> Tuple[bool, float]:
        """
        Detect voice activity in audio samples.

        Args:
            samples: Audio samples (float32, 16kHz)

        Returns:
            (is_speech, confidence) where confidence is 0.0–1.0
        """
        if self.model is None:
            return False, 0.0

        try:
            if self._use_onnx:
                return self._detect_onnx(samples)
            else:
                return self._detect_torch(samples)
        except Exception as e:
            print(f"[SileroVAD] Detection error: {e}")
            return False, 0.0

    def _detect_torch(self, samples: np.ndarray) -> Tuple[bool, float]:
        """PyTorch inference path."""
        audio = self._torch.FloatTensor(samples)
        # Process in 512-sample chunks, take max probability
        chunk_size = 512
        max_prob = 0.0
        for i in range(0, len(audio), chunk_size):
            chunk = audio[i:i + chunk_size]
            if len(chunk) < chunk_size:
                chunk = self._torch.nn.functional.pad(chunk, (0, chunk_size - len(chunk)))
            prob = self.model(chunk, self.sample_rate).item()
            max_prob = max(max_prob, prob)
        return max_prob > 0.5, float(max_prob)

    def _detect_onnx(self, samples: np.ndarray) -> Tuple[bool, float]:
        """ONNX runtime inference path."""
        # Simplified: feed full window, model handles internally
        audio = samples.astype(np.float32).reshape(1, -1)
        if self._state is None:
            # Initialize hidden state for ONNX model
            self._state = np.zeros((2, 1, 64), dtype=np.float32)
        ort_inputs = {
            'input': audio,
            'state': self._state,
            'sr': np.array([self.sample_rate], dtype=np.int64),
        }
        try:
            output, state_out = self.model.run(None, ort_inputs)
            self._state = state_out
            prob = float(output[0])
            return prob > 0.5, prob
        except Exception:
            # Fallback: simple energy check
            rms = float(np.sqrt(np.mean(samples ** 2)))
            return rms > 0.01, rms

    def reset(self):
        """Reset model state (call between sessions)."""
        self._state = None
        if self.model and not self._use_onnx:
            self.model.reset_states()


# ---------------------------------------------------------------------------
# Edge Filter: YAMNet — 521-class audio event classification
# ---------------------------------------------------------------------------

class YAMNetClassifier:
    """
    YAMNet audio event classification via TFLite.

    521 audio event categories: speech, music, dog bark, glass breaking,
    coughing, siren, footsteps, etc. Processes 0.975s windows at 16kHz.

    Critical for the canary function: detects falls, distress sounds,
    glass breaking, and other environmental safety events.
    """

    # Top canary-relevant YAMNet classes
    SAFETY_EVENTS = {
        'Glass', 'Shatter', 'Crash', 'Thud', 'Bang',
        'Screaming', 'Crying', 'Whimper', 'Groan', 'Moan',
        'Fall', 'Thump', 'Slam',
        'Fire alarm', 'Smoke detector', 'Siren', 'Alarm',
        'Cough', 'Choking', 'Gasp',
    }

    def __init__(self, model_path: str = 'yamnet.tflite', class_map_path: str = 'yamnet_class_map.csv'):
        self.interpreter = None
        self.class_names = []
        self._load_model(model_path, class_map_path)

    def _load_model(self, model_path: str, class_map_path: str):
        """Load YAMNet TFLite model and class map."""
        import os

        # Load class names
        if os.path.exists(class_map_path):
            with open(class_map_path, 'r') as f:
                # CSV format: index, mid, display_name
                lines = f.readlines()[1:]  # skip header
                self.class_names = [line.strip().split(',')[-1].strip('"') for line in lines]

        # Load TFLite model
        if os.path.exists(model_path):
            try:
                try:
                    import tensorflow as tf
                    self.interpreter = tf.lite.Interpreter(model_path=model_path)
                except ImportError:
                    import tflite_runtime.interpreter as tflite_rt
                    self.interpreter = tflite_rt.Interpreter(model_path=model_path)

                self.interpreter.allocate_tensors()
                self._input_details = self.interpreter.get_input_details()
                self._output_details = self.interpreter.get_output_details()
                print(f"[YAMNet] Loaded model with {len(self.class_names)} classes")
            except Exception as e:
                print(f"[YAMNet] Failed to load model: {e}")
        else:
            print(f"[YAMNet] Model not found at {model_path} — using heuristic fallback")

    def classify(self, samples: np.ndarray, sample_rate: int = 16000) -> Tuple[str, float, List[str]]:
        """
        Classify audio event.

        Returns:
            (top_class, confidence, top3_classes)
        """
        if self.interpreter is None or not self.class_names:
            return "", 0.0, []

        try:
            # YAMNet expects float32 waveform at 16kHz
            audio = samples.astype(np.float32)

            # Set input tensor
            self.interpreter.set_tensor(self._input_details[0]['index'], audio)
            self.interpreter.invoke()

            # Get output scores
            scores = self.interpreter.get_tensor(self._output_details[0]['index'])
            mean_scores = np.mean(scores, axis=0)  # average across time frames

            # Top 3
            top_indices = np.argsort(mean_scores)[-3:][::-1]
            top_class = self.class_names[top_indices[0]] if top_indices[0] < len(self.class_names) else ""
            confidence = float(mean_scores[top_indices[0]])
            top3 = [self.class_names[i] for i in top_indices if i < len(self.class_names)]

            return top_class, confidence, top3

        except Exception as e:
            print(f"[YAMNet] Classification error: {e}")
            return "", 0.0, []

    def is_safety_event(self, event_class: str) -> bool:
        """Check if the classified event is safety-relevant for canary."""
        return any(s.lower() in event_class.lower() for s in self.SAFETY_EVENTS)


# ---------------------------------------------------------------------------
# Edge Filter: Whisper STT — local speech-to-text (never sends audio to cloud)
# ---------------------------------------------------------------------------

class WhisperSTT:
    """
    Local speech-to-text via faster-whisper (CTranslate2 backend).

    Converts speech segments to text ON THE EDGE DEVICE.
    Raw audio never leaves the local machine — only transcribed text
    gets stored in the database or sent to the traffic cop.

    Model sizes:
      tiny  (~39MB, fastest, good enough for triage keywords)
      base  (~74MB, better accuracy)
      small (~244MB, good accuracy, slower)
    """

    def __init__(self, model_size: str = 'tiny', device: str = 'cpu', compute_type: str = 'int8'):
        self.model = None
        self.model_size = model_size
        try:
            from faster_whisper import WhisperModel
            self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
            print(f"[WhisperSTT] Loaded {model_size} model ({device}/{compute_type})")
        except Exception as e:
            print(f"[WhisperSTT] Failed to load: {e}")

    def transcribe(self, samples: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe speech audio to text.

        Args:
            samples: Audio samples (float32, 16kHz) — should be speech segment
            sample_rate: Sample rate (default 16kHz)

        Returns:
            Transcribed text string. Empty string if no speech detected.
        """
        if self.model is None:
            return ""

        try:
            segments, info = self.model.transcribe(
                samples,
                beam_size=1,       # fastest
                language=None,     # auto-detect
                vad_filter=False,  # we already ran Silero VAD
            )
            text = ' '.join([seg.text for seg in segments]).strip()
            return text
        except Exception as e:
            print(f"[WhisperSTT] Transcription error: {e}")
            return ""


# ---------------------------------------------------------------------------
# Tier 3: Canary — behavioral analysis + anomaly detection
# ---------------------------------------------------------------------------

class Tier3Canary:
    """
    Full behavioral audio canary.

    Monitors for:
    - Speech pattern changes (repetition, rate shifts, word-finding pauses)
    - Environmental sounds (falls, glass, prolonged silence)
    - Emotional prosody (pitch contour, energy patterns, distress)
    - Conversation quality (turn-taking, response latency, engagement)

    Flags anomalies against a rolling baseline for episodic consciousness monitoring.
    """

    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate

        # Rolling baselines for anomaly detection
        self.mfcc_history: deque = deque(maxlen=720)       # ~1 hour at 5s windows
        self.energy_history: deque = deque(maxlen=720)
        self.speech_segments: deque = deque(maxlen=720)     # VAD history
        self.pitch_history: deque = deque(maxlen=720)
        self.composite_history: deque = deque(maxlen=720)   # composite feature vector

        # Turn-taking state
        self._last_vad_state = False
        self._speech_onset_time = 0.0
        self._silence_onset_time = 0.0
        self._speech_durations: deque = deque(maxlen=60)    # last 5 min of segments
        self._silence_durations: deque = deque(maxlen=60)

    def extract(self, samples: np.ndarray, tier1: dict, tier2: Optional[dict] = None) -> dict:
        """
        Extract Tier 3 canary features.

        Uses Tier 1 and Tier 2 features as inputs (no redundant recomputation).
        """
        now = time.time()
        result = {
            'audio_event': self._classify_environment(tier1),
            'event_confidence': 0.0,
            'prosody_valence': 0.5,
            'speech_rate': 0.0,
            'repetition_score': 0.0,
            'turn_taking_score': 0.0,
            'response_latency': 0.0,
            'anomaly_flag': False,
            'baseline_deviation': 0.0,
        }

        # --- Prosody analysis (emotional tone from pitch + energy patterns) ---
        if tier2 and tier2.get('pitch_mean', 0) > 0:
            result['prosody_valence'] = self._analyze_prosody(tier2, tier1)

        # --- Speech rate estimation ---
        if tier2 and tier2.get('onset_rate', 0) > 0 and tier1.get('vad_state', False):
            # Approximate syllable rate from onset rate during speech
            result['speech_rate'] = tier2['onset_rate'] * 0.75

        # --- Repetition detection ---
        if tier2 and tier2.get('mfcc'):
            mfcc_vec = np.array(tier2['mfcc'])
            result['repetition_score'] = self._detect_repetition(mfcc_vec)
            self.mfcc_history.append(mfcc_vec)

        # --- Turn-taking and conversation quality ---
        vad_now = tier1.get('vad_state', False)
        turn_score, resp_latency = self._update_turn_taking(vad_now, now)
        result['turn_taking_score'] = turn_score
        result['response_latency'] = resp_latency

        # --- Anomaly detection (z-score vs rolling baseline) ---
        composite = self._build_composite(tier1, tier2)
        deviation = self._compute_deviation(composite)
        result['baseline_deviation'] = deviation
        result['anomaly_flag'] = abs(deviation) > 2.5  # >2.5σ = canary trigger
        self.composite_history.append(composite)

        # Track energy and speech for trend analysis
        self.energy_history.append(tier1.get('rms_energy', 0))
        self.speech_segments.append(vad_now)
        if tier2:
            self.pitch_history.append(tier2.get('pitch_mean', 0))

        return result

    def _classify_environment(self, tier1: dict) -> str:
        """
        Classify the audio environment from Tier 1 features.
        Lightweight heuristic classification (no ML model required).

        For full YAMNet 521-class classification, load the TFLite model.
        This heuristic covers the most critical canary categories.
        """
        rms = tier1.get('rms_energy', 0)
        vad = tier1.get('vad_state', False)
        flatness = tier1.get('spectral_flatness', 0)
        centroid = tier1.get('spectral_centroid', 0)
        zcr = tier1.get('zero_crossing_rate', 0)
        consonance = tier1.get('consonance_score', 0)

        if rms < 0.005:
            return "silence"
        elif vad and consonance > 0.6:
            return "speech_calm"
        elif vad and consonance < 0.4:
            return "speech_stressed"
        elif vad:
            return "speech"
        elif consonance > 0.7 and flatness < 0.3:
            return "music"
        elif flatness > 0.6:
            return "noise"
        elif centroid > 4000 and rms > 0.1:
            return "impact"  # sudden high-frequency energy (fall, crash)
        elif zcr > 0.15 and rms > 0.05:
            return "environmental_active"
        else:
            return "ambient"

    def _analyze_prosody(self, tier2: dict, tier1: dict) -> float:
        """
        Estimate emotional valence from prosody features.

        Higher pitch + higher energy + more variation = more aroused/positive
        Low pitch + low energy + flat contour = withdrawn/negative
        Very high pitch variance + high energy = distress

        Returns 0.0 (distressed) to 1.0 (positive).
        """
        pitch = tier2.get('pitch_mean', 0)
        pitch_var = tier2.get('pitch_variance', 0)
        energy = tier1.get('rms_energy', 0)

        # Establish pitch baseline
        if self.pitch_history:
            pitch_baseline = np.mean(list(self.pitch_history))
            pitch_std = np.std(list(self.pitch_history)) + 1e-6
            pitch_z = (pitch - pitch_baseline) / pitch_std
        else:
            pitch_z = 0.0

        # Very high variance + high energy = distress signal
        if pitch_var > 2000 and energy > 0.08:
            return 0.15  # likely distressed

        # Moderate elevation with moderate variance = engaged/positive
        if 0 < pitch_z < 1.5 and pitch_var < 1000:
            return 0.75

        # Low energy, flat pitch = withdrawn
        if energy < 0.01 and pitch_var < 100:
            return 0.3

        # Default: neutral
        return 0.5

    def _detect_repetition(self, current_mfcc: np.ndarray) -> float:
        """
        Detect repetitive audio patterns by comparing MFCC fingerprints.

        High repetition score = same sounds repeating (potential cognitive flag).
        Compares against last 60 windows (~5 minutes).
        """
        if len(self.mfcc_history) < 6:
            return 0.0

        # Compare current MFCC to recent history
        recent = list(self.mfcc_history)[-60:]  # last 5 minutes
        similarities = []
        for past_mfcc in recent:
            # Cosine similarity
            dot = np.dot(current_mfcc, past_mfcc)
            norm = (np.linalg.norm(current_mfcc) * np.linalg.norm(past_mfcc))
            if norm > 0:
                similarities.append(dot / norm)

        if not similarities:
            return 0.0

        # High mean similarity = repetitive pattern
        mean_sim = np.mean(similarities)
        # Score: 0 = all novel, 1 = exact repetition
        return float(np.clip((mean_sim - 0.7) / 0.3, 0.0, 1.0))

    def _update_turn_taking(self, vad_now: bool, now: float) -> Tuple[float, float]:
        """
        Track speech/silence transitions for conversation quality assessment.

        Good turn-taking: alternating speech/silence with moderate durations.
        Poor turn-taking: long monologues, very long silences, no transitions.
        """
        response_latency = 0.0

        # Detect transitions
        if vad_now and not self._last_vad_state:
            # Silence → Speech transition
            if self._silence_onset_time > 0:
                silence_dur = now - self._silence_onset_time
                self._silence_durations.append(silence_dur)
                response_latency = silence_dur
            self._speech_onset_time = now

        elif not vad_now and self._last_vad_state:
            # Speech → Silence transition
            if self._speech_onset_time > 0:
                speech_dur = now - self._speech_onset_time
                self._speech_durations.append(speech_dur)
            self._silence_onset_time = now

        self._last_vad_state = vad_now

        # Turn-taking score: based on regularity of transitions
        if len(self._speech_durations) < 3 or len(self._silence_durations) < 3:
            return 0.5, response_latency  # insufficient data

        speech_var = np.std(list(self._speech_durations))
        silence_var = np.std(list(self._silence_durations))
        n_transitions = len(self._speech_durations) + len(self._silence_durations)

        # More transitions + lower variance = better conversation quality
        transition_score = min(n_transitions / 20.0, 1.0)  # normalize to ~20 transitions/5min
        regularity_score = 1.0 / (1.0 + speech_var + silence_var)

        turn_score = float(np.clip(0.5 * transition_score + 0.5 * regularity_score, 0.0, 1.0))
        return turn_score, response_latency

    def _build_composite(self, tier1: dict, tier2: Optional[dict]) -> np.ndarray:
        """Build a composite numeric vector for baseline comparison."""
        vec = [
            tier1.get('consonance_score', 0),
            tier1.get('spectral_centroid', 0) / 8000.0,  # normalize
            tier1.get('spectral_bandwidth', 0) / 4000.0,
            tier1.get('spectral_flatness', 0),
            tier1.get('rms_energy', 0),
            tier1.get('zero_crossing_rate', 0),
            float(tier1.get('vad_state', False)),
        ]
        if tier2 and tier2.get('pitch_mean'):
            vec.extend([
                tier2['pitch_mean'] / 500.0,  # normalize
                tier2['pitch_variance'] / 5000.0,
                tier2['onset_rate'] / 10.0,
            ])
        return np.array(vec, dtype=np.float32)

    def _compute_deviation(self, current: np.ndarray) -> float:
        """Compute z-score deviation from rolling baseline."""
        if len(self.composite_history) < 30:
            return 0.0  # not enough baseline data

        history = np.array(list(self.composite_history))
        baseline_mean = np.mean(history, axis=0)
        baseline_std = np.std(history, axis=0) + 1e-6

        # Mean z-score across all features
        z_scores = (current[:len(baseline_mean)] - baseline_mean) / baseline_std
        return float(np.mean(np.abs(z_scores)))

    def get_canary_summary(self) -> dict:
        """Generate a summary of canary state for external monitoring."""
        speech_ratio = 0.0
        if self.speech_segments:
            speech_ratio = sum(1 for s in self.speech_segments if s) / len(self.speech_segments)

        energy_trend = 0.0
        if len(self.energy_history) > 10:
            recent = list(self.energy_history)[-10:]
            older = list(self.energy_history)[-60:-10] if len(self.energy_history) > 60 else list(self.energy_history)[:-10]
            if older:
                energy_trend = np.mean(recent) - np.mean(older)

        return {
            'speech_ratio_1h': speech_ratio,
            'energy_trend': float(energy_trend),
            'baseline_samples': len(self.composite_history),
            'mfcc_baseline_samples': len(self.mfcc_history),
        }


# ---------------------------------------------------------------------------
# Main Processor — Orchestrates tiers with progressive activation
# ---------------------------------------------------------------------------

class AudioProcessor:
    """
    Layered Audio Processing Pipeline.

    Processes audio in real-time across three tiers of increasing sophistication.
    Extracts features, stores vectors, NEVER stores raw audio.

    Usage:
        processor = AudioProcessor()
        processor.start()

        # In your main loop:
        features = processor.get_latest_features()
        consonance = processor.get_consonance_score()

        processor.stop()
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        window_seconds: float = 5.0,
        chunk_size: int = 4096,
        db_path: str = 'cognitive_observations.db',
        enable_tier2: bool = True,
        enable_tier3: bool = True,
    ):
        self.sample_rate = sample_rate
        self.window_seconds = window_seconds
        self.chunk_size = chunk_size
        self.db_path = db_path
        self.samples_per_window = int(sample_rate * window_seconds)

        # PyAudio
        self.pa = pyaudio.PyAudio()
        self.stream = None

        # Circular sample buffer (overwritten each window — never persisted)
        self._sample_ring = np.zeros(self.samples_per_window, dtype=np.float32)
        self._ring_pos = 0
        self._ring_filled = False  # True once we've filled one full window

        # Tier processors
        self.tier1 = Tier1Core(sample_rate)
        self.tier2 = Tier2Enhanced(sample_rate) if (enable_tier2 and LIBROSA_AVAILABLE) else None
        self.tier3 = Tier3Canary(sample_rate) if enable_tier3 else None

        # Edge filtering components (Lean Workflow Step 2)
        self.silero_vad = SileroVAD(sample_rate) if SILERO_AVAILABLE else None
        self.yamnet = YAMNetClassifier() if YAMNET_AVAILABLE else None
        self.whisper_stt = WhisperSTT() if WHISPER_AVAILABLE else None

        # Feature history (vectors only — this IS the memory, not raw audio)
        self.feature_history: deque[AudioFeatureVector] = deque(maxlen=3600)  # ~5h at 5s

        # Latest features (thread-safe access)
        self._latest: Optional[AudioFeatureVector] = None
        self._lock = threading.Lock()

        # Threading
        self._capture_thread: Optional[threading.Thread] = None
        self._process_thread: Optional[threading.Thread] = None
        self._running = False

        # CPU-adaptive tier activation
        self._tier2_active = self.tier2 is not None
        self._tier3_active = self.tier3 is not None
        self._last_process_time = 0.0  # track processing duration

        # Database
        self._init_database()

    def _init_database(self):
        """Add audio_features table to existing database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audio_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                tier INTEGER,
                consonance_score REAL,
                rms_energy REAL,
                vad_state INTEGER,
                vad_confidence REAL,
                audio_event TEXT,
                transcribed_text TEXT,
                anomaly_flag INTEGER,
                baseline_deviation REAL,
                feature_vector TEXT
            )
        ''')
        conn.commit()
        conn.close()

    def start(self):
        """Start audio capture and processing threads."""
        if self._running:
            return

        self._running = True

        # Open PyAudio stream
        self.stream = self.pa.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
        )

        # Capture thread: reads mic → ring buffer
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()

        # Process thread: ring buffer → features (every window_seconds)
        self._process_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._process_thread.start()

        tier_status = ["Tier1:Core"]
        if self.silero_vad and self.silero_vad.model:
            tier_status.append("SileroVAD")
        if self.tier2:
            tier_status.append("Tier2:Enhanced")
        if self.yamnet and self.yamnet.interpreter:
            tier_status.append("YAMNet")
        if self.whisper_stt and self.whisper_stt.model:
            tier_status.append(f"Whisper:{self.whisper_stt.model_size}")
        if self.tier3:
            tier_status.append("Tier3:Canary")
        print(f"[AudioProcessor] Started — {' | '.join(tier_status)}")

    def stop(self):
        """Stop audio processing and release resources."""
        self._running = False

        if self._capture_thread:
            self._capture_thread.join(timeout=2.0)
        if self._process_thread:
            self._process_thread.join(timeout=2.0)

        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

        self.pa.terminate()
        print("[AudioProcessor] Stopped")

    def _capture_loop(self):
        """Background thread: continuously read mic into ring buffer."""
        while self._running:
            try:
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                samples = np.frombuffer(data, dtype=np.float32)

                # Write to ring buffer (circular, overwrites old data)
                n = len(samples)
                end_pos = self._ring_pos + n

                if end_pos <= self.samples_per_window:
                    self._sample_ring[self._ring_pos:end_pos] = samples
                else:
                    # Wrap around
                    first_part = self.samples_per_window - self._ring_pos
                    self._sample_ring[self._ring_pos:] = samples[:first_part]
                    self._sample_ring[:n - first_part] = samples[first_part:]
                    self._ring_filled = True

                self._ring_pos = end_pos % self.samples_per_window

            except Exception as e:
                if self._running:
                    print(f"[AudioProcessor] Capture error: {e}")
                time.sleep(0.01)

    def _process_loop(self):
        """Background thread: process accumulated audio every window_seconds."""
        while self._running:
            time.sleep(self.window_seconds)

            if not self._ring_filled and self._ring_pos < self.samples_per_window * 0.5:
                continue  # not enough data yet

            # Snapshot the ring buffer (copy — ring continues filling)
            samples = self._sample_ring.copy()

            # Process through active tiers
            t_start = time.time()
            features = self._extract_features(samples)
            self._last_process_time = time.time() - t_start

            # Store and publish
            with self._lock:
                self._latest = features
            self.feature_history.append(features)

            # Persist to database (vector only, never raw audio)
            self._store_features(features)

            # Adaptive tier activation based on processing time budget
            budget = self.window_seconds * 0.5  # use at most 50% of window for processing
            if self._last_process_time > budget and self._tier2_active:
                print(f"[AudioProcessor] Processing slow ({self._last_process_time:.2f}s) — throttling Tier 2")
                self._tier2_active = False
            elif self._last_process_time < budget * 0.5 and self.tier2 and not self._tier2_active:
                print("[AudioProcessor] CPU headroom available — re-enabling Tier 2")
                self._tier2_active = True

    def _extract_features(self, samples: np.ndarray) -> AudioFeatureVector:
        """
        Run samples through active tiers + edge filters, return feature vector.

        Lean Workflow:
          1. Tier 1 Core (FFT features, always)
          2. Silero VAD (overrides heuristic VAD if available)
          3. Tier 2 Enhanced (librosa features, when CPU allows)
          4. YAMNet classification (edge filter)
          5. Whisper STT (speech → text, only when VAD=True)
          6. Tier 3 Canary (behavioral analysis)
        """
        now = time.time()
        fv = AudioFeatureVector(timestamp=now, tier=1)

        # --- Tier 1: Always (FFT spectral features) ---
        t1 = self.tier1.extract(samples)
        fv.consonance_score = t1['consonance_score']
        fv.spectral_centroid = t1['spectral_centroid']
        fv.spectral_bandwidth = t1['spectral_bandwidth']
        fv.spectral_flatness = t1['spectral_flatness']
        fv.spectral_rolloff = t1['spectral_rolloff']
        fv.rms_energy = t1['rms_energy']
        fv.zero_crossing_rate = t1['zero_crossing_rate']
        fv.vad_state = t1['vad_state']
        fv.peak_frequencies = t1['peak_frequencies']

        # --- Edge Filter: Silero VAD (overrides heuristic if available) ---
        if self.silero_vad and self.silero_vad.model is not None:
            try:
                is_speech, confidence = self.silero_vad.detect(samples)
                fv.vad_state = is_speech
                fv.vad_confidence = confidence
            except Exception as e:
                print(f"[AudioProcessor] Silero VAD error: {e}")
                # Falls back to heuristic VAD from Tier 1

        # --- Tier 2: Enhanced (when active, librosa features) ---
        t2 = None
        if self._tier2_active and self.tier2:
            try:
                t2 = self.tier2.extract(samples)
                fv.tier = 2
                fv.mfcc = t2['mfcc']
                fv.chroma = t2['chroma']
                fv.onset_rate = t2['onset_rate']
                fv.tempo_estimate = t2['tempo_estimate']
                fv.pitch_mean = t2['pitch_mean']
                fv.pitch_variance = t2['pitch_variance']
            except Exception as e:
                print(f"[AudioProcessor] Tier 2 error: {e}")

        # --- Edge Filter: YAMNet classification ---
        if self.yamnet and self.yamnet.interpreter is not None:
            try:
                event, confidence, top3 = self.yamnet.classify(samples, self.sample_rate)
                fv.audio_event = event
                fv.event_confidence = confidence
                fv.yamnet_top3 = top3

                # Safety event escalation: force Tier 3 if canary-relevant
                if self.yamnet.is_safety_event(event) and self.tier3:
                    self._tier3_active = True
            except Exception as e:
                print(f"[AudioProcessor] YAMNet error: {e}")

        # --- Edge Filter: Whisper STT (only when speech detected) ---
        if self.whisper_stt and self.whisper_stt.model is not None and fv.vad_state:
            try:
                fv.transcribed_text = self.whisper_stt.transcribe(samples, self.sample_rate)
            except Exception as e:
                print(f"[AudioProcessor] Whisper STT error: {e}")

        # --- Tier 3: Canary (behavioral analysis + anomaly detection) ---
        if self._tier3_active and self.tier3:
            try:
                t3 = self.tier3.extract(samples, t1, t2)
                fv.tier = 3
                # Don't override YAMNet classification if it ran
                if not fv.audio_event:
                    fv.audio_event = t3['audio_event']
                    fv.event_confidence = t3['event_confidence']
                fv.prosody_valence = t3['prosody_valence']
                fv.speech_rate = t3['speech_rate']
                fv.repetition_score = t3['repetition_score']
                fv.turn_taking_score = t3['turn_taking_score']
                fv.response_latency = t3['response_latency']
                fv.anomaly_flag = t3['anomaly_flag']
                fv.baseline_deviation = t3['baseline_deviation']
            except Exception as e:
                print(f"[AudioProcessor] Tier 3 error: {e}")

        return fv

    def _store_features(self, fv: AudioFeatureVector):
        """Persist feature vector to SQLite. Vector + text only — never raw audio."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                '''INSERT INTO audio_features
                   (timestamp, tier, consonance_score, rms_energy, vad_state,
                    vad_confidence, audio_event, transcribed_text,
                    anomaly_flag, baseline_deviation, feature_vector)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (
                    fv.timestamp, fv.tier, fv.consonance_score, fv.rms_energy,
                    int(fv.vad_state), fv.vad_confidence, fv.audio_event,
                    fv.transcribed_text, int(fv.anomaly_flag),
                    fv.baseline_deviation, json.dumps(fv.to_dict()),
                )
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[AudioProcessor] DB write error: {e}")

    # --- Public API (thread-safe) ---

    def get_latest_features(self) -> Optional[AudioFeatureVector]:
        """Get the most recent feature vector."""
        with self._lock:
            return self._latest

    def get_consonance_score(self) -> float:
        """Get the current consonance score (for Resonance Engine)."""
        with self._lock:
            return self._latest.consonance_score if self._latest else 0.0

    def get_canary_status(self) -> dict:
        """Get canary monitoring summary."""
        with self._lock:
            latest = self._latest

        status = {
            'anomaly_flag': False,
            'baseline_deviation': 0.0,
            'audio_event': '',
            'prosody_valence': 0.5,
            'repetition_score': 0.0,
            'tier_active': self._latest.tier if self._latest else 0,
            'processing_time_ms': self._last_process_time * 1000,
        }

        if latest:
            status['anomaly_flag'] = latest.anomaly_flag
            status['baseline_deviation'] = latest.baseline_deviation
            status['audio_event'] = latest.audio_event
            status['prosody_valence'] = latest.prosody_valence
            status['repetition_score'] = latest.repetition_score

        if self.tier3:
            status['canary_summary'] = self.tier3.get_canary_summary()

        return status

    def get_feature_history(self, last_n: int = 60) -> List[dict]:
        """Get recent feature history as list of dicts (for MCP/API)."""
        history = list(self.feature_history)[-last_n:]
        return [fv.to_dict() for fv in history]

    def __del__(self):
        if self._running:
            self.stop()


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("Audio Processor — Standalone Test (Lean Workflow)")
    print("=" * 60)
    print(f"Silero VAD:        {'Available' if SILERO_AVAILABLE else 'Not installed (pip install torch)'}")
    print(f"Tier 2 (librosa):  {'Available' if LIBROSA_AVAILABLE else 'Not installed (pip install librosa)'}")
    print(f"YAMNet:            {'Available' if YAMNET_AVAILABLE else 'Not installed (need yamnet.tflite)'}")
    print(f"Whisper STT:       {'Available' if WHISPER_AVAILABLE else 'Not installed (pip install faster-whisper)'}")
    print()

    processor = AudioProcessor(
        sample_rate=16000,
        window_seconds=5.0,
    )

    try:
        processor.start()
        print("\nListening... (Ctrl+C to stop)\n")

        while True:
            time.sleep(5.5)
            features = processor.get_latest_features()
            if features:
                vad_icon = f"[{features.vad_confidence:.0%}]" if features.vad_confidence > 0 else ""
                print(f"[t={features.timestamp:.0f}] Tier {features.tier}")
                print(f"  Consonance: {features.consonance_score:.3f} | "
                      f"Energy: {features.rms_energy:.4f} | "
                      f"VAD: {'SPEECH' if features.vad_state else 'silent'} {vad_icon}")
                print(f"  Centroid: {features.spectral_centroid:.0f} Hz | "
                      f"Flatness: {features.spectral_flatness:.3f} | "
                      f"Event: {features.audio_event}")

                if features.transcribed_text:
                    print(f"  STT: \"{features.transcribed_text}\"")

                if features.yamnet_top3:
                    print(f"  YAMNet: {' > '.join(features.yamnet_top3)}")

                if features.tier >= 2:
                    print(f"  Pitch: {features.pitch_mean:.0f} Hz | "
                          f"Tempo: {features.tempo_estimate:.0f} BPM | "
                          f"Onsets: {features.onset_rate:.1f}/s")

                if features.tier >= 3:
                    flag = "!! ANOMALY" if features.anomaly_flag else "OK"
                    print(f"  Canary: {flag} | "
                          f"Deviation: {features.baseline_deviation:.2f}s | "
                          f"Prosody: {features.prosody_valence:.2f} | "
                          f"Repetition: {features.repetition_score:.2f}")

                print()

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        processor.stop()
