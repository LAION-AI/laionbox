"""ASR-based audio trimmer using NemoTron 3.5.

After TTS generates audio for a segment, this module runs ASR to get
word-level timestamps, identifies which words correspond to the actual
dialogue (text inside double quotes), and trims away any instruction /
narration words that the model incorrectly vocalised.
"""
from __future__ import annotations

import importlib
import logging
import re
import sys
import tempfile
import threading
import time
import types
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

log = logging.getLogger("asr-trimmer")

# ---------------------------------------------------------------------------
# Mock onnx so NeMo can import (protobuf 3.19 vs onnx 1.20 conflict)
# ---------------------------------------------------------------------------

def _mock_onnx():
    """Inject a lightweight mock onnx module tree into sys.modules.

    NeMo 2.7's import chain pulls in ``nemo.core.classes.exportable`` which
    imports ``onnx``.  The installed onnx 1.20.1 requires protobuf >= 3.20
    but the env has 3.19.6.  Since we never export to ONNX at runtime, a
    thin mock is sufficient.
    """
    if "onnx" in sys.modules and hasattr(sys.modules["onnx"], "TensorProto"):
        return  # real onnx already loaded fine

    class _Mock(types.ModuleType):
        def __init__(self, name):
            super().__init__(name)
            self.__spec__ = importlib.machinery.ModuleSpec(name, None)
            self.__path__ = []
            self.__file__ = "/dev/null"

        def __getattr__(self, name):
            if name.startswith("_"):
                raise AttributeError(name)
            child = _Mock(f"{self.__name__}.{name}")
            setattr(self, name, child)
            return child

    _names = [
        "onnx", "onnx.external_data_helper", "onnx.onnx_pb",
        "onnx.onnx_ml_pb2", "onnx.shape_inference", "onnx.helper",
        "onnx.numpy_helper", "onnx.TensorProto", "onnx.mapping",
        "onnx.checker", "onnx.compose", "onnx.version_converter",
    ]
    for n in _names:
        sys.modules[n] = _Mock(n)


# Run the mock before any NeMo import
_mock_onnx()
from nemo.collections.asr.models.hybrid_rnnt_ctc_bpe_models_prompt import (  # noqa: E402
    EncDecHybridRNNTCTCBPEModelWithPrompt,
    HybridRNNTCTCPromptTranscribeConfig,
)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)


def _normalise(word: str) -> str:
    """Lowercase, strip punctuation / asterisks."""
    return _PUNCT_RE.sub("", word).strip().lower()


def extract_dialogue_words(segment_text: str) -> List[str]:
    """Return the normalised words from text inside double quotes.

    If no double-quoted region is found, returns an empty list.
    """
    parts = re.findall(r'"([^"]*)"', segment_text)
    if not parts:
        return []
    raw = " ".join(parts)
    return [w for w in (_normalise(t) for t in raw.split()) if w]


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate between reference and hypothesis text.

    Both strings are normalised (lowercased, punctuation stripped) before
    comparison.  Returns a float in [0, ∞) — 0.0 means perfect match.
    """
    ref_words = [w for w in (_normalise(t) for t in reference.split()) if w]
    hyp_words = [w for w in (_normalise(t) for t in hypothesis.split()) if w]

    if not ref_words:
        return 0.0 if not hyp_words else float(len(hyp_words))

    # Levenshtein distance on word sequences
    n, m = len(ref_words), len(hyp_words)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            ins = dp[j] + 1
            dele = dp[j - 1] + 1
            sub = prev + cost
            prev = dp[j]
            dp[j] = min(ins, dele, sub)
    return dp[m] / n


def _fuzzy_eq(a: str, b: str) -> bool:
    """Check if two normalised words are similar enough."""
    if a == b:
        return True
    # allow one char difference for short words
    if len(a) <= 2 or len(b) <= 2:
        return a == b
    # prefix match for cases like "Hatschi" vs "hatschi!" etc.
    shorter = min(len(a), len(b))
    if shorter >= 3 and a[:shorter] == b[:shorter]:
        return True
    # Levenshtein distance 1
    if abs(len(a) - len(b)) > 1:
        return False
    diffs = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            diffs += 1
    diffs += abs(len(a) - len(b))
    return diffs <= 1


# ---------------------------------------------------------------------------
# Word alignment
# ---------------------------------------------------------------------------

def find_dialogue_span(
    asr_words: List[Dict],
    dialogue_words: List[str],
) -> Optional[Tuple[float, float]]:
    """Find the time span of *dialogue_words* inside *asr_words*.

    Returns ``(start_sec, end_sec)`` or ``None`` if alignment fails.
    """
    if not dialogue_words or not asr_words:
        return None

    asr_norm = [_normalise(w["word"]) for w in asr_words]

    # --- find start: first ASR word matching first dialogue word ----------
    first_target = dialogue_words[0]
    start_idx = None
    for i, w in enumerate(asr_norm):
        if _fuzzy_eq(w, first_target):
            start_idx = i
            break

    # --- find end: last ASR word matching last dialogue word --------------
    last_target = dialogue_words[-1]
    end_idx = None
    for i in range(len(asr_norm) - 1, -1, -1):
        if _fuzzy_eq(asr_norm[i], last_target):
            end_idx = i
            break

    if start_idx is None or end_idx is None or end_idx < start_idx:
        return None

    # --- validate: enough dialogue words appear in the span ----
    span_words = set(asr_norm[start_idx : end_idx + 1])
    matches = sum(1 for dw in dialogue_words if any(_fuzzy_eq(dw, sw) for sw in span_words))
    # Relax threshold for long dialogues (>20 words) where ASR is more likely
    # to miss some words
    threshold = 0.3 if len(dialogue_words) > 20 else 0.5
    if matches < len(dialogue_words) * threshold:
        log.info(f"Alignment validation failed: {matches}/{len(dialogue_words)} words matched "
                 f"(threshold={threshold:.0%})")
        return None

    start_sec = asr_words[start_idx].get("start", 0.0)
    end_sec = asr_words[end_idx].get("end", asr_words[end_idx].get("start", 0.0))
    return (start_sec, end_sec)


# ---------------------------------------------------------------------------
# ASRTrimmer
# ---------------------------------------------------------------------------

class ASRTrimmer:
    """Wraps NemoTron 3.5 ASR for post-generation audio trimming and VoiceCLAP for scoring."""

    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self._lock = threading.Lock()

        log.info(f"Loading NemoTron 3.5 ASR on {device} ...")
        t0 = time.time()
        self.model = EncDecHybridRNNTCTCBPEModelWithPrompt.from_pretrained(
            model_name="nvidia/nemotron-3.5-asr-streaming-0.6b",
            strict=False,
        )
        self.model = self.model.to(device)
        self.model.eval()
        self._sample_rate = self.model.preprocessor._sample_rate  # typically 16000
        self._patch_manifest_processing()
        log.info(f"NemoTron ASR ready on {device} in {time.time()-t0:.1f}s "
                 f"(sr={self._sample_rate})")

        # Load VoiceCLAP for audio quality scoring
        self._clap = None
        self._clap_tok = None
        self._clap_text_cache: Dict[str, torch.Tensor] = {}
        try:
            from transformers import AutoModel, AutoTokenizer
            log.info(f"Loading VoiceCLAP-small on {device} ...")
            t0 = time.time()
            self._clap = AutoModel.from_pretrained(
                "laion/voiceclap-small", trust_remote_code=True
            ).eval().to(device)
            self._clap_tok = AutoTokenizer.from_pretrained("laion/voiceclap-small")
            log.info(f"VoiceCLAP-small ready on {device} in {time.time()-t0:.1f}s")
        except Exception as e:
            log.warning(f"VoiceCLAP loading failed (ranking disabled): {e}")

    # ------------------------------------------------------------------
    def clap_text_embedding(self, text: str) -> Optional[torch.Tensor]:
        """Get L2-normalised CLAP text embedding (cached)."""
        if self._clap is None:
            return None
        if text in self._clap_text_cache:
            return self._clap_text_cache[text]
        enc = self._clap_tok([text], padding=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            emb = self._clap.encode_text(enc.input_ids, enc.attention_mask)
        self._clap_text_cache[text] = emb
        return emb

    def clap_audio_score(self, waveform: torch.Tensor, sr: int,
                         text_emb: torch.Tensor) -> float:
        """Compute CLAP cosine similarity between audio and a pre-encoded text."""
        if self._clap is None:
            return 0.0
        import torchaudio as ta
        wav = waveform.cpu().float()
        if wav.ndim == 3:
            wav = wav.squeeze(0)
        if wav.ndim == 2:
            wav = wav.mean(0)  # mono
        if sr != 16000:
            wav = ta.functional.resample(wav, sr, 16000)
        with torch.no_grad():
            audio_emb = self._clap.encode_waveform(wav.to(self.device))
            sim = (audio_emb @ text_emb.T).item()
        return round(sim, 4)

    # ------------------------------------------------------------------
    def _patch_manifest_processing(self):
        """Monkey-patch _transcribe_input_manifest_processing to include language.

        NeMo's default creates manifest entries without a 'lang' field, so
        lhotse sets supervision.language=None → "Unknown prompt key: 'None'".
        This patch adds 'lang' to every manifest entry so the dataloader can
        create proper prompt tensors.
        """
        import json
        import os

        original = self.model._transcribe_input_manifest_processing.__func__

        def _patched(model_self, audio_files, temp_dir, trcfg):
            # Write manifest with language field
            manifest_path = os.path.join(temp_dir, 'manifest.json')
            with open(manifest_path, 'w', encoding='utf-8') as fp:
                for audio_file in audio_files:
                    if isinstance(audio_file, str):
                        entry = {
                            'audio_filepath': audio_file,
                            'duration': 100000,
                            'text': '',
                            'lang': getattr(trcfg, 'target_lang', 'en-US'),
                        }
                        fp.write(json.dumps(entry) + '\n')
                    elif isinstance(audio_file, dict):
                        if 'lang' not in audio_file:
                            audio_file['lang'] = getattr(trcfg, 'target_lang', 'en-US')
                        fp.write(json.dumps(audio_file) + '\n')

            # Call original for the ds_config part but we already wrote manifest
            from nemo.collections.asr.parts.mixins.transcription import (
                get_value_from_transcription_config,
            )
            ds_config = {
                'use_lhotse': get_value_from_transcription_config(trcfg, 'use_lhotse', True),
                'paths2audio_files': audio_files,
                'batch_size': get_value_from_transcription_config(trcfg, 'batch_size', 4),
                'temp_dir': temp_dir,
                'num_workers': get_value_from_transcription_config(trcfg, 'num_workers', 0),
                'channel_selector': get_value_from_transcription_config(trcfg, 'channel_selector', None),
                'text_field': get_value_from_transcription_config(trcfg, 'text_field', 'text'),
                'lang_field': get_value_from_transcription_config(trcfg, 'lang_field', 'lang'),
            }
            augmentor = get_value_from_transcription_config(trcfg, 'augmentor', None)
            if augmentor:
                ds_config['augmentor'] = augmentor
            return ds_config

        import types
        self.model._transcribe_input_manifest_processing = types.MethodType(
            _patched, self.model
        )

    # ------------------------------------------------------------------
    def _transcribe(self, waveform_np: np.ndarray, sr: int) -> Tuple[str, List[Dict]]:
        """Run ASR and return (full_text, word_timestamps).

        *waveform_np*: 1-D float32 numpy array at *sr* Hz.
        """
        import os
        import torchaudio

        # NeMo expects audio at self._sample_rate; resample on CPU if needed
        if sr != self._sample_rate:
            wav_t = torch.from_numpy(waveform_np).unsqueeze(0).float()
            wav_t = torchaudio.functional.resample(wav_t, sr, self._sample_rate)
            waveform_np = wav_t.squeeze(0).numpy()

        # Save to temp WAV — NeMo's file-path transcription creates a lhotse
        # dataloader that builds prompt tensors with correct encoder-aligned
        # dimensions.  Our patched _transcribe_input_manifest_processing adds
        # 'lang' to the manifest so supervision.language != None.
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        sf.write(tmp_path, waveform_np, self._sample_rate)

        trcfg = HybridRNNTCTCPromptTranscribeConfig(
            batch_size=1,
            return_hypotheses=True,
            verbose=False,
            timestamps=True,
            target_lang="auto",
        )

        with self._lock:
            torch.cuda.set_device(self.device)
            outputs = self.model.transcribe(
                [tmp_path],
                override_config=trcfg,
            )

        os.unlink(tmp_path)

        # NeMo returns (best_hyps, beam_hyps) or just best_hyps
        hyps = outputs[0] if isinstance(outputs, tuple) else outputs
        hyp = hyps[0] if isinstance(hyps, list) else hyps

        text = hyp.text if hasattr(hyp, "text") else str(hyp)

        # Build word-level timestamps from raw RNNT token timestamps.
        # NeMo's process_timestamp_outputs is broken for RNNT (expects dict,
        # gets tensor), so we reconstruct word boundaries from token data.
        word_ts = []
        if (hasattr(hyp, "timestamp") and isinstance(hyp.timestamp, torch.Tensor)
                and hasattr(hyp, "y_sequence") and len(hyp.timestamp) > 0):
            frames = hyp.timestamp.tolist()
            token_ids = hyp.y_sequence.tolist()
            sub = self.model.encoder.subsampling_factor
            ws = self.model.cfg["preprocessor"]["window_stride"]
            tok = self.model.tokenizer

            current_word = ""
            current_start = None
            current_end = None
            for tid, frame in zip(token_ids, frames):
                piece = tok.ids_to_tokens([tid])
                if isinstance(piece, list):
                    piece = piece[0] if piece else ""
                t = frame * sub * ws
                if piece.startswith("\u2581") or piece.startswith(" "):
                    if current_word:
                        word_ts.append({"word": current_word,
                                        "start": round(current_start, 3),
                                        "end": round(current_end, 3)})
                    current_word = piece.lstrip("\u2581 ")
                    current_start = t
                    current_end = t
                else:
                    current_word += piece
                    if current_start is None:
                        current_start = t
                    current_end = t
            if current_word:
                word_ts.append({"word": current_word,
                                "start": round(current_start, 3),
                                "end": round(current_end, 3)})
        elif hasattr(hyp, "timestep") and isinstance(hyp.timestep, dict):
            word_ts = hyp.timestep.get("word", [])

        return text, word_ts

    # ------------------------------------------------------------------
    def trim_segment(
        self,
        waveform: torch.Tensor,
        sr: int,
        segment_text: str,
        margin_ms: float = 30.0,
    ) -> Tuple[torch.Tensor, Dict]:
        """Analyse *waveform* with ASR and trim to dialogue span.

        Returns ``(trimmed_or_original_waveform, info_dict)``.
        """
        info: Dict = {
            "trimmed": False,
            "asr_text": "",
            "dialogue_words": [],
            "asr_ms": 0,
            "trim_start_s": 0.0,
            "trim_end_s": 0.0,
        }

        # 1. Extract expected dialogue words
        dialogue_words = extract_dialogue_words(segment_text)
        info["dialogue_words"] = dialogue_words
        if not dialogue_words:
            return waveform, info  # no quotes → nothing to trim

        # 1b. Skip ASR trimming for long audio (streaming ASR unreliable >30s)
        audio_dur = waveform.shape[-1] / sr
        if audio_dur > 30.0:
            log.info(f"Skipping ASR trim: audio too long ({audio_dur:.1f}s > 30s)")
            return waveform, info

        # 2. Convert waveform to numpy
        wav_np = waveform.cpu().float().numpy()
        if wav_np.ndim == 3:
            wav_np = wav_np.squeeze(0)
        if wav_np.ndim == 2:
            wav_np = wav_np[0]  # take first channel

        # 3. Run ASR
        t0 = time.time()
        try:
            asr_text, word_ts = self._transcribe(wav_np, sr)
        except Exception as e:
            log.warning(f"ASR failed: {e}")
            return waveform, info
        info["asr_ms"] = round((time.time() - t0) * 1000)
        info["asr_text"] = asr_text

        if not word_ts:
            log.warning("ASR returned no word timestamps")
            return waveform, info

        # 4. Check if trimming is needed
        asr_norm = [_normalise(w["word"]) for w in word_ts]
        # If ASR output already matches dialogue words closely, skip
        if len(asr_norm) > 0 and len(dialogue_words) > 0:
            # Check if there are extra words before or after
            span = find_dialogue_span(word_ts, dialogue_words)
            if span is None:
                log.info(f"ASR alignment failed, skipping trim. ASR: {asr_text}")
                return waveform, info

            start_sec, end_sec = span

            # Find indices of dialogue start/end in asr_words
            first_asr_start = word_ts[0].get("start", 0.0)
            last_asr_end = word_ts[-1].get("end", word_ts[-1].get("start", 0.0))

            # Only trim if there are clearly extra words outside the dialogue
            has_prefix_words = start_sec - first_asr_start > 0.15
            has_suffix_words = last_asr_end - end_sec > 0.15

            if not has_prefix_words and not has_suffix_words:
                log.info(f"No trimming needed — ASR matches dialogue. ASR: {asr_text}")
                return waveform, info

            # 5. Trim
            margin_s = margin_ms / 1000.0
            total_dur = wav_np.shape[-1] / sr
            trim_start = max(0.0, start_sec - margin_s)
            trim_end = min(total_dur, end_sec + margin_s)

            start_sample = int(trim_start * sr)
            end_sample = int(trim_end * sr)

            if end_sample <= start_sample:
                log.warning("Trim range invalid, skipping")
                return waveform, info

            trimmed = waveform[..., start_sample:end_sample]
            info["trimmed"] = True
            info["trim_start_s"] = round(trim_start, 3)
            info["trim_end_s"] = round(trim_end, 3)

            log.info(
                f"Trimmed: {trim_start:.2f}s–{trim_end:.2f}s "
                f"(removed {trim_start:.2f}s prefix, "
                f"{total_dur - trim_end:.2f}s suffix). "
                f"ASR: {asr_text}"
            )
            return trimmed, info

        return waveform, info
