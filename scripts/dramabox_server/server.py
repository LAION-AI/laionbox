#!/usr/bin/env python3
"""FastAPI DramaBox TTS inference server with multi-GPU parallel segment generation.

Loads 4 TTSServer instances (one per GPU), merges the best LoRA checkpoint,
and dispatches dialogue segments in parallel for pseudo-streaming playback.
"""
import asyncio
import base64
import faulthandler
import io
import json
import logging
import os
import sys
import tempfile
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

# Dump tracebacks on segfault / SIGABRT to stderr
faulthandler.enable()

import torch
# Disable cuDNN — runtime version (9.1.0) is incompatible with PyTorch's
# compiled version (9.8.0), causing SIGABRT on first cuDNN op.
torch.backends.cudnn.enabled = False
import torchaudio

# ---------------------------------------------------------------------------
# Path setup — import DramaBox internals
# ---------------------------------------------------------------------------
_BASE = Path(__file__).resolve().parent.parent          # Voice-Acting-Pipeline
_DRAMABOX = _BASE.parent / "DramaBox"
sys.path.insert(0, str(_DRAMABOX / "src"))
sys.path.insert(0, str(_DRAMABOX / "ltx2"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # prompt_parser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
log = logging.getLogger("dramabox-server")

# ---------------------------------------------------------------------------
# DramaBox imports (after path setup)
# ---------------------------------------------------------------------------
from inference_server import TTSServer, auto_rescale_for_cfg, DEFAULT_NEG, _equal_power_crossfade
from audio_conditioning import AudioConditionByReferenceLatent
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.patchifiers import AudioPatchifier
from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.model.transformer.model import X0Model
from ltx_core.tools import AudioLatentTools
from ltx_core.types import Audio, AudioLatentShape, VideoPixelShape
from ltx_core.model.audio_vae import encode_audio as vae_encode_audio
from ltx_pipelines.utils.media_io import decode_audio_from_file
from ltx_pipelines.utils.denoisers import GuidedDenoiser
from ltx_pipelines.utils.samplers import euler_denoising_loop
from duration_estimator import estimate_speech_duration
from prompt_parser import parse_prompt, Segment
from asr_trimmer import ASRTrimmer, extract_dialogue_words, compute_wer

# ---------------------------------------------------------------------------
# FastAPI imports
# ---------------------------------------------------------------------------
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from starlette.staticfiles import StaticFiles

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
NUM_GPUS = int(os.environ.get("NUM_GPUS", "4"))
GPU_OFFSET = int(os.environ.get("GPU_OFFSET", "4"))     # Start from cuda:4
LORA_PATH = os.environ.get(
    "LORA_PATH",
    str(_BASE / "finetune_output" / "combined_12ep" / "lora_epoch12.safetensors"),
)
LORA_RANK = int(os.environ.get("LORA_RANK", "128"))
PORT = int(os.environ.get("PORT", "8766"))
COMPILE_MODEL = os.environ.get("COMPILE_MODEL", "0") == "1"

# ---------------------------------------------------------------------------
# LoRA merger
# ---------------------------------------------------------------------------
def merge_lora(velocity_model, lora_path: str, rank: int = 128):
    """Apply LoRA weights to the velocity model and merge them in."""
    from peft import LoraConfig, get_peft_model
    from safetensors.torch import load_file as st_load

    log.info(f"Loading LoRA: {lora_path} (rank={rank})")
    lora_sd = st_load(lora_path)

    lora_config = LoraConfig(
        r=rank, lora_alpha=rank, lora_dropout=0.0, bias="none",
        target_modules=[
            "audio_attn1.to_k", "audio_attn1.to_q",
            "audio_attn1.to_v", "audio_attn1.to_out.0",
            "audio_ff.net.0.proj", "audio_ff.net.2",
        ],
    )
    model = get_peft_model(velocity_model, lora_config)

    # Remap PEFT key format
    mapped_sd = {}
    for k, v in lora_sd.items():
        new_key = k
        if ".lora_A.weight" in k and ".lora_A.default.weight" not in k:
            new_key = k.replace(".lora_A.weight", ".lora_A.default.weight")
        if ".lora_B.weight" in k and ".lora_B.default.weight" not in k:
            new_key = k.replace(".lora_B.weight", ".lora_B.default.weight")
        mapped_sd[new_key] = v

    missing, unexpected = model.load_state_dict(mapped_sd, strict=False)
    loaded = len(mapped_sd) - len(unexpected)
    log.info(f"Loaded {loaded} LoRA weights (missing={len(missing)}, unexpected={len(unexpected)})")

    model = model.merge_and_unload()
    log.info("Merged LoRA into model")
    return model


# ---------------------------------------------------------------------------
# Timed generation — same logic as TTSServer.generate() but returns timings
# ---------------------------------------------------------------------------
@torch.inference_mode()
def generate_timed(server: TTSServer, prompt: str, voice_ref: Optional[str] = None,
                   cfg_scale: float = 2.5, stg_scale: float = 1.5,
                   duration_multiplier: float = 1.1, seed: int = 42,
                   ref_duration: float = 10.0, gen_duration: float = 0.0,
                   denoise_ref: bool = False,
                   ref_latent_cpu: Optional[torch.Tensor] = None):
    """Generate audio with per-stage timing breakdown.

    If ref_latent_cpu is provided, uses that pre-encoded reference latent
    instead of re-encoding from voice_ref (avoids thread-unsafe torchaudio resample).

    Returns (waveform, sample_rate, timings_dict).
    """
    # Pin this thread to the server's CUDA device
    torch.cuda.set_device(server.device)

    timings = {}
    t_total = time.time()

    # Duration + target shape
    if gen_duration and gen_duration > 0:
        gen_dur = float(gen_duration)
    else:
        base = estimate_speech_duration(prompt)
        gen_dur = max(3.0, round(base * duration_multiplier, 1))
    fps = 25.0
    n_frames = int(round(gen_dur * fps)) + 1
    n_frames = ((n_frames - 1 + 4) // 8) * 8 + 1
    pixel_shape = VideoPixelShape(batch=1, frames=n_frames, height=64, width=64, fps=fps)
    target_shape = AudioLatentShape.from_video_pixel_shape(pixel_shape)
    audio_tools = AudioLatentTools(patchifier=server.patchifier, target_shape=target_shape)

    # Initial state
    state = audio_tools.create_initial_state(device=server.device, dtype=server.dtype)

    # Voice ref conditioning — use pre-encoded latent if available
    t0 = time.time()
    if ref_latent_cpu is not None:
        ref_latent = ref_latent_cpu.to(server.device, server.dtype)

        cond = AudioConditionByReferenceLatent(latent=ref_latent, strength=1.0)
        state = cond.apply_to(state, audio_tools)
    timings["ref_encode_ms"] = (time.time() - t0) * 1000

    # Noise
    gen = torch.Generator(device=server.device).manual_seed(seed)
    noiser = GaussianNoiser(generator=gen)
    state = noiser(state, noise_scale=1.0)

    # Prompt encode
    t0 = time.time()
    prompts = [prompt, DEFAULT_NEG] if cfg_scale > 1.0 else [prompt]
    ctx = server._prompt_encoder(prompts, streaming_prefetch_count=None)
    a_ctx = ctx[0].audio_encoding
    a_ctx_neg = ctx[1].audio_encoding if cfg_scale > 1.0 else None
    timings["text_encode_ms"] = (time.time() - t0) * 1000

    # Denoiser
    resc = auto_rescale_for_cfg(cfg_scale)
    guider = MultiModalGuider(
        params=MultiModalGuiderParams(
            cfg_scale=cfg_scale, stg_scale=stg_scale,
            stg_blocks=[29], rescale_scale=resc, modality_scale=1.0,
        ),
        negative_context=a_ctx_neg,
    )
    denoiser = GuidedDenoiser(
        v_context=None, a_context=a_ctx,
        video_guider=None, audio_guider=guider,
    )

    # Sigmas
    sigmas = LTX2Scheduler().execute(steps=30, latent=state.latent).to(server.device)

    # Denoise
    t0 = time.time()
    x0 = X0Model(server._velocity_model)
    _, audio_state = euler_denoising_loop(
        sigmas=sigmas, video_state=None, audio_state=state,
        stepper=EulerDiffusionStep(), transformer=x0, denoiser=denoiser,
    )
    timings["denoise_ms"] = (time.time() - t0) * 1000

    # Strip + unpatchify
    audio_state = audio_tools.clear_conditioning(audio_state)
    audio_state = audio_tools.unpatchify(audio_state)

    # End-of-clip silence-prior fix
    latent = audio_state.latent
    if latent.shape[2] > 513:
        f0, f1 = 511, 514
        patched = latent.clone()
        for f in (512, 513):
            t = (f - f0) / (f1 - f0)
            patched[:, :, f, :] = (1.0 - t) * latent[:, :, f0, :] + t * latent[:, :, f1, :]
        latent = patched

    # Decode
    t0 = time.time()
    decoded = server._audio_decoder(latent)
    out_waveform, out_sr = decoded.waveform, decoded.sampling_rate
    timings["decode_ms"] = (time.time() - t0) * 1000

    timings["total_ms"] = (time.time() - t_total) * 1000
    timings["audio_duration_s"] = round(out_waveform.shape[-1] / out_sr, 2)
    timings["gen_duration_target_s"] = round(gen_dur, 1)

    return out_waveform, out_sr, timings


# ---------------------------------------------------------------------------
# WAV encoding helper
# ---------------------------------------------------------------------------
def waveform_to_wav_base64(waveform: torch.Tensor, sr: int) -> str:
    """Encode waveform tensor to base64 WAV string."""
    wav_cpu = waveform.cpu().float()
    if wav_cpu.dim() == 3:
        wav_cpu = wav_cpu.squeeze(0)
    if wav_cpu.dim() == 1:
        wav_cpu = wav_cpu.unsqueeze(0)
    buf = io.BytesIO()
    torchaudio.save(buf, wav_cpu, sr, format="wav")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# GPU worker pool
# ---------------------------------------------------------------------------
class GPUPool:
    """Manages a pool of TTSServer instances across multiple GPUs."""

    def __init__(self, num_gpus: int, gpu_offset: int, lora_path: str, lora_rank: int,
                 compile_model: bool = False):
        self.servers: List[TTSServer] = []
        self.gpu_ids: List[int] = list(range(gpu_offset, gpu_offset + num_gpus))
        self._queue: asyncio.Queue = None  # Set up in async context
        self._executor = ThreadPoolExecutor(max_workers=num_gpus)
        self._locks: List[threading.Lock] = [threading.Lock() for _ in range(num_gpus)]
        self.compile_model = compile_model

        for gpu_id in self.gpu_ids:
            device = f"cuda:{gpu_id}"
            log.info(f"Loading TTSServer on {device}...")
            t0 = time.time()
            srv = TTSServer(device=device, dtype="bf16",
                            compile_model=compile_model, bnb_4bit=True)
            # Merge LoRA
            if lora_path and os.path.exists(lora_path):
                srv._velocity_model = merge_lora(srv._velocity_model, lora_path, lora_rank)
                if compile_model:
                    log.info(f"  Re-compiling merged model on {device}...")
                    srv._velocity_model = torch.compile(srv._velocity_model, mode="default", dynamic=True)
            log.info(f"  GPU {gpu_id} ready in {time.time()-t0:.1f}s")
            self.servers.append(srv)

        log.info(f"All {num_gpus} GPUs loaded and ready")

    async def init_queue(self):
        self._queue = asyncio.Queue()
        for i in range(len(self.servers)):
            await self._queue.put(i)

    async def acquire(self) -> int:
        """Get an available server index."""
        return await self._queue.get()

    async def release(self, idx: int):
        """Return a server to the pool."""
        await self._queue.put(idx)

    def get_server(self, idx: int) -> TTSServer:
        return self.servers[idx]


# ---------------------------------------------------------------------------
# Job storage for SSE streaming
# ---------------------------------------------------------------------------
class Job:
    def __init__(self, job_id: str, num_segments: int):
        self.job_id = job_id
        self.num_segments = num_segments
        self.segments: Dict[int, dict] = {}
        self.events: asyncio.Queue = asyncio.Queue()
        self.start_time = time.time()
        self.voice_ref_path: Optional[str] = None

    def segment_done(self, idx: int, data: dict):
        self.segments[idx] = data
        # Push event non-blocking
        try:
            self.events.put_nowait(data)
        except asyncio.QueueFull:
            pass

    def is_complete(self):
        return len(self.segments) == self.num_segments


jobs: Dict[str, Job] = {}

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(title="DramaBox TTS Server")
pool: Optional[GPUPool] = None
asr_trimmer: Optional[ASRTrimmer] = None


@app.on_event("startup")
async def startup():
    global pool, asr_trimmer
    log.info(f"Starting DramaBox server: {NUM_GPUS} GPUs (cuda:{GPU_OFFSET}..cuda:{GPU_OFFSET+NUM_GPUS-1})")
    log.info(f"LoRA: {LORA_PATH}")
    pool = GPUPool(NUM_GPUS, GPU_OFFSET, LORA_PATH, LORA_RANK, COMPILE_MODEL)
    await pool.init_queue()
    # Load ASR trimmer on the first inference GPU
    try:
        asr_trimmer = ASRTrimmer(device=f"cuda:{GPU_OFFSET}")
    except Exception as e:
        log.error(f"Failed to load ASR trimmer: {e}\n{traceback.format_exc()}")
        asr_trimmer = None
    log.info("Server ready")


@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(content=html_path.read_text())


@app.get("/status")
async def status():
    info = {
        "status": "ready",
        "num_gpus": NUM_GPUS,
        "gpu_ids": pool.gpu_ids if pool else [],
        "lora": LORA_PATH,
        "active_jobs": len([j for j in jobs.values() if not j.is_complete()]),
    }
    return JSONResponse(info)


@app.post("/generate")
async def generate(
    prompt: str = Form(...),
    voice_ref: Optional[UploadFile] = File(None),
    cfg_scale: float = Form(2.5),
    stg_scale: float = Form(1.5),
    duration_multiplier: float = Form(0.9),
    ref_duration: float = Form(10.0),
    seed: int = Form(42),
    streaming_mode: bool = Form(True),
    asr_trim: bool = Form(True),
    prefix: str = Form(""),
    duration_mode: str = Form("algorithmic"),
    fixed_duration: float = Form(5.0),
    num_candidates: int = Form(1),
):
    """Start a generation job. Returns a job_id for SSE streaming."""
    # Save uploaded voice ref to temp file
    voice_ref_path = None
    if voice_ref and voice_ref.filename:
        suffix = Path(voice_ref.filename).suffix or ".wav"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(await voice_ref.read())
        tmp.close()
        voice_ref_path = tmp.name

    job_id = str(uuid.uuid4())[:8]

    if streaming_mode:
        segments = parse_prompt(prompt, duration_multiplier)
    else:
        dur = estimate_speech_duration(prompt) * duration_multiplier
        segments = [Segment(index=0, text=prompt, estimated_duration=dur)]

    job = Job(job_id, len(segments))
    job.voice_ref_path = voice_ref_path
    jobs[job_id] = job

    log.info(f"Job {job_id}: {len(segments)} segments, streaming={streaming_mode}")
    for seg in segments:
        log.info(f"  Seg {seg.index}: {seg.estimated_duration:.1f}s — {seg.text[:80]}...")

    # Launch generation in background
    asyncio.create_task(_run_job(job, segments, voice_ref_path,
                                 cfg_scale, stg_scale, duration_multiplier,
                                 ref_duration, seed, asr_trim,
                                 prefix, duration_mode, fixed_duration,
                                 num_candidates))

    return JSONResponse({
        "job_id": job_id,
        "num_segments": len(segments),
        "segments": [
            {"index": s.index, "text": s.text, "estimated_duration": s.estimated_duration}
            for s in segments
        ],
    })


async def _run_job(job: Job, segments: List[Segment], voice_ref_path: Optional[str],
                   cfg_scale: float, stg_scale: float, duration_multiplier: float,
                   ref_duration: float, seed: int, asr_trim: bool = True,
                   prefix: str = "", duration_mode: str = "algorithmic",
                   fixed_duration: float = 5.0, num_candidates: int = 1):
    """Generate all segments in parallel across available GPUs."""
    loop = asyncio.get_event_loop()

    # Pre-encode the voice reference ONCE on a single GPU before dispatching
    # parallel work.  torchaudio's CUDA sinc-resample kernel is not thread-safe
    # across devices, so we serialize this step.
    ref_latent_cpu = None
    if voice_ref_path and os.path.exists(voice_ref_path):
        gpu_idx = await pool.acquire()
        try:
            server = pool.get_server(gpu_idx)
            log.info(f"Job {job.job_id}: pre-encoding voice ref on GPU {pool.gpu_ids[gpu_idx]}")

            def _encode_ref():
                torch.cuda.set_device(server.device)
                # Load audio to CPU first — torchaudio's CUDA sinc-resample
                # kernel is broken in this environment (SIGABRT from cudnn
                # symbol conflict). CPU resample is safe.
                voice = decode_audio_from_file(voice_ref_path, "cpu", 0.0, ref_duration)
                w = voice.waveform
                if w.dim() == 2:
                    if w.shape[0] == 1:
                        w = w.repeat(2, 1)
                    w = w.unsqueeze(0)
                elif w.dim() == 3 and w.shape[1] == 1:
                    w = w.repeat(1, 2, 1)
                target_samples = int(ref_duration * voice.sampling_rate)
                if w.shape[-1] < target_samples:
                    w = w.repeat(1, 1, (target_samples // w.shape[-1]) + 1)
                w = w[..., :target_samples]
                peak = w.abs().max()
                if peak > 0:
                    w = w * (10 ** (-4.0 / 20) / peak)
                # Pre-resample on CPU to the VAE's target rate (24000 Hz)
                # so the VAE encoder never calls torchaudio resample on GPU
                target_sr = 24000
                if voice.sampling_rate != target_sr:
                    w = torchaudio.functional.resample(w, voice.sampling_rate, target_sr)
                    voice = Audio(waveform=w, sampling_rate=target_sr)
                else:
                    voice = Audio(waveform=w, sampling_rate=voice.sampling_rate)
                # Move to GPU for VAE encoding
                voice = Audio(
                    waveform=voice.waveform.to(server.device),
                    sampling_rate=voice.sampling_rate,
                )
                with torch.inference_mode():
                    latent = server._audio_conditioner(lambda enc: vae_encode_audio(voice, enc, None))
                return latent.cpu()

            ref_latent_cpu = await loop.run_in_executor(pool._executor, _encode_ref)
            log.info(f"Job {job.job_id}: voice ref encoded, shape={ref_latent_cpu.shape}")
        except Exception as e:
            log.error(f"Job {job.job_id}: ref encode failed: {e}\n{traceback.format_exc()}")
        finally:
            await pool.release(gpu_idx)

    def _gen_one_candidate(gpu_idx: int, seg: Segment, cand_seed: int):
        """Generate a single candidate with a given seed."""
        gen_text = f"{prefix}{seg.text}" if prefix else seg.text
        gen_dur = fixed_duration if duration_mode == "fixed" else 0.0
        with pool._locks[gpu_idx]:
            server = pool.get_server(gpu_idx)
            waveform, sr, timings = generate_timed(
                server, gen_text, voice_ref_path,
                cfg_scale, stg_scale, duration_multiplier, cand_seed,
                ref_duration, gen_dur, False,
                ref_latent_cpu=ref_latent_cpu,
            )
        # ASR trimming + transcription
        if asr_trim and asr_trimmer is not None:
            try:
                with pool._locks[0]:
                    waveform, asr_info = asr_trimmer.trim_segment(waveform, sr, seg.text)
                timings["asr_ms"] = asr_info.get("asr_ms", 0)
                timings["asr_trimmed"] = asr_info.get("trimmed", False)
                timings["asr_text"] = asr_info.get("asr_text", "")
                if asr_info.get("trimmed"):
                    timings["audio_duration_s"] = round(waveform.shape[-1] / sr, 2)
            except Exception as e:
                log.warning(f"ASR trim failed for seg {seg.index}: {e}")
                timings["asr_ms"] = 0
                timings["asr_trimmed"] = False
                timings["asr_text"] = f"ERROR: {e}"
        elif num_candidates > 1 and asr_trimmer is not None:
            # Even without trim, we need ASR text for WER ranking
            try:
                import numpy as np
                wav_np = waveform.cpu().float().numpy()
                if wav_np.ndim == 3:
                    wav_np = wav_np.squeeze(0)
                if wav_np.ndim == 2:
                    wav_np = wav_np[0]
                with pool._locks[0]:
                    asr_text, _ = asr_trimmer._transcribe(wav_np, sr)
                timings["asr_text"] = asr_text
            except Exception as e:
                timings["asr_text"] = ""
        timings["seed"] = cand_seed
        wav_b64 = waveform_to_wav_base64(waveform, sr)
        return wav_b64, sr, timings

    def _gen_segment_sync(gpu_idx: int, seg: Segment):
        """Run generation synchronously in a thread — with GPU lock.

        When num_candidates > 1, generates multiple candidates with different
        seeds, computes WER for each, and returns the best (lowest WER) along
        with all candidates for the UI.
        """
        if num_candidates <= 1:
            return _gen_one_candidate(gpu_idx, seg, seed)

        # Multi-candidate mode: generate N candidates with different seeds
        # Score by: reward = (1 / (1 + WER)) * clap_similarity
        dialogue_words = extract_dialogue_words(seg.text)
        ref_text = " ".join(dialogue_words) if dialogue_words else seg.text

        # Pre-compute CLAP text embedding for the prefix
        clap_text_emb = None
        if asr_trimmer is not None and prefix:
            with pool._locks[0]:
                clap_text_emb = asr_trimmer.clap_text_embedding(prefix)

        candidates = []
        for ci in range(num_candidates):
            cand_seed = seed + ci
            wav_b64, sr, timings = _gen_one_candidate(gpu_idx, seg, cand_seed)
            asr_text = timings.get("asr_text", "")
            wer = compute_wer(ref_text, asr_text) if asr_text else 999.0
            timings["wer"] = round(wer, 3)

            # Compute CLAP audio-text similarity
            clap_sim = 0.0
            if clap_text_emb is not None and asr_trimmer is not None:
                try:
                    wav_bytes = base64.b64decode(wav_b64)
                    buf = io.BytesIO(wav_bytes)
                    wav_tensor, wav_sr = torchaudio.load(buf)
                    with pool._locks[0]:
                        clap_sim = asr_trimmer.clap_audio_score(wav_tensor, wav_sr, clap_text_emb)
                except Exception as e:
                    log.warning(f"CLAP scoring failed for candidate {ci}: {e}")
            timings["clap_sim"] = clap_sim

            # Combined reward: (1 / (1 + WER)) * max(0, clap_sim)
            # If no CLAP, fall back to 1/(1+WER) only
            wer_score = 1.0 / (1.0 + wer)
            if clap_text_emb is not None:
                reward = wer_score * max(0.0, clap_sim + 0.5)  # shift sim to be mostly positive
            else:
                reward = wer_score
            timings["reward"] = round(reward, 4)

            candidates.append((reward, wav_b64, sr, timings))
            log.info(f"  Candidate {ci+1}/{num_candidates} seed={cand_seed} "
                     f"WER={wer:.3f} CLAP={clap_sim:.3f} reward={reward:.4f}")

        # Sort by reward descending (best first)
        candidates.sort(key=lambda c: c[0], reverse=True)
        _, best_b64, best_sr, best_timings = candidates[0]
        # Attach all candidates to timings for UI display
        best_timings["candidates"] = [
            {"seed": c[3]["seed"], "wer": c[3]["wer"],
             "clap_sim": c[3].get("clap_sim", 0.0),
             "reward": c[3].get("reward", 0.0),
             "asr_text": c[3].get("asr_text", ""), "wav_base64": c[1]}
            for c in candidates
        ]
        return best_b64, best_sr, best_timings

    async def gen_segment(seg: Segment):
        gpu_idx = await pool.acquire()
        try:
            log.info(f"Job {job.job_id} seg {seg.index}: dispatched to GPU {pool.gpu_ids[gpu_idx]}")
            wav_b64, sr, timings = await loop.run_in_executor(
                pool._executor,
                _gen_segment_sync, gpu_idx, seg,
            )
            data = {
                "event": "segment",
                "index": seg.index,
                "text": seg.text,
                "wav_base64": wav_b64,
                "sample_rate": sr,
                "timings": timings,
                "gpu_id": pool.gpu_ids[gpu_idx],
            }
            job.segment_done(seg.index, data)
            log.info(f"Job {job.job_id} seg {seg.index}: done in {timings['total_ms']:.0f}ms "
                     f"(GPU {pool.gpu_ids[gpu_idx]})")
        except Exception as e:
            tb = traceback.format_exc()
            log.error(f"Job {job.job_id} seg {seg.index}: FAILED\n{tb}")
            job.segment_done(seg.index, {
                "event": "error",
                "index": seg.index,
                "text": seg.text,
                "error": str(e),
            })
        finally:
            await pool.release(gpu_idx)

    # Launch all segments concurrently
    tasks = [asyncio.create_task(gen_segment(seg)) for seg in segments]
    await asyncio.gather(*tasks)

    # Build concatenated full audio
    try:
        all_wavs = []
        sr = None
        for i in range(len(segments)):
            seg_data = job.segments.get(i)
            if seg_data and "wav_base64" in seg_data:
                wav_bytes = base64.b64decode(seg_data["wav_base64"])
                buf = io.BytesIO(wav_bytes)
                waveform, s = torchaudio.load(buf)
                all_wavs.append(waveform)
                sr = s

        if all_wavs and sr:
            combined = all_wavs[0]
            for wav in all_wavs[1:]:
                # Align channels
                if wav.shape[0] != combined.shape[0]:
                    if wav.shape[0] == 1:
                        wav = wav.repeat(combined.shape[0], 1)
                    elif combined.shape[0] == 1:
                        combined = combined.repeat(wav.shape[0], 1)
                combined = _equal_power_crossfade(combined, wav, sr, fade_ms=50.0)

            full_b64 = waveform_to_wav_base64(combined, sr)
            total_dur = combined.shape[-1] / sr
        else:
            full_b64 = None
            total_dur = 0
    except Exception as e:
        log.warning(f"Failed to concatenate: {e}")
        full_b64 = None
        total_dur = 0

    # Final event
    await job.events.put({
        "event": "done",
        "total_time_ms": (time.time() - job.start_time) * 1000,
        "total_audio_duration_s": round(total_dur, 2),
        "full_wav_base64": full_b64,
    })

    # Cleanup temp file
    if voice_ref_path:
        try:
            os.unlink(voice_ref_path)
        except OSError:
            pass


@app.get("/stream/{job_id}")
async def stream(job_id: str):
    """SSE endpoint: streams segment completion events."""
    job = jobs.get(job_id)
    if not job:
        return JSONResponse({"error": "Job not found"}, status_code=404)

    async def event_generator():
        delivered = 0
        total_events = job.num_segments + 1  # segments + final "done"
        while delivered < total_events:
            try:
                data = await asyncio.wait_for(job.events.get(), timeout=300)
                yield f"data: {json.dumps(data)}\n\n"
                delivered += 1
                if data.get("event") == "done":
                    break
            except asyncio.TimeoutError:
                yield f"data: {json.dumps({'event': 'heartbeat'})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    log.info(f"Starting on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
