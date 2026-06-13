#!/usr/bin/env python3
"""FAIR Comparison V2: All models with left-channel fix applied everywhere.

7 model columns + reference:
  1. Vanilla DramaBox (no LoRA, left channel)
  2. FT Decoder (best L1 decoder + orig vocoder, left channel)
  3. FT Vocoder (orig decoder + best mel-only vocoder, left channel)
  4. FT Combined (jointly trained decoder + vocoder, left channel)
  5. 6-Aux s80 LoRA (best saved nat from 6-aux run, left channel)
  6. 3-Aux V2 s170 LoRA (left channel)
  7. LaionBox v0.1-wip LoRA (production baseline, left channel)

All audio rendered as left-channel-only mono to eliminate stereo comb artifacts.

Usage:
    python scripts/run_fair_comparison_v2.py --gpus 1,4,5,7
"""

import argparse, json, os, sys, base64, subprocess, time, io, logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import torchaudio

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Paths ──
DRAMABOX_DIR = "/home/deployer/laion/DramaBox"
PIPELINE_DIR = "/home/deployer/laion/Voice-Acting-Pipeline"
INFERENCE_SCRIPT = os.path.join(DRAMABOX_DIR, "src", "inference.py")
BASE_CKPT = os.path.join(DRAMABOX_DIR, "models", "ltx-2.3-22b-dev-audio-only-v13-merged.safetensors")
FULL_CKPT = os.path.join(DRAMABOX_DIR, "models", "ltx-2.3-22b-dev.safetensors")
GEMMA_ROOT = "/home/deployer/.cache/dramabox/models--unsloth--gemma-3-12b-it-bnb-4bit/snapshots/826e729dbaeea4ecb143738eed2bcf3539ebf7bf"
PYTHON = "/home/deployer/miniconda3/envs/ml-general/bin/python"

REFS_DIR = "/home/deployer/laion/test-refs"
PROMPTS_FILE = os.path.join(PIPELINE_DIR, "scripts", "fair_eval_prompts.json")
OUTPUT_DIR = os.path.join(PIPELINE_DIR, "fair_comparison_v2")

# ── LoRA checkpoints ──
LORA_6AUX_S80 = os.path.join(PIPELINE_DIR, "finetune_output/lora_6aux_all_disc/lora_step_00080.safetensors")
LORA_3AUX_S170 = os.path.join(PIPELINE_DIR, "finetune_output/lora_3aux_artifact_v2/lora_step_00170.safetensors")
LORA_V01_WIP = "/home/deployer/.cache/huggingface/hub/models--laion--laionbox-v0.1-wip/snapshots/66176d2a653a013a7b71c1ccb7a7a4d4cf514b0d/lora_epoch5.safetensors"

# ── Fine-tuned decoder/vocoder checkpoints ──
FT_DECODER_BEST = os.path.join(PIPELINE_DIR, "vae_decoder_finetune_output/decoder_best.pt")
FT_VOCODER_BEST = os.path.join(PIPELINE_DIR, "vocoder_melonly_output/vocoder_best.pt")
FT_COMBINED_DECODER = os.path.join(PIPELINE_DIR, "combined_melonly_output/decoder_best.pt")
FT_COMBINED_VOCODER = os.path.join(PIPELINE_DIR, "combined_melonly_output/vocoder_best.pt")

# ── Reuse audio from previous comparisons ──
OLD_AUDIO_DIRS = [
    os.path.join(PIPELINE_DIR, "fair_4aux_comparison/audio"),
    os.path.join(PIPELINE_DIR, "fair_3aux_v2_comparison/audio"),
]

# ── Model definitions ──
MODELS = {
    "vanilla": {
        "label": "Vanilla DramaBox",
        "desc": "Base model, no fine-tuning",
        "lora": None,
        "color": "#8b949e",
        "type": "inference",
    },
    "ft_decoder": {
        "label": "FT Decoder",
        "desc": "Best L1 decoder (loss=0.226) + orig vocoder",
        "color": "#39d2c0",
        "type": "custom_pipeline",
        "decoder_ckpt": FT_DECODER_BEST,
        "vocoder_ckpt": None,
    },
    "ft_vocoder": {
        "label": "FT Vocoder",
        "desc": "Orig decoder + best mel-only vocoder (mel=0.352)",
        "color": "#f778ba",
        "type": "custom_pipeline",
        "decoder_ckpt": None,
        "vocoder_ckpt": FT_VOCODER_BEST,
    },
    "ft_combined": {
        "label": "FT Combined",
        "desc": "Jointly trained decoder+vocoder (total=0.270)",
        "color": "#3fb950",
        "type": "custom_pipeline",
        "decoder_ckpt": FT_COMBINED_DECODER,
        "vocoder_ckpt": FT_COMBINED_VOCODER,
    },
    "6aux_s80": {
        "label": "6-Aux Best Nat (s80)",
        "desc": "6-aux LoRA, best saved nat=0.529, spk=0.930",
        "lora": LORA_6AUX_S80,
        "color": "#bc8cff",
        "type": "inference",
        "metrics": {"nat": 0.529, "spk": 0.930, "loss": 0.550},
    },
    "3aux_s170": {
        "label": "3-Aux V2 (s170)",
        "desc": "3-aux-v2 LoRA step 170, artifact detector V2",
        "lora": LORA_3AUX_S170,
        "color": "#ffd700",
        "type": "inference",
        "metrics": {"nat": 0.099, "spk": 0.890},
    },
    "v01_wip": {
        "label": "v0.1-wip",
        "desc": "LaionBox v0.1-wip production LoRA (5 epochs)",
        "lora": LORA_V01_WIP,
        "color": "#58a6ff",
        "type": "inference",
    },
}

MODEL_ORDER = ["vanilla", "ft_decoder", "ft_vocoder", "ft_combined", "6aux_s80", "3aux_s170", "v01_wip"]


# ═══════════════════════════════════════════════════════════════════
# Phase 1: Generate audio via DramaBox inference (LoRA + vanilla)
# ═══════════════════════════════════════════════════════════════════

def run_inference(task, gpu_id):
    """Run a single DramaBox inference task."""
    os.makedirs(os.path.dirname(task["output"]) or ".", exist_ok=True)
    if os.path.exists(task["output"]):
        log.info(f"  SKIP  {Path(task['output']).name}")
        return True

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env.pop("LD_LIBRARY_PATH", None)

    cmd = [
        PYTHON, INFERENCE_SCRIPT,
        "--prompt", task["prompt"],
        "--output", task["output"],
        "--checkpoint", BASE_CKPT,
        "--full-checkpoint", FULL_CKPT,
        "--gemma-root", GEMMA_ROOT,
        "--seed", "42",
        "--no-watermark",
    ]
    if task.get("gen_duration"):
        cmd.extend(["--gen-duration", str(task["gen_duration"])])
    if task.get("lora"):
        cmd.extend(["--lora", task["lora"], "--lora-rank", "128"])
    if task.get("ref"):
        cmd.extend(["--voice-sample", task["ref"]])
    else:
        cmd.append("--no-ref")
    if task.get("save_latent"):
        cmd.extend(["--save-latent", task["save_latent"]])
    if task.get("skip_decode"):
        cmd.append("--skip-decode")

    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env, cwd=DRAMABOX_DIR)
        elapsed = time.time() - t0
        ok = result.returncode == 0
        if task.get("skip_decode"):
            ok = ok and os.path.exists(task["save_latent"])
        else:
            ok = ok and os.path.exists(task["output"])
        if not ok:
            err = result.stderr[-500:] if result.stderr else "unknown"
            log.error(f"  FAIL  gpu={gpu_id} {Path(task['output']).name}: {err}")
        else:
            log.info(f"  OK    gpu={gpu_id} {Path(task['output']).name} ({elapsed:.1f}s)")
        return ok
    except subprocess.TimeoutExpired:
        log.error(f"  TIMEOUT gpu={gpu_id} {Path(task['output']).name}")
        return False
    except Exception as e:
        log.error(f"  ERROR   gpu={gpu_id} {Path(task['output']).name}: {e}")
        return False


def generate_inference_audio(prompts, refs, gpu_ids):
    """Generate audio for inference-based models (LoRA + vanilla) and save vanilla latents."""
    audio_dir = os.path.join(OUTPUT_DIR, "audio")
    latent_dir = os.path.join(OUTPUT_DIR, "latents")
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(latent_dir, exist_ok=True)

    tasks = []
    reused = 0

    for model_key in MODEL_ORDER:
        model_info = MODELS[model_key]
        if model_info["type"] != "inference":
            continue

        for prompt in prompts:
            for ref_path in refs:
                ref_name = ref_path.stem
                fname = f"{model_key}__{prompt['id']}__{ref_name}.wav"
                out_path = os.path.join(audio_dir, fname)

                # Try to reuse from old comparisons
                if not os.path.exists(out_path):
                    for old_dir in OLD_AUDIO_DIRS:
                        old_path = os.path.join(old_dir, fname)
                        if os.path.exists(old_path):
                            os.link(old_path, out_path)
                            reused += 1
                            break

                task = {
                    "prompt": prompt["prompt"],
                    "prompt_id": prompt["id"],
                    "ref": str(ref_path),
                    "ref_name": ref_name,
                    "output": out_path,
                    "lora": model_info.get("lora"),
                    "gen_duration": prompt.get("gen_duration"),
                    "model_key": model_key,
                }

                # For vanilla, also save latents for custom pipeline variants
                if model_key == "vanilla":
                    latent_path = os.path.join(latent_dir, f"{prompt['id']}__{ref_name}.pt")
                    task["save_latent"] = latent_path

                tasks.append(task)

    # Also generate latent-only runs for vanilla if WAV already exists but latent doesn't
    latent_only_tasks = []
    for prompt in prompts:
        for ref_path in refs:
            ref_name = ref_path.stem
            latent_path = os.path.join(latent_dir, f"{prompt['id']}__{ref_name}.pt")
            wav_path = os.path.join(audio_dir, f"vanilla__{prompt['id']}__{ref_name}.wav")
            if os.path.exists(wav_path) and not os.path.exists(latent_path):
                latent_only_tasks.append({
                    "prompt": prompt["prompt"],
                    "prompt_id": prompt["id"],
                    "ref": str(ref_path),
                    "ref_name": ref_name,
                    "output": os.path.join(latent_dir, f"_dummy_{prompt['id']}__{ref_name}.wav"),
                    "lora": None,
                    "gen_duration": prompt.get("gen_duration"),
                    "model_key": "vanilla_latent_only",
                    "save_latent": latent_path,
                    "skip_decode": True,
                })

    all_tasks = tasks + latent_only_tasks
    log.info(f"Inference tasks: {len(all_tasks)} total (reused {reused} from previous runs)")

    # Filter to tasks that still need running
    pending = [t for t in all_tasks if not os.path.exists(t["output"])
               or (t.get("save_latent") and not os.path.exists(t["save_latent"]))]
    log.info(f"Pending: {len(pending)} tasks to generate on {len(gpu_ids)} GPUs")

    if not pending:
        return

    ok_count = fail_count = 0
    with ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
        futures = {}
        for i, task in enumerate(pending):
            gpu = gpu_ids[i % len(gpu_ids)]
            fut = pool.submit(run_inference, task, gpu)
            futures[fut] = task
        for fut in as_completed(futures):
            if fut.result():
                ok_count += 1
            else:
                fail_count += 1

    log.info(f"Generation: {ok_count} OK, {fail_count} FAIL")


# ═══════════════════════════════════════════════════════════════════
# Phase 2: Decode latents with fine-tuned decoder/vocoder
# ═══════════════════════════════════════════════════════════════════

def load_vae_decoder(device):
    """Load VAE decoder from DramaBox full checkpoint."""
    sys.path.insert(0, os.path.join(DRAMABOX_DIR, "ltx2"))
    sys.path.insert(0, os.path.join(DRAMABOX_DIR, "src"))
    from ltx_pipelines.utils.blocks import _materialize_meta_tensors
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
    from ltx_core.loader.registry import DummyRegistry
    from ltx_core.model.audio_vae.model_configurator import (
        AudioDecoderConfigurator, AUDIO_VAE_DECODER_COMFY_KEYS_FILTER)
    builder = Builder(model_path=FULL_CKPT,
                      model_class_configurator=AudioDecoderConfigurator,
                      model_sd_ops=AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
                      registry=DummyRegistry())
    return _materialize_meta_tensors(
        builder.build(device=device, dtype=torch.float32), dtype=torch.float32
    ).to(device).eval()


def load_base_vocoder(device):
    """Load vocoder from DramaBox full checkpoint."""
    sys.path.insert(0, os.path.join(DRAMABOX_DIR, "ltx2"))
    sys.path.insert(0, os.path.join(DRAMABOX_DIR, "src"))
    from ltx_pipelines.utils.blocks import _materialize_meta_tensors
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
    from ltx_core.loader.registry import DummyRegistry
    from ltx_core.model.audio_vae.model_configurator import VocoderConfigurator, VOCODER_COMFY_KEYS_FILTER
    from ltx_core.model.audio_vae.vocoder import VocoderWithBWE
    builder = Builder(model_path=FULL_CKPT,
                      model_class_configurator=VocoderConfigurator,
                      model_sd_ops=VOCODER_COMFY_KEYS_FILTER,
                      registry=DummyRegistry())
    model = _materialize_meta_tensors(
        builder.build(device=device, dtype=torch.float32), dtype=torch.float32
    ).to(device).eval()
    if isinstance(model, VocoderWithBWE):
        return model.vocoder
    return model


@torch.no_grad()
def decode_latent(decoder, vocoder, latent, device):
    """Decode latent [8, T, 16] → wav [2, samples]."""
    latent = latent.unsqueeze(0).to(device).float()
    mel = decoder(latent)
    wav = vocoder(mel.float())
    return wav.squeeze(0).cpu()


def generate_custom_pipeline_audio(prompts, refs, gpu_id):
    """Decode saved vanilla latents with fine-tuned decoder/vocoder."""
    audio_dir = os.path.join(OUTPUT_DIR, "audio")
    latent_dir = os.path.join(OUTPUT_DIR, "latents")

    # Check which custom pipeline models need generation
    custom_models = {k: v for k, v in MODELS.items() if v["type"] == "custom_pipeline"}
    if not custom_models:
        return

    # Check if all output already exists
    pending = []
    for model_key, model_info in custom_models.items():
        for prompt in prompts:
            for ref_path in refs:
                ref_name = ref_path.stem
                fname = f"{model_key}__{prompt['id']}__{ref_name}.wav"
                out_path = os.path.join(audio_dir, fname)
                latent_path = os.path.join(latent_dir, f"{prompt['id']}__{ref_name}.pt")
                if not os.path.exists(out_path) and os.path.exists(latent_path):
                    pending.append((model_key, model_info, prompt, ref_path, out_path, latent_path))

    if not pending:
        log.info("All custom pipeline audio exists, skipping")
        return

    log.info(f"Custom pipeline: {len(pending)} audio files to decode on GPU {gpu_id}")
    device = torch.device(f"cuda:{gpu_id}")

    # Load base decoder and vocoder
    log.info("Loading base decoder and vocoder...")
    base_decoder = load_vae_decoder(device)
    base_vocoder = load_base_vocoder(device)

    # Load fine-tuned checkpoints
    ft_decoders = {}
    ft_vocoders = {}

    for model_key, model_info in custom_models.items():
        dec_ckpt = model_info.get("decoder_ckpt")
        voc_ckpt = model_info.get("vocoder_ckpt")

        if dec_ckpt and os.path.exists(dec_ckpt) and dec_ckpt not in ft_decoders:
            log.info(f"Loading FT decoder: {dec_ckpt}")
            ft_dec = load_vae_decoder(device)
            ft_dec.load_state_dict(torch.load(dec_ckpt, map_location=device, weights_only=True))
            ft_dec.eval()
            ft_decoders[dec_ckpt] = ft_dec

        if voc_ckpt and os.path.exists(voc_ckpt) and voc_ckpt not in ft_vocoders:
            log.info(f"Loading FT vocoder: {voc_ckpt}")
            ft_voc = load_base_vocoder(device)
            ft_voc.load_state_dict(torch.load(voc_ckpt, map_location=device, weights_only=True))
            ft_voc.eval()
            ft_vocoders[voc_ckpt] = ft_voc

    # Decode all pending
    for model_key, model_info, prompt, ref_path, out_path, latent_path in pending:
        dec_ckpt = model_info.get("decoder_ckpt")
        voc_ckpt = model_info.get("vocoder_ckpt")
        decoder = ft_decoders.get(dec_ckpt, base_decoder)
        vocoder = ft_vocoders.get(voc_ckpt, base_vocoder)

        latent = torch.load(latent_path, map_location="cpu", weights_only=True).float()
        wav = decode_latent(decoder, vocoder, latent, device)

        # Save as 16kHz WAV
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        torchaudio.save(out_path, wav, 16000)
        log.info(f"  Decoded {Path(out_path).name} ({wav.shape[-1]/16000:.1f}s)")

    # Free GPU memory
    del base_decoder, base_vocoder, ft_decoders, ft_vocoders
    torch.cuda.empty_cache()
    log.info("Custom pipeline decoding complete")


# ═══════════════════════════════════════════════════════════════════
# Phase 3: Build HTML with left-channel extraction
# ═══════════════════════════════════════════════════════════════════

def wav_to_left_channel_base64(path):
    """Load WAV, extract left channel, return base64-encoded mono WAV."""
    if not os.path.exists(path):
        return None
    try:
        wav, sr = torchaudio.load(path)
        # Extract left channel
        if wav.shape[0] > 1:
            wav = wav[0:1]  # [1, T] — left channel only
        # Encode to WAV in memory
        buf = io.BytesIO()
        torchaudio.save(buf, wav, sr, format="wav")
        buf.seek(0)
        return "data:audio/wav;base64," + base64.b64encode(buf.read()).decode()
    except Exception as e:
        log.warning(f"Failed to process {path}: {e}")
        return None


def build_html(prompts, refs):
    """Build the comparison HTML with all models side by side."""
    audio_dir = os.path.join(OUTPUT_DIR, "audio")
    ref_names = [r.stem for r in refs]

    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LaionBox FAIR Comparison V2 — Left Channel Fix</title>
<style>
:root{--bg:#0d1117;--card:#161b22;--border:#30363d;--text:#c9d1d9;--muted:#8b949e;--accent:#f0883e}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:var(--bg);color:var(--text);padding:20px;max-width:2000px;margin:0 auto}
h1{color:#fff;font-size:1.8em;margin-bottom:4px}
h2{color:var(--accent);font-size:1.2em;margin:24px 0 10px}
.subtitle{color:var(--muted);margin-bottom:16px;font-size:.9em}
.section{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:16px;margin-bottom:16px}
.info-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px;margin-bottom:16px}
.info-card{background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:10px}
.info-card h4{font-size:.85em;margin-bottom:3px}
.info-card p{font-size:.75em;color:var(--muted);line-height:1.3}
.metric-row{display:flex;gap:6px;flex-wrap:wrap;margin-top:4px}
.metric{background:#21262d;padding:2px 5px;border-radius:3px;font-size:.7em;font-family:monospace}
.metric .k{color:var(--muted)}.metric .v{color:#3fb950;font-weight:600}
.prompt-box{background:var(--bg);border:1px solid var(--border);border-radius:8px;padding:12px;margin-bottom:12px}
.prompt-box .prompt-title{color:var(--accent);font-weight:600;font-size:.9em;margin-bottom:4px}
.prompt-box .prompt-text{color:var(--text);font-size:.82em;line-height:1.4;font-style:italic}
.comparison-grid{overflow-x:auto}
table{border-collapse:collapse;width:100%;min-width:1400px}
th,td{padding:6px 5px;border:1px solid var(--border);text-align:center;vertical-align:middle}
th{background:#21262d;color:#fff;font-size:.72em;position:sticky;top:0;z-index:2}
th.model-header{min-width:120px}
td.ref-header{text-align:left;font-weight:600;font-size:.8em;background:var(--card);min-width:150px}
td audio{width:110px;height:28px}
td.empty{background:#1a1a2e;color:var(--muted);font-size:.7em}
.group-sep{border-left:3px solid var(--accent) !important}
.note{font-size:.72em;color:var(--muted);margin-top:8px;padding:8px;background:#21262d;border-radius:6px}
.footer{margin-top:24px;padding:12px;border-top:1px solid var(--border);font-size:.72em;color:var(--muted)}
.badge{display:inline-block;padding:1px 6px;border-radius:3px;font-size:.65em;font-weight:600;margin-left:4px}
.badge-lora{background:#2d1b69;color:#bc8cff}
.badge-ft{background:#1b3d2d;color:#3fb950}
.badge-base{background:#2d2d2d;color:#8b949e}
</style>
</head>
<body>

<h1>LaionBox FAIR Comparison V2</h1>
<p class="subtitle">7 model variants &times; 3 dramatic scenes &times; 5 reference speakers = 105 audio samples | <b>All left-channel mono</b> (comb-filter fix applied)</p>

<div class="section">
<h2>Models</h2>
<div class="note">
<b>Left-channel fix:</b> All audio is rendered as left-channel mono. The DramaBox vocoder (BigVGAN v2) outputs stereo with
L-R correlation ~0.64. Mono summation creates destructive comb-filter interference above 2kHz, producing metallic artifacts.
Using the left channel alone eliminates this completely.<br><br>
<b>Two approaches compared:</b> Columns 1-4 modify the audio rendering pipeline (VAE decoder + vocoder) while keeping
the flow model (latent generation) unchanged. Columns 5-7 modify the flow model via LoRA fine-tuning while keeping
the decoder/vocoder unchanged. FT Decoder/Vocoder/Combined output at 16kHz; LoRA models output at 48kHz (with BWE).
</div>
<div class="info-grid">
"""

    for mk in MODEL_ORDER:
        mi = MODELS[mk]
        color = mi.get("color", "#ccc")
        mtype = mi["type"]
        badge_cls = "badge-base" if mk == "vanilla" else ("badge-ft" if mtype == "custom_pipeline" else "badge-lora")
        badge_text = "BASE" if mk == "vanilla" else ("FT" if mtype == "custom_pipeline" else "LoRA")
        html += f'<div class="info-card"><h4 style="color:{color}">{mi["label"]} <span class="badge {badge_cls}">{badge_text}</span></h4>'
        html += f'<p>{mi["desc"]}</p>'
        metrics = mi.get("metrics", {})
        if metrics:
            html += '<div class="metric-row">'
            for k, v in metrics.items():
                html += f'<span class="metric"><span class="k">{k}:</span> <span class="v">{v}</span></span>'
            html += '</div>'
        html += '</div>\n'
    html += '</div>\n</div>\n'

    # Per-prompt sections
    for prompt in prompts:
        pid = prompt["id"]
        html += f'<div class="section">\n<h2>{prompt["title"]}</h2>\n'
        html += f'<div class="prompt-box"><div class="prompt-title">Prompt:</div>'
        html += f'<div class="prompt-text">{prompt["prompt"]}</div>'
        if prompt.get("gen_duration"):
            html += f'<div style="margin-top:3px;font-size:.78em;color:var(--muted)">Duration: {prompt["gen_duration"]}s</div>'
        html += '</div>\n'

        html += '<div class="comparison-grid"><table>\n<thead><tr><th>Reference</th>'
        for i, mk in enumerate(MODEL_ORDER):
            mi = MODELS[mk]
            color = mi.get("color", "#ccc")
            sep = ' class="group-sep model-header"' if mk == "6aux_s80" else ' class="model-header"'
            html += f'<th{sep} style="color:{color}">{mi["label"]}</th>'
        html += '</tr></thead>\n<tbody>\n'

        for ref_name in ref_names:
            friendly = ref_name.replace("enh_", "").replace("-ref-enhanced", "").replace("_", " ")
            if len(friendly) > 22:
                friendly = friendly[:19] + "..."
            ref_path = os.path.join(REFS_DIR, ref_name + ".wav")
            ref_b64 = wav_to_left_channel_base64(ref_path)
            html += f'<tr><td class="ref-header">{friendly}'
            if ref_b64:
                html += f'<br><audio controls preload="none" src="{ref_b64}"></audio>'
            html += '</td>'

            for mk in MODEL_ORDER:
                fname = f"{mk}__{pid}__{ref_name}.wav"
                wav_path = os.path.join(audio_dir, fname)
                b64 = wav_to_left_channel_base64(wav_path)
                sep = ' class="group-sep"' if mk == "6aux_s80" else ""
                if b64:
                    html += f'<td{sep}><audio controls preload="none" src="{b64}"></audio></td>'
                else:
                    html += f'<td{sep} class="empty">missing</td>'
            html += '</tr>\n'

        html += '</tbody></table></div>\n</div>\n'

    html += f"""
<div class="footer">
  Generated {time.strftime("%Y-%m-%d %H:%M:%S")} | LaionBox FAIR Comparison V2<br>
  Left group (columns 1-4): Audio pipeline fine-tuning (decoder/vocoder) | Right group (columns 5-7): Flow model LoRA fine-tuning<br>
  All audio rendered as <b>left-channel mono</b> to eliminate stereo comb-filter artifacts
</div>
</body></html>"""

    out_path = os.path.join(OUTPUT_DIR, "fair_comparison.html")
    with open(out_path, "w") as f:
        f.write(html)
    sz = os.path.getsize(out_path) / 1e6
    log.info(f"HTML written: {out_path} ({sz:.1f} MB)")
    return out_path


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, default="1,4,5,7", help="Comma-separated GPU IDs for inference")
    args = parser.parse_args()

    gpu_ids = [int(g) for g in args.gpus.split(",")]
    log.info(f"Using GPUs: {gpu_ids}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(PROMPTS_FILE) as f:
        prompts = json.load(f)
    refs = sorted(Path(REFS_DIR).glob("*.wav"))
    log.info(f"Prompts: {len(prompts)}, Refs: {len(refs)}, Models: {len(MODELS)}")

    # Verify checkpoints exist
    for mk, mi in MODELS.items():
        if mi.get("lora") and not os.path.exists(mi["lora"]):
            log.error(f"Missing LoRA: {mi['lora']}")
            sys.exit(1)
        for ck in ["decoder_ckpt", "vocoder_ckpt"]:
            path = mi.get(ck)
            if path and not os.path.exists(path):
                log.warning(f"Missing {ck} for {mk}: {path}")

    t0 = time.time()

    # Phase 1: Generate inference-based audio
    log.info("\n" + "="*60)
    log.info("Phase 1: Generating inference audio (LoRA + vanilla)...")
    log.info("="*60)
    generate_inference_audio(prompts, refs, gpu_ids)

    # Phase 2: Decode latents with fine-tuned decoder/vocoder
    log.info("\n" + "="*60)
    log.info("Phase 2: Decoding latents with fine-tuned decoder/vocoder...")
    log.info("="*60)
    generate_custom_pipeline_audio(prompts, refs, gpu_ids[0])

    # Phase 3: Build HTML
    log.info("\n" + "="*60)
    log.info("Phase 3: Building HTML with left-channel extraction...")
    log.info("="*60)
    html_path = build_html(prompts, refs)

    elapsed = time.time() - t0
    log.info(f"\nDone in {elapsed:.0f}s")
    log.info(f"HTML: {html_path}")
    log.info(f"\nTo serve:")
    log.info(f"  Already served on port 8792 via Cloudflare tunnel")
    log.info(f"  URL: <tunnel>/fair_comparison_v2/fair_comparison.html")


if __name__ == "__main__":
    main()
