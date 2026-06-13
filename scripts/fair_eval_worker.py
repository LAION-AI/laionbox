#!/usr/bin/env python3
"""FAIR: Continuous evaluation worker for LaionBox training.

Watches a checkpoint directory, and for each new checkpoint:
1. Generates 3 scenes x N reference speakers = 15+ audio samples on 2 GPUs
2. Scores with CLAP naturalness, quality MLP, speaker similarity
3. Builds an accumulating HTML page with embedded audio
4. Pushes checkpoint to HuggingFace

Usage:
    CUDA_VISIBLE_DEVICES=6,7 python scripts/fair_eval_worker.py \
        --checkpoint-dir ./finetune_output/diff_nat_spk_quality_2ep_cont \
        --output-dir ./fair_eval \
        --hf-repo TTS-AGI/laionbox-checkpoints \
        --refs-dir /home/deployer/laion/test-refs \
        --num-gpus 2
"""

import argparse
import glob
import json
import logging
import os
import sys
import time
import base64
import subprocess
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VAP_DIR = os.path.dirname(SCRIPT_DIR)


def load_prompts():
    with open(os.path.join(SCRIPT_DIR, "fair_eval_prompts.json")) as f:
        return json.load(f)


def get_refs(refs_dir):
    return sorted(glob.glob(os.path.join(refs_dir, "*.wav")))


def generate_samples(checkpoint_path, prompts, refs, output_dir, num_gpus, step_name,
                     is_full_model=False):
    """Generate audio samples for one checkpoint using subprocess workers."""
    tasks = []
    # Use absolute paths — inference subprocesses run with cwd=DRAMABOX_DIR
    abs_output_dir = os.path.abspath(output_dir)
    abs_checkpoint = os.path.abspath(checkpoint_path)
    for p in prompts:
        for ref in refs:
            ref_name = Path(ref).stem
            out_path = os.path.join(abs_output_dir, "wavs", f"{step_name}__{p['id']}__{ref_name}.wav")
            task = {
                "prompt": p["prompt"],
                "prompt_id": p["id"],
                "ref": os.path.abspath(ref),
                "ref_name": ref_name,
                "output": out_path,
            }
            # Full model checkpoint: pass as --checkpoint directly (no LoRA)
            # LoRA checkpoint: pass as --lora with default base model
            if is_full_model:
                task["checkpoint"] = abs_checkpoint
            else:
                task["lora"] = abs_checkpoint
            if p.get("gen_duration"):
                task["gen_duration"] = p["gen_duration"]
            tasks.append(task)

    # Skip existing
    tasks_to_gen = [t for t in tasks if not os.path.exists(t["output"])]
    if not tasks_to_gen:
        logging.info(f"  All {len(tasks)} samples exist, skipping generation")
        return tasks

    logging.info(f"  Generating {len(tasks_to_gen)}/{len(tasks)} samples on {num_gpus} GPUs...")

    # Write tasks to temp file for workers
    tasks_file = os.path.join(output_dir, f"_gen_tasks_{step_name}.json")
    with open(tasks_file, "w") as f:
        json.dump(tasks_to_gen, f)

    # Launch generation worker — inherits CUDA_VISIBLE_DEVICES from parent
    worker_script = os.path.join(SCRIPT_DIR, "fair_gen_worker.py")
    env = os.environ.copy()
    proc = subprocess.run(
        [sys.executable, worker_script, tasks_file, str(num_gpus)],
        env=env, capture_output=True, text=True, timeout=1800,
    )
    if proc.returncode != 0:
        logging.error(f"  Generation failed (exit {proc.returncode})")
        if proc.stderr:
            logging.error(f"  stderr: {proc.stderr[-500:]}")
    else:
        logging.info(f"  Generation complete")
    if proc.stdout:
        for line in proc.stdout.strip().split("\n")[-5:]:
            logging.info(f"  gen> {line}")

    try:
        os.remove(tasks_file)
    except OSError:
        pass
    return tasks


def score_samples(tasks, classifiers_dir):
    """Score generated audio with CLAP-small, CLAP-7B, centroid, quality MLP, WavLM-SV.

    Models are loaded fresh each time to avoid holding GPU memory during generation.
    """
    import torch
    import torchaudio
    import torch.nn.functional as F
    import torch.nn as nn
    import soundfile as sf

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"  Loading scoring models on {device}...")

    # ── CLAP-small ──
    from transformers import AutoModel, AutoTokenizer, AutoFeatureExtractor
    clap_model = AutoModel.from_pretrained("laion/voiceclap-small", trust_remote_code=True).to(device).eval()
    clap_tokenizer = AutoTokenizer.from_pretrained("laion/voiceclap-small", trust_remote_code=True)

    # ── CLAP-large (VoiceCLAP 7B with INT4) ──
    logging.info("  Loading VoiceCLAP-7B (INT4)...")
    from sentence_transformers import SentenceTransformer
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    clap_large = SentenceTransformer(
        "gijs/voiceclap-lco-7b-lora",
        model_kwargs={"quantization_config": bnb_config, "torch_dtype": torch.bfloat16, "trust_remote_code": True},
        trust_remote_code=True,
    )
    logging.info(f"  VoiceCLAP-7B loaded, dim={clap_large.get_sentence_embedding_dimension()}")

    # ── Centroids (from CLAP-small embeddings) ──
    emb_data = torch.load(os.path.join(classifiers_dir, "clap_embeddings.pt"),
                          map_location="cpu", weights_only=False)
    dramabox_embs = emb_data["dramabox_embeddings"]
    emilia_embs = emb_data["emilia_embeddings"]
    n_train = int(len(dramabox_embs) * 0.8)
    synth_centroid = F.normalize(dramabox_embs[:n_train].float().mean(0, keepdim=True), p=2, dim=-1).to(device)
    real_centroid = F.normalize(emilia_embs[:n_train].float().mean(0, keepdim=True), p=2, dim=-1).to(device)

    # ── Quality MLP ──
    ckpt = torch.load(os.path.join(classifiers_dir, "quality_classifier.pt"),
                      map_location="cpu", weights_only=False)

    class BinaryMLP(nn.Module):
        def __init__(self, d_in, h1, h2):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_in, h1), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(h1, h2), nn.ReLU(), nn.Dropout(0.3),
                nn.Linear(h2, 1))
        def forward(self, x):
            return self.net(x)

    quality_mlp = BinaryMLP(ckpt["input_dim"], ckpt["hidden1"], ckpt["hidden2"])
    quality_mlp.load_state_dict(ckpt["model_state_dict"])
    quality_mlp.eval().to(device)

    # ── WavLM-SV ──
    from transformers import WavLMForXVector
    wavlm = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv").to(device).eval()
    wavlm_fe = AutoFeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")

    # ── Pre-encode text embeddings ──
    pos_text = "Realistic, genuine, spontaneous, authentic, sensual, natural voice with all imperfections and organic microdistractions a natural situation brings with it"
    neg_text = "distorted, unnatural, robotic, distortion"

    with torch.no_grad():
        pos_tok = clap_tokenizer(pos_text, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
        neg_tok = clap_tokenizer(neg_text, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
        pos_emb = clap_model.encode_text(pos_tok["input_ids"], pos_tok.get("attention_mask"))
        neg_emb = clap_model.encode_text(neg_tok["input_ids"], neg_tok.get("attention_mask"))
        pos_emb = pos_emb / pos_emb.norm(dim=-1, keepdim=True)
        neg_emb = neg_emb / neg_emb.norm(dim=-1, keepdim=True)

    with torch.no_grad():
        pos_emb_large = clap_large.encode([pos_text], convert_to_tensor=True)
        neg_emb_large = clap_large.encode([neg_text], convert_to_tensor=True)
        pos_emb_large = F.normalize(pos_emb_large, p=2, dim=-1)
        neg_emb_large = F.normalize(neg_emb_large, p=2, dim=-1)

    logging.info("  All scoring models loaded. Scoring...")

    # ── Score each sample ──
    scored = []
    for i, t in enumerate(tasks):
        if not os.path.exists(t["output"]):
            scored.append({**t, "scores": {}})
            continue
        try:
            waveform, sr = torchaudio.load(t["output"])
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            with torch.no_grad():
                if sr != 16000:
                    waveform_16k = torchaudio.functional.resample(waveform, sr, 16000)
                else:
                    waveform_16k = waveform

                # CLAP-small
                audio_emb = clap_model.encode_waveform(waveform_16k.to(device), sample_rate=16000)
                audio_emb = audio_emb / audio_emb.norm(dim=-1, keepdim=True)
                nat_small = (audio_emb @ pos_emb.T).item() - (audio_emb @ neg_emb.T).item()
                cent_score = (audio_emb @ real_centroid.T).item() - (audio_emb @ synth_centroid.T).item()
                quality_score = torch.sigmoid(quality_mlp(audio_emb)).item()

                # CLAP-large via temp file
                tmp_path = f"/dev/shm/fair_clap_{os.getpid()}.wav"
                try:
                    sf.write(tmp_path, waveform_16k.squeeze(0).numpy(), 16000)
                    audio_emb_large = clap_large.encode([{"audio": tmp_path}], convert_to_tensor=True)
                    audio_emb_large = F.normalize(audio_emb_large, p=2, dim=-1)
                    nat_large = (audio_emb_large @ pos_emb_large.T).item() - (audio_emb_large @ neg_emb_large.T).item()
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

                # Speaker similarity
                spk_score = None
                if t.get("ref"):
                    ref_wav, ref_sr = torchaudio.load(t["ref"])
                    if ref_wav.shape[0] > 1:
                        ref_wav = ref_wav.mean(dim=0, keepdim=True)
                    if ref_sr != 16000:
                        ref_wav_16k = torchaudio.functional.resample(ref_wav, ref_sr, 16000)
                    else:
                        ref_wav_16k = ref_wav

                    max_len = 16000 * 10
                    ref_wav_16k = ref_wav_16k[:, :max_len]
                    gen_wav_16k = waveform_16k[:, :max_len]

                    ref_in = wavlm_fe(ref_wav_16k.squeeze(0), sampling_rate=16000, return_tensors="pt", padding=True)
                    gen_in = wavlm_fe(gen_wav_16k.squeeze(0), sampling_rate=16000, return_tensors="pt", padding=True)
                    ref_emb_sv = wavlm(**{k: v.to(device) for k, v in ref_in.items()}).embeddings
                    gen_emb_sv = wavlm(**{k: v.to(device) for k, v in gen_in.items()}).embeddings
                    ref_emb_sv = ref_emb_sv / ref_emb_sv.norm(dim=-1, keepdim=True)
                    gen_emb_sv = gen_emb_sv / gen_emb_sv.norm(dim=-1, keepdim=True)
                    spk_score = (ref_emb_sv @ gen_emb_sv.T).item()

                scores = {
                    "naturalness_small": round(nat_small, 4),
                    "naturalness_large": round(nat_large, 4),
                    "centroid": round(cent_score, 4),
                    "quality": round(quality_score, 4),
                    "speaker_sim": round(spk_score, 4) if spk_score is not None else None,
                }
            scored.append({**t, "scores": scores})
        except Exception as e:
            logging.warning(f"  Score failed for {t['output']}: {e}")
            import traceback
            traceback.print_exc()
            scored.append({**t, "scores": {}})

        if (i + 1) % 5 == 0:
            logging.info(f"  Scored {i+1}/{len(tasks)}")

    logging.info(f"  Scoring complete: {sum(1 for s in scored if s.get('scores'))} scored")

    # Free GPU memory for next generation round
    del clap_model, clap_tokenizer, clap_large, quality_mlp, wavlm, wavlm_fe
    del pos_emb, neg_emb, pos_emb_large, neg_emb_large, synth_centroid, real_centroid
    torch.cuda.empty_cache()

    return scored


def push_to_hf(checkpoint_path, hf_repo, step_name, metrics_file=None):
    """Push checkpoint to HuggingFace."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()

        api.upload_file(
            path_or_fileobj=checkpoint_path,
            path_in_repo=f"run14/{step_name}.safetensors",
            repo_id=hf_repo,
            repo_type="model",
        )
        logging.info(f"  Pushed {step_name} to {hf_repo}")

        if metrics_file and os.path.exists(metrics_file):
            api.upload_file(
                path_or_fileobj=metrics_file,
                path_in_repo=f"run14/metrics.jsonl",
                repo_id=hf_repo,
                repo_type="model",
            )
    except Exception as e:
        logging.error(f"  HF push failed: {e}")


COMMON_CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 20px; background: #1a1a2e; color: #e0e0e0; }
h1 { color: #e94560; text-align: center; }
h2 { background: #16213e; padding: 12px; border-radius: 8px; margin-top: 30px; color: #e94560; }
h3 { margin-top: 20px; color: #53a8b6; }
.scores-summary { display: flex; gap: 15px; flex-wrap: wrap; margin: 10px 0; }
.score-card { background: #0f3460; padding: 10px 15px; border-radius: 8px; text-align: center; min-width: 120px; }
.score-card .label { font-size: 11px; color: #999; text-transform: uppercase; }
.score-card .value { font-size: 20px; font-weight: bold; color: #e94560; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; }
th { background: #0f3460; color: #e94560; padding: 8px 12px; text-align: left; }
td { padding: 8px 12px; border-bottom: 1px solid #2a2a4a; }
tr:hover { background: #1f1f3a; }
audio { width: 280px; height: 36px; }
.best-row { background: #1a3a1a !important; }
.scene-title { font-weight: bold; color: #53a8b6; }
.nav { text-align: center; margin: 20px 0; }
.nav a { color: #53a8b6; margin: 0 8px; text-decoration: none; padding: 4px 10px; border: 1px solid #2a2a4a; border-radius: 4px; }
.nav a:hover { color: #e94560; border-color: #e94560; }
.back { display: inline-block; margin-bottom: 15px; color: #53a8b6; text-decoration: none; }
.back:hover { color: #e94560; }
"""


def _compute_step_avgs(scored):
    avgs = {}
    for metric in ["naturalness_small", "naturalness_large", "centroid", "quality", "speaker_sim"]:
        vals = [s["scores"].get(metric) for s in scored if s.get("scores") and s["scores"].get(metric) is not None]
        avgs[metric] = sum(vals) / len(vals) if vals else 0
    composite = (avgs["naturalness_small"] + avgs["quality"] + (avgs["speaker_sim"] or 0)) / 3
    avgs["composite"] = composite
    return avgs


def _build_checkpoint_page(step_name, scored, output_dir, prompts):
    """Build a per-checkpoint subpage with embedded audio."""
    os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)
    page_path = os.path.join(output_dir, "checkpoints", f"{step_name}.html")

    avgs = _compute_step_avgs(scored)

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>FAIR - {step_name}</title>
<style>{COMMON_CSS}</style></head><body>
<a class="back" href="../fair.html">&larr; Back to Overview</a>
<h1>{step_name}</h1>
<div class="scores-summary">
"""
    for m, label in [("naturalness_small", "Nat-S"), ("naturalness_large", "Nat-L"),
                     ("centroid", "Centroid"), ("quality", "Quality"), ("speaker_sim", "SpkSim"),
                     ("composite", "Composite")]:
        html += f'<div class="score-card"><div class="label">{label}</div><div class="value">{avgs[m]:.4f}</div></div>\n'
    html += '</div>\n'

    for p in prompts:
        scene_samples = [s for s in scored if s.get("prompt_id") == p["id"]]
        if not scene_samples:
            continue
        html += f'<h2 class="scene-title">{p["title"]}</h2>\n'
        # Show only the acting instruction part of the prompt (after the quality prefix)
        prompt_display = p["prompt"]
        if "High Quality Recording." in prompt_display:
            prompt_display = prompt_display.split("High Quality Recording.")[-1].strip()
        html += f'<p style="color:#777; font-size:12px; margin:2px 0 8px 0;"><em>{prompt_display[:200]}</em></p>\n'
        if p.get("gen_duration"):
            html += f'<p style="color:#555; font-size:11px;">Duration: {p["gen_duration"]}s</p>\n'
        html += '<table><tr><th>Speaker</th><th>Audio</th><th>Nat-S</th><th>Nat-L</th><th>Quality</th><th>SpkSim</th></tr>\n'

        for s in scene_samples:
            ref_name = s.get("ref_name", "?")
            wav_path = s.get("output", "")

            audio_html = ""
            if os.path.exists(wav_path):
                try:
                    with open(wav_path, "rb") as af:
                        audio_b64 = base64.b64encode(af.read()).decode()
                    audio_html = f'<audio controls preload="none"><source src="data:audio/wav;base64,{audio_b64}" type="audio/wav"></audio>'
                except Exception:
                    audio_html = "(error)"
            else:
                audio_html = "(missing)"

            scores = s.get("scores", {})
            html += f'<tr><td>{ref_name}</td><td>{audio_html}</td>'
            for m in ["naturalness_small", "naturalness_large", "quality", "speaker_sim"]:
                v = scores.get(m)
                html += f'<td>{v:.4f}</td>' if v is not None else '<td>-</td>'
            html += '</tr>\n'
        html += '</table>\n'

    html += '</body></html>'

    with open(page_path, "w") as f:
        f.write(html)
    return page_path


def build_html(all_results, output_dir, prompts):
    """Build lightweight index page + per-checkpoint subpages with audio."""
    html_path = os.path.join(output_dir, "fair.html")
    by_step = dict(all_results)

    # Build per-checkpoint subpages
    for step_name, scored in by_step.items():
        _build_checkpoint_page(step_name, scored, output_dir, prompts)

    # Build index page (no embedded audio — lightweight)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>FAIR - Fine-tuning Audio Inspection Report</title>
<meta http-equiv="refresh" content="60">
<style>{COMMON_CSS}</style></head><body>
<h1>FAIR - Fine-tuning Audio Inspection Report</h1>
<p style="text-align:center; color:#999;">Continuous evaluation of LaionBox (Nat+Spk+Quality, 6-GPU training)</p>
<p style="text-align:center; color:#666;">Last updated: {now_str} &mdash; {len(by_step)} checkpoints evaluated</p>
"""

    if by_step:
        # Compute averages
        step_avgs = {}
        for step_name in sorted(by_step.keys()):
            step_avgs[step_name] = _compute_step_avgs(by_step[step_name])

        best_step = max(step_avgs, key=lambda s: step_avgs[s]["composite"]) if step_avgs else None

        html += '<h2>Score Summary</h2>\n'
        html += '<table><tr><th>Checkpoint</th><th>Nat-Small</th><th>Nat-Large</th><th>Centroid</th><th>Quality</th><th>Speaker Sim</th><th>Composite</th><th></th></tr>\n'

        for step_name in sorted(step_avgs.keys()):
            avgs = step_avgs[step_name]
            row_class = ' class="best-row"' if step_name == best_step else ''
            short = step_name.replace("lora_step_", "s").replace("r14_lora_step_", "r14_s").replace("model_step_", "ft_s").replace("model_epoch", "ft_ep")
            html += f'<tr{row_class}><td>{short}</td>'
            for m in ["naturalness_small", "naturalness_large", "centroid", "quality", "speaker_sim"]:
                html += f'<td>{avgs[m]:.4f}</td>'
            html += f'<td><b>{avgs["composite"]:.4f}</b></td>'
            html += f'<td><a href="checkpoints/{step_name}.html">Listen</a></td>'
            html += '</tr>\n'
        html += '</table>\n'

        # Navigation links
        html += '<div class="nav">\n'
        for step_name in sorted(by_step.keys()):
            short = step_name.replace("lora_step_", "s").replace("r14_lora_step_", "r14_s").replace("model_step_", "ft_s").replace("model_epoch", "ft_ep")
            html += f'<a href="checkpoints/{step_name}.html">{short}</a>\n'
        html += '</div>\n'

    html += '</body></html>'

    with open(html_path, "w") as f:
        f.write(html)
    logging.info(f"  HTML index updated: {html_path} ({len(html)} bytes), {len(by_step)} subpages")
    return html_path


def main():
    parser = argparse.ArgumentParser(description="FAIR continuous eval worker")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", default=os.path.join(VAP_DIR, "fair_eval"))
    parser.add_argument("--hf-repo", default="TTS-AGI/laionbox-ablation-checkpoints")
    parser.add_argument("--refs-dir", default="/home/deployer/laion/test-refs")
    parser.add_argument("--classifiers-dir", default=os.path.join(VAP_DIR, "classifiers"))
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument("--keep-local", type=int, default=3, help="Keep best N checkpoints locally")
    parser.add_argument("--poll-interval", type=int, default=30, help="Seconds between polling")
    parser.add_argument("--no-push", action="store_true", help="Skip pushing checkpoints to HuggingFace")
    args = parser.parse_args()

    # Ensure absolute paths to avoid cwd issues in subprocesses
    args.output_dir = os.path.abspath(args.output_dir)
    args.checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    args.refs_dir = os.path.abspath(args.refs_dir)
    args.classifiers_dir = os.path.abspath(args.classifiers_dir)
    os.makedirs(os.path.join(args.output_dir, "wavs"), exist_ok=True)

    prompts = load_prompts()
    refs = get_refs(args.refs_dir)
    logging.info(f"FAIR eval worker started")
    logging.info(f"  Checkpoint dir: {args.checkpoint_dir}")
    logging.info(f"  Prompts: {len(prompts)} scenes")
    logging.info(f"  Refs: {len(refs)} speakers")
    logging.info(f"  Samples per checkpoint: {len(prompts) * len(refs)}")
    logging.info(f"  GPUs: {args.num_gpus}")
    logging.info(f"  HF repo: {args.hf_repo}")

    # Track evaluated checkpoints
    state_file = os.path.join(args.output_dir, "eval_state.json")
    if os.path.exists(state_file):
        with open(state_file) as f:
            state = json.load(f)
    else:
        state = {"evaluated": {}, "all_results": {}}

    all_results = state.get("all_results", {})
    metrics_file = os.path.join(args.checkpoint_dir, "metrics.jsonl")

    while True:
        # Find all checkpoints — LoRA and full model
        ckpts = sorted(glob.glob(os.path.join(args.checkpoint_dir, "lora_step_*.safetensors")))
        ckpts += sorted(glob.glob(os.path.join(args.checkpoint_dir, "r14_lora_step_*.safetensors")))
        ckpts += sorted(glob.glob(os.path.join(args.checkpoint_dir, "lora_epoch*.safetensors")))
        ckpts += sorted(glob.glob(os.path.join(args.checkpoint_dir, "model_step_*.safetensors")))
        ckpts += sorted(glob.glob(os.path.join(args.checkpoint_dir, "model_epoch*.safetensors")))

        new_ckpts = [c for c in ckpts if c not in state.get("evaluated", {})]

        if not new_ckpts:
            # Check if training is done
            status_file = os.path.join(args.checkpoint_dir, "status.json")
            if os.path.exists(status_file):
                with open(status_file) as f:
                    status = json.load(f)
                if status.get("step", 0) >= status.get("total_steps", 999999):
                    logging.info("Training complete, final poll...")
                    time.sleep(5)
                    final_ckpts = sorted(glob.glob(os.path.join(args.checkpoint_dir, "lora_*.safetensors")))
                    final_ckpts += sorted(glob.glob(os.path.join(args.checkpoint_dir, "r14_lora_*.safetensors")))
                    final_ckpts += sorted(glob.glob(os.path.join(args.checkpoint_dir, "model_*.safetensors")))
                    new_final = [c for c in final_ckpts if c not in state.get("evaluated", {})]
                    if not new_final:
                        logging.info("All checkpoints evaluated. Exiting.")
                        break

            time.sleep(args.poll_interval)
            continue

        for ckpt_path in new_ckpts:
            step_name = Path(ckpt_path).stem
            is_full_model = step_name.startswith("model_")
            logging.info(f"\n{'='*60}")
            logging.info(f"Evaluating: {step_name} ({'full model' if is_full_model else 'LoRA'})")
            logging.info(f"{'='*60}")

            # 1. Push to HuggingFace first
            if not args.no_push:
                push_to_hf(ckpt_path, args.hf_repo, step_name, metrics_file)

            # 2. Generate samples (subprocess — uses both GPUs)
            tasks = generate_samples(
                ckpt_path, prompts, refs,
                args.output_dir, args.num_gpus, step_name,
                is_full_model=is_full_model,
            )

            # 3. Score (loads models on GPU, frees after)
            logging.info(f"  Scoring {len(tasks)} samples...")
            scored = score_samples(tasks, args.classifiers_dir)

            # 4. Store results
            all_results[step_name] = scored
            state["evaluated"][ckpt_path] = {
                "step_name": step_name,
                "timestamp": datetime.now().isoformat(),
                "n_samples": len(scored),
            }
            state["all_results"] = all_results

            # 5. Build HTML
            build_html(all_results, args.output_dir, prompts)

            # 6. Save state
            with open(state_file, "w") as f:
                json.dump(state, f, indent=2, default=str)

            # 7. Cleanup: keep only best N checkpoints locally
            if len(state["evaluated"]) > args.keep_local:
                ckpt_scores = {}
                for cp, info in state["evaluated"].items():
                    sn = info["step_name"]
                    if sn in all_results:
                        vals = [s["scores"].get("naturalness_small", 0) or 0
                                for s in all_results[sn] if s.get("scores")]
                        ckpt_scores[cp] = sum(vals) / max(len(vals), 1)

                ranked = sorted(ckpt_scores.items(), key=lambda x: x[1])
                to_delete = ranked[:len(ranked) - args.keep_local]

                for cp, score in to_delete:
                    if os.path.exists(cp):
                        os.remove(cp)
                        logging.info(f"  Cleaned up {Path(cp).name} (score={score:.4f})")

        logging.info(f"\nWaiting for new checkpoints...")
        time.sleep(args.poll_interval)

    logging.info("FAIR eval worker finished.")


if __name__ == "__main__":
    main()
