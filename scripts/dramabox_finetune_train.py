#!/usr/bin/env python3
"""
DramaBox Voice Cloning LoRA Fine-Tuning

Three-mode training: predict part2 from part1 reference (Mode A),
predict part1 from part2 reference (Mode B), or predict full sequence
without reference (Mode C).

Uses the IC-LoRA pattern from DramaBox: reference audio tokens appended
to end of target with asymmetric attention mask.

Usage (single GPU test):
    python scripts/dramabox_finetune_train.py --config configs/finetune.yaml --test

Usage (multi-GPU):
    accelerate launch --num_processes=8 scripts/dramabox_finetune_train.py \
        --config configs/finetune.yaml

Usage (overfit sanity check):
    python scripts/dramabox_finetune_train.py --config configs/finetune.yaml \
        --test --overfit 16 --steps 2000
"""

import os
import sys

# Filter out conda ml-general paths that break native cuDNN libraries
_ld = os.environ.get("LD_LIBRARY_PATH", "")
if _ld:
    _filtered = [p for p in _ld.split(":") if "ml-general" not in p]
    os.environ["LD_LIBRARY_PATH"] = ":".join(_filtered)

import argparse
import http.server
import json
import logging
import math
import random
import glob as glob_mod
import shutil
import subprocess as _subprocess
import threading
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ── Path setup ─────────────────────────────────────────────────────────

def setup_paths(dramabox_dir: str):
    """Add DramaBox to sys.path for imports."""
    sys.path.insert(0, os.path.join(dramabox_dir, "ltx2"))
    sys.path.insert(0, os.path.join(dramabox_dir, "src"))


# ── Timestep Samplers (from DramaBox train.py) ────────────────────────

class ShiftedLogitNormalTimestepSampler:
    """Shifted logit-normal distribution, shift depends on sequence length."""

    def __init__(self, std: float = 1.0, eps: float = 1e-3, uniform_prob: float = 0.1):
        self.std = std
        self.eps = eps
        self.uniform_prob = uniform_prob
        self.normal_999_percentile = 3.0902 * std
        self.normal_005_percentile = -2.5758 * std

    def sample(self, batch_size: int, seq_length: int, device: torch.device = None) -> torch.Tensor:
        mu = self._get_shift(seq_length)
        normal = torch.randn(batch_size, device=device) * self.std + mu
        logitnormal = torch.sigmoid(normal)
        p999 = torch.sigmoid(torch.tensor(mu + self.normal_999_percentile, device=device))
        p005 = torch.sigmoid(torch.tensor(mu + self.normal_005_percentile, device=device))
        stretched = (logitnormal - p005) / (p999 - p005)
        stretched = torch.where(stretched >= self.eps, stretched, 2 * self.eps - stretched)
        stretched = stretched.clamp(0, 1)
        uniform = (1 - self.eps) * torch.rand(batch_size, device=device) + self.eps
        prob = torch.rand(batch_size, device=device)
        return torch.where(prob > self.uniform_prob, stretched, uniform)

    @staticmethod
    def _get_shift(seq_length, min_tok=1024, max_tok=4096, min_s=0.95, max_s=2.05):
        m = (max_s - min_s) / (max_tok - min_tok)
        return m * seq_length + (min_s - m * min_tok)


class DistilledTimestepSampler:
    SIGMAS = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]

    def __init__(self, jitter: float = 0.02):
        self.jitter = jitter

    def sample(self, batch_size: int, seq_length: int = None, device: torch.device = None) -> torch.Tensor:
        n_intervals = len(self.SIGMAS) - 1
        interval_idx = torch.randint(0, n_intervals, (batch_size,), device=device)
        t = torch.rand(batch_size, device=device)
        sigma_high = torch.tensor([self.SIGMAS[i] for i in interval_idx], device=device)
        sigma_low = torch.tensor([self.SIGMAS[i + 1] for i in interval_idx], device=device)
        sigma = sigma_low + t * (sigma_high - sigma_low)
        return sigma.clamp(0.01, 0.99)


# ── Dataset ────────────────────────────────────────────────────────────

class DramaBoxFinetuneDataset(Dataset):
    """Three-mode dataset for DramaBox voice cloning fine-tuning.

    Modes:
        A (voice_clone_fwd): target=part2, ref=part1, text=scene2_expected_text
        B (voice_clone_rev): target=part1, ref=part2, text=scene1_expected_text
        C (unconditional):   target=full,  ref=None,  text=full_prompt
    """

    def __init__(self, preprocessed_dir: str, mode_weights: dict = None,
                 max_ref_tokens: int = 250, overfit_n: int = 0,
                 expand_all_modes: bool = False, source_weights: dict = None,
                 index_file: str = None):
        self.data_dir = Path(preprocessed_dir)
        self.latent_dir = self.data_dir / "audio_latents"
        self.cond_dir = self.data_dir / "conditions"

        self.max_ref_tokens = max_ref_tokens
        self.expand_all_modes = expand_all_modes
        self.source_weights = source_weights  # e.g. {"dramabox": 0.4, "emolia": 0.2, "podcast": 0.4}
        self.mode_weights = mode_weights or {
            "voice_clone_fwd": 0.33,
            "voice_clone_rev": 0.33,
            "unconditional": 0.34,
        }

        # Load index
        idx_path = self.data_dir / (index_file or "index.json")
        with open(idx_path) as f:
            index = json.load(f)

        self.groups = index.get("groups", {})  # prompt_id -> [sample_indices]
        self.group_keys = list(self.groups.keys())
        self.all_samples = index["samples"]  # list of sample metadata

        # Filter to requested sources only
        if self.source_weights:
            allowed = set(self.source_weights.keys())
            self.all_samples = [s for s in self.all_samples
                                if s.get("source", "dramabox") in allowed]
            valid_idx = {s["index"] for s in self.all_samples}
            self.groups = {k: [i for i in v if i in valid_idx]
                          for k, v in self.groups.items()}
            self.groups = {k: v for k, v in self.groups.items() if v}
            self.group_keys = list(self.groups.keys())

        if overfit_n > 0:
            # Limit to first N groups for overfitting test
            self.group_keys = self.group_keys[:overfit_n]
            valid_indices = set()
            for k in self.group_keys:
                valid_indices.update(self.groups[k])
            self.all_samples = [s for s in self.all_samples if s["index"] in valid_indices]

        # Build source lookup
        self._idx_to_source = {s["index"]: s.get("source", "dramabox")
                               for s in self.all_samples}

        # Build flat sample list
        self._build_items()

        # Log per-source item counts
        if self.source_weights:
            src_counts = defaultdict(int)
            for item_key, mode in self.items:
                src = self._idx_to_source.get(item_key, "unknown")
                src_counts[src] += 1
            logging.info(f"Dataset: {len(self.items)} weighted items, "
                         f"per-source: {dict(src_counts)}")
        else:
            logging.info(f"Dataset: {len(self.items)} items from "
                         f"{len(self.group_keys)} groups, "
                         f"{len(self.all_samples)} total samples")

    def _build_items(self):
        """Build flat item list with optional source-weighted proportions.

        If expand_all_modes=True: 3 modes per individual sample (all modes
        for every sample in every epoch). Items are (sample_index, mode).
        Otherwise: 3 modes per group with random sample selection.
        Items are (group_key, mode).

        Source-aware: Emolia samples (source="emolia") always use only
        voice_clone_fwd mode since their pair direction is pre-defined.
        DramaBox/podcast samples get all 3 modes when expand_all_modes is set.

        If source_weights is set, builds a proportionally-weighted item list
        so that each source contributes the specified fraction of items.
        """
        modes = list(self.mode_weights.keys())

        # Build per-source item lists
        source_items = defaultdict(list)
        if self.expand_all_modes:
            for sample in self.all_samples:
                source = sample.get("source", "dramabox")
                if source == "emolia":
                    source_items[source].append((sample["index"], "voice_clone_fwd"))
                else:
                    for mode in modes:
                        source_items[source].append((sample["index"], mode))
        else:
            for group_key in self.group_keys:
                # Determine source from first sample in group
                if self.groups[group_key]:
                    first_idx = self.groups[group_key][0]
                    source = self._idx_to_source.get(first_idx, "dramabox")
                else:
                    source = "dramabox"
                for mode in modes:
                    source_items[source].append((group_key, mode))

        if self.source_weights and len(source_items) > 1:
            # Weighted proportional item list
            # Find epoch size: largest (source_items / weight) determines it
            epoch_size = 0
            for src, weight in self.source_weights.items():
                if src in source_items and weight > 0:
                    needed = len(source_items[src]) / weight
                    epoch_size = max(epoch_size, needed)
            epoch_size = int(epoch_size)

            self.items = []
            for src, weight in self.source_weights.items():
                if src not in source_items or weight <= 0:
                    continue
                target_count = int(epoch_size * weight)
                src_list = source_items[src]
                # Repeat source items to fill target count
                full_repeats = target_count // len(src_list)
                remainder = target_count % len(src_list)
                for _ in range(full_repeats):
                    self.items.extend(src_list)
                if remainder > 0:
                    self.items.extend(random.sample(src_list, remainder))
        else:
            # Original behavior: concatenate all sources
            self.items = []
            for items in source_items.values():
                self.items.extend(items)

    def __len__(self):
        return len(self.items)

    def _load_latent(self, idx: int, part: str) -> torch.Tensor:
        path = self.latent_dir / f"sample_{idx:06d}_{part}.pt"
        lat = torch.load(path, weights_only=True).detach()
        return lat

    def _load_condition(self, idx: int, part: str):
        path = self.cond_dir / f"sample_{idx:06d}_{part}.pt"
        cond = torch.load(path, weights_only=False)
        audio_feats = cond.get("audio_prompt_embeds", cond.get("prompt_embeds")).detach()
        attn_mask = cond.get("prompt_attention_mask").detach()

        # Pad to multiple of 128 for audio_connector
        REG = 128
        L = audio_feats.shape[0]
        target_L = ((L + REG - 1) // REG) * REG
        if target_L != L:
            pad_len = target_L - L
            pad_emb = torch.zeros(pad_len, audio_feats.shape[1], dtype=audio_feats.dtype)
            pad_mask = torch.zeros(pad_len, dtype=attn_mask.dtype)
            audio_feats = torch.cat([pad_emb, audio_feats], dim=0)
            attn_mask = torch.cat([pad_mask, attn_mask], dim=0)

        return audio_feats, attn_mask

    def __getitem__(self, idx):
        try:
            return self._load_item(idx)
        except Exception as e:
            # On corrupted file, fall back to a random valid item
            logging.warning(f"Failed to load item {idx}: {e}, retrying random")
            for _ in range(5):
                alt = random.randint(0, len(self.items) - 1)
                try:
                    return self._load_item(alt)
                except Exception:
                    continue
            raise RuntimeError(f"Cannot load any item after 5 retries")

    def _load_item(self, idx):
        item_key, mode = self.items[idx]

        if self.expand_all_modes:
            # item_key is a direct sample index
            sample_idx = item_key
        else:
            # item_key is a group key; pick random sample from group
            sample_indices = self.groups[item_key]
            sample_idx = random.choice(sample_indices)

        if mode == "voice_clone_fwd":
            # Target = part2, Reference = part1
            tgt_latent = self._load_latent(sample_idx, "part2")
            ref_latent = self._load_latent(sample_idx, "part1")
            audio_feats, attn_mask = self._load_condition(sample_idx, "part2")
        elif mode == "voice_clone_rev":
            # Target = part1, Reference = part2
            tgt_latent = self._load_latent(sample_idx, "part1")
            ref_latent = self._load_latent(sample_idx, "part2")
            audio_feats, attn_mask = self._load_condition(sample_idx, "part1")
        else:  # unconditional
            # Target = full, no reference
            tgt_latent = self._load_latent(sample_idx, "full")
            C, F_dim = tgt_latent.shape[0], tgt_latent.shape[2]
            ref_latent = torch.zeros(C, 0, F_dim, dtype=tgt_latent.dtype)
            audio_feats, attn_mask = self._load_condition(sample_idx, "full")

        # Cap reference length
        if ref_latent.shape[1] > self.max_ref_tokens:
            ref_latent = ref_latent[:, :self.max_ref_tokens, :]

        return {
            "tgt_latent": tgt_latent,       # [C=8, T, F=16]
            "ref_latent": ref_latent,        # [C=8, T_ref, F=16] or [C=8, 0, F=16]
            "audio_features": audio_feats,
            "attention_mask": attn_mask,
            "mode": mode,
        }


def collate_fn(batch):
    """Pad variable-length audio to max in batch."""
    max_tgt_T = max(b["tgt_latent"].shape[1] for b in batch)
    max_ref_T = max(b["ref_latent"].shape[1] for b in batch)
    C = batch[0]["tgt_latent"].shape[0]
    F_dim = batch[0]["tgt_latent"].shape[2]

    tgt_list, ref_list, feat_list, mask_list = [], [], [], []
    tgt_lengths, ref_lengths = [], []
    modes = []

    for b in batch:
        tgt = b["tgt_latent"]
        ref = b["ref_latent"]
        tgt_lengths.append(tgt.shape[1])
        ref_lengths.append(ref.shape[1])

        if tgt.shape[1] < max_tgt_T:
            pad = torch.zeros(C, max_tgt_T - tgt.shape[1], F_dim, dtype=tgt.dtype)
            tgt = torch.cat([tgt, pad], dim=1)
        tgt_list.append(tgt)

        if ref.shape[1] < max_ref_T:
            pad = torch.zeros(C, max_ref_T - ref.shape[1], F_dim, dtype=ref.dtype)
            ref = torch.cat([ref, pad], dim=1)
        ref_list.append(ref)

        feat_list.append(b["audio_features"])
        mask_list.append(b["attention_mask"])
        modes.append(b["mode"])

    return {
        "tgt_latent": torch.stack(tgt_list),
        "ref_latent": torch.stack(ref_list),
        "audio_features": torch.stack(feat_list),
        "attention_mask": torch.stack(mask_list),
        "tgt_lengths": torch.tensor(tgt_lengths),
        "ref_lengths": torch.tensor(ref_lengths),
        "modes": modes,
    }


# ── Model Building (from DramaBox train.py) ───────────────────────────

def build_audio_only_model(checkpoint_path, device, dtype):
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
    from ltx_core.loader.registry import DummyRegistry
    from ltx_core.loader.sd_ops import SDOps
    from ltx_core.model.transformer.model import LTXModel, LTXModelType
    from ltx_core.model.transformer.attention import AttentionFunction
    from ltx_core.model.transformer.rope import LTXRopeType

    sd_ops = SDOps("AO").with_matching(
        prefix="model.diffusion_model."
    ).with_replacement("model.diffusion_model.", "")

    class Cfg:
        @classmethod
        def from_config(cls, config):
            from ltx_core.model.model_protocol import ModelConfigurator
            t = config.get("transformer", {})
            cp = None
            if not t.get("caption_proj_before_connector", False):
                from ltx_core.model.transformer.text_projection import create_caption_projection
                with torch.device("meta"):
                    cp = create_caption_projection(t, audio=True)
            return LTXModel(
                model_type=LTXModelType.AudioOnly,
                audio_num_attention_heads=t.get("audio_num_attention_heads", 32),
                audio_attention_head_dim=t.get("audio_attention_head_dim", 64),
                audio_in_channels=t.get("audio_in_channels", 128),
                audio_out_channels=t.get("audio_out_channels", 128),
                num_layers=t.get("num_layers", 48),
                audio_cross_attention_dim=t.get("audio_cross_attention_dim", 2048),
                norm_eps=t.get("norm_eps", 1e-6),
                attention_type=AttentionFunction(t.get("attention_type", "default")),
                positional_embedding_theta=t.get("positional_embedding_theta", 10000.0),
                audio_positional_embedding_max_pos=t.get("audio_positional_embedding_max_pos", [20]),
                timestep_scale_multiplier=t.get("timestep_scale_multiplier", 1000),
                use_middle_indices_grid=t.get("use_middle_indices_grid", True),
                rope_type=LTXRopeType(t.get("rope_type", "interleaved")),
                double_precision_rope=t.get("frequencies_precision", False) == "float64",
                apply_gated_attention=t.get("apply_gated_attention", False),
                audio_caption_projection=cp,
                cross_attention_adaln=t.get("cross_attention_adaln", False),
            )

    builder = Builder(model_path=checkpoint_path, model_class_configurator=Cfg,
                      model_sd_ops=sd_ops, registry=DummyRegistry())
    return builder.build(device=device, dtype=dtype)


def load_audio_connector(checkpoint_path, device, dtype):
    from ltx_trainer.model_loader import load_embeddings_processor
    emb_proc = load_embeddings_processor(checkpoint_path, device=device, dtype=dtype)
    connector = emb_proc.audio_connector
    del emb_proc
    return connector


def apply_lora(model, rank, alpha, dropout=0.0):
    from peft import LoraConfig, get_peft_model
    config = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=dropout, bias="none",
        target_modules=[
            "audio_attn1.to_k", "audio_attn1.to_q",
            "audio_attn1.to_v", "audio_attn1.to_out.0",
            "audio_ff.net.0.proj", "audio_ff.net.2",
        ],
    )
    model = get_peft_model(model, config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logging.info(f"LoRA: {trainable:,} trainable / {total:,} total ({100*trainable/total:.1f}%)")
    return model


@torch.no_grad()
def prepare_audio_context(audio_connector, audio_features, attention_mask, device, dtype):
    from ltx_core.text_encoders.gemma.embeddings_processor import convert_to_additive_mask
    audio_features = audio_features.to(device=device, dtype=dtype)
    attention_mask = attention_mask.to(device=device)
    if audio_features.shape[0] > 1:
        results = []
        for i in range(audio_features.shape[0]):
            feat_i = audio_features[i:i+1]
            mask_i = attention_mask[i:i+1]
            additive = convert_to_additive_mask(mask_i, feat_i.dtype)
            enc_i, _ = audio_connector(feat_i, additive)
            results.append(enc_i)
        return torch.cat(results, dim=0)
    additive_mask = convert_to_additive_mask(attention_mask, audio_features.dtype)
    audio_encoded, _ = audio_connector(audio_features, additive_mask)
    return audio_encoded


def _unwrap_model_safe(model):
    while hasattr(model, "module"):
        model = model.module
    return model


def save_training_state(output_dir, step, epoch, optimizer, scheduler, best_loss,
                        best_step, model, accelerator, tag=""):
    """Save full training state for resumability."""
    suffix = f"_epoch{epoch}" if tag == "" else f"_{tag}"
    state = {
        "step": step,
        "epoch": epoch,
        "best_loss": best_loss,
        "best_step": best_step,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng_python": random.getstate(),
        "rng_torch": torch.random.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state(),
    }
    state_path = os.path.join(output_dir, f"training_state{suffix}.pt")
    torch.save(state, state_path)
    # Save LoRA weights alongside
    unwrapped = _unwrap_model_safe(model)
    unwrapped.save_pretrained(output_dir)
    adapter = os.path.join(output_dir, "adapter_model.safetensors")
    lora_path = os.path.join(output_dir, f"lora_epoch{epoch}.safetensors")
    if os.path.exists(adapter):
        shutil.copy(adapter, lora_path)
    logging.info(f"Saved full training state: {state_path}, LoRA: {lora_path}")
    return lora_path


def load_training_state(resume_dir, optimizer, scheduler):
    """Load full training state from checkpoint directory."""
    # Find latest state file
    state_files = sorted(glob_mod.glob(os.path.join(resume_dir, "training_state*.pt")))
    if not state_files:
        raise FileNotFoundError(f"No training state found in {resume_dir}")
    state_path = state_files[-1]
    logging.info(f"Resuming from: {state_path}")
    state = torch.load(state_path, weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    random.setstate(state["rng_python"])
    torch.random.set_rng_state(state["rng_torch"])
    torch.cuda.set_rng_state(state["rng_cuda"])
    return state["step"], state["epoch"], state["best_loss"], state["best_step"]


def launch_validation_async(script_dir, output_dir, lora_path, epoch, args):
    """Launch epoch validation as a background subprocess."""
    val_script = os.path.join(script_dir, "run_epoch_validation.py")
    if not os.path.exists(val_script):
        logging.warning(f"Validation script not found: {val_script}")
        return None
    cmd = [
        sys.executable, val_script,
        "--lora", lora_path,
        "--epoch", str(epoch),
        "--output-dir", output_dir,
        "--dramabox-dir", args.dramabox_dir,
        "--checkpoint", args.checkpoint,
        "--full-checkpoint", args.full_checkpoint,
        "--gemma-root", getattr(args, "gemma_root",
            "/home/deployer/.cache/dramabox/models--unsloth--gemma-3-12b-it-bnb-4bit/snapshots/826e729dbaeea4ecb143738eed2bcf3539ebf7bf"),
        "--lora-rank", str(args.lora_rank),
        "--val-samples", str(args.val_samples),
        "--val-refs-dir", args.val_refs_dir,
        "--preprocessed-dir", args.preprocessed_dir,
    ]
    log_path = os.path.join(output_dir, f"val_epoch{epoch}.log")
    log_f = open(log_path, "w")
    proc = _subprocess.Popen(cmd, stdout=log_f, stderr=_subprocess.STDOUT,
                             env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
    logging.info(f"Launched validation for epoch {epoch} (PID {proc.pid}), log: {log_path}")
    return proc


# ── Metrics Server ────────────────────────────────────────────────────

class MetricsHandler(http.server.SimpleHTTPRequestHandler):
    """Serves the training monitor HTML and metrics data."""

    metrics_dir = None

    def do_GET(self):
        # Strip query string for route matching
        path = self.path.split("?")[0]

        if path == "/" or path == "/index.html":
            monitor_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "scripts", "dramabox_finetune_monitor.html"
            )
            if os.path.exists(monitor_path):
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                with open(monitor_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404, "Monitor HTML not found")
        elif path == "/metrics":
            metrics_path = os.path.join(self.metrics_dir, "metrics.jsonl")
            if os.path.exists(metrics_path):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                with open(metrics_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(b"")
        elif path == "/status":
            status_path = os.path.join(self.metrics_dir, "status.json")
            if os.path.exists(status_path):
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                with open(status_path, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404)
        elif path.startswith("/val/"):
            # Serve validation HTML and audio files
            rel = path[5:]  # strip /val/
            fpath = os.path.join(self.metrics_dir, "val", rel)
            if os.path.exists(fpath):
                self.send_response(200)
                if fpath.endswith(".html"):
                    ct = "text/html"
                elif fpath.endswith(".wav"):
                    ct = "audio/wav"
                elif fpath.endswith(".json"):
                    ct = "application/json"
                else:
                    ct = "application/octet-stream"
                self.send_header("Content-Type", ct)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                with open(fpath, "rb") as f:
                    self.wfile.write(f.read())
            else:
                self.send_error(404)
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass  # Suppress request logging


def start_metrics_server(output_dir: str, port: int = 8765):
    MetricsHandler.metrics_dir = output_dir
    server = http.server.HTTPServer(("0.0.0.0", port), MetricsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logging.info(f"Training monitor serving on http://0.0.0.0:{port}")
    return server


# ── Args ──────────────────────────────────────────────────────────────

def parse_args():
    import yaml
    cfg_parser = argparse.ArgumentParser(add_help=False)
    cfg_parser.add_argument("--config", default=None)
    cfg_args, remaining = cfg_parser.parse_known_args()

    yaml_defaults = {}
    if cfg_args.config:
        with open(cfg_args.config) as f:
            yaml_defaults = yaml.safe_load(f) or {}
        yaml_defaults = {k.replace("-", "_"): v for k, v in yaml_defaults.items()}

    def _y(name, fallback):
        return yaml_defaults.get(name, fallback)

    p = argparse.ArgumentParser(parents=[cfg_parser], description="DramaBox Voice Cloning LoRA Training")
    p.add_argument("--preprocessed-dir", default=_y("preprocessed_dir", "./finetune_data"))
    p.add_argument("--output-dir", default=_y("output_dir", "./finetune_output"))
    p.add_argument("--dramabox-dir", default=_y("dramabox_dir", "/home/deployer/laion/DramaBox"))
    p.add_argument("--checkpoint", default=_y("checkpoint",
                   "/home/deployer/laion/DramaBox/models/ltx-2.3-22b-dev-audio-only-v13-merged.safetensors"))
    p.add_argument("--full-checkpoint", default=_y("full_checkpoint",
                   "/home/deployer/laion/DramaBox/models/ltx-2.3-22b-dev.safetensors"))
    p.add_argument("--gemma-root", default=_y("gemma_root",
                   "/home/deployer/.cache/dramabox/models--unsloth--gemma-3-12b-it-bnb-4bit/snapshots/826e729dbaeea4ecb143738eed2bcf3539ebf7bf"))
    p.add_argument("--base-model", choices=["distilled", "dev"], default=_y("base_model", "dev"))
    p.add_argument("--lora-rank", type=int, default=_y("lora_rank", 128))
    p.add_argument("--lora-alpha", type=int, default=_y("lora_alpha", 128))
    p.add_argument("--lora-dropout", type=float, default=_y("lora_dropout", 0.05))
    p.add_argument("--resume-lora", default=_y("resume_lora", None))
    p.add_argument("--max-ref-tokens", type=int, default=_y("max_ref_tokens", 250))
    p.add_argument("--text-dropout", type=float, default=_y("text_dropout", 0.1))
    p.add_argument("--steps", type=int, default=_y("steps", 15000))
    p.add_argument("--lr", type=float, default=_y("lr", 5e-5))
    p.add_argument("--lr-scheduler", choices=["cosine", "linear", "constant"],
                   default=_y("lr_scheduler", "cosine"))
    p.add_argument("--batch-size", type=int, default=_y("batch_size", 1))
    p.add_argument("--grad-accum", type=int, default=_y("grad_accum", 4))
    p.add_argument("--max-grad-norm", type=float, default=_y("max_grad_norm", 1.0))
    p.add_argument("--save-every", type=int, default=_y("save_every", 500))
    p.add_argument("--log-every", type=int, default=_y("log_every", 25))
    p.add_argument("--seed", type=int, default=_y("seed", 42))
    p.add_argument("--warmup-steps", type=int, default=_y("warmup_steps", 500))
    p.add_argument("--monitor-port", type=int, default=_y("monitor_port", 8765))
    p.add_argument("--test", action="store_true", help="Quick test: 1 GPU, 100 steps")
    p.add_argument("--overfit", type=int, default=0,
                   help="Overfit on N prompt groups (sanity check)")
    p.add_argument("--expand-all-modes", action="store_true",
                   default=bool(_y("expand_all_modes", False)),
                   help="Use all 3 modes per sample (not per group). "
                        "~3x more items per epoch for better coverage.")
    p.add_argument("--epochs", type=int, default=_y("epochs", 0),
                   help="Train for N epochs (overrides --steps)")
    p.add_argument("--val-samples", type=int, default=_y("val_samples", 10),
                   help="Number of validation samples per epoch")
    p.add_argument("--val-refs-dir", default=_y("val_refs_dir", "/home/deployer/laion/test-refs"),
                   help="Directory with reference audio WAVs for validation")
    p.add_argument("--resume-dir", default=_y("resume_dir", None),
                   help="Resume from full training state (dir with training_state.pt)")
    p.add_argument("--index-file", default=_y("index_file", None),
                   help="Index JSON file name (relative to preprocessed-dir)")
    p.add_argument("--source-weights", default=_y("source_weights", None),
                   help="JSON dict of source weights, e.g. "
                        "'{\"dramabox\":0.4,\"emolia\":0.2,\"podcast\":0.4}'")

    return p.parse_args(remaining)


# ── Main Training Loop ────────────────────────────────────────────────

def main():
    from accelerate import Accelerator
    from accelerate.utils import set_seed

    args = parse_args()
    setup_paths(args.dramabox_dir)

    from audio_conditioning import AudioConditionByReferenceLatent
    from ltx_core.components.patchifiers import AudioPatchifier
    from ltx_core.guidance.perturbations import BatchedPerturbationConfig
    from ltx_core.tools import AudioLatentTools
    from ltx_core.types import AudioLatentShape, LatentState
    from ltx_pipelines.utils.helpers import modality_from_latent_state

    if args.test:
        args.steps = min(args.steps, 100)
        args.save_every = 50
        args.log_every = 5

    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum,
        mixed_precision="bf16",
    )

    is_main = accelerator.is_main_process
    if is_main:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

    set_seed(args.seed)
    device = accelerator.device
    dtype = torch.bfloat16

    os.makedirs(args.output_dir, exist_ok=True)

    # Start metrics server on main process
    if is_main:
        try:
            start_metrics_server(args.output_dir, args.monitor_port)
        except OSError as e:
            logging.warning(f"Could not start metrics server: {e}")

    # Save training args
    if is_main:
        import yaml
        args_dict = vars(args).copy()
        args_dict["_meta"] = {
            "world_size": accelerator.num_processes,
            "dtype": str(dtype),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "script": "dramabox_finetune_train.py",
            "pattern": "IC-LoRA 3-mode voice cloning",
        }
        with open(os.path.join(args.output_dir, "training_args.yaml"), "w") as f:
            yaml.dump(args_dict, f, default_flow_style=False, sort_keys=False)

    # Build model
    if is_main:
        logging.info("Loading audio-only model...")
    model = build_audio_only_model(args.checkpoint, device, dtype)

    if is_main:
        logging.info("Loading audio connector...")
    audio_connector = load_audio_connector(args.full_checkpoint, device, dtype)
    audio_connector.eval()
    for p in audio_connector.parameters():
        p.requires_grad = False

    if is_main:
        logging.info(f"Applying LoRA (rank={args.lora_rank}, alpha={args.lora_alpha})...")
    model = apply_lora(model, args.lora_rank, args.lora_alpha, args.lora_dropout)

    # Resume LoRA
    if args.resume_lora:
        from safetensors.torch import load_file as st_load
        if is_main:
            logging.info(f"Resuming from: {args.resume_lora}")
        lora_sd = st_load(args.resume_lora)
        mapped = {}
        for k, v in lora_sd.items():
            nk = k.replace(".lora_A.weight", ".lora_A.default.weight").replace(
                ".lora_B.weight", ".lora_B.default.weight")
            mapped[nk] = v
        model.load_state_dict(mapped, strict=False)

    model.train()
    model.base_model.model.set_gradient_checkpointing(True)

    # Dataset
    mode_weights = {
        "voice_clone_fwd": 0.33,
        "voice_clone_rev": 0.33,
        "unconditional": 0.34,
    }
    expand_all = getattr(args, "expand_all_modes", False)

    # Parse source weights if provided
    source_weights = None
    if args.source_weights:
        if isinstance(args.source_weights, str):
            source_weights = json.loads(args.source_weights)
        else:
            source_weights = args.source_weights
        if is_main:
            logging.info(f"Source weights: {source_weights}")

    dataset = DramaBoxFinetuneDataset(
        preprocessed_dir=args.preprocessed_dir,
        mode_weights=mode_weights,
        max_ref_tokens=args.max_ref_tokens,
        overfit_n=args.overfit,
        expand_all_modes=expand_all,
        source_weights=source_weights,
        index_file=getattr(args, "index_file", None),
    )

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True, collate_fn=collate_fn,
    )

    # Compute epochs -> steps
    steps_per_epoch = 0
    if args.epochs > 0 and not args.overfit:
        items_per_gpu = math.ceil(len(dataset) / max(accelerator.num_processes, 1))
        forward_per_epoch = items_per_gpu  # batch_size=1
        steps_per_epoch = math.ceil(forward_per_epoch / args.grad_accum)
        args.steps = steps_per_epoch * args.epochs
        if is_main:
            logging.info(f"Epoch-based: {args.epochs} epochs × {steps_per_epoch} steps/epoch "
                         f"= {args.steps} total optimizer steps")
            logging.info(f"  ({len(dataset)} items, {accelerator.num_processes} GPUs, "
                         f"grad_accum={args.grad_accum})")

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01,
    )

    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, ConstantLR
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=args.warmup_steps)
    remaining = args.steps - args.warmup_steps
    if args.lr_scheduler == "cosine":
        hold_steps = max(remaining // 5, 0)
        decay_steps = max(remaining - hold_steps, 1)
        hold_sched = ConstantLR(optimizer, factor=1.0, total_iters=hold_steps)
        decay_sched = CosineAnnealingLR(optimizer, T_max=decay_steps, eta_min=1e-6)
        scheduler = SequentialLR(
            optimizer,
            [warmup, hold_sched, decay_sched],
            milestones=[args.warmup_steps, args.warmup_steps + hold_steps],
        )
    elif args.lr_scheduler == "linear":
        main_sched = LinearLR(optimizer, start_factor=1.0, end_factor=0.01,
                              total_iters=max(remaining, 1))
        scheduler = SequentialLR(optimizer, [warmup, main_sched], milestones=[args.warmup_steps])
    else:
        main_sched = ConstantLR(optimizer, factor=1.0, total_iters=max(remaining, 1))
        scheduler = SequentialLR(optimizer, [warmup, main_sched], milestones=[args.warmup_steps])

    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

    # Resume from full training state
    resume_step = 0
    resume_epoch = 0
    if args.resume_dir and is_main:
        try:
            resume_step, resume_epoch, best_loss_r, best_step_r = load_training_state(
                args.resume_dir, optimizer, scheduler)
            logging.info(f"Resumed: step={resume_step}, epoch={resume_epoch}, "
                         f"best_loss={best_loss_r:.4f}, best_step={best_step_r}")
        except Exception as e:
            logging.error(f"Failed to resume: {e}")
            resume_step = 0
            resume_epoch = 0

    patchifier = AudioPatchifier(patch_size=1)

    # Timestep sampler
    if args.base_model == "distilled":
        timestep_sampler = DistilledTimestepSampler()
    else:
        timestep_sampler = ShiftedLogitNormalTimestepSampler()

    # Load silence frame for padding
    silence_frame = None
    sf_path = os.path.join(args.dramabox_dir, "assets", "silence_latent_frame.pt")
    if os.path.exists(sf_path):
        silence_frame = torch.load(sf_path, weights_only=True)
        if is_main:
            logging.info(f"Loaded silence latent from {sf_path}")

    # Metrics file
    metrics_path = os.path.join(args.output_dir, "metrics.jsonl")

    if is_main:
        logging.info(f"Training: {args.steps} steps, lr={args.lr}, scheduler={args.lr_scheduler}, "
                     f"batch={args.batch_size}, grad_accum={args.grad_accum}, "
                     f"world_size={accelerator.num_processes}, "
                     f"max_ref_tokens={args.max_ref_tokens}")
        logging.info(f"3-mode IC-LoRA: voice_clone_fwd/rev + unconditional")
        if args.overfit:
            logging.info(f"OVERFIT MODE: training on {args.overfit} groups only")

    data_iter = iter(dataloader)
    step = 0
    accum_loss = 0.0
    best_loss = float("inf")
    best_step = 0
    t0 = time.time()
    mode_counts = defaultdict(int)
    current_epoch = 0
    last_val_epoch = -1
    val_procs = []
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Handle resume
    if resume_step > 0:
        step = resume_step
        current_epoch = resume_epoch
        if 'best_loss_r' in dir():
            best_loss = best_loss_r
            best_step = best_step_r
        if is_main:
            logging.info(f"Skipping to step {step}, epoch {current_epoch}")

    total_micro_steps = args.steps * args.grad_accum
    start_micro = step * args.grad_accum

    for micro_step in range(start_micro, total_micro_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        is_opt_step = (micro_step + 1) % args.grad_accum == 0
        if is_opt_step:
            step += 1

        with accelerator.accumulate(model):
            tgt_latent = batch["tgt_latent"].to(dtype=dtype)  # [B, C, T, F]
            ref_latent = batch["ref_latent"].to(dtype=dtype)
            tgt_lengths = batch["tgt_lengths"].to(device=device)
            B = tgt_latent.shape[0]

            # Track mode distribution
            for m in batch["modes"]:
                mode_counts[m] += 1

            # Random silence padding (0-25 frames)
            max_pad = 25
            pad_frames = random.randint(0, max_pad)
            if pad_frames > 0:
                C, F_dim = tgt_latent.shape[1], tgt_latent.shape[3]
                if silence_frame is not None:
                    sf = silence_frame.to(dtype=dtype, device=device)
                    silence_pad = sf.unsqueeze(0).expand(B, -1, pad_frames, -1)
                else:
                    silence_pad = torch.zeros(B, C, pad_frames, F_dim, dtype=dtype, device=device)
                tgt_latent = torch.cat([silence_pad, tgt_latent], dim=2)

            # Cap reference tokens
            ref_T_frames = min(ref_latent.shape[2], args.max_ref_tokens)
            ref_latent = ref_latent[:, :, :ref_T_frames, :]
            tgt_T_frames = tgt_latent.shape[2]

            # Create target state
            tgt_shape = AudioLatentShape(
                batch=B, channels=tgt_latent.shape[1],
                frames=tgt_T_frames, mel_bins=tgt_latent.shape[3],
            )
            audio_tools = AudioLatentTools(patchifier=patchifier, target_shape=tgt_shape)
            state = audio_tools.create_initial_state(device=device, dtype=dtype, initial_latent=tgt_latent)
            tgt_T = audio_tools.target_shape.token_count()

            # Sample noise + sigma
            total_tokens = tgt_T + ref_T_frames
            sigma = timestep_sampler.sample(B, total_tokens, device=device)
            sigma_exp = sigma.view(-1, 1, 1)

            noise = torch.randn_like(state.latent)
            noisy_tgt = (1 - sigma_exp) * state.latent + sigma_exp * noise

            state = LatentState(
                latent=noisy_tgt,
                denoise_mask=state.denoise_mask,
                positions=state.positions,
                clean_latent=state.clean_latent,
                attention_mask=state.attention_mask,
            )

            # Append reference (skip if ref_T=0 for unconditional mode)
            if ref_T_frames > 0:
                ref_conditioning = AudioConditionByReferenceLatent(
                    latent=ref_latent, strength=1.0,
                )
                state = ref_conditioning.apply_to(latent_state=state, latent_tools=audio_tools)

            # Loss mask
            loss_mask = torch.zeros(B, tgt_T, device=device)
            for b_idx in range(B):
                real_len = min(tgt_lengths[b_idx].item() + pad_frames, tgt_T)
                loss_mask[b_idx, :real_len] = 1.0

            # Text context
            with torch.no_grad():
                audio_context = prepare_audio_context(
                    audio_connector, batch["audio_features"],
                    batch["attention_mask"], device, dtype)
                if args.text_dropout > 0 and random.random() < args.text_dropout:
                    audio_context = torch.zeros_like(audio_context)

            # Build modality
            audio_mod = modality_from_latent_state(
                state=state, context=audio_context, sigma=sigma, enabled=True,
            )

            # Forward pass
            perturbations = BatchedPerturbationConfig.empty(B)
            with torch.autocast(device_type="cuda", dtype=dtype):
                _, velocity_pred = model(video=None, audio=audio_mod, perturbations=perturbations)

            # Loss (IC-LoRA: only on target tokens)
            tgt_patchified = audio_tools.patchifier.patchify(tgt_latent)
            target_velocity = noise - tgt_patchified
            pred_tgt = velocity_pred[:, :tgt_T]
            per_token_mse = (pred_tgt - target_velocity).pow(2).mean(dim=-1)
            loss = per_token_mse.mul(loss_mask).div(loss_mask.mean().clamp(min=1e-6)).mean()

            accelerator.backward(loss)

            if accelerator.sync_gradients and args.max_grad_norm > 0:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            optimizer.zero_grad()
            if accelerator.sync_gradients:
                scheduler.step()

        accum_loss += loss.item()

        # Logging
        if is_opt_step and step % args.log_every == 0 and is_main:
            avg_loss = accum_loss / (args.log_every * args.grad_accum)
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            sps = step / elapsed if elapsed > 0 else 0
            eta = (args.steps - step) / sps if sps > 0 else 0

            total_modes = sum(mode_counts.values()) or 1
            mode_pcts = {k: f"{100*v/total_modes:.0f}%" for k, v in mode_counts.items()}

            logging.info(
                f"Step {step}/{args.steps} | loss={avg_loss:.4f} | lr={lr:.2e} | "
                f"tgt_T={tgt_T} ref_T={ref_T_frames} | "
                f"{sps:.2f} steps/s | ETA {eta/60:.0f}min | modes={mode_pcts}"
            )

            # Write metrics
            metric = {
                "step": step,
                "loss": round(avg_loss, 6),
                "lr": lr,
                "tgt_tokens": tgt_T,
                "ref_tokens": ref_T_frames,
                "steps_per_sec": round(sps, 3),
                "elapsed_sec": round(elapsed, 1),
                "eta_sec": round(eta, 1),
                "mode_counts": dict(mode_counts),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            with open(metrics_path, "a") as f:
                f.write(json.dumps(metric) + "\n")

            # Update status file
            status = {
                "step": step,
                "total_steps": args.steps,
                "epoch": current_epoch,
                "total_epochs": args.epochs if args.epochs > 0 else 0,
                "steps_per_epoch": steps_per_epoch,
                "loss": round(avg_loss, 6),
                "best_loss": round(best_loss, 6),
                "best_step": best_step,
                "lr": lr,
                "elapsed_sec": round(elapsed, 1),
                "eta_sec": round(eta, 1),
                "steps_per_sec": round(sps, 3),
                "world_size": accelerator.num_processes,
                "mode_counts": dict(mode_counts),
            }
            with open(os.path.join(args.output_dir, "status.json"), "w") as f:
                json.dump(status, f, indent=2)

            # Best checkpoint
            if avg_loss < best_loss:
                best_loss = avg_loss
                old_best = os.path.join(args.output_dir, f"best_step_{best_step:05d}.safetensors")
                best_step = step
                new_best = os.path.join(args.output_dir, f"best_step_{best_step:05d}.safetensors")
                unwrapped = _unwrap_model_safe(model)
                unwrapped.save_pretrained(args.output_dir)
                adapter = os.path.join(args.output_dir, "adapter_model.safetensors")
                if os.path.exists(adapter):
                    shutil.copy(adapter, new_best)
                if old_best != new_best and os.path.exists(old_best):
                    os.remove(old_best)
                logging.info(f"  New best: loss={best_loss:.4f} at step {best_step}")

            accum_loss = 0.0

        # Periodic save (with optimizer state for resumability)
        if is_opt_step and step % args.save_every == 0 and is_main:
            save_path = os.path.join(args.output_dir, f"lora_step_{step:05d}.safetensors")
            logging.info(f"Saving: {save_path}")
            unwrapped = _unwrap_model_safe(model)
            unwrapped.save_pretrained(args.output_dir)
            adapter = os.path.join(args.output_dir, "adapter_model.safetensors")
            if os.path.exists(adapter):
                shutil.copy(adapter, save_path)
            # Save optimizer + scheduler state (per-step for full resumability)
            opt_state = {
                "step": step,
                "epoch": current_epoch,
                "best_loss": best_loss,
                "best_step": best_step,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rng_python": random.getstate(),
                "rng_torch": torch.random.get_rng_state(),
                "rng_cuda": torch.cuda.get_rng_state(),
            }
            state_path = os.path.join(args.output_dir, f"training_state_step_{step:05d}.pt")
            torch.save(opt_state, state_path)
            # Also keep a "latest" symlink for convenience
            latest_path = os.path.join(args.output_dir, "training_state.pt")
            if os.path.islink(latest_path) or os.path.exists(latest_path):
                os.remove(latest_path)
            os.symlink(os.path.basename(state_path), latest_path)

        # Epoch boundary detection
        if is_opt_step and steps_per_epoch > 0 and is_main:
            new_epoch = step // steps_per_epoch
            if new_epoch > current_epoch:
                current_epoch = new_epoch
                logging.info(f"{'='*60}")
                logging.info(f"EPOCH {current_epoch}/{args.epochs} COMPLETE (step {step})")
                logging.info(f"{'='*60}")
                # Save full state for this epoch
                lora_path = save_training_state(
                    args.output_dir, step, current_epoch, optimizer,
                    scheduler, best_loss, best_step, model, accelerator)
                # Launch validation in background
                if args.val_samples > 0:
                    proc = launch_validation_async(
                        script_dir, args.output_dir, lora_path,
                        current_epoch, args)
                    if proc:
                        val_procs.append(proc)

    # Final save
    if is_main:
        final_epoch = args.epochs if args.epochs > 0 else 0
        lora_path = save_training_state(
            args.output_dir, step, final_epoch, optimizer,
            scheduler, best_loss, best_step, model, accelerator, tag="final")
        logging.info(f"Training complete! {step} steps in {time.time()-t0:.0f}s")
        logging.info(f"Best loss: {best_loss:.4f} at step {best_step}")
        # Launch final validation
        if args.val_samples > 0 and final_epoch > 0:
            proc = launch_validation_async(
                script_dir, args.output_dir, lora_path,
                final_epoch, args)
            if proc:
                val_procs.append(proc)
        # Wait for all validation processes
        for proc in val_procs:
            proc.wait()
            logging.info(f"Validation PID {proc.pid} finished (rc={proc.returncode})")


if __name__ == "__main__":
    main()
