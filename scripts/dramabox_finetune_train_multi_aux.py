#!/usr/bin/env python3
"""
DramaBox Voice Cloning LoRA Fine-Tuning with Multi-Auxiliary Losses (Epochs 15-17)

Extends dramabox_finetune_train_clap.py with THREE auxiliary losses:
  1. CLAP Naturalness — CLAP text similarity (positive-negative) + quality MLP
  2. Centroid Real/Fake — cos(emb, real_centroid) - cos(emb, synth_centroid)
  3. Speaker Similarity — WavLM-SV cosine similarity between ref and pred speaker embs

Each auxiliary loss is individually normalized via EMA-based adaptive coefficients
to have approximately the same magnitude as the flow matching loss.

Usage:
    accelerate launch --num_processes=8 scripts/dramabox_finetune_train_multi_aux.py \
        --config configs/finetune_multi_aux.yaml
"""

import os
import sys

# Filter out conda ml-general paths that break native cuDNN libraries
_ld = os.environ.get("LD_LIBRARY_PATH", "")
if _ld:
    _filtered = [p for p in _ld.split(":") if "ml-general" not in p]
    os.environ["LD_LIBRARY_PATH"] = ":".join(_filtered)

# Auto-accept trust_remote_code for HuggingFace models (non-interactive multi-GPU)
os.environ["HF_HUB_TRUST_REMOTE_CODE"] = "1"
os.environ["TRUST_REMOTE_CODE"] = "1"

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
from collections import defaultdict, deque
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler

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
                 expand_all_modes: bool = False, index_file: str = None):
        self.data_dir = Path(preprocessed_dir)
        self.latent_dir = self.data_dir / "audio_latents"
        self.cond_dir = self.data_dir / "conditions"

        self.max_ref_tokens = max_ref_tokens
        self.expand_all_modes = expand_all_modes
        self.mode_weights = mode_weights or {
            "voice_clone_fwd": 0.33,
            "voice_clone_rev": 0.33,
            "unconditional": 0.34,
        }

        # Load index — support custom index file for filtered datasets
        index_path = index_file if index_file else str(self.data_dir / "index.json")
        with open(index_path) as f:
            index = json.load(f)

        self.groups = index["groups"]  # prompt_id -> [sample_indices]
        self.group_keys = list(self.groups.keys())
        self.all_samples = index["samples"]  # list of sample metadata

        if overfit_n > 0:
            self.group_keys = self.group_keys[:overfit_n]
            valid_indices = set()
            for k in self.group_keys:
                valid_indices.update(self.groups[k])
            self.all_samples = [s for s in self.all_samples if s["index"] in valid_indices]

        self._build_items()

        logging.info(f"Dataset: {len(self.items)} items from {len(self.group_keys)} groups, "
                     f"{len(self.all_samples)} total samples (index: {os.path.basename(index_path)})")

    def _build_items(self):
        """Build flat item list."""
        self.items = []
        modes = list(self.mode_weights.keys())
        if self.expand_all_modes:
            for sample in self.all_samples:
                source = sample.get("source", "dramabox")
                if source in ("emolia", "augmented", "podcast"):
                    # Emolia/augmented/podcast pairs have direction baked in; use fwd only
                    self.items.append((sample["index"], "voice_clone_fwd"))
                else:
                    for mode in modes:
                        self.items.append((sample["index"], mode))
        else:
            for group_key in self.group_keys:
                for mode in modes:
                    self.items.append((group_key, mode))

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
        item_key, mode = self.items[idx]

        if self.expand_all_modes:
            sample_idx = item_key
        else:
            sample_indices = self.groups[item_key]
            sample_idx = random.choice(sample_indices)

        if mode == "voice_clone_fwd":
            tgt_latent = self._load_latent(sample_idx, "part2")
            ref_latent = self._load_latent(sample_idx, "part1")
            audio_feats, attn_mask = self._load_condition(sample_idx, "part2")
        elif mode == "voice_clone_rev":
            tgt_latent = self._load_latent(sample_idx, "part1")
            ref_latent = self._load_latent(sample_idx, "part2")
            audio_feats, attn_mask = self._load_condition(sample_idx, "part1")
        else:  # unconditional
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


# ── Bucket-Weighted Sampler ───────────────────────────────────────────

class BucketWeightedSampler(Sampler):
    """Weighted sampling from source buckets with round-robin within each bucket.

    Divides dataset items into buckets by source (dramabox, podcast, emolia,
    augmented). Each epoch produces items in proportions matching bucket_weights,
    cycling through each bucket's items in shuffled order.

    One "epoch" = each item in the anchor bucket (podcast) seen once.
    Total training length = anchor_items / anchor_weight.
    """

    def __init__(self, dataset, bucket_weights: dict, anchor_bucket: str = "podcast",
                 seed: int = 42):
        """
        Args:
            dataset: DramaBoxFinetuneDataset with expand_all_modes=True
            bucket_weights: source -> weight, e.g. {"dramabox": 0.5, "podcast": 0.35, ...}
            anchor_bucket: which bucket defines one epoch (each item seen once per epoch)
            seed: base random seed
        """
        super().__init__(dataset)
        self.bucket_weights = bucket_weights
        self.seed = seed
        self._epoch = 0

        # Build source -> item_indices mapping
        sample_idx_to_source = {
            s["index"]: s.get("source", "dramabox") for s in dataset.all_samples
        }
        self.buckets = {}
        for source in bucket_weights:
            self.buckets[source] = []
        for i, (item_key, mode) in enumerate(dataset.items):
            source = sample_idx_to_source.get(item_key, "dramabox")
            if source in self.buckets:
                self.buckets[source].append(i)
            else:
                # Unknown source falls into dramabox bucket
                self.buckets.setdefault("dramabox", []).append(i)

        # Epoch size: anchor bucket items seen exactly once per epoch
        anchor_items = len(self.buckets.get(anchor_bucket, []))
        anchor_weight = bucket_weights.get(anchor_bucket, 0.35)
        self.anchor = anchor_bucket
        self._epoch_size = int(anchor_items / anchor_weight) if anchor_items > 0 else len(dataset)

        # Log bucket info
        info = []
        for source, weight in bucket_weights.items():
            n = len(self.buckets.get(source, []))
            per_epoch = int(self._epoch_size * weight)
            repeats = per_epoch / n if n > 0 else 0
            info.append(f"  {source}: {n} items × {repeats:.1f}× = {per_epoch}/epoch (weight={weight})")
        logging.info(f"BucketWeightedSampler: {self._epoch_size} items/epoch, anchor={self.anchor}")
        for line in info:
            logging.info(line)

    def __len__(self):
        return self._epoch_size

    def set_epoch(self, epoch):
        self._epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1  # auto-increment for next iter()

        indices = []
        for source, weight in self.bucket_weights.items():
            bucket = self.buckets.get(source, [])
            if not bucket or weight <= 0:
                continue
            count = int(self._epoch_size * weight)
            # Repeat bucket items enough times, shuffle, take count
            repeats = (count // len(bucket)) + 1
            pool = []
            for _ in range(repeats):
                shuffled = bucket.copy()
                rng.shuffle(shuffled)
                pool.extend(shuffled)
            indices.extend(pool[:count])

        # Final shuffle to interleave all sources
        rng.shuffle(indices)
        return iter(indices)


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


_checkpoint_metadata = {}  # populated during model loading for full FT saves


def _save_full_ft_model(model, accelerator, save_path):
    """Save full fine-tuning model weights.
    Uses unwrapped model state_dict directly (works with DDP and FSDP NO_SHARD).
    Only needs to be called from main process."""
    from safetensors.torch import save_file as st_save
    unwrapped = _unwrap_model_safe(model)
    sd = unwrapped.state_dict()
    prefixed = {}
    for k, v in sd.items():
        if v.is_floating_point():
            v = v.to(torch.bfloat16)
        prefixed[f"model.diffusion_model.{k}"] = v
    st_save(prefixed, save_path, metadata=_checkpoint_metadata or None)


def save_training_state(output_dir, step, epoch, optimizer, scheduler, best_loss,
                        best_step, model, accelerator, tag="", full_ft=False):
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

    if full_ft:
        model_path = os.path.join(output_dir, f"model_epoch{epoch}.safetensors")
        _save_full_ft_model(model, accelerator, model_path)
        logging.info(f"Saved full training state: {state_path}, model: {model_path}")
        return model_path
    else:
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
    """Load full training state from checkpoint directory (picks highest step)."""
    state_files = glob_mod.glob(os.path.join(resume_dir, "training_state*.pt"))
    # Filter out broken symlinks
    state_files = [f for f in state_files if os.path.isfile(f)]
    if not state_files:
        raise FileNotFoundError(f"No training state found in {resume_dir}")
    # Pick the checkpoint with the highest step number
    best_path, best_step = None, -1
    for sf in state_files:
        try:
            s = torch.load(sf, map_location="cpu", weights_only=False)
            if s["step"] > best_step:
                best_step = s["step"]
                best_path = sf
            del s
        except Exception as e:
            logging.warning(f"Skipping unreadable state file {sf}: {e}")
    if best_path is None:
        raise FileNotFoundError(f"No valid training state found in {resume_dir}")
    state_path = best_path
    logging.info(f"Resuming from: {state_path} (step {best_step})")
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
            rel = path[5:]
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
        pass


def start_metrics_server(output_dir: str, port: int = 8765):
    MetricsHandler.metrics_dir = output_dir
    server = http.server.HTTPServer(("0.0.0.0", port), MetricsHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logging.info(f"Training monitor serving on http://0.0.0.0:{port}")
    return server


# ── CLAP + Audio Decoder Setup ────────────────────────────────────────

def setup_clap_models(full_checkpoint, device, dtype):
    """Load AudioDecoder + VoiceCLAP-small for auxiliary losses.

    All models are frozen and in eval mode.
    Returns: (audio_decoder, clap_model, clap_tokenizer)
    """
    from ltx_pipelines.utils.blocks import AudioDecoder

    logging.info("Loading AudioDecoder (VAE decoder + vocoder)...")
    audio_decoder = AudioDecoder(
        checkpoint_path=full_checkpoint,
        dtype=dtype,
        device=device,
        warm=True,
    )

    logging.info("Loading VoiceCLAP-small...")
    from transformers import AutoModel, AutoTokenizer
    clap_model = AutoModel.from_pretrained(
        "laion/voiceclap-small", trust_remote_code=True
    ).eval().to(device)
    clap_tokenizer = AutoTokenizer.from_pretrained("laion/voiceclap-small",
                                                     trust_remote_code=True)

    for p in clap_model.parameters():
        p.requires_grad = False

    logging.info("CLAP + AudioDecoder loaded successfully")
    return audio_decoder, clap_model, clap_tokenizer


def setup_clap_models_large(full_checkpoint, device, dtype, args):
    """Load AudioDecoder + large VoiceCLAP via SentenceTransformer with optional quantization.

    For 7B+ models, INT4 quantization via bitsandbytes keeps VRAM manageable (~4 GB).
    Returns: (audio_decoder, st_model, None) — no separate tokenizer needed.
    """
    from ltx_pipelines.utils.blocks import AudioDecoder

    logging.info("Loading AudioDecoder (VAE decoder + vocoder)...")
    audio_decoder = AudioDecoder(
        checkpoint_path=full_checkpoint,
        dtype=dtype,
        device=device,
        warm=True,
    )

    logging.info(f"Loading large CLAP model: {args.clap_model} (quantize={args.clap_quantize})...")
    from sentence_transformers import SentenceTransformer

    model_kwargs = {"torch_dtype": torch.bfloat16, "trust_remote_code": True}

    if args.clap_quantize in ("int4", "int8"):
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=(args.clap_quantize == "int4"),
            load_in_8bit=(args.clap_quantize == "int8"),
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model_kwargs["quantization_config"] = bnb_config

    st_model = SentenceTransformer(
        args.clap_model,
        model_kwargs=model_kwargs,
        trust_remote_code=True,
    )

    # Log embedding dimension
    test_emb = st_model.encode(["test"], convert_to_tensor=True)
    emb_dim = test_emb.shape[-1]
    logging.info(f"Large CLAP loaded: embedding dim = {emb_dim}")

    for p in st_model.parameters():
        p.requires_grad = False

    logging.info("Large CLAP + AudioDecoder loaded successfully")
    return audio_decoder, st_model, None


def encode_text_st(text, st_model):
    """Encode text with SentenceTransformer CLAP, returns normalized embedding [1, D]."""
    with torch.no_grad():
        emb = st_model.encode([text], convert_to_tensor=True)
    emb = F.normalize(emb, p=2, dim=-1)
    return emb  # [1, D]


def encode_audio_st(waveform_np, sr, st_model, rank=0):
    """Encode audio with SentenceTransformer CLAP via temp file, returns normalized [1, D].

    Args:
        waveform_np: numpy array of audio samples (mono)
        sr: sample rate
        st_model: SentenceTransformer model
        rank: GPU rank for unique temp file naming
    """
    import soundfile as sf
    tmp_path = f"/dev/shm/clap_tmp_{rank}.wav"
    try:
        sf.write(tmp_path, waveform_np, sr)
        with torch.no_grad():
            emb = st_model.encode([{"audio": tmp_path}], convert_to_tensor=True)
        emb = F.normalize(emb, p=2, dim=-1)
        return emb  # [1, D]
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def encode_clap_text(text, clap_model, clap_tokenizer, device):
    """Encode text with VoiceCLAP-small, returns normalized embedding [1, 768]."""
    enc = clap_tokenizer([text], padding=True, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = clap_model.encode_text(enc.input_ids, enc.attention_mask)
    emb = F.normalize(emb, p=2, dim=-1)
    return emb  # [1, 768]


def encode_clap_waveform_differentiable(waveform, clap_model):
    """Differentiable CLAP audio encoding — bypasses @torch.no_grad() on compute_log_mel.

    VoiceCLAP-small's encode_waveform has @torch.no_grad() on its mel computation,
    which breaks gradient flow. This function replicates the mel computation with
    gradients enabled, then feeds into the audio encoder normally.

    Args:
        waveform: [B, T] or [T] at 16kHz, with requires_grad from upstream
        clap_model: VoiceCLAP-small model (frozen but grad flows through ops)

    Returns:
        Normalized audio embedding [B, 768] with gradient chain intact
    """
    _CHUNK_SAMPLES = 30 * 16000  # 480000

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    B, T = waveform.shape
    device = waveform.device

    # Pad to 30s chunks (same as original)
    n_chunks = max(1, (T + _CHUNK_SAMPLES - 1) // _CHUNK_SAMPLES)
    pad = n_chunks * _CHUNK_SAMPLES - T
    if pad > 0:
        waveform = F.pad(waveform, (0, pad))
    chunks = waveform.view(B, n_chunks, _CHUNK_SAMPLES).reshape(B * n_chunks, _CHUNK_SAMPLES)

    # ── Differentiable mel spectrogram (NO @torch.no_grad()) ──
    chunks_f32 = chunks.to(dtype=torch.float32)
    window = torch.hann_window(400, device=device)
    stft = torch.stft(chunks_f32, n_fft=400, hop_length=160, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    # Use CLAP's registered mel_filters buffer
    mel_filters = clap_model.mel_filters.to(magnitudes.dtype)
    mel = mel_filters @ magnitudes

    log_spec = torch.clamp(mel, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.amax(dim=(-2, -1), keepdim=True) - 8.0)
    log_spec = (log_spec + 4.0) / 4.0

    # ── Audio encoder (already differentiable) ──
    feats = clap_model.audio_encoder(log_spec)   # (B*n_chunks, T', D)
    feats = feats.mean(dim=1)                     # clip-level mean
    feats = clap_model.audio_proj(feats)
    feats = F.normalize(feats, dim=-1)

    if n_chunks == 1:
        return feats
    feats = feats.view(B, n_chunks, -1).mean(dim=1)
    return F.normalize(feats, dim=-1)


# ── Speaker Similarity Model Setup ────────────────────────────────────

def setup_speaker_model(device, dtype):
    """Load WavLM-SV for speaker similarity.

    WavLM-base-plus-sv produces 512-dim speaker embeddings. It is a superior
    alternative to ECAPA-TDNN from the same HuggingFace ecosystem with no extra
    dependencies (uses transformers WavLMForXVector class).
    """
    from transformers import WavLMForXVector, Wav2Vec2FeatureExtractor

    logging.info("Loading WavLM-base-plus-sv for speaker similarity...")
    model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv")
    model.eval().to(device=device, dtype=torch.float32)  # keep float32 for stability
    for p in model.parameters():
        p.requires_grad = False

    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("microsoft/wavlm-base-plus-sv")

    logging.info("WavLM-SV loaded successfully (~200MB)")
    return model, feature_extractor


def wavlm_extract_embeddings(model, input_values):
    """Wrapper for WavLM that returns tensor (for grad_checkpoint compatibility)."""
    return model(input_values).embeddings


# ── Centroid + Quality MLP Setup ──────────────────────────────────────

def setup_centroid_and_quality(classifiers_dir, device, dtype):
    """Load centroid embeddings and quality MLP.

    Returns: (real_centroid [1, 768], synth_centroid [1, 768], quality_mlp)
    """
    import torch.nn as nn

    classifiers_dir = Path(classifiers_dir)

    # Centroid embeddings
    logging.info("Loading CLAP centroid embeddings...")
    emb_data = torch.load(classifiers_dir / "clap_embeddings.pt", map_location="cpu",
                          weights_only=False)
    dramabox_embs = emb_data["dramabox_embeddings"]  # [3247, 768]
    emilia_embs = emb_data["emilia_embeddings"]       # [3247, 768]

    # Use 80% for centroids (same train split as classifier training)
    n_train = int(len(dramabox_embs) * 0.8)
    synth_centroid = F.normalize(dramabox_embs[:n_train].float().mean(0, keepdim=True), p=2, dim=-1)
    real_centroid = F.normalize(emilia_embs[:n_train].float().mean(0, keepdim=True), p=2, dim=-1)
    synth_centroid = synth_centroid.to(device=device, dtype=torch.float32)
    real_centroid = real_centroid.to(device=device, dtype=torch.float32)
    logging.info(f"Centroids computed from {n_train} train samples each")

    # Quality MLP
    logging.info("Loading quality classifier MLP...")
    ckpt = torch.load(classifiers_dir / "quality_classifier.pt", map_location="cpu",
                      weights_only=False)

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
    quality_mlp.eval().to(device=device, dtype=torch.float32)  # keep float32 for stability
    for p in quality_mlp.parameters():
        p.requires_grad = False
    logging.info("Quality MLP loaded (102K params)")

    return real_centroid, synth_centroid, quality_mlp


# ── Comb Filter Detector ─────────────────────────────────────────────

class CombFilterDetector(torch.nn.Module):
    """Lightweight CNN to detect comb-filter artifacts in DramaBox latent space.

    Input: [B, 8, T, 16] latent tensor
    Output: [B, 1] (logits)
    """
    def __init__(self, in_channels=8):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32), torch.nn.GELU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(64), torch.nn.GELU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128), torch.nn.GELU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(128), torch.nn.GELU(),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
            torch.nn.Linear(128, 64), torch.nn.GELU(),
            torch.nn.Dropout(0.2), torch.nn.Linear(64, 1),
        )

    def forward(self, x):
        x = x.float()
        return self.classifier(self.features(x))

    def comb_score(self, x):
        """Differentiable comb score: 0=clean, 1=comb-filtered."""
        return torch.sigmoid(self.forward(x)).squeeze(-1)


def setup_comb_detector(ckpt_path, device, label="CombFilterDetector"):
    """Load trained CombFilterDetector from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = CombFilterDetector(in_channels=ckpt.get("in_channels", 8))
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device=device, dtype=torch.float32)
    for p in model.parameters():
        p.requires_grad = False
    n_params = sum(p.numel() for p in model.parameters())
    logging.info(f"{label} loaded: {n_params:,} params, val_acc={ckpt.get('val_acc', 'N/A')}")
    return model


class CLAPArtifactMLP(torch.nn.Module):
    """MLP artifact detector operating on CLAP embeddings (768-dim).
    Auto-detects architecture from checkpoint state dict shape."""
    def __init__(self, layers):
        super().__init__()
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.float())

    def artifact_score(self, x):
        """Differentiable artifact score: 0=clean, 1=artifact."""
        return torch.sigmoid(self.forward(x)).squeeze(-1)


def setup_clap_artifact_mlp(ckpt_path, device):
    """Load CLAP-based artifact detector MLP from checkpoint.

    Checkpoint format: {'state_dict': ..., 'variant': 'small'|'medium'|'large', ...}
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))
    variant = ckpt.get("variant", "unknown")

    # Known architectures keyed by state_dict structure
    architectures = {
        "small": lambda: torch.nn.Sequential(
            torch.nn.Linear(768, 256), torch.nn.GELU(), torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 64), torch.nn.GELU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 1)),
        "medium": lambda: torch.nn.Sequential(
            torch.nn.Linear(768, 512), torch.nn.LayerNorm(512),
            torch.nn.GELU(), torch.nn.Dropout(0.3),
            torch.nn.Linear(512, 256), torch.nn.LayerNorm(256),
            torch.nn.GELU(), torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 64), torch.nn.GELU(), torch.nn.Dropout(0.1),
            torch.nn.Linear(64, 1)),
    }

    model = None

    # Strip "net." prefix if present (saved from CLAPArtifactMLP wrapper)
    if all(k.startswith("net.") for k in sd.keys()):
        sd = {k[len("net."):]: v for k, v in sd.items()}

    # Try variant hint first, then brute-force all
    order = [variant] + [k for k in architectures if k != variant]
    for var in order:
        if var not in architectures:
            continue
        try:
            m = architectures[var]()
            m.load_state_dict(sd)
            model = CLAPArtifactMLP(list(m.children()))
            logging.info(f"CLAP artifact MLP: matched '{var}' architecture")
            break
        except Exception:
            continue

    if model is None:
        raise ValueError(f"Could not load CLAP artifact MLP from {ckpt_path}. "
                         f"variant={variant}, keys={list(sd.keys())[:10]}")

    model.eval().to(device=device, dtype=torch.float32)
    for p in model.parameters():
        p.requires_grad = False
    n_params = sum(p.numel() for p in model.parameters())
    logging.info(f"CLAPArtifactMLP loaded: {n_params:,} params, val_acc={ckpt.get('val_acc', 'N/A'):.4f}")
    return model


# ── Adversarial Training Components ──────────────────────────────────

class _SEBlock(torch.nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(channels, channels // reduction),
            torch.nn.GELU(),
            torch.nn.Linear(channels // reduction, channels),
            torch.nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


class _ResBlock(torch.nn.Module):
    """Residual block with optional channel change + SE attention."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(out_ch)
        self.conv2 = torch.nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = torch.nn.BatchNorm2d(out_ch)
        self.skip = torch.nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else torch.nn.Identity()
        self.se = _SEBlock(out_ch)

    def forward(self, x):
        identity = self.skip(x)
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.gelu(out + identity)


class ArtifactDetectorLarge(torch.nn.Module):
    """Large CNN (~10.6M params) with residual blocks + SE attention.

    Operates on latent space [B, 8, T, 16]. Same task as CombFilterDetector
    but much higher capacity for adversarial co-training.
    """
    def __init__(self, in_channels=8):
        super().__init__()
        self.stem = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 64, 3, padding=1),
            torch.nn.BatchNorm2d(64), torch.nn.GELU(),
        )
        self.blocks = torch.nn.Sequential(
            _ResBlock(64, 64), torch.nn.MaxPool2d(2, 2),
            _ResBlock(64, 128), torch.nn.MaxPool2d(2, 2),
            _ResBlock(128, 256), torch.nn.MaxPool2d(2, 2),
            _ResBlock(256, 384), torch.nn.MaxPool2d(2, 2),
            _ResBlock(384, 512),
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.BatchNorm2d(512), torch.nn.GELU(),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
            torch.nn.Linear(512, 256), torch.nn.GELU(), torch.nn.Dropout(0.3),
            torch.nn.Linear(256, 64), torch.nn.GELU(), torch.nn.Dropout(0.2),
            torch.nn.Linear(64, 1),
        )

    def forward(self, x):
        x = x.float()
        x = self.stem(x)
        x = self.blocks(x)
        return self.classifier(x)

    def comb_score(self, x):
        """Sigmoid output: probability of being an artifact."""
        return torch.sigmoid(self.forward(x)).squeeze(-1)


def setup_adversarial_disc(ckpt_path, device, label="DiscOnline"):
    """Load ArtifactDetectorLarge from checkpoint for adversarial training."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    model = ArtifactDetectorLarge()
    model.load_state_dict(sd, strict=True)
    model.to(device=device, dtype=torch.float32)
    n_params = sum(p.numel() for p in model.parameters())
    val_acc = ckpt.get("best_val_acc", ckpt.get("val_acc", "N/A"))
    logging.info(f"{label} loaded: {n_params:,} params, val_acc={val_acc}")
    return model


def train_discriminator_online(disc, disc_optimizer, real_buffer, fake_buffer,
                                epochs=3, device="cuda"):
    """Train discriminator on accumulated real/fake latent pairs.

    Args:
        disc: ArtifactDetectorLarge in train() mode
        disc_optimizer: Adam optimizer for disc
        real_buffer: list of detached real latent tensors [8, T, 16]
        fake_buffer: list of detached fake latent tensors [8, T, 16]
        epochs: number of training epochs on the buffer
        device: target device
    Returns:
        dict with training stats
    """
    disc.train()
    n_real = len(real_buffer)
    n_fake = len(fake_buffer)
    n_total = n_real + n_fake
    if n_total == 0:
        return {"disc_loss": 0.0, "disc_acc": 0.0, "n_samples": 0}

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for epoch in range(epochs):
        # Shuffle indices
        indices = list(range(n_total))
        random.shuffle(indices)

        for idx in indices:
            if idx < n_real:
                latent = real_buffer[idx].unsqueeze(0).to(device)
                label = torch.zeros(1, 1, device=device)  # real = 0
            else:
                latent = fake_buffer[idx - n_real].unsqueeze(0).to(device)
                label = torch.ones(1, 1, device=device)   # fake = 1

            logit = disc(latent)
            loss = F.binary_cross_entropy_with_logits(logit, label)

            disc_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
            disc_optimizer.step()

            total_loss += loss.item()
            pred = (torch.sigmoid(logit) > 0.5).float()
            total_correct += (pred == label).sum().item()
            total_samples += 1

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    disc.eval()
    return {"disc_loss": avg_loss, "disc_acc": accuracy, "n_samples": n_total,
            "n_real": n_real, "n_fake": n_fake, "epochs": epochs}


def update_ema_weights(ema_model, online_model, decay=0.995):
    """Exponential moving average update: θ_ema = decay * θ_ema + (1-decay) * θ_online."""
    with torch.no_grad():
        for ema_p, online_p in zip(ema_model.parameters(), online_model.parameters()):
            ema_p.data.mul_(decay).add_(online_p.data, alpha=1.0 - decay)
        # Also update buffers (BatchNorm running mean/var)
        for ema_b, online_b in zip(ema_model.buffers(), online_model.buffers()):
            ema_b.data.copy_(online_b.data)


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

    p = argparse.ArgumentParser(parents=[cfg_parser],
                                description="DramaBox LoRA Training with Multi-Auxiliary Losses")
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
    p.add_argument("--full-ft", action="store_true", default=_y("full_ft", False),
                   help="Full fine-tuning (no LoRA). Model loaded from --checkpoint is trained directly.")
    p.add_argument("--lora-rank", type=int, default=_y("lora_rank", 128))
    p.add_argument("--lora-alpha", type=int, default=_y("lora_alpha", 128))
    p.add_argument("--lora-dropout", type=float, default=_y("lora_dropout", 0.05))
    p.add_argument("--resume-lora", default=_y("resume_lora", None))
    p.add_argument("--max-ref-tokens", type=int, default=_y("max_ref_tokens", 250))
    p.add_argument("--text-dropout", type=float, default=_y("text_dropout", 0.1))
    p.add_argument("--steps", type=int, default=_y("steps", 15000))
    p.add_argument("--lr", type=float, default=_y("lr", 3e-5))
    p.add_argument("--lr-scheduler", choices=["cosine", "linear", "constant"],
                   default=_y("lr_scheduler", "cosine"))
    p.add_argument("--batch-size", type=int, default=_y("batch_size", 1))
    p.add_argument("--grad-accum", type=int, default=_y("grad_accum", 4))
    p.add_argument("--max-grad-norm", type=float, default=_y("max_grad_norm", 1.0))
    p.add_argument("--save-every", type=int, default=_y("save_every", 500))
    p.add_argument("--log-every", type=int, default=_y("log_every", 25))
    p.add_argument("--seed", type=int, default=_y("seed", 42))
    p.add_argument("--warmup-steps", type=int, default=_y("warmup_steps", 161))
    p.add_argument("--monitor-port", type=int, default=_y("monitor_port", 8765))
    p.add_argument("--test", action="store_true", help="Quick test: 1 GPU, 100 steps")
    p.add_argument("--overfit", type=int, default=0,
                   help="Overfit on N prompt groups (sanity check)")
    p.add_argument("--expand-all-modes", action="store_true",
                   default=bool(_y("expand_all_modes", False)),
                   help="Use all 3 modes per sample (not per group).")
    p.add_argument("--epochs", type=int, default=_y("epochs", 3),
                   help="Train for N epochs (overrides --steps)")
    p.add_argument("--val-samples", type=int, default=_y("val_samples", 10),
                   help="Number of validation samples per epoch")
    p.add_argument("--val-refs-dir", default=_y("val_refs_dir", "/home/deployer/laion/test-refs"),
                   help="Directory with reference audio WAVs for validation")
    p.add_argument("--resume-dir", default=_y("resume_dir", None),
                   help="Resume from full training state (dir with training_state.pt)")

    # Index file
    p.add_argument("--index-file", default=_y("index_file", None),
                   help="Path to custom index.json (e.g. index_filtered.json)")

    # Bucket-weighted sampling
    bw_default = _y("bucket_weights", None)
    p.add_argument("--bucket-weights", default=bw_default,
                   help="Source->weight dict for bucket sampling (YAML dict or JSON string)")
    p.add_argument("--podcast-epochs", type=int, default=_y("podcast_epochs", 0),
                   help="Train until podcast bucket seen N times (overrides --epochs when > 0)")

    # CLAP / multi-auxiliary loss args
    clap_default = _y("clap_loss", True)
    p.add_argument("--clap-loss", action="store_true", default=clap_default,
                   help="Enable CLAP auxiliary losses (default)")
    p.add_argument("--no-clap", action="store_true", default=False,
                   help="Disable all auxiliary losses (for ablation)")
    p.add_argument("--aux-target-ratio", type=float,
                   default=_y("aux_target_ratio", 1.0),
                   help="Target ratio of EACH aux loss to flow matching (default 1.0)")
    p.add_argument("--speaker-sim-ratio", type=float,
                   default=_y("speaker_sim_ratio", 0.0),
                   help="Override target ratio for speaker similarity loss (0 = use aux-target-ratio)")
    p.add_argument("--coeff-cap", type=float,
                   default=_y("coeff_cap", 10.0),
                   help="Max value for adaptive aux coefficients (default 10.0)")
    p.add_argument("--positive-text", default=_y("positive_text",
                   "Realistic, genuine, spontaneous, authentic, sensual, natural voice "
                   "with all imperfections and organic microdistractions a natural situation brings with it"),
                   help="CLAP positive text")
    p.add_argument("--negative-text", default=_y("negative_text",
                   "distorted, unnatural, robotic, distortion"),
                   help="CLAP negative text")
    p.add_argument("--classifiers-dir", default=_y("classifiers_dir", "./classifiers"),
                   help="Directory with quality_classifier.pt and clap_embeddings.pt")

    p.add_argument("--aux-sigma-max", type=float,
                   default=_y("aux_sigma_max", 0.4),
                   help="Only compute aux losses when sigma < this threshold (default 0.4)")
    p.add_argument("--rejection-sampling", action="store_true",
                   default=_y("rejection_sampling", False),
                   help="Enable rejection sampling: only train on above-median reward samples")
    p.add_argument("--rejection-percentile", type=float,
                   default=_y("rejection_percentile", 50.0),
                   help="Percentile threshold for rejection (default 50 = median)")
    p.add_argument("--differentiable-reward", action="store_true",
                   default=_y("differentiable_reward", False),
                   help="Backprop through decoder/CLAP/WavLM for true differentiable rewards (ReFL-style)")
    p.add_argument("--diff-reward-checkpoint", action="store_true",
                   default=_y("diff_reward_checkpoint", False),
                   help="Use gradient checkpointing on decoder/CLAP for VRAM savings")

    # Individual loss toggles (read defaults from YAML)
    p.add_argument("--no-speaker-sim", action="store_true",
                   default=bool(_y("no_speaker_sim", False)),
                   help="Disable speaker similarity loss")
    p.add_argument("--no-centroid", action="store_true",
                   default=bool(_y("no_centroid", False)),
                   help="Disable centroid real/fake loss")
    p.add_argument("--no-quality-mlp", action="store_true",
                   default=bool(_y("no_quality_mlp", False)),
                   help="Disable quality MLP (keep CLAP text similarity only)")

    # Large CLAP model support
    p.add_argument("--clap-model", default=_y("clap_model", "laion/voiceclap-small"),
                   help="CLAP model name (default: laion/voiceclap-small)")
    p.add_argument("--clap-quantize", default=_y("clap_quantize", "none"),
                   choices=["none", "int8", "int4"],
                   help="Quantization for CLAP model: none/int8/int4 (default: none)")
    p.add_argument("--keep-last-n", type=int, default=_y("keep_last_n", 0),
                   help="Rolling checkpoint window (0 = keep all)")
    p.add_argument("--no-save-state", action="store_true",
                   default=bool(_y("no_save_state", False)),
                   help="Skip saving optimizer state .pt files (saves disk)")

    # Comb filter detector (aux loss 4)
    p.add_argument("--comb-detector-path", default=_y("comb_detector_path", None),
                   help="Path to trained CombFilterDetector .pt checkpoint")
    p.add_argument("--no-comb-detector", action="store_true",
                   default=bool(_y("no_comb_detector", False)),
                   help="Disable comb filter detector aux loss")
    p.add_argument("--comb-target-ratio", type=float,
                   default=_y("comb_target_ratio", 3.0),
                   help="Target ratio for comb detector loss (default 3.0, higher than other aux losses)")

    # Artifact detector V2 CNN (aux loss 5) — second latent-space detector
    p.add_argument("--artifact-v2-path", default=_y("artifact_v2_path", None),
                   help="Path to artifact detector V2 .pt checkpoint (latent-space CNN)")
    p.add_argument("--artifact-v2-ratio", type=float,
                   default=_y("artifact_v2_ratio", 3.0),
                   help="Target ratio for artifact V2 detector")

    # CLAP-based artifact MLP (aux loss 6)
    p.add_argument("--clap-artifact-path", default=_y("clap_artifact_path", None),
                   help="Path to CLAP-based artifact detector MLP .pt checkpoint")
    p.add_argument("--clap-artifact-ratio", type=float,
                   default=_y("clap_artifact_ratio", 3.0),
                   help="Target ratio for CLAP artifact detector")

    # ── Adversarial training mode ──
    p.add_argument("--adversarial", action="store_true",
                   default=bool(_y("adversarial", False)),
                   help="Enable adversarial training with online discriminator")
    p.add_argument("--adv-disc-path", default=_y("adv_disc_path", None),
                   help="Path to large ArtifactDetectorLarge .pt checkpoint for adversarial training")
    p.add_argument("--adv-disc-lr", type=float, default=_y("adv_disc_lr", 3e-4),
                   help="Discriminator learning rate (default 3e-4)")
    p.add_argument("--adv-disc-epochs", type=int, default=_y("adv_disc_epochs", 3),
                   help="Disc training epochs per update (default 3)")
    p.add_argument("--adv-disc-interval", type=int, default=_y("adv_disc_interval", 8),
                   help="Train disc every N optimizer steps (default 8)")
    p.add_argument("--adv-ema-decay", type=float, default=_y("adv_ema_decay", 0.995),
                   help="EMA decay for discriminator (default 0.995)")
    p.add_argument("--adv-flow-weight", type=float, default=_y("adv_flow_weight", 0.7),
                   help="Flow loss weight in total loss (default 0.7)")
    p.add_argument("--adv-disc-weight", type=float, default=_y("adv_disc_weight", 0.3),
                   help="Discriminator loss weight in total loss (default 0.3)")
    p.add_argument("--adv-buffer-min", type=int, default=_y("adv_buffer_min", 50),
                   help="Minimum samples in buffer before disc training (default 50)")

    args = p.parse_args(remaining)

    # --no-clap overrides everything
    if args.no_clap:
        args.clap_loss = False

    # Parse bucket_weights if provided as JSON string
    if isinstance(args.bucket_weights, str):
        args.bucket_weights = json.loads(args.bucket_weights)

    return args


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
            "script": "dramabox_finetune_train_multi_aux.py",
            "pattern": "IC-LoRA 3-mode + 3 auxiliary losses (naturalness, centroid, speaker sim)",
            "aux_enabled": args.clap_loss,
        }
        with open(os.path.join(args.output_dir, "training_args.yaml"), "w") as f:
            yaml.dump(args_dict, f, default_flow_style=False, sort_keys=False)

    # Build model
    if is_main:
        logging.info("Loading audio-only model...")
    model = build_audio_only_model(args.checkpoint, device, dtype)

    # DramaBox model builder may leave some sub-modules on meta device (e.g.
    # caption_projection created with `torch.device("meta")`). Materialize them
    # now, BEFORE LoRA is applied, so that LoRA wraps real tensors.
    _meta_count = 0
    for name, module in model.named_modules():
        for pname, param in list(module.named_parameters(recurse=False)):
            if param.device.type == "meta":
                new_param = torch.nn.Parameter(
                    torch.zeros(param.shape, dtype=param.dtype, device=device),
                    requires_grad=param.requires_grad)
                setattr(module, pname, new_param)
                _meta_count += 1
        for bname, buf in list(module.named_buffers(recurse=False)):
            if buf.device.type == "meta":
                module.register_buffer(bname,
                    torch.zeros(buf.shape, dtype=buf.dtype, device=device))
                _meta_count += 1
    if _meta_count > 0 and is_main:
        logging.warning(f"Materialized {_meta_count} meta tensors to {device}")

    # Integrity check: critical weights must not be all-zero (catches corrupted checkpoints)
    if is_main:
        _critical = ['audio_patchify_proj.weight', 'audio_proj_out.weight']
        for _cn in _critical:
            for _n, _p in model.named_parameters():
                if _n == _cn:
                    if _p.float().norm().item() == 0.0:
                        raise RuntimeError(
                            f"CORRUPTED CHECKPOINT: {_cn} is all zeros! "
                            f"Re-run merge_lora_into_base.py to fix.")
                    break

    # Extract safetensors metadata for full FT checkpoint saves
    if args.full_ft:
        try:
            import struct as _struct
            with open(args.checkpoint, "rb") as _f:
                _hs = _struct.unpack("<Q", _f.read(8))[0]
                _hdr = json.loads(_f.read(_hs).decode())
            _checkpoint_metadata.update(_hdr.get("__metadata__", {}))
            if is_main:
                logging.info(f"Extracted checkpoint metadata: {list(_checkpoint_metadata.keys())}")
        except Exception as e:
            if is_main:
                logging.warning(f"Could not extract checkpoint metadata: {e}")

    if is_main:
        logging.info("Loading audio connector...")
    audio_connector = load_audio_connector(args.full_checkpoint, device, dtype)
    audio_connector.eval()
    for p in audio_connector.parameters():
        p.requires_grad = False

    use_lora = not args.full_ft
    if use_lora:
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
    else:
        # Full fine-tuning: all diffusion transformer parameters are trainable
        for p in model.parameters():
            p.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        if is_main:
            logging.info(f"Full FT: {trainable:,} trainable / {total:,} total ({100*trainable/total:.1f}%)")

    model.train()
    if use_lora:
        model.base_model.model.set_gradient_checkpointing(True)
    else:
        # Re-enabled: the zero-grad bug was caused by a corrupted merged checkpoint
        # (non-LoRA params were all zeros), NOT by gradient checkpointing.
        model.set_gradient_checkpointing(True)
        if is_main:
            logging.info("Full FT: gradient checkpointing ENABLED")

    # ── Multi-auxiliary loss setup ────────────────────────────────────
    aux_enabled = args.clap_loss
    audio_decoder_clap = None
    clap_model = None
    clap_tokenizer = None
    st_model = None  # SentenceTransformer model for large CLAP
    pos_text_emb = None
    neg_text_emb = None
    real_centroid = None
    synth_centroid = None
    quality_mlp = None
    wavlm_sv = None
    wavlm_fe = None
    use_large_clap = False
    comb_detector = None
    artifact_v2 = None
    clap_artifact_mlp = None

    # EMA trackers for adaptive coefficients
    ema_alpha = 0.95
    ema_flow = 0.0
    ema_aux1 = 0.0
    ema_aux2 = 0.0
    ema_aux3 = 0.0
    ema_aux4 = 0.0  # comb filter detector
    ema_aux5 = 0.0  # artifact detector V2
    ema_aux6 = 0.0  # CLAP artifact MLP
    target_ratio = args.aux_target_ratio
    speaker_sim_ratio = args.speaker_sim_ratio if args.speaker_sim_ratio > 0 else target_ratio
    comb_target_ratio = args.comb_target_ratio
    artifact_v2_ratio = getattr(args, 'artifact_v2_ratio', 3.0)
    clap_artifact_ratio = getattr(args, 'clap_artifact_ratio', 3.0)
    coeff_cap = args.coeff_cap
    aux_sigma_max = args.aux_sigma_max

    use_centroid = not args.no_centroid
    use_quality_mlp = not args.no_quality_mlp
    use_speaker_sim = not args.no_speaker_sim
    use_comb_detector = (not args.no_comb_detector and args.comb_detector_path is not None)
    use_artifact_v2 = (getattr(args, 'artifact_v2_path', None) is not None)
    use_clap_artifact = (getattr(args, 'clap_artifact_path', None) is not None)

    if aux_enabled:
        if is_main:
            logging.info("Setting up multi-auxiliary loss models...")

        # Determine CLAP model type
        use_large_clap = (args.clap_model != "laion/voiceclap-small")

        if use_large_clap:
            # Large CLAP via SentenceTransformer (e.g. 7B with INT4)
            audio_decoder_clap, st_model, _ = setup_clap_models_large(
                args.full_checkpoint, device, dtype, args)
            pos_text_emb = encode_text_st(args.positive_text, st_model)
            neg_text_emb = encode_text_st(args.negative_text, st_model)

            # If centroid or quality_mlp is enabled, also load CLAP-small
            # (these classifiers were trained on 768-dim CLAP-small embeddings)
            if use_centroid or use_quality_mlp:
                if is_main:
                    logging.info("Also loading CLAP-small for centroid/quality_mlp compatibility...")
                from transformers import AutoModel, AutoTokenizer
                clap_model = AutoModel.from_pretrained(
                    "laion/voiceclap-small", trust_remote_code=True
                ).to(device).eval()
                for p in clap_model.parameters():
                    p.requires_grad_(False)
                if is_main:
                    logging.info("CLAP-small loaded alongside CLAP-large for aux classifiers")

            # Warn if differentiable reward requested with large CLAP
            if getattr(args, 'differentiable_reward', False) and is_main:
                logging.warning("Differentiable reward is not supported with large CLAP models "
                                "(7B+ too large for activation storage). "
                                "Falling back to non-differentiable mode.")
                args.differentiable_reward = False
        else:
            # Original VoiceCLAP-small path
            audio_decoder_clap, clap_model, clap_tokenizer = setup_clap_models(
                args.full_checkpoint, device, dtype)
            pos_text_emb = encode_clap_text(args.positive_text, clap_model, clap_tokenizer, device)
            neg_text_emb = encode_clap_text(args.negative_text, clap_model, clap_tokenizer, device)

        # 2. Centroid embeddings + Quality MLP
        if use_centroid or use_quality_mlp:
            real_centroid, synth_centroid, quality_mlp = setup_centroid_and_quality(
                args.classifiers_dir, device, dtype)
            if not use_centroid:
                real_centroid = None
                synth_centroid = None
            if not use_quality_mlp:
                quality_mlp = None

        # 3. WavLM-SV for speaker similarity
        if use_speaker_sim:
            wavlm_sv, wavlm_fe = setup_speaker_model(device, dtype)

        # 4. Comb filter detector CNN (operates on latent space directly)
        if use_comb_detector:
            comb_detector = setup_comb_detector(args.comb_detector_path, device, label="CombFilterDetector-v1")

        # 5. Artifact detector V2 CNN (second latent-space detector)
        if use_artifact_v2:
            artifact_v2 = setup_comb_detector(args.artifact_v2_path, device, label="ArtifactDetectorV2")

        # 6. CLAP-based artifact MLP (operates on CLAP embeddings)
        if use_clap_artifact:
            clap_artifact_mlp = setup_clap_artifact_mlp(args.clap_artifact_path, device)

        if is_main:
            logging.info(f"Auxiliary losses enabled:")
            logging.info(f"  CLAP model: {args.clap_model} (large={use_large_clap}, quantize={args.clap_quantize})")
            logging.info(f"  Loss 1 (Naturalness): CLAP text={True}, quality_mlp={use_quality_mlp}")
            logging.info(f"  Loss 2 (Centroid): {use_centroid}")
            logging.info(f"  Loss 3 (Speaker Sim): {use_speaker_sim}")
            logging.info(f"  Loss 4 (Comb Detector v1): {use_comb_detector}")
            logging.info(f"  Loss 5 (Artifact V2 CNN): {use_artifact_v2}")
            logging.info(f"  Loss 6 (CLAP Artifact MLP): {use_clap_artifact}")
            if use_comb_detector:
                logging.info(f"    → CNN on latent space, no decoder needed, gradient flows directly")
                logging.info(f"    → Target ratio: {comb_target_ratio}")
            if use_artifact_v2:
                logging.info(f"    → Artifact V2 ratio: {artifact_v2_ratio}")
            if use_clap_artifact:
                logging.info(f"    → CLAP artifact ratio: {clap_artifact_ratio}")
            logging.info(f"  Target ratio (naturalness/centroid): {target_ratio}")
            logging.info(f"  Speaker sim ratio: {speaker_sim_ratio}")
            logging.info(f"  Coefficient cap: {coeff_cap}")
            logging.info(f"  Aux sigma threshold: {aux_sigma_max} (skip aux when sigma >= this)")
            _rej = getattr(args, 'rejection_sampling', False)
            _rej_pct = getattr(args, 'rejection_percentile', 50.0)
            _diff = getattr(args, 'differentiable_reward', False)
            if _diff:
                logging.info(f"  DIFFERENTIABLE REWARD: enabled (ReFL-style backprop through decoder/CLAP/WavLM)")
                logging.info(f"    → Gradients flow: pred_tgt → decoder → waveform → CLAP/WavLM → loss")
                logging.info(f"    → Gradient checkpointing on aux models: {getattr(args, 'diff_reward_checkpoint', False)}")
            elif _rej:
                logging.info(f"  REJECTION SAMPLING: enabled, percentile={_rej_pct}%")
                logging.info(f"    → Only train on top {100-_rej_pct:.0f}% reward samples")
                logging.info(f"    → Flow loss boosted by {100.0/max(100.0-_rej_pct, 1.0):.1f}x to compensate")
            else:
                logging.info(f"  Rejection sampling: disabled (using reward-weighted aux losses)")
            logging.info(f"  Positive text: {args.positive_text[:80]}...")
            logging.info(f"  Negative text: {args.negative_text}")
            if getattr(args, 'keep_last_n', 0) > 0:
                logging.info(f"  Checkpoint management: rolling window of {args.keep_last_n} + keep better older ones")

    # ── Adversarial discriminator setup ──
    adv_disc_online = None
    adv_disc_ema = None
    adv_disc_optimizer = None
    adv_real_buffer = []
    adv_fake_buffer = []
    adv_disc_train_stats = {}
    adversarial_mode = getattr(args, 'adversarial', False)

    if adversarial_mode and args.adv_disc_path:
        if is_main:
            logging.info("="*60)
            logging.info("ADVERSARIAL TRAINING MODE ENABLED")
            logging.info("="*60)

        # Load online discriminator (will be trained)
        adv_disc_online = setup_adversarial_disc(args.adv_disc_path, device, label="DiscOnline")
        adv_disc_online.eval()
        for p in adv_disc_online.parameters():
            p.requires_grad = False  # frozen during generator training, unfrozen for disc training

        # Load EMA discriminator (never directly trained, updated via EMA)
        adv_disc_ema = setup_adversarial_disc(args.adv_disc_path, device, label="DiscEMA")
        adv_disc_ema.eval()
        for p in adv_disc_ema.parameters():
            p.requires_grad = False  # always frozen

        # Disc optimizer (only for online disc training phase)
        adv_disc_optimizer = torch.optim.Adam(adv_disc_online.parameters(), lr=args.adv_disc_lr)

        if is_main:
            n_disc_params = sum(p.numel() for p in adv_disc_online.parameters())
            logging.info(f"  Online disc: {n_disc_params:,} params, lr={args.adv_disc_lr}")
            logging.info(f"  EMA disc: decay={args.adv_ema_decay}")
            logging.info(f"  Disc training: every {args.adv_disc_interval} steps, "
                         f"{args.adv_disc_epochs} epochs per update, "
                         f"min buffer={args.adv_buffer_min}")
            logging.info(f"  Loss split: {args.adv_flow_weight:.0%} flow + "
                         f"{args.adv_disc_weight:.0%} disc")
            logging.info(f"  Other aux losses: DISABLED (pure adversarial)")
            logging.info("="*60)

        # In adversarial mode, disable all traditional aux losses
        use_centroid = False
        use_quality_mlp = False
        use_speaker_sim = False
        use_comb_detector = False
        use_artifact_v2 = False
        use_clap_artifact = False

    # Dataset
    mode_weights = {
        "voice_clone_fwd": 0.33,
        "voice_clone_rev": 0.33,
        "unconditional": 0.34,
    }
    expand_all = getattr(args, "expand_all_modes", False)
    dataset = DramaBoxFinetuneDataset(
        preprocessed_dir=args.preprocessed_dir,
        mode_weights=mode_weights,
        max_ref_tokens=args.max_ref_tokens,
        overfit_n=args.overfit,
        expand_all_modes=expand_all,
        index_file=args.index_file,
    )

    # Build DataLoader (with optional bucket-weighted sampling)
    bucket_sampler = None
    if args.bucket_weights and isinstance(args.bucket_weights, dict):
        bucket_sampler = BucketWeightedSampler(
            dataset, bucket_weights=args.bucket_weights, seed=args.seed,
        )
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, sampler=bucket_sampler,
            num_workers=2, pin_memory=True, drop_last=True, collate_fn=collate_fn,
        )
    else:
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=2, pin_memory=True, drop_last=True, collate_fn=collate_fn,
        )

    # Compute epochs -> steps
    steps_per_epoch = 0
    if bucket_sampler is not None and args.podcast_epochs > 0:
        # Bucket mode: 1 epoch = 1 pass through sampler (anchor bucket seen once)
        # podcast_epochs = how many times podcast is fully seen
        args.epochs = args.podcast_epochs
        items_per_gpu = math.ceil(len(bucket_sampler) / max(accelerator.num_processes, 1))
        steps_per_epoch = math.ceil(items_per_gpu / args.grad_accum)
        args.steps = steps_per_epoch * args.epochs
        if is_main:
            logging.info(f"Bucket-weighted: {args.epochs} podcast-epochs x {steps_per_epoch} steps/epoch "
                         f"= {args.steps} total optimizer steps")
            logging.info(f"  ({len(bucket_sampler)} items/epoch, {accelerator.num_processes} GPUs, "
                         f"grad_accum={args.grad_accum})")
    elif args.epochs > 0 and not args.overfit:
        items_per_gpu = math.ceil(len(dataset) / max(accelerator.num_processes, 1))
        forward_per_epoch = items_per_gpu  # batch_size=1
        steps_per_epoch = math.ceil(forward_per_epoch / args.grad_accum)
        args.steps = steps_per_epoch * args.epochs
        if is_main:
            logging.info(f"Epoch-based: {args.epochs} epochs x {steps_per_epoch} steps/epoch "
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

    # NOTE: device_placement must NOT be [False, True, True] — that makes
    # accelerate wrap the optimizer in a no-op shim that silently skips steps.
    # Meta tensors are materialized before this point, so .to(device) is safe.
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
        logging.info(f"Multi-auxiliary losses: {'ENABLED' if aux_enabled else 'DISABLED'}")
        if args.overfit:
            logging.info(f"OVERFIT MODE: training on {args.overfit} groups only")

    data_iter = iter(dataloader)
    step = 0
    accum_loss = 0.0
    accum_total_loss = 0.0
    # Per-aux accumulators
    accum_aux1 = 0.0
    accum_aux2 = 0.0
    accum_aux3 = 0.0
    accum_aux4 = 0.0  # comb filter detector
    accum_aux5 = 0.0  # artifact detector V2
    accum_aux6 = 0.0  # CLAP artifact MLP
    accum_coeff1 = 0.0
    accum_coeff2 = 0.0
    accum_coeff3 = 0.0
    accum_coeff4 = 0.0
    accum_coeff5 = 0.0
    accum_coeff6 = 0.0
    accum_clap_text_reward = 0.0
    accum_quality_prob = 0.0
    accum_naturalness_reward = 0.0
    accum_centroid_score = 0.0
    accum_speaker_sim = 0.0
    accum_comb_score = 0.0
    accum_artifact_v2_score = 0.0
    accum_clap_artifact_score = 0.0
    aux_log_count = 0
    speaker_sim_count = 0
    comb_log_count = 0
    # Rejection sampling state
    rejection_enabled = getattr(args, 'rejection_sampling', False)
    rejection_percentile = getattr(args, 'rejection_percentile', 50.0)
    reward_buffer = deque(maxlen=512)  # rolling window for percentile
    rejection_count = 0  # number of rejected micro-batches in log window
    acceptance_count = 0  # number of accepted micro-batches in log window
    # Differentiable reward mode
    differentiable_reward = getattr(args, 'differentiable_reward', False)
    diff_checkpoint = getattr(args, 'diff_reward_checkpoint', False)
    vram_peak_mb = 0.0
    best_loss = float("inf")
    best_step = 0
    t0 = time.time()
    mode_counts = defaultdict(int)
    current_epoch = 0
    last_val_epoch = -1
    val_procs = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Smart checkpoint management
    keep_last_n = getattr(args, 'keep_last_n', 0)
    recent_checkpoints = deque(maxlen=keep_last_n) if keep_last_n > 0 else None
    checkpoint_rewards = {}  # {checkpoint_path: naturalness_reward}

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

            # ── Multi-Auxiliary Losses ─────────────────────────────────
            aux1_val = 0.0
            aux2_val = 0.0
            aux3_val = 0.0
            aux4_val = 0.0  # comb filter detector
            coeff1_val = 0.0
            coeff2_val = 0.0
            coeff3_val = 0.0
            coeff4_val = 0.0
            clap_text_reward_val = 0.0
            quality_prob_val = 0.0
            naturalness_reward_val = 0.0
            centroid_score_val = 0.0
            comb_score_val = 0.0
            speaker_sim_val = None

            # Only compute aux losses at low sigma where x0 prediction is clean enough
            # to produce meaningful decoded audio for reward computation
            sigma_val = sigma.item()
            aux_active = aux_enabled and sigma_val < aux_sigma_max

            # ── Adversarial discriminator loss (replaces standard aux losses) ──
            adv_score_online_val = 0.0
            adv_score_ema_val = 0.0
            if adversarial_mode and adv_disc_online is not None and sigma_val < aux_sigma_max:
                try:
                    # Compute x0 prediction WITH gradient
                    noisy_tgt_tokens = state.latent[:, :tgt_T].detach()
                    x0_pred_tokens = noisy_tgt_tokens - pred_tgt * sigma_exp

                    x0_latent_adv = patchifier.unpatchify(
                        x0_pred_tokens,
                        AudioLatentShape(
                            batch=B, channels=tgt_latent.shape[1],
                            frames=tgt_T_frames, mel_bins=tgt_latent.shape[3],
                        ),
                    )

                    # Generator loss: minimize disc scores (want x0 to look "real")
                    # Both discs in eval mode, params frozen — grad flows through x0_latent_adv only
                    score_online = adv_disc_online.comb_score(x0_latent_adv).mean()  # [0,1]
                    score_ema = adv_disc_ema.comb_score(x0_latent_adv).mean()        # [0,1]

                    gen_disc_loss = 0.5 * score_online + 0.5 * score_ema

                    # Weighted total loss
                    fw = args.adv_flow_weight
                    dw = args.adv_disc_weight
                    total_loss = fw * loss + dw * gen_disc_loss

                    if torch.isnan(total_loss) or torch.isinf(total_loss):
                        accelerator.backward(loss)
                        total_loss = loss
                    else:
                        accelerator.backward(total_loss)

                    adv_score_online_val = score_online.item()
                    adv_score_ema_val = score_ema.item()
                    accum_total_loss += total_loss.item()
                    aux_log_count += 1

                    # Collect samples for disc training buffer (detached, on CPU to save VRAM)
                    with torch.no_grad():
                        # Real = ground truth latent
                        adv_real_buffer.append(tgt_latent[0].detach().cpu())
                        # Fake = model prediction
                        adv_fake_buffer.append(x0_latent_adv[0].detach().cpu())

                    # Track metrics
                    accum_naturalness_reward += 0.0  # no naturalness in adversarial mode
                    accum_centroid_score += 0.0

                    # Track VRAM
                    cur_vram = torch.cuda.max_memory_allocated(device) / 1024**2
                    if cur_vram > vram_peak_mb:
                        vram_peak_mb = cur_vram

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        torch.cuda.empty_cache()
                        if is_main:
                            logging.warning(f"OOM in adversarial loss! Falling back to flow-only.")
                        accelerator.backward(loss)
                    else:
                        raise

            elif aux_active and differentiable_reward and tgt_T_frames <= 250:
                # ════════════════════════════════════════════════════��══════
                # DIFFERENTIABLE REWARD MODE (ReFL-style)
                # Gradients flow: pred_tgt → decoder → waveform → CLAP/WavLM → loss
                # ═══════════════════════════════════════════════════════════
                import torchaudio
                from torch.utils.checkpoint import checkpoint as grad_checkpoint

                try:
                    torch.cuda.empty_cache()
                    # ── x0 prediction WITH gradient (pred_tgt has grad) ──
                    noisy_tgt_tokens = state.latent[:, :tgt_T].detach()
                    x0_pred_tokens = noisy_tgt_tokens - pred_tgt * sigma_exp

                    x0_latent = patchifier.unpatchify(
                        x0_pred_tokens,
                        AudioLatentShape(
                            batch=B, channels=tgt_latent.shape[1],
                            frames=tgt_T_frames, mel_bins=tgt_latent.shape[3],
                        ),
                    )

                    # ── Decode prediction → waveform (grad flows through decoder) ──
                    if diff_checkpoint:
                        # Gradient checkpointing: recompute decoder activations during backward
                        decoded = grad_checkpoint(
                            audio_decoder_clap, x0_latent[0:1].to(dtype=dtype),
                            use_reentrant=False)
                    else:
                        decoded = audio_decoder_clap(x0_latent[0:1].to(dtype=dtype))

                    pred_wav = decoded.waveform.squeeze(0).float()
                    audio_sr = decoded.sampling_rate
                    pred_mono = pred_wav.mean(0) if pred_wav.ndim > 1 else pred_wav
                    if audio_sr != 16000:
                        pred_mono = torchaudio.functional.resample(pred_mono, audio_sr, 16000)

                    # ── CLAP embedding WITH gradient (differentiable mel computation) ──
                    if diff_checkpoint:
                        audio_emb = grad_checkpoint(
                            encode_clap_waveform_differentiable,
                            pred_mono.unsqueeze(0).to(device),
                            clap_model,
                            use_reentrant=False)
                    else:
                        audio_emb = encode_clap_waveform_differentiable(
                            pred_mono.unsqueeze(0).to(device), clap_model)
                    audio_emb_norm = F.normalize(audio_emb, p=2, dim=-1)  # [1, 768]

                    # ── Differentiable Loss 1: CLAP Naturalness ──
                    clap_pos_sim = (audio_emb_norm @ pos_text_emb.T).squeeze()
                    clap_neg_sim = (audio_emb_norm @ neg_text_emb.T).squeeze()
                    naturalness_loss = -clap_pos_sim + clap_neg_sim

                    # Quality MLP (differentiable — small, no checkpointing needed)
                    if use_quality_mlp and quality_mlp is not None:
                        quality_logit_t = quality_mlp(audio_emb_norm.float()).squeeze()
                        quality_loss = -quality_logit_t  # maximize P(real)
                        aux1_loss = naturalness_loss + 0.5 * quality_loss
                    else:
                        quality_logit_t = torch.tensor(0.0, device=device)
                        aux1_loss = naturalness_loss

                    # ── Differentiable Loss 2: Centroid Real/Fake ──
                    aux2_loss = torch.tensor(0.0, device=device)
                    if use_centroid and real_centroid is not None:
                        cos_real_t = (audio_emb_norm.float() @ real_centroid.T).squeeze()
                        cos_synth_t = (audio_emb_norm.float() @ synth_centroid.T).squeeze()
                        aux2_loss = -cos_real_t + cos_synth_t

                    # ── Differentiable Loss 3: Speaker Similarity ──
                    aux3_loss = torch.tensor(0.0, device=device)
                    if use_speaker_sim and wavlm_sv is not None and ref_T_frames > 0:
                        # Reference decoding (no grad needed for ref)
                        with torch.no_grad():
                            ref_decoded = audio_decoder_clap(ref_latent[0:1].to(dtype=dtype))
                            ref_wav = ref_decoded.waveform.squeeze(0).float()
                            ref_mono = ref_wav.mean(0) if ref_wav.ndim > 1 else ref_wav
                            if ref_decoded.sampling_rate != 16000:
                                ref_mono = torchaudio.functional.resample(
                                    ref_mono, ref_decoded.sampling_rate, 16000)
                            # WavLM normalization (replaces feature_extractor for differentiability)
                            ref_norm = (ref_mono - ref_mono.mean()) / (ref_mono.std() + 1e-7)
                            ref_spk = wavlm_sv(
                                ref_norm.unsqueeze(0).to(device=device, dtype=torch.float32)
                            ).embeddings
                            ref_spk = F.normalize(ref_spk.float(), p=2, dim=-1)

                        # Prediction speaker embedding WITH grad
                        pred_std = pred_mono.std().detach()  # detach std for stability
                        pred_norm = (pred_mono - pred_mono.mean().detach()) / (pred_std + 1e-7)
                        pred_input = pred_norm.unsqueeze(0).to(device=device, dtype=torch.float32)
                        if diff_checkpoint:
                            pred_spk_emb = grad_checkpoint(
                                wavlm_extract_embeddings,
                                wavlm_sv,
                                pred_input,
                                use_reentrant=False)
                        else:
                            pred_spk_emb = wavlm_sv(pred_input).embeddings
                        pred_spk = F.normalize(pred_spk_emb.float(), p=2, dim=-1)
                        speaker_sim_t = (pred_spk @ ref_spk.T).squeeze()
                        aux3_loss = -speaker_sim_t  # maximize similarity

                    # ── Differentiable Loss 4: Comb Filter Detector ──
                    # Operates DIRECTLY on x0_latent — no decoder needed!
                    # Gradient path: pred_tgt → x0_latent → CNN → loss
                    aux4_loss = torch.tensor(0.0, device=device)
                    if use_comb_detector and comb_detector is not None:
                        comb_score_t = comb_detector.comb_score(x0_latent)  # [B] in [0,1]
                        aux4_loss = comb_score_t.mean()  # minimize = push away from comb artifacts
                        comb_score_val = comb_score_t.mean().item()

                    # ── Differentiable Loss 5: Artifact Detector V2 CNN ──
                    aux5_loss = torch.tensor(0.0, device=device)
                    artifact_v2_score_val = 0.0
                    if use_artifact_v2 and artifact_v2 is not None:
                        av2_score_t = artifact_v2.comb_score(x0_latent)  # [B] in [0,1]
                        aux5_loss = av2_score_t.mean()
                        artifact_v2_score_val = av2_score_t.mean().item()

                    # ── Differentiable Loss 6: CLAP Artifact MLP ──
                    aux6_loss = torch.tensor(0.0, device=device)
                    clap_artifact_score_val = 0.0
                    if use_clap_artifact and clap_artifact_mlp is not None:
                        clap_art_score_t = clap_artifact_mlp.artifact_score(
                            audio_emb_norm)  # [B] in [0,1]
                        aux6_loss = clap_art_score_t.mean()
                        clap_artifact_score_val = clap_art_score_t.mean().item()

                    # ── Adaptive coefficients (EMA-based) ──
                    ema_flow = ema_alpha * ema_flow + (1 - ema_alpha) * loss.item()
                    aux1_item = aux1_loss.item()
                    ema_aux1 = ema_alpha * ema_aux1 + (1 - ema_alpha) * abs(aux1_item)
                    coeff1_val = min(target_ratio * ema_flow / max(ema_aux1, 1e-8), coeff_cap)

                    coeff2_val = 0.0
                    if use_centroid and real_centroid is not None:
                        aux2_item = aux2_loss.item()
                        ema_aux2 = ema_alpha * ema_aux2 + (1 - ema_alpha) * abs(aux2_item)
                        coeff2_val = min(target_ratio * ema_flow / max(ema_aux2, 1e-8), coeff_cap)

                    coeff3_val = 0.0
                    if aux3_loss.requires_grad or aux3_loss.item() != 0.0:
                        aux3_item = aux3_loss.item()
                        ema_aux3 = ema_alpha * ema_aux3 + (1 - ema_alpha) * abs(aux3_item)
                        coeff3_val = min(speaker_sim_ratio * ema_flow / max(ema_aux3, 1e-8), coeff_cap)

                    coeff4_val = 0.0
                    if aux4_loss.requires_grad or aux4_loss.item() != 0.0:
                        aux4_item = aux4_loss.item()
                        ema_aux4 = ema_alpha * ema_aux4 + (1 - ema_alpha) * abs(aux4_item)
                        coeff4_val = min(comb_target_ratio * ema_flow / max(ema_aux4, 1e-8), coeff_cap)

                    coeff5_val = 0.0
                    if aux5_loss.requires_grad or aux5_loss.item() != 0.0:
                        aux5_item = aux5_loss.item()
                        ema_aux5 = ema_alpha * ema_aux5 + (1 - ema_alpha) * abs(aux5_item)
                        coeff5_val = min(artifact_v2_ratio * ema_flow / max(ema_aux5, 1e-8), coeff_cap)

                    coeff6_val = 0.0
                    if aux6_loss.requires_grad or aux6_loss.item() != 0.0:
                        aux6_item = aux6_loss.item()
                        ema_aux6 = ema_alpha * ema_aux6 + (1 - ema_alpha) * abs(aux6_item)
                        coeff6_val = min(clap_artifact_ratio * ema_flow / max(ema_aux6, 1e-8), coeff_cap)

                    total_loss = (loss + coeff1_val * aux1_loss + coeff2_val * aux2_loss
                                  + coeff3_val * aux3_loss + coeff4_val * aux4_loss
                                  + coeff5_val * aux5_loss + coeff6_val * aux6_loss)

                    if torch.isnan(total_loss) or torch.isinf(total_loss):
                        accelerator.backward(loss)
                        total_loss = loss
                    else:
                        accelerator.backward(total_loss)

                    # Log values
                    clap_text_reward_val = (clap_pos_sim - clap_neg_sim).item()
                    quality_prob_val = torch.sigmoid(quality_logit_t.detach()).item()
                    naturalness_reward_val = clap_text_reward_val
                    if use_centroid and real_centroid is not None:
                        centroid_score_val = (cos_real_t - cos_synth_t).item()
                    if aux3_loss.item() != 0.0:
                        speaker_sim_val = -aux3_loss.item()

                    aux1_val = aux1_loss.item()
                    aux2_val = aux2_loss.item()
                    aux3_val = aux3_loss.item()
                    aux4_val = aux4_loss.item()
                    accum_total_loss += total_loss.item()
                    accum_aux1 += aux1_val
                    accum_aux2 += aux2_val
                    accum_aux3 += aux3_val
                    accum_aux4 += aux4_val
                    aux5_val = aux5_loss.item()
                    aux6_val = aux6_loss.item()
                    accum_aux5 += aux5_val
                    accum_aux6 += aux6_val
                    accum_coeff1 += coeff1_val
                    accum_coeff2 += coeff2_val
                    accum_coeff3 += coeff3_val
                    accum_coeff4 += coeff4_val
                    accum_coeff5 += coeff5_val
                    accum_coeff6 += coeff6_val
                    if comb_score_val > 0:
                        accum_comb_score += comb_score_val
                        comb_log_count += 1
                    if artifact_v2_score_val > 0:
                        accum_artifact_v2_score += artifact_v2_score_val
                    if clap_artifact_score_val > 0:
                        accum_clap_artifact_score += clap_artifact_score_val

                    # Track VRAM
                    cur_vram = torch.cuda.max_memory_allocated(device) / 1024**2
                    if cur_vram > vram_peak_mb:
                        vram_peak_mb = cur_vram

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        # OOM: log and fall back to flow-only for this step
                        torch.cuda.empty_cache()
                        if is_main:
                            logging.warning(f"OOM in differentiable reward! VRAM peak: "
                                          f"{torch.cuda.max_memory_allocated(device)/1024**2:.0f}MB. "
                                          f"Falling back to flow-only. Consider --diff-reward-checkpoint")
                        accelerator.backward(loss)
                    else:
                        raise

                # Track reward metrics
                accum_clap_text_reward += clap_text_reward_val
                accum_quality_prob += quality_prob_val
                accum_naturalness_reward += naturalness_reward_val
                accum_centroid_score += centroid_score_val
                aux_log_count += 1
                if speaker_sim_val is not None:
                    accum_speaker_sim += speaker_sim_val
                    speaker_sim_count += 1

            elif aux_active:
                # ══════════════════════���════════════════════════════════════
                # NON-DIFFERENTIABLE MODE (scalar rewards, no grad through aux models)
                # ═══════════════════════════════════════════════════════════
                import torchaudio

                with torch.no_grad():
                    # ── Recover x0 prediction ──
                    noisy_tgt_tokens = state.latent[:, :tgt_T]
                    x0_pred_tokens_ng = noisy_tgt_tokens - pred_tgt.detach() * sigma_exp

                    # Unpatchify to latent shape: [B, 8, T, 16]
                    x0_latent = patchifier.unpatchify(
                        x0_pred_tokens_ng,
                        AudioLatentShape(
                            batch=B, channels=tgt_latent.shape[1],
                            frames=tgt_T_frames, mel_bins=tgt_latent.shape[3],
                        ),
                    )

                    # ── Decode prediction → waveform (first sample only) ──
                    decoded = audio_decoder_clap(x0_latent[0:1].to(dtype=dtype))
                    pred_wav = decoded.waveform.squeeze(0).float()
                    audio_sr = decoded.sampling_rate
                    pred_mono = pred_wav.mean(0) if pred_wav.ndim > 1 else pred_wav
                    if audio_sr != 16000:
                        pred_mono = torchaudio.functional.resample(pred_mono, audio_sr, 16000)

                    # ── CLAP embedding for naturalness ──
                    if use_large_clap and st_model is not None:
                        import numpy as np
                        wav_np = pred_mono.cpu().numpy()
                        audio_emb = encode_audio_st(
                            wav_np, 16000, st_model,
                            rank=accelerator.process_index)
                    else:
                        audio_emb = clap_model.encode_waveform(pred_mono.unsqueeze(0).to(device))
                        audio_emb = F.normalize(audio_emb, p=2, dim=-1)  # [1, 768]

                    # ── CLAP-small embedding for quality_mlp/centroid (768-dim) ──
                    # When using large CLAP, the quality MLP and centroids are trained
                    # on 768-dim CLAP-small embeddings — compute a separate embedding
                    audio_emb_small = None
                    if use_large_clap and clap_model is not None and (use_quality_mlp or use_centroid):
                        audio_emb_small = clap_model.encode_waveform(pred_mono.unsqueeze(0).to(device))
                        audio_emb_small = F.normalize(audio_emb_small, p=2, dim=-1)  # [1, 768]

                    # ── Reward 1: CLAP Naturalness (uses large or small CLAP) ──
                    clap_pos_sim = (audio_emb @ pos_text_emb.T).item()
                    clap_neg_sim = (audio_emb @ neg_text_emb.T).item()
                    clap_text_reward_val = clap_pos_sim - clap_neg_sim  # ~[-0.5, +0.5]

                    if use_quality_mlp and quality_mlp is not None:
                        # Quality MLP uses CLAP-small embeddings (768-dim)
                        emb_for_mlp = audio_emb_small if audio_emb_small is not None else audio_emb
                        quality_logit = quality_mlp(emb_for_mlp.float()).item()
                        quality_prob_val = torch.sigmoid(torch.tensor(quality_logit)).item()
                        naturalness_reward_val = (0.5 * clap_text_reward_val +
                                                  0.5 * (2 * quality_prob_val - 1))
                    else:
                        quality_prob_val = 0.5
                        naturalness_reward_val = clap_text_reward_val

                    # ── Reward 2: Centroid Real/Fake (uses CLAP-small embeddings) ──
                    if use_centroid and real_centroid is not None:
                        emb_for_cent = audio_emb_small if audio_emb_small is not None else audio_emb
                        cos_real = (emb_for_cent @ real_centroid.T).item()
                        cos_synth = (emb_for_cent @ synth_centroid.T).item()
                        centroid_score_val = cos_real - cos_synth  # ~[-0.8, +0.9]

                    # ── Reward 3: Speaker Similarity (only with reference) ──
                    if use_speaker_sim and wavlm_sv is not None and ref_T_frames > 0:
                        # Decode reference → waveform
                        ref_decoded = audio_decoder_clap(ref_latent[0:1].to(dtype=dtype))
                        ref_wav = ref_decoded.waveform.squeeze(0).float()
                        ref_mono = ref_wav.mean(0) if ref_wav.ndim > 1 else ref_wav
                        if ref_decoded.sampling_rate != 16000:
                            ref_mono = torchaudio.functional.resample(
                                ref_mono, ref_decoded.sampling_rate, 16000)

                        # Extract speaker embeddings via WavLM-SV
                        # Process through feature extractor for proper normalization
                        pred_inputs = wavlm_fe(
                            pred_mono.cpu().numpy(), sampling_rate=16000,
                            return_tensors="pt", padding=True)
                        ref_inputs = wavlm_fe(
                            ref_mono.cpu().numpy(), sampling_rate=16000,
                            return_tensors="pt", padding=True)

                        pred_spk = wavlm_sv(
                            pred_inputs.input_values.to(device=device, dtype=torch.float32)
                        ).embeddings
                        ref_spk = wavlm_sv(
                            ref_inputs.input_values.to(device=device, dtype=torch.float32)
                        ).embeddings

                        pred_spk = F.normalize(pred_spk.float(), p=2, dim=-1)
                        ref_spk = F.normalize(ref_spk.float(), p=2, dim=-1)
                        speaker_sim_val = (pred_spk @ ref_spk.T).item()  # [-1, +1]

                    # ── Reward 4: Comb Filter Detector (latent-space, no decoder needed) ──
                    if use_comb_detector and comb_detector is not None:
                        comb_score_val = comb_detector.comb_score(x0_latent).mean().item()

                    # ── Reward 5: Artifact Detector V2 (latent-space) ──
                    artifact_v2_score_val = 0.0
                    if use_artifact_v2 and artifact_v2 is not None:
                        artifact_v2_score_val = artifact_v2.comb_score(x0_latent).mean().item()

                    # ── Reward 6: CLAP Artifact MLP ──
                    clap_artifact_score_val = 0.0
                    if use_clap_artifact and clap_artifact_mlp is not None:
                        emb_for_art = audio_emb_small if audio_emb_small is not None else audio_emb
                        clap_artifact_score_val = clap_artifact_mlp.artifact_score(
                            emb_for_art).mean().item()

                # ── Compute composite reward for rejection sampling ──
                composite_reward = naturalness_reward_val
                n_rewards = 1
                if use_centroid and real_centroid is not None:
                    composite_reward += centroid_score_val
                    n_rewards += 1
                if speaker_sim_val is not None:
                    composite_reward += speaker_sim_val
                    n_rewards += 1
                composite_reward /= n_rewards  # normalize to ~[-0.5, +1.0]

                # ── Rejection sampling mode ──
                if rejection_enabled:
                    reward_buffer.append(composite_reward)

                    # Need at least 32 samples before we can reject meaningfully
                    if len(reward_buffer) >= 32:
                        sorted_buf = sorted(reward_buffer)
                        idx = int(len(sorted_buf) * rejection_percentile / 100.0)
                        idx = min(idx, len(sorted_buf) - 1)
                        threshold = sorted_buf[idx]
                    else:
                        threshold = -999.0  # accept everything during warmup

                    if composite_reward >= threshold:
                        # ACCEPTED: train on this sample (flow loss with 2x boost
                        # to compensate for ~50% rejection rate)
                        boost = 100.0 / max(100.0 - rejection_percentile, 1.0)
                        accelerator.backward(loss * boost)
                        acceptance_count += 1
                    else:
                        # REJECTED: zero gradient for this micro-batch
                        accelerator.backward(loss * 0.0)
                        rejection_count += 1

                    total_loss = loss  # for logging purposes
                    accum_total_loss += loss.item()

                else:
                    # ── Original aux loss mode (reward-weighted reconstruction) ──
                    x0_clean = patchifier.patchify(tgt_latent)
                    x0_pred_grad = state.latent[:, :tgt_T].detach() - pred_tgt * sigma_exp
                    x0_recon_loss = ((x0_pred_grad - x0_clean.detach()).pow(2).mean(dim=-1)
                                     * loss_mask).div(loss_mask.mean().clamp(min=1e-6)).mean()

                    # Loss 1: Naturalness
                    w1 = torch.clamp(
                        torch.tensor(0.5 - naturalness_reward_val, device=device),
                        min=0.05, max=2.0)
                    aux1 = w1 * x0_recon_loss

                    # Loss 2: Centroid
                    if use_centroid and real_centroid is not None:
                        w2 = torch.clamp(
                            torch.tensor(0.5 - centroid_score_val, device=device),
                            min=0.05, max=2.0)
                        aux2 = w2 * x0_recon_loss
                    else:
                        aux2 = torch.tensor(0.0, device=device)

                    # Loss 3: Speaker sim
                    if speaker_sim_val is not None:
                        w3 = torch.clamp(
                            torch.tensor(0.5 - speaker_sim_val, device=device),
                            min=0.05, max=2.0)
                        aux3 = w3 * x0_recon_loss
                    else:
                        aux3 = torch.tensor(0.0, device=device)

                    # Loss 4: Comb filter (reward-weighted: higher comb score → more penalty)
                    if use_comb_detector and comb_score_val > 0:
                        w4 = torch.clamp(torch.tensor(comb_score_val, device=device), min=0.05, max=2.0)
                        aux4 = w4 * x0_recon_loss
                    else:
                        aux4 = torch.tensor(0.0, device=device)

                    # Loss 5: Artifact V2 (reward-weighted)
                    if use_artifact_v2 and artifact_v2_score_val > 0:
                        w5 = torch.clamp(torch.tensor(artifact_v2_score_val, device=device), min=0.05, max=2.0)
                        aux5 = w5 * x0_recon_loss
                    else:
                        aux5 = torch.tensor(0.0, device=device)

                    # Loss 6: CLAP artifact (reward-weighted)
                    if use_clap_artifact and clap_artifact_score_val > 0:
                        w6 = torch.clamp(torch.tensor(clap_artifact_score_val, device=device), min=0.05, max=2.0)
                        aux6 = w6 * x0_recon_loss
                    else:
                        aux6 = torch.tensor(0.0, device=device)

                    # Adaptive coefficients (EMA-based)
                    ema_flow = ema_alpha * ema_flow + (1 - ema_alpha) * loss.item()
                    ema_aux1 = ema_alpha * ema_aux1 + (1 - ema_alpha) * aux1.item()
                    coeff1_val = min(target_ratio * ema_flow / max(ema_aux1, 1e-8), coeff_cap)

                    coeff2_val = 0.0
                    if use_centroid and real_centroid is not None:
                        ema_aux2 = ema_alpha * ema_aux2 + (1 - ema_alpha) * aux2.item()
                        coeff2_val = min(target_ratio * ema_flow / max(ema_aux2, 1e-8), coeff_cap)

                    coeff3_val = 0.0
                    if speaker_sim_val is not None:
                        ema_aux3 = ema_alpha * ema_aux3 + (1 - ema_alpha) * aux3.item()
                        coeff3_val = min(speaker_sim_ratio * ema_flow / max(ema_aux3, 1e-8), coeff_cap)

                    coeff4_val = 0.0
                    if use_comb_detector and comb_score_val > 0:
                        ema_aux4 = ema_alpha * ema_aux4 + (1 - ema_alpha) * aux4.item()
                        coeff4_val = min(comb_target_ratio * ema_flow / max(ema_aux4, 1e-8), coeff_cap)

                    coeff5_val = 0.0
                    if use_artifact_v2 and artifact_v2_score_val > 0:
                        ema_aux5 = ema_alpha * ema_aux5 + (1 - ema_alpha) * aux5.item()
                        coeff5_val = min(artifact_v2_ratio * ema_flow / max(ema_aux5, 1e-8), coeff_cap)

                    coeff6_val = 0.0
                    if use_clap_artifact and clap_artifact_score_val > 0:
                        ema_aux6 = ema_alpha * ema_aux6 + (1 - ema_alpha) * aux6.item()
                        coeff6_val = min(clap_artifact_ratio * ema_flow / max(ema_aux6, 1e-8), coeff_cap)

                    total_loss = (loss + coeff1_val * aux1 + coeff2_val * aux2
                                  + coeff3_val * aux3 + coeff4_val * aux4
                                  + coeff5_val * aux5 + coeff6_val * aux6)

                    if torch.isnan(total_loss) or torch.isinf(total_loss):
                        accelerator.backward(loss)
                        total_loss = loss
                    else:
                        accelerator.backward(total_loss)

                    aux1_val = aux1.item() if not math.isnan(aux1.item()) else 0.0
                    aux2_val = aux2.item() if not math.isnan(aux2.item()) else 0.0
                    aux3_val = aux3.item() if not math.isnan(aux3.item()) else 0.0
                    aux4_val = aux4.item() if not math.isnan(aux4.item()) else 0.0
                    aux5_val = aux5.item() if not math.isnan(aux5.item()) else 0.0
                    aux6_val = aux6.item() if not math.isnan(aux6.item()) else 0.0
                    accum_total_loss += total_loss.item() if not math.isnan(total_loss.item()) else loss.item()
                    accum_aux1 += aux1_val
                    accum_aux2 += aux2_val
                    accum_aux3 += aux3_val
                    accum_aux4 += aux4_val
                    accum_aux5 += aux5_val
                    accum_aux6 += aux6_val
                    accum_coeff1 += coeff1_val
                    accum_coeff2 += coeff2_val
                    accum_coeff3 += coeff3_val
                    accum_coeff4 += coeff4_val
                    accum_coeff5 += coeff5_val
                    accum_coeff6 += coeff6_val
                    if comb_score_val > 0:
                        accum_comb_score += comb_score_val
                        comb_log_count += 1
                    if artifact_v2_score_val > 0:
                        accum_artifact_v2_score += artifact_v2_score_val
                    if clap_artifact_score_val > 0:
                        accum_clap_artifact_score += clap_artifact_score_val

                # Track reward metrics for logging (both modes)
                accum_clap_text_reward += clap_text_reward_val
                accum_quality_prob += quality_prob_val
                accum_naturalness_reward += naturalness_reward_val
                accum_centroid_score += centroid_score_val
                aux_log_count += 1
                if speaker_sim_val is not None:
                    accum_speaker_sim += speaker_sim_val
                    speaker_sim_count += 1
            else:
                # aux_enabled=False OR sigma >= aux_sigma_max: flow-only backward
                accelerator.backward(loss)

            if accelerator.sync_gradients and args.max_grad_norm > 0:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            # One-time gradient diagnostic (first optimizer step only)
            if is_opt_step and step == 1 and is_main:
                grad_stats = {}
                for n, p in model.named_parameters():
                    if p.grad is not None:
                        gn = p.grad.float().norm().item()
                        grad_stats[n] = gn
                if grad_stats:
                    sorted_grads = sorted(grad_stats.items(), key=lambda x: x[1], reverse=True)
                    logging.info(f"GRAD DIAG: {len(grad_stats)}/{sum(1 for _ in model.parameters())} params have grads")
                    logging.info(f"  Top 5 grad norms: {[(n.split('.')[-2]+'.'+n.split('.')[-1], f'{v:.6f}') for n,v in sorted_grads[:5]]}")
                    logging.info(f"  Bottom 5 grad norms: {[(n.split('.')[-2]+'.'+n.split('.')[-1], f'{v:.6f}') for n,v in sorted_grads[-5:]]}")
                    zero_grads = sum(1 for v in grad_stats.values() if v == 0.0)
                    logging.info(f"  Zero-grad params: {zero_grads}/{len(grad_stats)}")
                else:
                    logging.warning("GRAD DIAG: NO parameters have gradients!")

            optimizer.step()
            optimizer.zero_grad()
            if accelerator.sync_gradients:
                scheduler.step()

            # ── Adversarial: train discriminator periodically ──
            if (adversarial_mode and adv_disc_online is not None
                    and step % args.adv_disc_interval == 0
                    and len(adv_real_buffer) >= args.adv_buffer_min):
                # Enable disc training
                for p in adv_disc_online.parameters():
                    p.requires_grad_(True)
                adv_disc_online.train()

                # Gather buffers from all GPUs (each GPU has its own samples)
                # For simplicity, train on local samples only (each GPU trains its own disc copy)
                adv_disc_train_stats = train_discriminator_online(
                    adv_disc_online, adv_disc_optimizer,
                    adv_real_buffer, adv_fake_buffer,
                    epochs=args.adv_disc_epochs, device=device,
                )

                # Update EMA discriminator
                update_ema_weights(adv_disc_ema, adv_disc_online, decay=args.adv_ema_decay)

                if is_main:
                    logging.info(
                        f"  DISC UPDATE step {step}: loss={adv_disc_train_stats['disc_loss']:.4f} "
                        f"acc={adv_disc_train_stats['disc_acc']:.3f} "
                        f"({adv_disc_train_stats['n_real']}r+{adv_disc_train_stats['n_fake']}f "
                        f"× {adv_disc_train_stats['epochs']}ep)")

                # Freeze disc again for generator training
                for p in adv_disc_online.parameters():
                    p.requires_grad_(False)
                adv_disc_online.eval()

                # Clear buffers
                adv_real_buffer.clear()
                adv_fake_buffer.clear()

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

            log_msg = (
                f"Step {step}/{args.steps} | loss={avg_loss:.4f} | lr={lr:.2e} | "
                f"tgt_T={tgt_T} ref_T={ref_T_frames} | "
                f"{sps:.2f} steps/s | ETA {eta/60:.0f}min | modes={mode_pcts}"
            )

            # Build metrics dict
            metric = {
                "step": step,
                "flow_loss": round(avg_loss, 6),
                "lr": lr,
                "tgt_tokens": tgt_T,
                "ref_tokens": ref_T_frames,
                "steps_per_sec": round(sps, 3),
                "elapsed_sec": round(elapsed, 1),
                "eta_sec": round(eta, 1),
                "mode_counts": dict(mode_counts),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            # Add multi-aux metrics
            # aux_log_count = number of micro-batches where sigma < aux_sigma_max
            total_micro_batches = args.log_every * args.grad_accum

            # Adversarial mode logging
            if adversarial_mode and aux_log_count > 0:
                avg_total = accum_total_loss / aux_log_count
                disc_loss_str = f"d_loss={adv_disc_train_stats.get('disc_loss', 0):.4f}" if adv_disc_train_stats else "d_loss=N/A"
                disc_acc_str = f"d_acc={adv_disc_train_stats.get('disc_acc', 0):.3f}" if adv_disc_train_stats else "d_acc=N/A"
                log_msg += (
                    f" | d_online={adv_score_online_val:.3f} d_ema={adv_score_ema_val:.3f}"
                    f" | {disc_loss_str} {disc_acc_str}"
                    f" | total={avg_total:.4f}"
                    f" | aux_hit={aux_log_count}/{total_micro_batches}"
                    f" | buf={len(adv_real_buffer)}"
                    f" | VRAM={vram_peak_mb/1024:.1f}GB"
                )
                metric.update({
                    "disc_score_online": round(adv_score_online_val, 4),
                    "disc_score_ema": round(adv_score_ema_val, 4),
                    "disc_train_loss": round(adv_disc_train_stats.get("disc_loss", 0), 4),
                    "disc_train_acc": round(adv_disc_train_stats.get("disc_acc", 0), 4),
                    "total_loss": round(avg_total, 6),
                    "disc_buffer_size": len(adv_real_buffer),
                    "vram_peak_mb": round(vram_peak_mb, 0),
                })

            elif aux_enabled and aux_log_count > 0:
                avg_total = accum_total_loss / aux_log_count
                avg_aux1 = accum_aux1 / aux_log_count
                avg_aux2 = accum_aux2 / aux_log_count
                avg_aux3 = accum_aux3 / aux_log_count
                avg_aux4 = accum_aux4 / aux_log_count
                avg_aux5 = accum_aux5 / aux_log_count
                avg_aux6 = accum_aux6 / aux_log_count
                avg_coeff1 = accum_coeff1 / aux_log_count
                avg_coeff2 = accum_coeff2 / aux_log_count
                avg_coeff3 = accum_coeff3 / aux_log_count
                avg_coeff4 = accum_coeff4 / aux_log_count
                avg_coeff5 = accum_coeff5 / aux_log_count
                avg_coeff6 = accum_coeff6 / aux_log_count
                avg_clap_text = accum_clap_text_reward / aux_log_count
                avg_quality = accum_quality_prob / aux_log_count
                avg_naturalness = accum_naturalness_reward / aux_log_count
                avg_centroid = accum_centroid_score / aux_log_count
                avg_speaker = (accum_speaker_sim / speaker_sim_count
                               if speaker_sim_count > 0 else None)
                avg_comb = (accum_comb_score / comb_log_count
                            if comb_log_count > 0 else None)
                avg_artifact_v2 = (accum_artifact_v2_score / aux_log_count
                                    if accum_artifact_v2_score > 0 else None)
                avg_clap_artifact = (accum_clap_artifact_score / aux_log_count
                                      if accum_clap_artifact_score > 0 else None)

                log_msg += (
                    f" | nat={avg_naturalness:.3f} cent={avg_centroid:.3f}"
                )
                if avg_speaker is not None:
                    log_msg += f" spk={avg_speaker:.3f}"
                if avg_comb is not None:
                    log_msg += f" comb={avg_comb:.3f}"
                if avg_artifact_v2 is not None:
                    log_msg += f" artv2={avg_artifact_v2:.3f}"
                if avg_clap_artifact is not None:
                    log_msg += f" clapart={avg_clap_artifact:.3f}"
                if rejection_enabled:
                    total_scored = acceptance_count + rejection_count
                    accept_pct = 100 * acceptance_count / max(total_scored, 1)
                    log_msg += (
                        f" | reject={rejection_count}/{total_scored}"
                        f" ({accept_pct:.0f}% accepted)"
                        f" | aux_hit={aux_log_count}/{total_micro_batches}"
                    )
                elif differentiable_reward:
                    coeff_str = f"c1={avg_coeff1:.2f} c2={avg_coeff2:.2f} c3={avg_coeff3:.2f} c4={avg_coeff4:.2f}"
                    if use_artifact_v2: coeff_str += f" c5={avg_coeff5:.2f}"
                    if use_clap_artifact: coeff_str += f" c6={avg_coeff6:.2f}"
                    log_msg += (
                        f" | {coeff_str}"
                        f" | total={avg_total:.4f}"
                        f" | aux_hit={aux_log_count}/{total_micro_batches}"
                        f" | VRAM={vram_peak_mb/1024:.1f}GB"
                    )
                else:
                    coeff_str = f"c1={avg_coeff1:.2f} c2={avg_coeff2:.2f} c3={avg_coeff3:.2f} c4={avg_coeff4:.2f}"
                    if use_artifact_v2: coeff_str += f" c5={avg_coeff5:.2f}"
                    if use_clap_artifact: coeff_str += f" c6={avg_coeff6:.2f}"
                    log_msg += (
                        f" | {coeff_str}"
                        f" | total={avg_total:.4f}"
                        f" | aux_hit={aux_log_count}/{total_micro_batches}"
                    )

                metric.update({
                    "clap_text_reward": round(avg_clap_text, 4),
                    "quality_prob": round(avg_quality, 4),
                    "naturalness_reward": round(avg_naturalness, 4),
                    "centroid_score": round(avg_centroid, 4),
                    "aux1_loss": round(avg_aux1, 6),
                    "aux2_loss": round(avg_aux2, 6),
                    "aux3_loss": round(avg_aux3, 6),
                    "aux4_loss": round(avg_aux4, 6),
                    "aux5_loss": round(avg_aux5, 6),
                    "aux6_loss": round(avg_aux6, 6),
                    "coeff1": round(avg_coeff1, 4),
                    "coeff2": round(avg_coeff2, 4),
                    "coeff3": round(avg_coeff3, 4),
                    "coeff4": round(avg_coeff4, 4),
                    "coeff5": round(avg_coeff5, 4),
                    "coeff6": round(avg_coeff6, 4),
                    "total_loss": round(avg_total, 6),
                })
                if avg_speaker is not None:
                    metric["speaker_sim"] = round(avg_speaker, 4)
                if avg_comb is not None:
                    metric["comb_score"] = round(avg_comb, 4)
                if avg_artifact_v2 is not None:
                    metric["artifact_v2_score"] = round(avg_artifact_v2, 4)
                if avg_clap_artifact is not None:
                    metric["clap_artifact_score"] = round(avg_clap_artifact, 4)
                if differentiable_reward and vram_peak_mb > 0:
                    metric["vram_peak_mb"] = round(vram_peak_mb, 0)

                # Reset accumulators
                accum_total_loss = 0.0
                accum_aux1 = 0.0
                accum_aux2 = 0.0
                accum_aux3 = 0.0
                accum_aux4 = 0.0
                accum_aux5 = 0.0
                accum_aux6 = 0.0
                accum_coeff1 = 0.0
                accum_coeff2 = 0.0
                accum_coeff3 = 0.0
                accum_coeff4 = 0.0
                accum_coeff5 = 0.0
                accum_coeff6 = 0.0
                accum_clap_text_reward = 0.0
                accum_quality_prob = 0.0
                accum_naturalness_reward = 0.0
                accum_centroid_score = 0.0
                accum_speaker_sim = 0.0
                accum_comb_score = 0.0
                accum_artifact_v2_score = 0.0
                accum_clap_artifact_score = 0.0
                aux_log_count = 0
                speaker_sim_count = 0
                comb_log_count = 0

            logging.info(log_msg)

            with open(metrics_path, "a") as f:
                f.write(json.dumps(metric) + "\n")

            # Update status file
            status = {
                "step": step,
                "total_steps": args.steps,
                "epoch": current_epoch,
                "total_epochs": args.epochs if args.epochs > 0 else 0,
                "steps_per_epoch": steps_per_epoch,
                "flow_loss": round(avg_loss, 6),
                "best_loss": round(best_loss, 6),
                "best_step": best_step,
                "lr": lr,
                "elapsed_sec": round(elapsed, 1),
                "eta_sec": round(eta, 1),
                "steps_per_sec": round(sps, 3),
                "world_size": accelerator.num_processes,
                "mode_counts": dict(mode_counts),
                "aux_enabled": aux_enabled,
                "aux_losses": [l for l, e in [
                    ("naturalness", True), ("quality_mlp", use_quality_mlp),
                    ("centroid", use_centroid), ("speaker_sim", use_speaker_sim),
                    ("comb_detector", use_comb_detector),
                ] if e],
            }
            with open(os.path.join(args.output_dir, "status.json"), "w") as f:
                json.dump(status, f, indent=2)

            # Best checkpoint (skip separate best save for full FT to save disk)
            if avg_loss < best_loss:
                best_loss = avg_loss
                old_best = os.path.join(args.output_dir, f"best_step_{best_step:05d}.safetensors")
                best_step = step
                new_best = os.path.join(args.output_dir, f"best_step_{best_step:05d}.safetensors")
                if args.full_ft:
                    pass  # periodic saves are sufficient for full FT (6 GB each)
                else:
                    unwrapped = _unwrap_model_safe(model)
                    unwrapped.save_pretrained(args.output_dir)
                    adapter = os.path.join(args.output_dir, "adapter_model.safetensors")
                    if os.path.exists(adapter):
                        shutil.copy(adapter, new_best)
                if not args.full_ft and old_best != new_best and os.path.exists(old_best):
                    os.remove(old_best)
                logging.info(f"  New best: loss={best_loss:.4f} at step {best_step}")

            accum_loss = 0.0
            rejection_count = 0
            acceptance_count = 0

        # Periodic save (with optimizer state for resumability + smart checkpoint management)
        if is_opt_step and step % args.save_every == 0 and is_main:
            prefix = "model_step" if args.full_ft else "lora_step"
            save_path = os.path.join(args.output_dir, f"{prefix}_{step:05d}.safetensors")
            logging.info(f"Saving: {save_path}")
            if args.full_ft:
                _save_full_ft_model(model, accelerator, save_path)
            else:
                unwrapped = _unwrap_model_safe(model)
                unwrapped.save_pretrained(args.output_dir)
                adapter = os.path.join(args.output_dir, "adapter_model.safetensors")
                if os.path.exists(adapter):
                    shutil.copy(adapter, save_path)
            if not getattr(args, 'no_save_state', False):
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
                latest_path = os.path.join(args.output_dir, "training_state.pt")
                if os.path.islink(latest_path) or os.path.exists(latest_path):
                    os.remove(latest_path)
                os.symlink(os.path.basename(state_path), latest_path)

            # Smart checkpoint management: rolling window + keep better older ones
            if recent_checkpoints is not None:
                # Record reward for this checkpoint (use latest naturalness_reward)
                current_reward = naturalness_reward_val
                checkpoint_rewards[save_path] = current_reward

                # Check if an older checkpoint is being evicted from the deque
                evicted_path = None
                if len(recent_checkpoints) == recent_checkpoints.maxlen:
                    evicted_path = recent_checkpoints[0]  # will be pushed out

                recent_checkpoints.append(save_path)

                if evicted_path and evicted_path in checkpoint_rewards:
                    evicted_reward = checkpoint_rewards[evicted_path]
                    # Compare to min reward in current deque
                    deque_rewards = [checkpoint_rewards.get(p, float('-inf'))
                                     for p in recent_checkpoints]
                    min_deque_reward = min(deque_rewards) if deque_rewards else float('-inf')

                    if evicted_reward < min_deque_reward:
                        # Worse than all current → delete
                        if os.path.exists(evicted_path):
                            os.remove(evicted_path)
                            logging.info(f"  Deleted checkpoint: {os.path.basename(evicted_path)} "
                                         f"(reward={evicted_reward:.4f} < min_deque={min_deque_reward:.4f})")
                        # Delete corresponding training state
                        evicted_step = evicted_path.split("step_")[1].split(".")[0]
                        evicted_state = os.path.join(
                            args.output_dir, f"training_state_step_{evicted_step}.pt")
                        if os.path.exists(evicted_state):
                            os.remove(evicted_state)
                        del checkpoint_rewards[evicted_path]
                    else:
                        logging.info(f"  Keeping older checkpoint: {os.path.basename(evicted_path)} "
                                     f"(reward={evicted_reward:.4f} >= min_deque={min_deque_reward:.4f})")

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
                    scheduler, best_loss, best_step, model, accelerator,
                    full_ft=args.full_ft)
                # NOTE: Validation disabled at epoch boundary to prevent OOM —
                # the validation subprocess uses ~14GB on GPU 0/4, which combined
                # with differentiable reward (~66GB) exceeds 80GB and crashes.
                # Validation runs only after training completes (see below).
                # if args.val_samples > 0:
                #     proc = launch_validation_async(
                #         script_dir, args.output_dir, lora_path,
                #         current_epoch, args)
                #     if proc:
                #         val_procs.append(proc)

    # Final save
    if is_main:
        final_epoch = args.epochs if args.epochs > 0 else 0
        lora_path = save_training_state(
            args.output_dir, step, final_epoch, optimizer,
            scheduler, best_loss, best_step, model, accelerator, tag="final",
            full_ft=args.full_ft)
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
