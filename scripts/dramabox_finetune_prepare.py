#!/usr/bin/env python3
"""
DramaBox Fine-Tuning Data Preparation

Downloads annotated TARs from HuggingFace, ranks samples by reward,
selects top-K per prompt group, and pre-encodes audio latents + text
conditions for efficient training.

Usage:
    python scripts/dramabox_finetune_prepare.py --config configs/finetune.yaml
    python scripts/dramabox_finetune_prepare.py --config configs/finetune.yaml --phase rank-only
    python scripts/dramabox_finetune_prepare.py --config configs/finetune.yaml --phase encode-only
"""

import os
import sys

# Filter out conda ml-general paths that break native cuDNN libraries
_ld = os.environ.get("LD_LIBRARY_PATH", "")
if _ld:
    _filtered = [p for p in _ld.split(":") if "ml-general" not in p]
    os.environ["LD_LIBRARY_PATH"] = ":".join(_filtered)

import argparse
import json
import logging
import struct
import tarfile
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import torch
import torchaudio
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_hf_token() -> str:
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        token_path = Path.home() / ".cache" / "huggingface" / "token"
        if token_path.exists():
            token = token_path.read_text().strip()
    return token


# ── Phase 1: Download + Rank + Select ────────────────────────────────

def download_and_rank(config: dict):
    """Download all annotated TARs, extract JSON annotations, rank by reward."""
    from huggingface_hub import hf_hub_download

    repo = config["annotated_repo"]
    total = config["total_batches"]
    top_k = config.get("top_k", 3)
    min_reward = config.get("min_reward", 0.05)
    skip_singing = config.get("skip_singing", True)
    require_cut_to = config.get("require_cut_to", True)
    max_dur = config.get("max_duration_sec", 20.0)
    min_dur = config.get("min_duration_sec", 2.0)

    out_dir = Path(config["preprocessed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.json"

    # Check for existing manifest
    if manifest_path.exists():
        log.info(f"Loading existing manifest from {manifest_path}")
        with open(manifest_path) as f:
            manifest = json.load(f)
        log.info(f"Loaded {len(manifest['selected_samples'])} samples from manifest")
        return manifest

    token = get_hf_token()
    all_annotations = []

    log.info(f"Downloading annotations from {repo} ({total} batches)...")

    for batch_idx in range(total):
        tar_name = f"data/batch_{batch_idx:06d}.tar"
        try:
            tar_path = hf_hub_download(
                repo_id=repo,
                filename=tar_name,
                repo_type="dataset",
                token=token or None,
            )
        except Exception as e:
            log.warning(f"Batch {batch_idx}: download failed: {e}")
            continue

        # Extract only JSON files
        try:
            with tarfile.open(tar_path, "r") as tf:
                for member in tf.getmembers():
                    if not member.name.endswith(".json"):
                        continue
                    f = tf.extractfile(member)
                    if f is None:
                        continue
                    try:
                        ann = json.loads(f.read().decode("utf-8"))
                    except Exception:
                        continue

                    # Apply filters
                    if skip_singing and ann.get("singing_flag", False):
                        continue
                    if require_cut_to and not ann.get("has_cut_to", False):
                        continue

                    full_dur = ann.get("full_duration_sec", 0)
                    p1_dur = ann.get("part1_duration_sec", 0)
                    p2_dur = ann.get("part2_duration_sec", 0)
                    if full_dur > max_dur or full_dur < min_dur:
                        continue
                    if p1_dur < min_dur or p2_dur < min_dur:
                        continue

                    reward = ann.get("reward_full", 0)
                    if reward < min_reward:
                        continue

                    # Extract prompt group (prompt_id without seed)
                    prompt_id = ann.get("prompt_id", "")
                    seed = ann.get("seed", 0)

                    basename = os.path.splitext(os.path.basename(member.name))[0]

                    all_annotations.append({
                        "batch_idx": batch_idx,
                        "prompt_id": prompt_id,
                        "seed": seed,
                        "basename": basename,
                        "reward_full": reward,
                        "reward_part1": ann.get("reward_part1", 0),
                        "reward_part2": ann.get("reward_part2", 0),
                        "full_duration_sec": full_dur,
                        "part1_duration_sec": p1_dur,
                        "part2_duration_sec": p2_dur,
                        "original_prompt": ann.get("original_prompt", ""),
                        "modified_prompt": ann.get("modified_prompt", ""),
                        "scene1_expected_text": ann.get("scene1_expected_text", ""),
                        "scene2_expected_text": ann.get("scene2_expected_text", ""),
                        "format": ann.get("format", ""),
                    })
        except Exception as e:
            log.warning(f"Batch {batch_idx}: extract failed: {e}")
            continue

        if (batch_idx + 1) % 50 == 0:
            log.info(f"  Scanned {batch_idx + 1}/{total} batches, {len(all_annotations)} annotations so far")

    log.info(f"Total annotations after filtering: {len(all_annotations)}")

    # Group by prompt_id and rank
    groups = defaultdict(list)
    for ann in all_annotations:
        groups[ann["prompt_id"]].append(ann)

    log.info(f"Prompt groups: {len(groups)}")

    # Select top-K from each group
    selected = []
    for prompt_id, samples in groups.items():
        samples.sort(key=lambda x: x["reward_full"], reverse=True)
        for rank, sample in enumerate(samples[:top_k]):
            sample["rank"] = rank
            selected.append(sample)

    log.info(f"Selected {len(selected)} samples (top-{top_k} from {len(groups)} groups)")

    manifest = {
        "total_annotations": len(all_annotations),
        "total_groups": len(groups),
        "top_k": top_k,
        "selected_samples": selected,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    log.info(f"Saved manifest to {manifest_path}")

    return manifest


# ── Phase 2: Audio VAE Encoding ──────────────────────────────────────

def encode_audio_samples(config: dict, manifest: dict):
    """Encode all selected audio (full, part1, part2) through Audio VAE."""
    dramabox_dir = config["dramabox_dir"]
    sys.path.insert(0, os.path.join(dramabox_dir, "ltx2"))
    sys.path.insert(0, os.path.join(dramabox_dir, "src"))

    from huggingface_hub import hf_hub_download
    from ltx_core.model.audio_vae import encode_audio as vae_encode_audio
    from ltx_core.types import Audio
    from ltx_pipelines.utils.blocks import AudioConditioner

    repo = config["annotated_repo"]
    token = get_hf_token()
    out_dir = Path(config["preprocessed_dir"])
    latent_dir = out_dir / "audio_latents"
    latent_dir.mkdir(parents=True, exist_ok=True)

    full_ckpt = config["full_checkpoint"]
    device = torch.device("cuda")
    dtype = torch.bfloat16

    # Check what's already done
    selected = manifest["selected_samples"]
    todo = []
    for i, sample in enumerate(selected):
        full_path = latent_dir / f"sample_{i:06d}_full.pt"
        p1_path = latent_dir / f"sample_{i:06d}_part1.pt"
        p2_path = latent_dir / f"sample_{i:06d}_part2.pt"
        if full_path.exists() and p1_path.exists() and p2_path.exists():
            continue
        todo.append((i, sample))

    if not todo:
        log.info("All audio latents already encoded, skipping.")
        return

    log.info(f"Encoding {len(todo)} samples through Audio VAE...")

    # Load VAE
    ac = AudioConditioner(checkpoint_path=full_ckpt, dtype=dtype, device=device)

    # Group by batch to download each TAR only once
    batch_groups = defaultdict(list)
    for idx, sample in todo:
        batch_groups[sample["batch_idx"]].append((idx, sample))

    processed = 0
    for batch_idx in sorted(batch_groups.keys()):
        items = batch_groups[batch_idx]
        tar_name = f"data/batch_{batch_idx:06d}.tar"

        try:
            tar_path = hf_hub_download(
                repo_id=repo, filename=tar_name,
                repo_type="dataset", token=token or None,
            )
        except Exception as e:
            log.warning(f"Batch {batch_idx}: download failed: {e}")
            continue

        # Extract needed files
        with tempfile.TemporaryDirectory() as tmpdir:
            basenames = {s["basename"] for _, s in items}
            with tarfile.open(tar_path, "r") as tf:
                for member in tf.getmembers():
                    name = os.path.basename(member.name)
                    name_no_ext = os.path.splitext(name)[0]
                    # Check if this file belongs to a needed sample
                    base = name_no_ext.replace("_full", "").replace("_part1", "").replace("_part2", "")
                    if base in basenames and name.endswith(".mp3"):
                        tf.extract(member, tmpdir)

            # Encode each sample
            for idx, sample in items:
                basename = sample["basename"]
                # Find extracted files
                mp3_files = {}
                for root, dirs, files in os.walk(tmpdir):
                    for fname in files:
                        if fname.startswith(basename) and fname.endswith(".mp3"):
                            if "_full.mp3" in fname:
                                mp3_files["full"] = os.path.join(root, fname)
                            elif "_part1.mp3" in fname:
                                mp3_files["part1"] = os.path.join(root, fname)
                            elif "_part2.mp3" in fname:
                                mp3_files["part2"] = os.path.join(root, fname)

                if len(mp3_files) < 3:
                    log.warning(f"Sample {idx} ({basename}): missing MP3 files, got {list(mp3_files.keys())}")
                    continue

                try:
                    for part_name, mp3_path in mp3_files.items():
                        out_path = latent_dir / f"sample_{idx:06d}_{part_name}.pt"
                        if out_path.exists():
                            continue

                        # Load and encode
                        wav, sr = torchaudio.load(mp3_path)
                        if wav.shape[0] == 1:
                            wav = wav.repeat(2, 1)
                        wav = wav.unsqueeze(0).to(device)  # [1, 2, samples]

                        audio_obj = Audio(waveform=wav, sampling_rate=sr)
                        latent = ac(lambda enc: vae_encode_audio(audio_obj, enc, None))
                        torch.save(latent.squeeze(0).cpu(), out_path)  # [8, T, 16]

                    processed += 1
                    if processed % 100 == 0:
                        log.info(f"  Encoded {processed}/{len(todo)} samples")
                except Exception as e:
                    log.warning(f"Sample {idx} ({basename}): encode failed: {e}")
                    continue

    del ac
    torch.cuda.empty_cache()
    log.info(f"Audio encoding complete: {processed} samples")


# ── Phase 3: Text Encoding ───────────────────────────────────────────

def _load_gemma_bnb4bit(gemma_root: str, device, dtype):
    """Load Gemma text encoder using bitsandbytes 4-bit quantization.

    The standard load_text_encoder uses SingleGPUModelBuilder which doesn't
    handle pre-quantized bnb-4bit checkpoints. This uses from_pretrained
    which correctly loads the packed 4-bit weights.
    """
    from transformers import Gemma3ForConditionalGeneration
    from ltx_core.text_encoders.gemma.tokenizer import LTXVGemmaTokenizer
    from ltx_core.text_encoders.gemma.encoders.base_encoder import GemmaTextEncoder
    from ltx_core.utils import find_matching_file

    # Detect if checkpoint is pre-quantized
    cfg_path = os.path.join(gemma_root, "config.json")
    prequantized = False
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            prequantized = "quantization_config" in json.load(f)

    from_kwargs = {"device_map": str(device), "torch_dtype": dtype}
    if prequantized:
        log.info("Loading pre-quantized Gemma (bnb-4bit)...")
    else:
        from transformers import BitsAndBytesConfig
        log.info("Loading Gemma with runtime bnb 4-bit quantization...")
        from_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
        )

    hf_model = Gemma3ForConditionalGeneration.from_pretrained(gemma_root, **from_kwargs)
    tokenizer_root = str(find_matching_file(gemma_root, "tokenizer.model").parent)
    tokenizer = LTXVGemmaTokenizer(tokenizer_root, 1024)
    encoder = GemmaTextEncoder(model=hf_model, tokenizer=tokenizer, dtype=dtype)
    mem_gb = torch.cuda.memory_allocated(device) / 1e9
    log.info(f"Gemma 4-bit loaded: {mem_gb:.1f}GB VRAM")
    return encoder


def encode_text_conditions(config: dict, manifest: dict):
    """Encode all text prompts through Gemma + Feature Extractor."""
    dramabox_dir = config["dramabox_dir"]
    sys.path.insert(0, os.path.join(dramabox_dir, "ltx2"))
    sys.path.insert(0, os.path.join(dramabox_dir, "src"))

    from ltx_trainer.model_loader import load_embeddings_processor

    out_dir = Path(config["preprocessed_dir"])
    cond_dir = out_dir / "conditions"
    cond_dir.mkdir(parents=True, exist_ok=True)

    full_ckpt = config["full_checkpoint"]
    gemma_root = config.get("gemma_root", "gemma-3-12b-it-qat-q4_0-unquantized")
    device = torch.device("cuda")
    dtype = torch.bfloat16

    selected = manifest["selected_samples"]

    # Check what's already done
    todo = []
    for i, sample in enumerate(selected):
        full_path = cond_dir / f"sample_{i:06d}_full.pt"
        p1_path = cond_dir / f"sample_{i:06d}_part1.pt"
        p2_path = cond_dir / f"sample_{i:06d}_part2.pt"
        if full_path.exists() and p1_path.exists() and p2_path.exists():
            continue
        todo.append((i, sample))

    if not todo:
        log.info("All text conditions already encoded, skipping.")
        return

    log.info(f"Encoding {len(todo)} samples' text through Gemma...")

    # Load text encoder via bnb-4bit (handles pre-quantized checkpoints)
    text_encoder = _load_gemma_bnb4bit(gemma_root, device, dtype)

    # Load feature extractor from embeddings processor
    emb_proc = load_embeddings_processor(full_ckpt, device=device, dtype=dtype)
    feature_extractor = emb_proc.feature_extractor

    # The video_aggregate_embed may be on meta device (not in audio-only checkpoint).
    # Replace with a dummy since we only need audio features.
    fe = feature_extractor
    if hasattr(fe, 'video_aggregate_embed'):
        vae = fe.video_aggregate_embed
        if any(p.device.type == 'meta' for p in vae.parameters()):
            out_f = vae.out_features
            in_f = vae.in_features
            class _DummyLinear(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.out_features = out_f
                    self.in_features = in_f
                def forward(self, x):
                    return torch.zeros(*x.shape[:-1], self.out_features, device=x.device, dtype=x.dtype)
            fe.video_aggregate_embed = _DummyLinear()
            log.info("Replaced meta video_aggregate_embed with dummy (audio-only mode)")

    del emb_proc
    torch.cuda.empty_cache()

    processed = 0
    for idx, sample in todo:
        try:
            # Three text prompts per sample
            prompts = {
                "full": sample.get("modified_prompt") or sample.get("original_prompt", ""),
                "part1": sample.get("scene1_expected_text", ""),
                "part2": sample.get("scene2_expected_text", ""),
            }

            for part_name, text in prompts.items():
                out_path = cond_dir / f"sample_{idx:06d}_{part_name}.pt"
                if out_path.exists():
                    continue
                if not text.strip():
                    # Use full prompt as fallback
                    text = prompts["full"]

                with torch.no_grad():
                    hidden_states, attention_mask = text_encoder.encode(text)
                    _, audio_feats = feature_extractor(
                        hidden_states, attention_mask, "left"
                    )
                    if audio_feats is None:
                        audio_feats = hidden_states

                torch.save({
                    "audio_prompt_embeds": audio_feats.squeeze(0).cpu(),
                    "prompt_attention_mask": attention_mask.squeeze(0).bool().cpu(),
                }, out_path)

            processed += 1
            if processed % 100 == 0:
                log.info(f"  Encoded {processed}/{len(todo)} text conditions")

        except Exception as e:
            log.warning(f"Sample {idx}: text encode failed: {e}")
            continue

    del text_encoder, feature_extractor
    torch.cuda.empty_cache()
    log.info(f"Text encoding complete: {processed} samples")


# ── Phase 4: Build Index ─────────────────────────────────────────────

def build_index(config: dict, manifest: dict):
    """Create index file mapping sample indices to metadata for the dataset."""
    out_dir = Path(config["preprocessed_dir"])
    latent_dir = out_dir / "audio_latents"
    cond_dir = out_dir / "conditions"
    index_path = out_dir / "index.json"

    selected = manifest["selected_samples"]
    valid_samples = []

    for i, sample in enumerate(selected):
        # Verify both audio latent and text condition files exist
        files_ok = True
        for part in ["full", "part1", "part2"]:
            if not (latent_dir / f"sample_{i:06d}_{part}.pt").exists():
                files_ok = False
                break
            if not (cond_dir / f"sample_{i:06d}_{part}.pt").exists():
                files_ok = False
                break

        if files_ok:
            valid_samples.append({
                "index": i,
                "prompt_id": sample["prompt_id"],
                "seed": sample["seed"],
                "rank": sample["rank"],
                "reward_full": sample["reward_full"],
                "full_duration_sec": sample["full_duration_sec"],
                "part1_duration_sec": sample["part1_duration_sec"],
                "part2_duration_sec": sample["part2_duration_sec"],
            })

    # Group by prompt_id for epoch sampling
    groups = defaultdict(list)
    for s in valid_samples:
        groups[s["prompt_id"]].append(s)

    index = {
        "total_valid": len(valid_samples),
        "total_groups": len(groups),
        "samples": valid_samples,
        "groups": {k: [s["index"] for s in v] for k, v in groups.items()},
    }

    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)

    log.info(f"Built index: {len(valid_samples)} valid samples in {len(groups)} groups")
    log.info(f"Saved to {index_path}")


# ── Phase 5: Upload to HuggingFace ───────────────────────────────────

def upload_preprocessed(config: dict):
    """Upload preprocessed tensors and metadata to HuggingFace."""
    from huggingface_hub import HfApi

    repo = config.get("annotated_repo", "laion/dramabox-voice-acting-data-annotated")
    token = get_hf_token()
    out_dir = Path(config["preprocessed_dir"])

    if not out_dir.exists():
        log.error(f"Preprocessed dir does not exist: {out_dir}")
        return

    api = HfApi(token=token)

    # Upload the entire preprocessed directory as a folder
    log.info(f"Uploading preprocessed data from {out_dir} to {repo}...")
    log.info("  This includes: manifest.json, index.json, audio_latents/, conditions/")

    api.upload_folder(
        repo_id=repo,
        repo_type="dataset",
        folder_path=str(out_dir),
        path_in_repo="finetune_preprocessed",
        commit_message="Add preprocessed fine-tuning tensors (audio latents + text conditions + index)",
    )
    log.info(f"Upload complete: {repo}/finetune_preprocessed/")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare DramaBox fine-tuning data")
    parser.add_argument("--config", required=True, help="Path to finetune.yaml")
    parser.add_argument("--phase", choices=["all", "rank-only", "encode-only", "text-only", "index-only", "upload-only"],
                        default="all", help="Which phase to run")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device for encoding")
    parser.add_argument("--upload", action="store_true", help="Upload preprocessed data to HuggingFace after processing")
    args = parser.parse_args()

    config = load_config(args.config)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))

    if args.phase == "upload-only":
        upload_preprocessed(config)
        return

    if args.phase in ("all", "rank-only"):
        manifest = download_and_rank(config)
    else:
        manifest_path = Path(config["preprocessed_dir"]) / "manifest.json"
        with open(manifest_path) as f:
            manifest = json.load(f)

    if args.phase in ("all", "encode-only"):
        encode_audio_samples(config, manifest)
        encode_text_conditions(config, manifest)

    if args.phase == "text-only":
        encode_text_conditions(config, manifest)

    if args.phase in ("all", "text-only", "index-only"):
        build_index(config, manifest)

    log.info("Data preparation complete!")

    if args.upload or args.phase == "all":
        upload_preprocessed(config)


if __name__ == "__main__":
    main()
