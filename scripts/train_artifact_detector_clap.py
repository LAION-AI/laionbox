#!/usr/bin/env python3
"""
Train an artifact detector on VoiceCLAP-small embeddings.

Pipeline:
  1. Load AudioDecoder (VAE decoder + vocoder) + VoiceCLAP-small on a single GPU
  2. Decode all latents → waveforms → CLAP embeddings (768-dim)
  3. Cache embeddings to disk
  4. Train MLP classifiers on CLAP embeddings

Data sources (same as artifact_detector_v2):
  Label 1 (artifact):
    - Model predictions: artifact_detector_v2_data/latents/{model_id}/*.pt  (4 × 1500 = 6000)
    - Comb-augmented: comb_filter_detector/latents/*_comb.pt                (6000)

  Label 0 (clean):
    - Ground truth targets: finetune_data_combined/audio_latents/*_part2.pt (1500)
    - Clean comb pairs: comb_filter_detector/latents/*_clean.pt             (6000)

Usage:
    python scripts/train_artifact_detector_clap.py [--gpu 5] [--batch_size 4] [--skip_embed]
"""

import argparse
import glob
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path("/home/deployer/laion/Voice-Acting-Pipeline")
PRED_DIR = BASE_DIR / "artifact_detector_v2_data" / "latents"
TARGET_DIR = BASE_DIR / "finetune_data_combined" / "audio_latents"
COMB_DIR = BASE_DIR / "comb_filter_detector" / "latents"
OUTPUT_DIR = BASE_DIR / "artifact_detector_clap"
EMBED_CACHE_DIR = OUTPUT_DIR / "clap_embeddings"

MODEL_IDS = ["v01_wip", "4aux_s080", "4aux_s150", "4aux_s180"]
FULL_CHECKPOINT = "/home/deployer/laion/DramaBox/models/ltx-2.3-22b-dev.safetensors"

SEED = 42
VAL_SIZE = 400  # 200 per class


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Gather all latent file paths ─────────────────────────────────────────────

def gather_items():
    """Gather all (path, label, source_tag) tuples — same logic as V2 trainer."""
    items = []
    stats = defaultdict(int)

    # Label 1: Model predictions
    for model_id in MODEL_IDS:
        model_dir = PRED_DIR / model_id
        if not model_dir.exists():
            log.warning(f"Missing: {model_dir}")
            continue
        files = sorted(model_dir.glob("*.pt"))
        for f in files:
            items.append((str(f), 1, f"pred_{model_id}"))
        stats[f"pred_{model_id}"] = len(files)

    # Label 1: Comb-augmented
    comb_files = sorted(COMB_DIR.glob("*_comb.pt"))
    for f in comb_files:
        items.append((str(f), 1, "comb_augmented"))
    stats["comb_augmented"] = len(comb_files)

    # Label 0: Ground truth targets
    pred_sample_ids = set()
    for model_id in MODEL_IDS:
        model_dir = PRED_DIR / model_id
        if model_dir.exists():
            for f in model_dir.glob("*.pt"):
                pred_sample_ids.add(f.stem)
            if pred_sample_ids:
                break
    target_count = 0
    for sample_id in sorted(pred_sample_ids):
        target_path = TARGET_DIR / f"{sample_id}_part2.pt"
        if target_path.exists():
            items.append((str(target_path), 0, "ground_truth"))
            target_count += 1
    stats["ground_truth"] = target_count

    # Label 0: Clean comb pairs
    clean_files = sorted(COMB_DIR.glob("*_clean.pt"))
    for f in clean_files:
        items.append((str(f), 0, "comb_clean"))
    stats["comb_clean"] = len(clean_files)

    log.info("Dataset composition:")
    for source, count in sorted(stats.items()):
        label = 1 if source.startswith("pred_") or source == "comb_augmented" else 0
        log.info(f"  {source}: {count} (label={label})")
    log.info(f"  Total: {len(items)}")

    return items, dict(stats)


# ── Latent → Audio → CLAP embedding ─────────────────────────────────────────

def setup_decoder_and_clap(device, dtype=torch.bfloat16):
    """Load AudioDecoder + VoiceCLAP-small on specified device."""
    # Add DramaBox source dirs to path
    sys.path.insert(0, "/home/deployer/laion/DramaBox/src")
    sys.path.insert(0, "/home/deployer/laion/DramaBox/ltx2")
    from ltx_pipelines.utils.blocks import AudioDecoder

    log.info("Loading AudioDecoder (VAE decoder + vocoder)...")
    audio_decoder = AudioDecoder(
        checkpoint_path=FULL_CHECKPOINT,
        dtype=dtype,
        device=device,
        warm=True,
    )

    log.info("Loading VoiceCLAP-small...")
    from transformers import AutoModel, AutoTokenizer
    clap_model = AutoModel.from_pretrained(
        "laion/voiceclap-small", trust_remote_code=True
    ).eval().to(device)
    for p in clap_model.parameters():
        p.requires_grad = False

    log.info(f"AudioDecoder + CLAP loaded on {device}")
    return audio_decoder, clap_model


def decode_latent_to_audio(latent, audio_decoder, device, dtype=torch.bfloat16):
    """Decode a single latent [8, T, 16] to mono waveform [samples] at 16kHz.

    The AudioDecoder returns an Audio object with .waveform [B, 2, samples] at 24kHz.
    We convert to mono and resample to 16kHz for CLAP.
    """
    import torchaudio
    with torch.no_grad():
        # Add batch dim: [8, T, 16] → [1, 8, T, 16]
        lat = latent.unsqueeze(0).to(device=device, dtype=dtype)
        decoded = audio_decoder(lat)  # returns Audio object
        wav = decoded.waveform.squeeze(0).float()  # [2, samples] or [samples]
        sr = decoded.sampling_rate  # 24000

        # Stereo → mono
        if wav.ndim > 1:
            wav = wav.mean(0)  # [samples]

        # Resample to 16kHz
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)

    return wav  # [samples] at 16kHz on device


def encode_clap_audio(waveform_16k, clap_model, device):
    """Encode a mono 16kHz waveform → CLAP embedding [768]."""
    with torch.no_grad():
        wav = waveform_16k.to(device).float()
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)  # [1, T]

        emb = clap_model.encode_waveform(wav)  # [1, 768]
        emb = F.normalize(emb, p=2, dim=-1)

    return emb.squeeze(0).cpu()  # [768]


def embed_all_items(items, audio_decoder, clap_model, device, batch_size=1):
    """Convert all latents → CLAP embeddings, caching to disk.

    Returns dict: {path: (embedding_768, label, source_tag)}
    """
    EMBED_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = EMBED_CACHE_DIR / "all_embeddings.pt"

    if cache_file.exists():
        log.info(f"Loading cached embeddings from {cache_file}")
        cached = torch.load(cache_file, map_location="cpu", weights_only=False)
        log.info(f"  {len(cached['embeddings'])} embeddings loaded")
        return cached

    embeddings = []
    labels = []
    sources = []
    paths = []

    total = len(items)
    t0 = time.time()
    errors = 0

    for i, (path, label, source) in enumerate(items):
        try:
            latent = torch.load(path, weights_only=True, map_location="cpu").detach()
            waveform = decode_latent_to_audio(latent, audio_decoder, device)
            emb = encode_clap_audio(waveform, clap_model, device)

            embeddings.append(emb)
            labels.append(label)
            sources.append(source)
            paths.append(path)

        except Exception as e:
            errors += 1
            if errors <= 10:
                log.warning(f"Error on {path}: {e}")
            embeddings.append(torch.zeros(768))
            labels.append(label)
            sources.append(source)
            paths.append(path)
            # Clear cache on error (might be OOM)
            torch.cuda.empty_cache()

        if (i + 1) % 100 == 0 or i == total - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (total - i - 1) / rate if rate > 0 else 0
            vram = torch.cuda.memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0
            log.info(f"  [{i+1}/{total}] {rate:.1f} items/s | ETA {eta:.0f}s | "
                     f"errors={errors} | VRAM={vram:.1f}GB")

        # Periodic save every 2000 items
        if (i + 1) % 2000 == 0:
            partial = {
                "embeddings": torch.stack(embeddings),
                "labels": torch.tensor(labels),
                "sources": sources,
                "paths": paths,
            }
            partial_path = EMBED_CACHE_DIR / f"partial_{i+1}.pt"
            torch.save(partial, partial_path)
            log.info(f"  Saved partial checkpoint: {partial_path}")

    result = {
        "embeddings": torch.stack(embeddings),  # [N, 768]
        "labels": torch.tensor(labels),          # [N]
        "sources": sources,
        "paths": paths,
    }

    torch.save(result, cache_file)
    log.info(f"Saved {len(embeddings)} embeddings to {cache_file} "
             f"({os.path.getsize(cache_file) / 1e6:.1f} MB, {errors} errors)")

    return result


# ── MLP Classifiers ──────────────────────────────────────────────────────────

class ArtifactMLPSmall(nn.Module):
    """Small MLP: 768 → 256 → 64 → 1 (~200K params)"""
    def __init__(self, input_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


class ArtifactMLPMedium(nn.Module):
    """Medium MLP: 768 → 512 → 256 → 64 → 1 (~530K params)"""
    def __init__(self, input_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)


class ArtifactMLPLarge(nn.Module):
    """Large MLP with residual: 768 → 768 → 384 → 128 → 1 (~900K params)"""
    def __init__(self, input_dim=768):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Linear(input_dim, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(0.3),
        )
        self.block2 = nn.Sequential(
            nn.Linear(768, 384),
            nn.LayerNorm(384),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.head = nn.Sequential(
            nn.Linear(384, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
        )
        # Residual projection for block1
        self.res_proj = nn.Linear(input_dim, 768) if input_dim != 768 else nn.Identity()

    def forward(self, x):
        h = self.block1(x) + self.res_proj(x)
        h = self.block2(h)
        return self.head(h)


VARIANT_CLASSES = {
    "small": ArtifactMLPSmall,
    "medium": ArtifactMLPMedium,
    "large": ArtifactMLPLarge,
}


# ── Training ─────────────────────────────────────────────────────────────────

def split_train_val(embeddings, labels, sources, val_size=VAL_SIZE, seed=SEED):
    """Split into train/val with balanced classes."""
    rng = random.Random(seed)
    n = len(labels)
    indices = list(range(n))

    pos_idx = [i for i in indices if labels[i] == 1]
    neg_idx = [i for i in indices if labels[i] == 0]

    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    val_per_class = val_size // 2
    val_idx = set(pos_idx[:val_per_class] + neg_idx[:val_per_class])
    train_idx = [i for i in indices if i not in val_idx]
    val_idx = list(val_idx)

    return train_idx, val_idx


def train_classifier(variant_name, model_class, train_emb, train_labels,
                     val_emb, val_labels, sources_train, sources_val,
                     device, epochs=60, lr=3e-4, batch_size=256):
    """Train one MLP variant and return metrics."""
    log.info(f"\n{'='*60}")
    log.info(f"Training {variant_name} MLP")

    model = model_class().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"  Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    train_ds = TensorDataset(train_emb.to(device), train_labels.to(device))
    val_ds = TensorDataset(val_emb.to(device), val_labels.to(device))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val_acc = 0.0
    best_epoch = 0
    best_state = None
    patience = 15
    no_improve = 0

    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        for emb_batch, lab_batch in train_dl:
            logits = model(emb_batch).squeeze(-1)
            loss = criterion(logits, lab_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(lab_batch)
            preds = (logits > 0).float()
            train_correct += (preds == lab_batch).sum().item()
            train_total += len(lab_batch)
        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for emb_batch, lab_batch in val_dl:
                logits = model(emb_batch).squeeze(-1)
                loss = criterion(logits, lab_batch)
                val_loss += loss.item() * len(lab_batch)
                preds = (logits > 0).float()
                val_correct += (preds == lab_batch).sum().item()
                val_total += len(lab_batch)

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        avg_train_loss = train_loss / train_total
        avg_val_loss = val_loss / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if (epoch + 1) % 5 == 0 or val_acc >= 1.0:
            log.info(f"  Epoch {epoch+1:3d}/{epochs} | "
                     f"train_loss={avg_train_loss:.4f} acc={train_acc:.4f} | "
                     f"val_loss={avg_val_loss:.4f} acc={val_acc:.4f} | "
                     f"best={best_val_acc:.4f}@{best_epoch}")

        if val_acc >= 1.0 and epoch >= 5:
            log.info(f"  Perfect validation accuracy at epoch {epoch+1}, stopping")
            break

        if no_improve >= patience:
            log.info(f"  No improvement for {patience} epochs, stopping")
            break

    # Load best model and do per-source evaluation
    model.load_state_dict(best_state)
    model.eval()

    return model, best_val_acc, best_epoch, n_params


def per_source_eval(model, embeddings, labels, sources, device, n_per_source=50):
    """Evaluate model per source category."""
    model.eval()
    source_groups = defaultdict(list)
    for i, src in enumerate(sources):
        source_groups[src].append(i)

    log.info(f"\n  Per-source accuracy ({n_per_source} samples each):")
    results = {}
    for src in sorted(source_groups.keys()):
        indices = source_groups[src]
        if len(indices) > n_per_source:
            indices = random.Random(SEED).sample(indices, n_per_source)

        emb = embeddings[indices].to(device)
        lab = labels[indices].to(device)

        with torch.no_grad():
            logits = model(emb).squeeze(-1)
            scores = torch.sigmoid(logits)
            preds = (logits > 0).float()
            acc = (preds == lab).float().mean().item()

        expected_label = 1 if src.startswith("pred_") or src == "comb_augmented" else 0
        log.info(f"    {src:25s} | label={expected_label} | acc={acc:.3f} | "
                 f"score: mean={scores.mean():.3f} min={scores.min():.3f} max={scores.max():.3f}")
        results[src] = {"acc": acc, "mean_score": scores.mean().item(),
                        "min_score": scores.min().item(), "max_score": scores.max().item()}

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=5, help="GPU to use for embedding + training")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for latent decoding")
    parser.add_argument("--skip_embed", action="store_true", help="Skip embedding, use cached")
    parser.add_argument("--variants", type=str, default="small,medium,large",
                        help="Comma-separated MLP variants to train")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # Check disk space
    stat = os.statvfs("/home/deployer/laion")
    free_gb = stat.f_bavail * stat.f_frsize / 1e9
    log.info(f"Disk free: {free_gb:.1f} GB")
    if free_gb < 5:
        log.error("Less than 5 GB free, aborting")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "models").mkdir(exist_ok=True)

    # Step 1: Gather all latent paths
    items, stats = gather_items()

    # Step 2: Embed all latents → CLAP embeddings
    cache_file = EMBED_CACHE_DIR / "all_embeddings.pt"

    if args.skip_embed and cache_file.exists():
        log.info("Loading cached embeddings (--skip_embed)")
        data = torch.load(cache_file, map_location="cpu", weights_only=False)
    else:
        log.info("Starting embedding pipeline: latent → audio → CLAP...")
        vram_before = torch.cuda.memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0
        audio_decoder, clap_model = setup_decoder_and_clap(device)
        vram_after = torch.cuda.memory_allocated(device) / 1e9 if torch.cuda.is_available() else 0
        log.info(f"Models loaded: VRAM {vram_before:.1f} → {vram_after:.1f} GB")

        data = embed_all_items(items, audio_decoder, clap_model, device)

        # Free decoder + CLAP to recover VRAM
        del audio_decoder, clap_model
        torch.cuda.empty_cache()
        log.info("Freed AudioDecoder + CLAP from GPU")

    embeddings = data["embeddings"]  # [N, 768]
    labels_tensor = data["labels"]   # [N]
    sources = data["sources"]
    paths = data["paths"]

    log.info(f"Embeddings: {embeddings.shape}, Labels: {labels_tensor.shape}")
    log.info(f"  Positive (artifact): {(labels_tensor == 1).sum().item()}")
    log.info(f"  Negative (clean):    {(labels_tensor == 0).sum().item()}")

    # Check disk space after embedding
    stat = os.statvfs("/home/deployer/laion")
    free_gb = stat.f_bavail * stat.f_frsize / 1e9
    log.info(f"Disk free after embedding: {free_gb:.1f} GB")

    # Step 3: Split train/val
    train_idx, val_idx = split_train_val(
        embeddings, labels_tensor.tolist(), sources
    )
    train_emb = embeddings[train_idx].float()
    train_labels = labels_tensor[train_idx].float()
    val_emb = embeddings[val_idx].float()
    val_labels = labels_tensor[val_idx].float()
    train_sources = [sources[i] for i in train_idx]
    val_sources = [sources[i] for i in val_idx]

    log.info(f"Train: {len(train_idx)} | Val: {len(val_idx)}")
    log.info(f"  Train pos: {(train_labels == 1).sum().item()} neg: {(train_labels == 0).sum().item()}")
    log.info(f"  Val pos:   {(val_labels == 1).sum().item()} neg: {(val_labels == 0).sum().item()}")

    # Step 4: Train MLP variants
    variants = args.variants.split(",")
    results_summary = {}

    for variant_name in variants:
        variant_name = variant_name.strip()
        if variant_name not in VARIANT_CLASSES:
            log.warning(f"Unknown variant: {variant_name}")
            continue

        model, best_acc, best_epoch, n_params = train_classifier(
            variant_name=variant_name,
            model_class=VARIANT_CLASSES[variant_name],
            train_emb=train_emb, train_labels=train_labels,
            val_emb=val_emb, val_labels=val_labels,
            sources_train=train_sources, sources_val=val_sources,
            device=device, epochs=args.epochs, lr=args.lr,
        )

        # Per-source evaluation on full dataset
        src_results = per_source_eval(model, embeddings, labels_tensor, sources, device)

        # Save model
        save_path = OUTPUT_DIR / "models" / f"best_clap_{variant_name}.pt"
        torch.save({
            "state_dict": model.state_dict(),
            "variant": variant_name,
            "params": n_params,
            "val_acc": best_acc,
            "best_epoch": best_epoch,
            "per_source": src_results,
            "embedding_dim": 768,
            "clap_model": "laion/voiceclap-small",
        }, save_path)
        log.info(f"  Saved: {save_path} ({os.path.getsize(save_path) / 1e3:.1f} KB)")

        results_summary[variant_name] = {
            "params": n_params,
            "val_acc": best_acc,
            "best_epoch": best_epoch,
            "per_source": src_results,
        }

    # Summary
    log.info(f"\n{'='*60}")
    log.info("SUMMARY")
    log.info(f"{'='*60}")
    for name, r in results_summary.items():
        log.info(f"  {name:10s} | params={r['params']:>10,} | val_acc={r['val_acc']:.4f} | epoch={r['best_epoch']}")

    # Save summary
    summary_path = OUTPUT_DIR / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    log.info(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
