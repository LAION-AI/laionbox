#!/usr/bin/env python3
"""
Train an artifact detector CNN (V2) that distinguishes clean latents from
model-generated predictions and comb-filtered corruptions.

Data sources:
  Label 1 (artifact):
    - Model predictions: artifact_detector_v2_data/latents/{model_id}/sample_XXXXXX.pt
      (4 models × 1500 samples = 6000 latents)
    - Comb-augmented: comb_filter_detector/latents/XXXXXX_comb.pt (6000 latents)

  Label 0 (clean):
    - Ground truth targets: finetune_data_combined/audio_latents/sample_XXXXXX_part2.pt
      (1500 latents, one per selected sample)
    - Clean comb pairs: comb_filter_detector/latents/XXXXXX_clean.pt (6000 latents)

Three CNN variants are trained and compared:
  - small:  4 conv blocks, ~251K params (same as comb detector V1)
  - medium: 5 conv blocks + SE attention, ~750K params
  - large:  6 conv blocks + residual + SE attention, ~2M params

Usage:
    python scripts/train_artifact_detector_v2.py [--variants small,medium,large] [--gpus 0,1,2]
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
from torch.utils.data import Dataset, DataLoader, Subset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path("/home/deployer/laion/Voice-Acting-Pipeline")
PRED_DIR = BASE_DIR / "artifact_detector_v2_data" / "latents"
TARGET_DIR = BASE_DIR / "finetune_data_combined" / "audio_latents"
COMB_DIR = BASE_DIR / "comb_filter_detector" / "latents"
OUTPUT_DIR = BASE_DIR / "artifact_detector_v2"
MODEL_IDS = ["v01_wip", "4aux_s080", "4aux_s150", "4aux_s180"]

# ── Training config ──────────────────────────────────────────────────────────
BATCH_SIZE = 64
LR = 3e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 40
VAL_SIZE = 200  # number of validation samples per class
SEED = 42
MAX_T = 200  # max time frames (random crop if longer, pad if shorter)


# ── CNN Architectures ────────────────────────────────────────────────────────

class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // reduction),
            nn.GELU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.fc(x).unsqueeze(-1).unsqueeze(-1)
        return x * w


class ArtifactDetectorSmall(nn.Module):
    """Small CNN (~251K params) — same architecture as comb detector V1."""
    def __init__(self, in_channels=8):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x.float()))


class ArtifactDetectorMedium(nn.Module):
    """Medium CNN (~750K params) with SE attention."""
    def __init__(self, in_channels=8):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: [8, T, 16] → [48, T/2, 8]
            nn.Conv2d(in_channels, 48, 3, padding=1), nn.BatchNorm2d(48), nn.GELU(),
            nn.MaxPool2d(2, 2),
            # Block 2: → [96, T/4, 4]
            nn.Conv2d(48, 96, 3, padding=1), nn.BatchNorm2d(96), nn.GELU(),
            SEBlock(96),
            nn.MaxPool2d(2, 2),
            # Block 3: → [192, T/8, 2]
            nn.Conv2d(96, 192, 3, padding=1), nn.BatchNorm2d(192), nn.GELU(),
            SEBlock(192),
            nn.MaxPool2d(2, 2),
            # Block 4: → [256, T/16, 1]
            nn.Conv2d(192, 256, 3, padding=1), nn.BatchNorm2d(256), nn.GELU(),
            SEBlock(256),
            nn.MaxPool2d(2, 2),
            # Block 5: → [256, T/32, 1] (pooled down further)
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x.float()))


class ResBlock(nn.Module):
    """Residual block with optional channel change."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.se = SEBlock(out_ch)

    def forward(self, x):
        identity = self.skip(x)
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.gelu(out + identity)


class ArtifactDetectorLarge(nn.Module):
    """Large CNN (~2M params) with residual blocks + SE attention."""
    def __init__(self, in_channels=8):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.BatchNorm2d(64), nn.GELU(),
        )
        self.blocks = nn.Sequential(
            # [64, T, 16] → [64, T/2, 8]
            ResBlock(64, 64), nn.MaxPool2d(2, 2),
            # → [128, T/4, 4]
            ResBlock(64, 128), nn.MaxPool2d(2, 2),
            # → [256, T/8, 2]
            ResBlock(128, 256), nn.MaxPool2d(2, 2),
            # → [384, T/16, 1]
            ResBlock(256, 384), nn.MaxPool2d(2, 2),
            # → [512, T/32, 1]
            ResBlock(384, 512),
            # Extra conv
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = x.float()
        x = self.stem(x)
        x = self.blocks(x)
        return self.classifier(x)


VARIANT_CLASSES = {
    "small": ArtifactDetectorSmall,
    "medium": ArtifactDetectorMedium,
    "large": ArtifactDetectorLarge,
}


# ── Dataset ──────────────────────────────────────────────────────────────────

class ArtifactLatentDataset(Dataset):
    """Unified dataset for artifact detection training.

    Each item is (latent_tensor [8, T, 16], label) where:
      label=1: artifact (model prediction or comb-filtered)
      label=0: clean (ground truth target or clean comb pair)
    """

    def __init__(self, items, max_T=MAX_T):
        """
        Args:
            items: list of (path, label, source_tag) tuples
            max_T: maximum time dimension (crop/pad)
        """
        self.items = items
        self.max_T = max_T

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label, _ = self.items[idx]
        latent = torch.load(path, weights_only=True, map_location="cpu").detach().float()  # [8, T, 16]

        T = latent.shape[1]
        if T > self.max_T:
            start = random.randint(0, T - self.max_T)
            latent = latent[:, start:start + self.max_T, :]
        elif T < self.max_T:
            pad = torch.zeros(8, self.max_T - T, 16)
            latent = torch.cat([latent, pad], dim=1)

        return latent, torch.tensor(label, dtype=torch.float32)


def build_dataset():
    """Build the full dataset from all sources.

    Returns:
        items: list of (path, label, source_tag)
        stats: dict with counts per source
    """
    items = []
    stats = defaultdict(int)

    # ── Label 1: Model predictions (artifact) ──
    for model_id in MODEL_IDS:
        model_dir = PRED_DIR / model_id
        if not model_dir.exists():
            log.warning(f"Model dir not found: {model_dir}")
            continue
        files = sorted(model_dir.glob("*.pt"))
        for f in files:
            items.append((str(f), 1, f"pred_{model_id}"))
        stats[f"pred_{model_id}"] = len(files)
        log.info(f"  pred_{model_id}: {len(files)} files (label=1)")

    # ── Label 1: Comb-augmented (artifact) ──
    comb_files = sorted(COMB_DIR.glob("*_comb.pt"))
    for f in comb_files:
        items.append((str(f), 1, "comb_augmented"))
    stats["comb_augmented"] = len(comb_files)
    log.info(f"  comb_augmented: {len(comb_files)} files (label=1)")

    # ── Label 0: Ground truth targets (clean) ──
    # Get the list of sample IDs from any complete model's predictions
    pred_sample_ids = set()
    for model_id in MODEL_IDS:
        model_dir = PRED_DIR / model_id
        if model_dir.exists():
            for f in model_dir.glob("*.pt"):
                pred_sample_ids.add(f.stem)  # e.g., "sample_003317"
            if pred_sample_ids:
                break

    target_count = 0
    for sample_id in sorted(pred_sample_ids):
        target_path = TARGET_DIR / f"{sample_id}_part2.pt"
        if target_path.exists():
            items.append((str(target_path), 0, "ground_truth"))
            target_count += 1
    stats["ground_truth"] = target_count
    log.info(f"  ground_truth: {target_count} files (label=0)")

    # ── Label 0: Clean comb pairs (clean) ──
    clean_files = sorted(COMB_DIR.glob("*_clean.pt"))
    for f in clean_files:
        items.append((str(f), 0, "comb_clean"))
    stats["comb_clean"] = len(clean_files)
    log.info(f"  comb_clean: {len(clean_files)} files (label=0)")

    return items, dict(stats)


def split_train_val(items, val_size=VAL_SIZE, seed=SEED):
    """Split items into train/val ensuring balanced classes and
    no sample-level leakage (a sample's target and predictions don't
    cross the split boundary)."""
    rng = random.Random(seed)

    # Group by label
    pos_items = [(i, it) for i, it in enumerate(items) if it[1] == 1]
    neg_items = [(i, it) for i, it in enumerate(items) if it[1] == 0]

    rng.shuffle(pos_items)
    rng.shuffle(neg_items)

    val_per_class = val_size // 2
    val_indices = set()
    train_indices = set()

    # Pick val items
    for idx, _ in pos_items[:val_per_class]:
        val_indices.add(idx)
    for idx, _ in pos_items[val_per_class:]:
        train_indices.add(idx)

    for idx, _ in neg_items[:val_per_class]:
        val_indices.add(idx)
    for idx, _ in neg_items[val_per_class:]:
        train_indices.add(idx)

    return sorted(train_indices), sorted(val_indices)


# ── Training ─────────────────────────────────────────────────────────────────

def train_one_variant(variant_name, model_class, train_ds, val_ds, device, output_dir):
    """Train a single CNN variant and return metrics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = model_class(in_channels=8).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"[{variant_name}] Params: {n_params:,} ({n_params * 4 / 1e6:.1f} MB)")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.BCEWithLogitsLoss()

    best_val_acc = 0.0
    best_epoch = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        # ── Train ──
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for latents, labels in train_loader:
            latents = latents.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1)

            logits = model(latents)
            loss = criterion(logits, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * latents.size(0)
            preds = (logits > 0.0).float()
            train_correct += (preds == labels).sum().item()
            train_total += latents.size(0)

        scheduler.step()
        train_acc = train_correct / max(train_total, 1)

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_tp = val_fp = val_tn = val_fn = 0

        with torch.no_grad():
            for latents, labels in val_loader:
                latents = latents.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).unsqueeze(1)
                logits = model(latents)
                loss = criterion(logits, labels)
                val_loss += loss.item() * latents.size(0)
                preds = (logits > 0.0).float()
                val_correct += (preds == labels).sum().item()
                val_total += latents.size(0)

                val_tp += ((preds == 1) & (labels == 1)).sum().item()
                val_fp += ((preds == 1) & (labels == 0)).sum().item()
                val_tn += ((preds == 0) & (labels == 0)).sum().item()
                val_fn += ((preds == 0) & (labels == 1)).sum().item()

        val_acc = val_correct / max(val_total, 1)
        precision = val_tp / max(val_tp + val_fp, 1)
        recall = val_tp / max(val_tp + val_fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        elapsed = time.time() - t0
        entry = {
            "epoch": epoch,
            "train_loss": train_loss / max(train_total, 1),
            "train_acc": train_acc,
            "val_loss": val_loss / max(val_total, 1),
            "val_acc": val_acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "lr": scheduler.get_last_lr()[0],
            "time": elapsed,
        }
        history.append(entry)

        # Log every 5 epochs or on improvement
        if epoch % 5 == 0 or val_acc > best_val_acc or epoch == 1:
            log.info(
                f"[{variant_name}] Epoch {epoch:2d}/{EPOCHS} | "
                f"train_loss={entry['train_loss']:.4f} acc={train_acc:.3f} | "
                f"val_loss={entry['val_loss']:.4f} acc={val_acc:.3f} "
                f"P={precision:.3f} R={recall:.3f} F1={f1:.3f} | "
                f"lr={entry['lr']:.2e} {elapsed:.1f}s"
            )

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "variant": variant_name,
                "epoch": epoch,
                "val_acc": val_acc,
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "n_params": n_params,
                "max_T": MAX_T,
            }, output_dir / f"best_{variant_name}.pt")

    # Save final
    torch.save({
        "model_state_dict": model.state_dict(),
        "variant": variant_name,
        "epoch": EPOCHS,
        "val_acc": val_acc,
        "f1": f1,
        "n_params": n_params,
        "max_T": MAX_T,
    }, output_dir / f"final_{variant_name}.pt")

    # Save history
    with open(output_dir / f"history_{variant_name}.json", "w") as f:
        json.dump(history, f, indent=2)

    log.info(
        f"[{variant_name}] Done. Best: epoch {best_epoch}, "
        f"val_acc={best_val_acc:.4f}, params={n_params:,}"
    )

    return {
        "variant": variant_name,
        "n_params": n_params,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "best_f1": max(h["f1"] for h in history),
        "final_val_acc": history[-1]["val_acc"],
    }


def train_variant_on_gpu(variant_name, model_class, items, train_indices, val_indices, gpu_id):
    """Train a variant on a specific GPU (for parallel execution)."""
    device = torch.device(f"cuda:{gpu_id}")
    ds = ArtifactLatentDataset(items, max_T=MAX_T)
    train_ds = Subset(ds, train_indices)
    val_ds = Subset(ds, val_indices)

    log.info(f"[{variant_name}] GPU {gpu_id}: train={len(train_ds)}, val={len(val_ds)}")
    result = train_one_variant(
        variant_name, model_class, train_ds, val_ds, device,
        OUTPUT_DIR / "models",
    )
    return result


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", default="small,medium,large",
                   help="Comma-separated list of variants to train")
    p.add_argument("--gpus", default="0,1,2",
                   help="Comma-separated GPU IDs for parallel training")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--val-size", type=int, default=VAL_SIZE)
    p.add_argument("--sequential", action="store_true",
                   help="Train variants sequentially instead of in parallel")
    return p.parse_args()


def main():
    args = parse_args()
    global EPOCHS, BATCH_SIZE, LR, VAL_SIZE
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LR = args.lr
    VAL_SIZE = args.val_size

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    variants = [v.strip() for v in args.variants.split(",")]
    gpus = [int(g) for g in args.gpus.split(",")]

    log.info("=== Building dataset ===")
    items, stats = build_dataset()

    label_1_count = sum(1 for _, l, _ in items if l == 1)
    label_0_count = sum(1 for _, l, _ in items if l == 0)
    log.info(f"Total: {len(items)} items (artifact={label_1_count}, clean={label_0_count})")
    log.info(f"Stats: {json.dumps(stats, indent=2)}")

    # Save dataset manifest
    manifest_path = OUTPUT_DIR / "dataset_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "stats": stats,
            "total": len(items),
            "label_1": label_1_count,
            "label_0": label_0_count,
            "items": [(p, l, s) for p, l, s in items],
        }, f)
    log.info(f"Saved manifest: {manifest_path}")

    # Split
    log.info(f"=== Splitting: val_size={args.val_size} ===")
    train_indices, val_indices = split_train_val(items, val_size=args.val_size, seed=SEED)
    log.info(f"Train: {len(train_indices)}, Val: {len(val_indices)}")

    # Verify val class balance
    val_labels = [items[i][1] for i in val_indices]
    val_pos = sum(val_labels)
    val_neg = len(val_labels) - val_pos
    log.info(f"Val split: artifact={val_pos}, clean={val_neg}")

    # Save split indices so subprocesses can reuse
    split_path = OUTPUT_DIR / "split_indices.json"
    with open(split_path, "w") as f:
        json.dump({"train": train_indices, "val": val_indices}, f)

    if args.sequential or len(variants) == 1:
        # Sequential training
        results = []
        gpu_id = gpus[0]
        for variant_name in variants:
            if variant_name not in VARIANT_CLASSES:
                log.warning(f"Unknown variant: {variant_name}, skipping")
                continue
            result = train_variant_on_gpu(
                variant_name, VARIANT_CLASSES[variant_name],
                items, train_indices, val_indices, gpu_id,
            )
            results.append(result)
    else:
        # Parallel training via subprocesses (avoids pickle issues with spawn)
        import subprocess
        results = []
        procs = []

        for i, variant_name in enumerate(variants):
            if variant_name not in VARIANT_CLASSES:
                log.warning(f"Unknown variant: {variant_name}, skipping")
                continue
            gpu_id = gpus[i % len(gpus)]
            log_path = f"/tmp/artifact_v2_{variant_name}.log"
            cmd = [
                sys.executable, __file__,
                "--variants", variant_name,
                "--gpus", str(gpu_id),
                "--epochs", str(EPOCHS),
                "--batch-size", str(BATCH_SIZE),
                "--lr", str(LR),
                "--val-size", str(args.val_size),
                "--sequential",
            ]
            log.info(f"Launching {variant_name} on GPU {gpu_id} → {log_path}")
            with open(log_path, "w") as lf:
                p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT)
            procs.append((variant_name, p, log_path))

        # Wait for all
        for variant_name, p, log_path in procs:
            p.wait()
            log.info(f"{variant_name} finished (exit={p.returncode})")
            # Read result from saved model checkpoint
            result_path = OUTPUT_DIR / "models" / f"best_{variant_name}.pt"
            if result_path.exists():
                ckpt = torch.load(result_path, map_location="cpu", weights_only=True)
                results.append({
                    "variant": variant_name,
                    "n_params": ckpt.get("n_params", 0),
                    "best_epoch": ckpt.get("epoch", 0),
                    "best_val_acc": ckpt.get("val_acc", 0),
                    "best_f1": ckpt.get("f1", 0),
                    "final_val_acc": ckpt.get("val_acc", 0),
                })
            else:
                results.append({"variant": variant_name, "error": f"No checkpoint found, see {log_path}"})

    # ── Summary ──
    log.info("\n" + "=" * 70)
    log.info("RESULTS SUMMARY")
    log.info("=" * 70)
    log.info(f"{'Variant':<10} {'Params':>10} {'Best Epoch':>10} {'Val Acc':>10} {'F1':>10}")
    log.info("-" * 55)
    best_result = None
    for r in sorted(results, key=lambda x: x.get("best_val_acc", 0), reverse=True):
        if "error" in r:
            log.info(f"{r['variant']:<10} FAILED: {r['error']}")
            continue
        log.info(
            f"{r['variant']:<10} {r['n_params']:>10,} {r['best_epoch']:>10} "
            f"{r['best_val_acc']:>10.4f} {r['best_f1']:>10.4f}"
        )
        if best_result is None or r["best_val_acc"] > best_result.get("best_val_acc", 0):
            best_result = r

    if best_result and "error" not in best_result:
        # Copy best model as the "winner"
        import shutil
        src = OUTPUT_DIR / "models" / f"best_{best_result['variant']}.pt"
        dst = OUTPUT_DIR / "models" / "best_artifact_detector_v2.pt"
        if src.exists():
            shutil.copy2(src, dst)
            log.info(f"\nBest model: {best_result['variant']} → {dst}")

    # Save summary
    with open(OUTPUT_DIR / "training_summary.json", "w") as f:
        json.dump({
            "dataset_stats": stats,
            "total_items": len(items),
            "train_size": len(train_indices),
            "val_size": len(val_indices),
            "results": results,
            "best_variant": best_result["variant"] if best_result else None,
        }, f, indent=2)

    log.info(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
