#!/usr/bin/env python3
"""
Train a lightweight CNN to detect comb-filter artifacts in DramaBox latent space.

Pipeline:
1. Load DramaBox VAE encoder
2. For each audio file: encode clean + comb-filtered versions → latent pairs
3. Train a small CNN discriminator on the latent pairs
4. Save the trained model for use as a reward signal

Usage:
    python scripts/train_comb_filter_detector.py [--encode-only] [--train-only]
"""

import os
import sys
import json
import random
import logging
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import soundfile as sf

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ── Config ──────────────────────────────────────────────────────────────────
DRAMABOX_DIR = "/home/deployer/laion/DramaBox"
FULL_CKPT = "/home/deployer/laion/DramaBox/models/ltx-2.3-22b-dev.safetensors"

AUDIO_SOURCES = {
    "podcast": "/home/deployer/laion/Voice-Acting-Pipeline/finetune_data_podcast/audio",
    "emolia": "/home/deployer/laion/Voice-Acting-Pipeline/finetune_data_emolia/raw_audio",
    "emolia_exp": "/home/deployer/laion/Voice-Acting-Pipeline/finetune_data_emolia_expanded/raw_audio",
}

OUTPUT_DIR = Path("/home/deployer/laion/Voice-Acting-Pipeline/comb_filter_detector")
LATENT_DIR = OUTPUT_DIR / "latents"
MODEL_DIR = OUTPUT_DIR / "models"

# Comb filter params
DELAY_RANGE_MS = (1, 8)
WET_RANGE = (0.5, 0.95)
TARGET_SR = 16000  # DramaBox VAE expects 16kHz
MAX_AUDIO_SEC = 20.0
MAX_SAMPLES = 5000  # max samples per source (to keep dataset manageable)

# Training params
BATCH_SIZE = 64
LR = 3e-4
EPOCHS = 30
VAL_SPLIT = 0.1
SEED = 42


# ── Comb Filter ─────────────────────────────────────────────────────────────

def apply_comb_filter(audio, sr, delay_ms, wet):
    """Feedforward comb filter: y[n] = x[n] + wet * x[n - delay]"""
    delay_samples = int(sr * delay_ms / 1000.0)
    if delay_samples == 0:
        return audio.copy()
    out = audio.copy()
    out[delay_samples:] += wet * audio[:-delay_samples]
    peak = np.abs(out).max()
    if peak > 0.99:
        out *= 0.99 / peak
    return out


def random_comb_filter(audio, sr):
    """Apply comb filter with random parameters."""
    delay_ms = random.uniform(*DELAY_RANGE_MS)
    wet = random.uniform(*WET_RANGE)
    return apply_comb_filter(audio, sr, delay_ms, wet), delay_ms, wet


# ── Audio Loading ────────────────────────────────────────────────────────────

def find_audio_files():
    """Find all available audio files across sources."""
    all_files = []
    for source_name, audio_dir in AUDIO_SOURCES.items():
        audio_path = Path(audio_dir)
        if not audio_path.exists():
            log.warning(f"Source '{source_name}' not found at {audio_dir}, skipping")
            continue
        # Get wav files, skip _ref files
        wavs = sorted([
            f for f in audio_path.glob("*.wav")
            if "_ref" not in f.name
        ])
        if len(wavs) > MAX_SAMPLES:
            random.seed(SEED)
            wavs = random.sample(wavs, MAX_SAMPLES)
        all_files.extend([(source_name, f) for f in wavs])
        log.info(f"  {source_name}: {len(wavs)} files")
    return all_files


def load_audio(path, target_sr=TARGET_SR, max_sec=MAX_AUDIO_SEC):
    """Load audio, convert to mono, resample to target_sr."""
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    max_samples = int(max_sec * target_sr)
    return audio[:max_samples], target_sr


# ── VAE Encoding ─────────────────────────────────────────────────────────────

def encode_batch(encoder, waveforms_np, sr, device, dtype):
    """Encode a batch of numpy waveforms through the VAE encoder.

    Args:
        encoder: The AudioEncoder instance
        waveforms_np: list of 1D numpy arrays (mono, 16kHz)
        sr: sample rate
        device, dtype: torch device/dtype

    Returns:
        list of latent tensors [8, T, 16] on CPU
    """
    sys.path.insert(0, os.path.join(DRAMABOX_DIR, "ltx2"))
    sys.path.insert(0, os.path.join(DRAMABOX_DIR, "src"))
    from ltx_core.model.audio_vae import encode_audio as vae_encode_audio
    from ltx_core.types import Audio

    latents = []
    for wav_np in waveforms_np:
        # Convert to [1, 2, samples] stereo tensor
        wav_t = torch.from_numpy(wav_np).float()
        wav_t = wav_t.unsqueeze(0).unsqueeze(0).repeat(1, 2, 1).to(device)
        audio_obj = Audio(waveform=wav_t, sampling_rate=sr)
        with torch.no_grad():
            latent = vae_encode_audio(audio_obj, encoder, None)
        latents.append(latent.squeeze(0).cpu())  # [8, T, 16]
    return latents


# ── Phase 1: Generate Latent Pairs ──────────────────────────────────────────

def generate_latent_pairs(audio_files, device="cuda:0"):
    """Generate clean + comb-filtered latent pairs."""
    sys.path.insert(0, os.path.join(DRAMABOX_DIR, "ltx2"))
    sys.path.insert(0, os.path.join(DRAMABOX_DIR, "src"))
    from ltx_pipelines.utils.blocks import AudioConditioner

    LATENT_DIR.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16

    # Check what's already done
    done_file = LATENT_DIR / "manifest.json"
    if done_file.exists():
        with open(done_file) as f:
            manifest = json.load(f)
        done_indices = {m["idx"] for m in manifest}
    else:
        manifest = []
        done_indices = set()

    todo = [(i, src, f) for i, (src, f) in enumerate(audio_files) if i not in done_indices]
    if not todo:
        log.info("All latent pairs already generated.")
        return manifest

    log.info(f"Generating latent pairs: {len(todo)} remaining of {len(audio_files)} total")

    # Load VAE encoder (keep it resident)
    log.info("Loading VAE encoder...")
    ac = AudioConditioner(checkpoint_path=FULL_CKPT, dtype=dtype, device=device, warm=True)

    processed = 0
    errors = 0
    for idx, source, audio_path in todo:
        try:
            audio, sr = load_audio(audio_path)
            if len(audio) < sr * 1.0:  # skip < 1 second
                continue

            # Apply random comb filter
            comb_audio, delay_ms, wet = random_comb_filter(audio, sr)

            # Encode both through VAE
            clean_latent, comb_latent = ac(
                lambda enc: encode_batch(enc, [audio, comb_audio], sr, device, dtype)
            )

            # Save
            clean_path = LATENT_DIR / f"{idx:06d}_clean.pt"
            comb_path = LATENT_DIR / f"{idx:06d}_comb.pt"
            torch.save(clean_latent, clean_path)
            torch.save(comb_latent, comb_path)

            manifest.append({
                "idx": idx,
                "source": source,
                "audio": str(audio_path),
                "delay_ms": round(delay_ms, 2),
                "wet": round(wet, 3),
                "clean": str(clean_path),
                "comb": str(comb_path),
                "clean_shape": list(clean_latent.shape),
            })

            processed += 1
            if processed % 50 == 0:
                log.info(f"  Encoded {processed}/{len(todo)} pairs")
                # Save manifest periodically
                with open(done_file, "w") as f:
                    json.dump(manifest, f)

        except Exception as e:
            errors += 1
            if errors < 5:
                log.warning(f"  Sample {idx} ({audio_path.name}): {e}")
            elif errors == 5:
                log.warning("  (suppressing further warnings)")

    # Final save
    with open(done_file, "w") as f:
        json.dump(manifest, f)

    del ac
    torch.cuda.empty_cache()
    log.info(f"Encoding complete: {processed} pairs, {errors} errors")
    return manifest


# ── CNN Architecture ─────────────────────────────────────────────────────────

class CombFilterDetector(nn.Module):
    """Lightweight CNN to detect comb-filter artifacts in DramaBox latent space.

    Input: [B, 8, T, 16] latent tensor
    Output: [B, 1] probability of comb filter presence (sigmoid)

    Total params: ~120K - fast enough for reward signal use.
    """
    def __init__(self, in_channels=8):
        super().__init__()
        # Conv blocks operating on [8, T, 16] "images"
        self.features = nn.Sequential(
            # Block 1: [8, T, 16] → [32, T/2, 8]
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: [32, T/4, 4]
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: [64, T/8, 2]
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4: [128, T/16, 1]
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )

        # Global average pooling → FC
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        """
        Args:
            x: [B, 8, T, 16] latent tensor (bfloat16 or float32)
        Returns:
            logits: [B, 1] (raw logits, apply sigmoid for probability)
        """
        x = x.float()  # ensure float32 for CNN
        x = self.features(x)
        return self.classifier(x)

    def predict_proba(self, x):
        """Return probability of comb filter presence."""
        return torch.sigmoid(self.forward(x))

    def comb_score(self, x):
        """Differentiable comb-filter score for reward signal use.
        Returns 0.0 for clean, 1.0 for heavily comb-filtered.
        """
        return torch.sigmoid(self.forward(x)).squeeze(-1)


# ── Dataset ──────────────────────────────────────────────────────────────────

class CombFilterLatentDataset(Dataset):
    """Dataset of clean/comb-filtered latent pairs."""

    def __init__(self, manifest, max_T=200):
        self.items = []
        self.max_T = max_T
        for m in manifest:
            self.items.append((m["clean"], 0))  # clean → label 0
            self.items.append((m["comb"], 1))    # comb → label 1

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label = self.items[idx]
        latent = torch.load(path, weights_only=True).float()  # [8, T, 16]

        # Pad or truncate T dimension
        T = latent.shape[1]
        if T > self.max_T:
            # Random crop
            start = random.randint(0, T - self.max_T)
            latent = latent[:, start:start + self.max_T, :]
        elif T < self.max_T:
            pad = torch.zeros(8, self.max_T - T, 16)
            latent = torch.cat([latent, pad], dim=1)

        return latent, torch.tensor(label, dtype=torch.float32)


# ── Training ─────────────────────────────────────────────────────────────────

def train_detector(manifest):
    """Train the comb filter detector CNN."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Determine max_T from data
    sample_latent = torch.load(manifest[0]["clean"], weights_only=True)
    typical_T = sample_latent.shape[1]
    max_T = min(200, typical_T)  # cap at 200 frames (~16 sec)
    log.info(f"Typical T={typical_T}, using max_T={max_T}")

    dataset = CombFilterLatentDataset(manifest, max_T=max_T)
    log.info(f"Dataset: {len(dataset)} samples ({len(manifest)} clean + {len(manifest)} comb)")

    # Split
    val_size = max(100, int(len(dataset) * VAL_SPLIT))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)

    device = torch.device("cuda:0")
    model = CombFilterDetector(in_channels=8).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.BCEWithLogitsLoss()

    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for latents, labels in train_loader:
            latents = latents.to(device)
            labels = labels.to(device).unsqueeze(1)

            logits = model(latents)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * latents.size(0)
            preds = (torch.sigmoid(logits) > 0.5).float()
            train_correct += (preds == labels).sum().item()
            train_total += latents.size(0)

        scheduler.step()
        train_acc = train_correct / max(train_total, 1)

        # Validate
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for latents, labels in val_loader:
                latents = latents.to(device)
                labels = labels.to(device).unsqueeze(1)
                logits = model(latents)
                loss = criterion(logits, labels)
                val_loss += loss.item() * latents.size(0)
                preds = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += latents.size(0)

        val_acc = val_correct / max(val_total, 1)

        log.info(
            f"Epoch {epoch}/{EPOCHS} | "
            f"train_loss={train_loss / train_total:.4f} train_acc={train_acc:.3f} | "
            f"val_loss={val_loss / val_total:.4f} val_acc={val_acc:.3f} | "
            f"lr={scheduler.get_last_lr()[0]:.2e}"
        )

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_acc": val_acc,
                "n_params": n_params,
                "max_T": max_T,
                "config": {
                    "delay_range_ms": DELAY_RANGE_MS,
                    "wet_range": WET_RANGE,
                    "in_channels": 8,
                },
            }, MODEL_DIR / "best_comb_detector.pt")
            log.info(f"  → New best: val_acc={val_acc:.4f}")

    # Save final
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": EPOCHS,
        "val_acc": val_acc,
        "n_params": n_params,
        "max_T": max_T,
        "config": {
            "delay_range_ms": DELAY_RANGE_MS,
            "wet_range": WET_RANGE,
            "in_channels": 8,
        },
    }, MODEL_DIR / "final_comb_detector.pt")

    log.info(f"\nTraining complete. Best val_acc={best_val_acc:.4f}")
    log.info(f"Model saved to {MODEL_DIR / 'best_comb_detector.pt'}")
    log.info(f"Model size: {n_params:,} params ({n_params * 4 / 1024 / 1024:.1f} MB float32)")

    return model


def _update_max_samples(val):
    global MAX_SAMPLES
    MAX_SAMPLES = val


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--encode-only", action="store_true",
                        help="Only generate latent pairs, skip training")
    parser.add_argument("--train-only", action="store_true",
                        help="Only train CNN, skip encoding (latents must exist)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-samples-per-source", type=int, default=MAX_SAMPLES)
    args = parser.parse_args()

    _update_max_samples(args.max_samples_per_source)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)
    torch.manual_seed(SEED)

    manifest_path = LATENT_DIR / "manifest.json"

    if not args.train_only:
        log.info("=== Phase 1: Finding audio files ===")
        audio_files = find_audio_files()
        log.info(f"Total: {len(audio_files)} audio files")

        log.info("=== Phase 2: Generating latent pairs ===")
        manifest = generate_latent_pairs(audio_files, device=args.device)
    else:
        log.info("=== Loading existing manifest ===")
        with open(manifest_path) as f:
            manifest = json.load(f)
        log.info(f"Loaded {len(manifest)} pairs from manifest")

    if args.encode_only:
        log.info("Encoding complete. Exiting (--encode-only).")
        return

    if len(manifest) < 100:
        log.error(f"Only {len(manifest)} pairs available, need at least 100. Run encoding first.")
        return

    log.info(f"\n=== Phase 3: Training CNN ({len(manifest)} pairs) ===")
    train_detector(manifest)


if __name__ == "__main__":
    main()
