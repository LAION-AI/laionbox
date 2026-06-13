#!/usr/bin/env python3
"""
Train two binary classifiers on top of VoiceCLAP-small embeddings:

1. Real vs Synthetic: Emilia (real=1) vs DramaBox (synthetic=0)
2. Quality detector: Clean (good=1) vs Distorted (bad=0)

Usage:
  python scripts/train_binary_classifiers.py --gpu 0 --batch-size 64
"""

import argparse
import json
import logging
import os
import random
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── Paths ──────────────────────────────────────────────────────────────
COMBINED_MANIFEST = "finetune_data_combined/manifest.json"
EMOLIA_RAW_AUDIO = "finetune_data_emolia/raw_audio"
DRAMABOX_CACHE = os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--laion--dramabox-voice-acting-data-annotated/"
    "snapshots/acdc136e02d34241257346b13fc33ea5cdf00993/data"
)
OUTPUT_DIR = "classifiers"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0, help="GPU to use (-1 for CPU)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-dramabox", type=int, default=3247, help="Max DramaBox samples")
    p.add_argument("--max-emilia", type=int, default=3247, help="Max Emilia samples (matched)")
    p.add_argument("--val-per-class", type=int, default=100)
    p.add_argument("--distortion-samples", type=int, default=1000, help="Samples per class for distortion classifier")
    p.add_argument("--epochs-real-fake", type=int, default=100)
    p.add_argument("--epochs-quality", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--extract-dir", default="/tmp/dramabox_audio_for_clap", help="Temp dir for extracted DramaBox audio")
    return p.parse_args()


# ── Audio extraction ──────────────────────────────────────────────────
def extract_dramabox_audio(manifest_path, extract_dir, max_samples=3247):
    """Extract DramaBox _full.mp3 files from cached HF tars."""
    log.info("Loading manifest...")
    with open(manifest_path) as f:
        manifest = json.load(f)

    samples = manifest["selected_samples"]
    dramabox = [s for s in samples if s.get("batch_idx", -1) >= 0]
    random.shuffle(dramabox)
    dramabox = dramabox[:max_samples]

    os.makedirs(extract_dir, exist_ok=True)

    # Group by batch
    batch_to_basenames = {}
    for s in dramabox:
        bid = s["batch_idx"]
        if bid not in batch_to_basenames:
            batch_to_basenames[bid] = []
        batch_to_basenames[bid].append(s["basename"])

    # Check which files already extracted
    already_done = set()
    for f in os.listdir(extract_dir):
        if f.endswith("_full.mp3"):
            already_done.add(f.replace("_full.mp3", ""))

    need_batches = {}
    for bid, basenames in batch_to_basenames.items():
        needed = [bn for bn in basenames if bn not in already_done]
        if needed:
            need_batches[bid] = needed

    if not need_batches:
        log.info(f"All {len(dramabox)} DramaBox files already extracted")
    else:
        log.info(f"Extracting from {len(need_batches)} batches...")
        for i, (bid, basenames) in enumerate(need_batches.items()):
            tar_path = os.path.join(DRAMABOX_CACHE, f"batch_{bid:06d}.tar")
            if not os.path.exists(tar_path):
                # Try downloading from HF
                try:
                    from huggingface_hub import hf_hub_download
                    tar_path = hf_hub_download(
                        repo_id="laion/dramabox-voice-acting-data-annotated",
                        filename=f"data/batch_{bid:06d}.tar",
                        repo_type="dataset",
                    )
                except Exception as e:
                    log.warning(f"Batch {bid} not available: {e}")
                    continue

            bn_set = set(basenames)
            try:
                with tarfile.open(tar_path) as tf:
                    for member in tf.getmembers():
                        name = os.path.basename(member.name)
                        if not name.endswith("_full.mp3"):
                            continue
                        bn = name.replace("_full.mp3", "")
                        if bn in bn_set:
                            member.name = name
                            tf.extract(member, extract_dir)
                            bn_set.discard(bn)
            except Exception as e:
                log.warning(f"Error extracting batch {bid}: {e}")

            if (i + 1) % 50 == 0:
                log.info(f"  Extracted {i+1}/{len(need_batches)} batches")

    # Return list of (filepath, basename) for successfully extracted files
    result = []
    for s in dramabox:
        fp = os.path.join(extract_dir, f"{s['basename']}_full.mp3")
        if os.path.exists(fp):
            result.append((fp, s["basename"]))
    log.info(f"DramaBox: {len(result)} audio files ready")
    return result


def get_emilia_audio(raw_dir, max_samples=3247):
    """Get list of Emilia audio files."""
    files = sorted(Path(raw_dir).glob("*.wav"))
    random.shuffle(files)
    files = files[:max_samples]
    result = [(str(f), f.stem) for f in files]
    log.info(f"Emilia: {len(result)} audio files ready")
    return result


# ── CLAP embedding extraction ─────────────────────────────────────────
def load_clap_model(device):
    """Load VoiceCLAP-small model."""
    from transformers import AutoModel, AutoTokenizer

    model = AutoModel.from_pretrained("laion/voiceclap-small", trust_remote_code=True)
    model = model.eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained("laion/voiceclap-small")
    return model, tokenizer


def extract_clap_embeddings(audio_files, model, device, batch_size=32):
    """Extract CLAP embeddings for a list of audio files. Returns (embeddings, valid_indices)."""
    import torchaudio

    embeddings = []
    valid_indices = []

    for i in range(0, len(audio_files), batch_size):
        batch_files = audio_files[i : i + batch_size]
        batch_wavs = []
        batch_valid = []

        for j, (fp, _) in enumerate(batch_files):
            try:
                wav, sr = torchaudio.load(fp)
                if sr != 16000:
                    wav = torchaudio.functional.resample(wav, sr, 16000)
                wav = wav.mean(0)  # mono

                # Truncate to 30s max for CLAP
                max_len = 30 * 16000
                if wav.shape[0] > max_len:
                    wav = wav[:max_len]
                # Pad to at least 1s
                if wav.shape[0] < 16000:
                    wav = F.pad(wav, (0, 16000 - wav.shape[0]))

                batch_wavs.append(wav)
                batch_valid.append(i + j)
            except Exception as e:
                continue

        if not batch_wavs:
            continue

        # Pad batch to same length
        max_samples = max(w.shape[0] for w in batch_wavs)
        padded = torch.zeros(len(batch_wavs), max_samples)
        for k, w in enumerate(batch_wavs):
            padded[k, : w.shape[0]] = w

        with torch.no_grad():
            emb = model.encode_waveform(padded.to(device))  # [B, 768]
            emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb.cpu())
            valid_indices.extend(batch_valid)

        if (i // batch_size + 1) % 20 == 0:
            log.info(f"  Embedded {min(i + batch_size, len(audio_files))}/{len(audio_files)}")

    if embeddings:
        embeddings = torch.cat(embeddings, dim=0)
    else:
        embeddings = torch.zeros(0, 768)

    return embeddings, valid_indices


# ── Audio distortion augmentation ─────────────────────────────────────
def apply_distortions(wav, sr=16000):
    """Apply random distortion effects to make audio sound degraded.
    Randomly applies one or more of: clipping, distortion, subtle reverb.
    """
    effects = []

    # Randomly select 1-2 effects
    available = ["clip", "overdrive", "subtle_reverb"]
    n_effects = random.randint(1, 2)
    chosen = random.sample(available, min(n_effects, len(available)))

    for effect in chosen:
        if effect == "clip":
            # Simulate too-loud recording that clips
            gain = random.uniform(2.0, 6.0)
            wav = wav * gain
            wav = torch.clamp(wav, -1.0, 1.0)
            effects.append(f"clip(gain={gain:.1f})")

        elif effect == "overdrive":
            # Soft clipping / overdrive distortion
            gain = random.uniform(1.5, 4.0)
            wav = torch.tanh(wav * gain) / gain * 1.5
            effects.append(f"overdrive(gain={gain:.1f})")

        elif effect == "subtle_reverb":
            # Very subtle reverb — short early reflections, low mix
            # Simulates slight room coloration, not an obvious echo
            ir_len = int(sr * random.uniform(0.005, 0.025))  # 5-25ms
            ir = torch.randn(ir_len) * torch.exp(-torch.linspace(0, 8, ir_len))
            ir[0] = 1.0  # strong direct signal
            ir = ir / ir.abs().max()

            # Convolve
            wav_padded = F.pad(wav.unsqueeze(0).unsqueeze(0), (0, ir_len - 1))
            ir_kernel = ir.flip(0).unsqueeze(0).unsqueeze(0)
            reverbed = F.conv1d(wav_padded, ir_kernel).squeeze()

            # Mix: mostly dry with just a touch of wet
            mix = random.uniform(0.08, 0.25)
            wav_len = min(wav.shape[0], reverbed.shape[0])
            wav = (1 - mix) * wav[:wav_len] + mix * reverbed[:wav_len]
            effects.append(f"reverb(ir={ir_len/sr*1000:.0f}ms,mix={mix:.2f})")

    # Normalize to prevent silence or extreme levels
    peak = wav.abs().max()
    if peak > 0:
        wav = wav / peak * 0.9

    return wav, effects


def prepare_distortion_dataset(dramabox_files, emilia_files, model, device,
                                n_per_class=1000, batch_size=32):
    """Create distorted vs clean dataset with CLAP embeddings.

    Clean: 500 DramaBox + 500 Emilia (label=1)
    Distorted: 500 DramaBox + 500 Emilia with distortions applied (label=0)
    """
    import torchaudio

    n_half = n_per_class // 2

    # Select files
    random.shuffle(dramabox_files)
    random.shuffle(emilia_files)

    # Clean samples: 500 DramaBox + 500 Emilia
    clean_db = dramabox_files[:n_half]
    clean_em = emilia_files[:n_half]

    # Distorted samples: 500 DramaBox + 500 Emilia (different from clean set)
    dist_db = dramabox_files[n_half : n_half * 2]
    dist_em = emilia_files[n_half : n_half * 2]

    # If not enough unique samples, allow overlap
    if len(dist_db) < n_half:
        dist_db = dramabox_files[:n_half]
    if len(dist_em) < n_half:
        dist_em = emilia_files[:n_half]

    log.info(f"Quality dataset: {len(clean_db)+len(clean_em)} clean, {len(dist_db)+len(dist_em)} distorted")

    all_embeddings = []
    all_labels = []

    def process_batch(files, apply_dist, label):
        batch_wavs = []
        for fp, _ in files:
            try:
                wav, sr = torchaudio.load(fp)
                if sr != 16000:
                    wav = torchaudio.functional.resample(wav, sr, 16000)
                wav = wav.mean(0)
                max_len = 30 * 16000
                if wav.shape[0] > max_len:
                    wav = wav[:max_len]
                if wav.shape[0] < 16000:
                    wav = F.pad(wav, (0, 16000 - wav.shape[0]))

                if apply_dist:
                    wav, _ = apply_distortions(wav, 16000)

                batch_wavs.append(wav)
            except:
                continue

        # Embed in batches
        embs = []
        for bi in range(0, len(batch_wavs), batch_size):
            chunk = batch_wavs[bi : bi + batch_size]
            max_s = max(w.shape[0] for w in chunk)
            padded = torch.zeros(len(chunk), max_s)
            for k, w in enumerate(chunk):
                padded[k, : w.shape[0]] = w
            with torch.no_grad():
                e = model.encode_waveform(padded.to(device))
                e = e / e.norm(dim=-1, keepdim=True)
                embs.append(e.cpu())
        if embs:
            embs = torch.cat(embs)
            all_embeddings.append(embs)
            all_labels.append(torch.full((embs.shape[0],), label, dtype=torch.float32))

    log.info("Embedding clean DramaBox samples...")
    process_batch(clean_db, False, 1.0)
    log.info("Embedding clean Emilia samples...")
    process_batch(clean_em, False, 1.0)
    log.info("Embedding distorted DramaBox samples...")
    process_batch(dist_db, True, 0.0)
    log.info("Embedding distorted Emilia samples...")
    process_batch(dist_em, True, 0.0)

    embeddings = torch.cat(all_embeddings)
    labels = torch.cat(all_labels)
    log.info(f"Quality dataset: {embeddings.shape[0]} total ({labels.sum().int()} clean, {(1-labels).sum().int()} distorted)")
    return embeddings, labels


# ── MLP classifier ────────────────────────────────────────────────────
class BinaryMLP(nn.Module):
    """Small MLP binary classifier on top of 768-dim embeddings.
    Architecture: 768 -> 128 -> 32 -> 1  (~26K parameters)
    """
    def __init__(self, input_dim=768, hidden1=128, hidden2=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def train_classifier(name, train_emb, train_labels, val_emb, val_labels,
                     epochs=100, lr=1e-3, device="cpu"):
    """Train a binary MLP classifier."""
    model = BinaryMLP().to(device)
    log.info(f"[{name}] Model params: {model.count_params():,}")

    train_ds = TensorDataset(train_emb.to(device), train_labels.to(device))
    val_ds = TensorDataset(val_emb.to(device), val_labels.to(device))
    train_dl = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=256)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.BCEWithLogitsLoss()

    best_val_acc = 0
    best_state = None
    patience = 20
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for emb, lbl in train_dl:
            logits = model(emb)
            loss = criterion(logits, lbl)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * emb.shape[0]
            preds = (logits > 0).float()
            correct += (preds == lbl).sum().item()
            total += emb.shape[0]
        scheduler.step()

        train_acc = correct / total
        train_loss = total_loss / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0
        with torch.no_grad():
            for emb, lbl in val_dl:
                logits = model(emb)
                loss = criterion(logits, lbl)
                val_loss += loss.item() * emb.shape[0]
                preds = (logits > 0).float()
                val_correct += (preds == lbl).sum().item()
                val_total += emb.shape[0]

        val_acc = val_correct / val_total
        val_loss = val_loss / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            log.info(f"[{name}] Epoch {epoch+1}/{epochs}: "
                     f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                     f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                     f"best_val_acc={best_val_acc:.4f}")

        if patience_counter >= patience:
            log.info(f"[{name}] Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    return model, best_val_acc


def main():
    args = parse_args()
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Step 1: Get audio file lists ──
    log.info("=" * 60)
    log.info("Step 1: Preparing audio file lists")
    log.info("=" * 60)

    dramabox_files = extract_dramabox_audio(COMBINED_MANIFEST, args.extract_dir, args.max_dramabox)
    emilia_files = get_emilia_audio(EMOLIA_RAW_AUDIO, args.max_emilia)

    if len(dramabox_files) < 200 or len(emilia_files) < 200:
        log.error(f"Not enough audio: DramaBox={len(dramabox_files)}, Emilia={len(emilia_files)}")
        sys.exit(1)

    # ── Step 2: Load CLAP model ──
    log.info("=" * 60)
    log.info("Step 2: Loading VoiceCLAP-small")
    log.info("=" * 60)
    clap_model, clap_tokenizer = load_clap_model(device)

    # ── Step 3: Extract embeddings for Real vs Synthetic classifier ──
    log.info("=" * 60)
    log.info("Step 3: Extracting CLAP embeddings for Real vs Synthetic")
    log.info("=" * 60)

    log.info(f"Embedding {len(dramabox_files)} DramaBox files...")
    db_emb, db_valid = extract_clap_embeddings(dramabox_files, clap_model, device, args.batch_size)
    log.info(f"  Got {db_emb.shape[0]} embeddings")

    log.info(f"Embedding {len(emilia_files)} Emilia files...")
    em_emb, em_valid = extract_clap_embeddings(emilia_files, clap_model, device, args.batch_size)
    log.info(f"  Got {em_emb.shape[0]} embeddings")

    # Balance classes
    n_min = min(db_emb.shape[0], em_emb.shape[0])
    db_emb = db_emb[:n_min]
    em_emb = em_emb[:n_min]
    log.info(f"Balanced to {n_min} per class")

    # Split train/val
    val_n = args.val_per_class
    db_train, db_val = db_emb[val_n:], db_emb[:val_n]
    em_train, em_val = em_emb[val_n:], em_emb[:val_n]

    train_emb_rf = torch.cat([db_train, em_train])
    train_labels_rf = torch.cat([
        torch.zeros(db_train.shape[0]),  # DramaBox = 0 (synthetic)
        torch.ones(em_train.shape[0]),   # Emilia = 1 (real)
    ])
    val_emb_rf = torch.cat([db_val, em_val])
    val_labels_rf = torch.cat([
        torch.zeros(db_val.shape[0]),
        torch.ones(em_val.shape[0]),
    ])

    # Shuffle
    perm = torch.randperm(train_emb_rf.shape[0])
    train_emb_rf = train_emb_rf[perm]
    train_labels_rf = train_labels_rf[perm]

    log.info(f"Real/Fake - Train: {train_emb_rf.shape[0]}, Val: {val_emb_rf.shape[0]}")

    # ── Step 4: Train Real vs Synthetic classifier ──
    log.info("=" * 60)
    log.info("Step 4: Training Real vs Synthetic classifier")
    log.info("=" * 60)

    rf_model, rf_best_acc = train_classifier(
        "RealFake", train_emb_rf, train_labels_rf, val_emb_rf, val_labels_rf,
        epochs=args.epochs_real_fake, lr=args.lr, device=device,
    )
    log.info(f"Real/Fake best val accuracy: {rf_best_acc:.4f}")

    # Save
    rf_path = os.path.join(OUTPUT_DIR, "real_fake_classifier.pt")
    torch.save({
        "model_state_dict": rf_model.state_dict(),
        "val_accuracy": rf_best_acc,
        "n_train": train_emb_rf.shape[0],
        "n_val": val_emb_rf.shape[0],
        "input_dim": 768,
        "hidden1": 128,
        "hidden2": 32,
        "label_map": {"0": "synthetic_dramabox", "1": "real_emilia"},
    }, rf_path)
    log.info(f"Saved: {rf_path}")

    # ── Step 5: Prepare distortion dataset ──
    log.info("=" * 60)
    log.info("Step 5: Preparing distortion/quality dataset")
    log.info("=" * 60)

    # Use a separate subset of files for distortion classifier
    random.shuffle(dramabox_files)
    random.shuffle(emilia_files)

    quality_emb, quality_labels = prepare_distortion_dataset(
        dramabox_files, emilia_files, clap_model, device,
        n_per_class=args.distortion_samples, batch_size=args.batch_size,
    )

    # Split: 100 clean + 100 distorted for val
    # Group by label
    clean_idx = (quality_labels == 1.0).nonzero(as_tuple=True)[0]
    dist_idx = (quality_labels == 0.0).nonzero(as_tuple=True)[0]

    val_clean = clean_idx[:100]
    val_dist = dist_idx[:100]
    train_clean = clean_idx[100:]
    train_dist = dist_idx[100:]

    val_idx_q = torch.cat([val_clean, val_dist])
    train_idx_q = torch.cat([train_clean, train_dist])

    perm_t = torch.randperm(train_idx_q.shape[0])
    train_idx_q = train_idx_q[perm_t]

    train_emb_q = quality_emb[train_idx_q]
    train_labels_q = quality_labels[train_idx_q]
    val_emb_q = quality_emb[val_idx_q]
    val_labels_q = quality_labels[val_idx_q]

    log.info(f"Quality - Train: {train_emb_q.shape[0]}, Val: {val_emb_q.shape[0]}")

    # ── Step 6: Train quality classifier ──
    log.info("=" * 60)
    log.info("Step 6: Training Quality/Distortion classifier")
    log.info("=" * 60)

    q_model, q_best_acc = train_classifier(
        "Quality", train_emb_q, train_labels_q, val_emb_q, val_labels_q,
        epochs=args.epochs_quality, lr=args.lr, device=device,
    )
    log.info(f"Quality best val accuracy: {q_best_acc:.4f}")

    # Save
    q_path = os.path.join(OUTPUT_DIR, "quality_classifier.pt")
    torch.save({
        "model_state_dict": q_model.state_dict(),
        "val_accuracy": q_best_acc,
        "n_train": train_emb_q.shape[0],
        "n_val": val_emb_q.shape[0],
        "input_dim": 768,
        "hidden1": 128,
        "hidden2": 32,
        "label_map": {"0": "distorted", "1": "clean"},
        "distortion_types": ["clipping", "overdrive", "subtle_reverb"],
    }, q_path)
    log.info(f"Saved: {q_path}")

    # Save embeddings for future use
    emb_path = os.path.join(OUTPUT_DIR, "clap_embeddings.pt")
    torch.save({
        "dramabox_embeddings": db_emb,
        "emilia_embeddings": em_emb,
        "quality_embeddings": quality_emb,
        "quality_labels": quality_labels,
    }, emb_path)
    log.info(f"Saved embeddings: {emb_path}")

    # ── Summary ──
    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)
    log.info(f"Real vs Synthetic classifier:")
    log.info(f"  Params: {rf_model.count_params():,}")
    log.info(f"  Train: {train_emb_rf.shape[0]} samples")
    log.info(f"  Val accuracy: {rf_best_acc:.4f}")
    log.info(f"  Saved: {rf_path}")
    log.info(f"")
    log.info(f"Quality/Distortion classifier:")
    log.info(f"  Params: {q_model.count_params():,}")
    log.info(f"  Train: {train_emb_q.shape[0]} samples")
    log.info(f"  Val accuracy: {q_best_acc:.4f}")
    log.info(f"  Saved: {q_path}")


if __name__ == "__main__":
    main()
