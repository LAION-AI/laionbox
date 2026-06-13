#!/usr/bin/env python3
"""
Retrain just the quality/distortion classifier with updated (more subtle) reverb.
Reuses cached CLAP model, re-embeds distorted samples with new augmentation params.
Then regenerates the quality HTML grid.
"""

import json
import logging
import os
import random
import shutil
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

EMOLIA_RAW = "finetune_data_emolia/raw_audio"
DRAMABOX_EXTRACT = "/tmp/dramabox_audio_for_clap"
CLASSIFIERS_DIR = "classifiers"
GRID_DIR = "classifiers/grids"


class BinaryMLP(nn.Module):
    def __init__(self, input_dim=768, hidden1=128, hidden2=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(hidden1, hidden2), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(hidden2, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)
    def count_params(self):
        return sum(p.numel() for p in self.parameters())


def apply_distortions(wav, sr=16000):
    """Apply random distortion effects with SUBTLE reverb."""
    effects = []
    available = ["clip", "overdrive", "subtle_reverb"]
    n_effects = random.randint(1, 2)
    chosen = random.sample(available, min(n_effects, len(available)))

    for effect in chosen:
        if effect == "clip":
            gain = random.uniform(2.0, 6.0)
            wav = wav * gain
            wav = torch.clamp(wav, -1.0, 1.0)
            effects.append(f"clip (gain {gain:.1f}x)")
        elif effect == "overdrive":
            gain = random.uniform(1.5, 4.0)
            wav = torch.tanh(wav * gain) / gain * 1.5
            effects.append(f"overdrive ({gain:.1f}x)")
        elif effect == "subtle_reverb":
            # Very short early reflections, low wet mix
            ir_len = int(sr * random.uniform(0.005, 0.025))  # 5-25ms
            ir = torch.randn(ir_len) * torch.exp(-torch.linspace(0, 8, ir_len))
            ir[0] = 1.0  # strong direct signal
            ir = ir / ir.abs().max()
            wav_padded = F.pad(wav.unsqueeze(0).unsqueeze(0), (0, ir_len - 1))
            ir_kernel = ir.flip(0).unsqueeze(0).unsqueeze(0)
            reverbed = F.conv1d(wav_padded, ir_kernel).squeeze()
            mix = random.uniform(0.08, 0.25)
            wav_len = min(wav.shape[0], reverbed.shape[0])
            wav = (1 - mix) * wav[:wav_len] + mix * reverbed[:wav_len]
            effects.append(f"reverb ({ir_len/sr*1000:.0f}ms, {mix:.0%} wet)")

    peak = wav.abs().max()
    if peak > 0:
        wav = wav / peak * 0.9
    return wav, effects


def load_clap(device):
    from transformers import AutoModel, AutoTokenizer
    model = AutoModel.from_pretrained("laion/voiceclap-small", trust_remote_code=True)
    model.eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained("laion/voiceclap-small")
    return model, tokenizer


def embed_batch(wavs, model, device, batch_size=64):
    embs = []
    for i in range(0, len(wavs), batch_size):
        chunk = wavs[i:i+batch_size]
        max_s = max(w.shape[0] for w in chunk)
        padded = torch.zeros(len(chunk), max_s)
        for k, w in enumerate(chunk):
            padded[k, :w.shape[0]] = w
        with torch.no_grad():
            e = model.encode_waveform(padded.to(device))
            e = e / e.norm(dim=-1, keepdim=True)
            embs.append(e.cpu())
    return torch.cat(embs) if embs else torch.zeros(0, 768)


def load_and_preprocess(path, sr_target=16000, max_sec=30):
    wav, sr = torchaudio.load(path)
    if sr != sr_target:
        wav = torchaudio.functional.resample(wav, sr, sr_target)
    wav = wav.mean(0)
    max_len = max_sec * sr_target
    if wav.shape[0] > max_len:
        wav = wav[:max_len]
    if wav.shape[0] < sr_target:
        wav = F.pad(wav, (0, sr_target - wav.shape[0]))
    return wav


def train_classifier(name, train_emb, train_labels, val_emb, val_labels,
                     epochs=100, lr=1e-3, device="cpu"):
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
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        total_loss = correct = total = 0
        for emb, lbl in train_dl:
            logits = model(emb)
            loss = criterion(logits, lbl)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * emb.shape[0]
            correct += ((logits > 0).float() == lbl).sum().item()
            total += emb.shape[0]
        scheduler.step()

        model.eval()
        val_correct = val_total = val_loss = 0
        with torch.no_grad():
            for emb, lbl in val_dl:
                logits = model(emb)
                val_loss += criterion(logits, lbl).item() * emb.shape[0]
                val_correct += ((logits > 0).float() == lbl).sum().item()
                val_total += emb.shape[0]

        val_acc = val_correct / val_total
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            log.info(f"[{name}] Epoch {epoch+1}/{epochs}: "
                     f"train_loss={total_loss/total:.4f} train_acc={correct/total:.4f} "
                     f"val_loss={val_loss/val_total:.4f} val_acc={val_acc:.4f} "
                     f"best={best_val_acc:.4f}")

        if patience_counter >= 20:
            log.info(f"[{name}] Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    return model, best_val_acc


def get_duration(path):
    info = torchaudio.info(path)
    return info.num_frames / info.sample_rate


def generate_quality_grid(samples, output_html, audio_subdir):
    clean = sorted([s for s in samples if s["true_label"] == "clean"],
                   key=lambda s: s["p_clean"])
    distorted = sorted([s for s in samples if s["true_label"] == "distorted"],
                       key=lambda s: s["p_clean"], reverse=True)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Audio Quality / Distortion Classifier Grid (v2 — subtle reverb)</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0d1117;color:#c9d1d9;padding:20px;line-height:1.6}}
h1{{color:#58a6ff;margin-bottom:4px}}
h2{{color:#f0883e;margin:24px 0 12px;font-size:1.2em}}
.sub{{color:#8b949e;margin-bottom:20px;font-size:.9em}}
.explainer{{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:20px;margin-bottom:24px;font-size:.88em;line-height:1.7}}
.explainer p{{margin-bottom:10px}}
.explainer code{{background:#0d1117;padding:2px 6px;border-radius:4px;font-size:.9em;color:#79c0ff}}
.explainer ul{{margin:8px 0 12px 20px}}
.explainer li{{margin-bottom:4px}}
.section-desc{{color:#8b949e;font-size:.85em;margin-bottom:16px;font-style:italic}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:12px;margin-bottom:32px}}
.card{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:14px;position:relative}}
.card .rank{{position:absolute;top:8px;right:10px;font-size:.7em;color:#484f58;font-weight:600}}
.card .name{{font-size:.85em;font-weight:600;color:#c9d1d9;margin-bottom:4px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.card .meta{{font-size:.75em;color:#8b949e;margin-bottom:4px}}
.card .effects{{font-size:.72em;color:#d29922;margin-bottom:8px;font-style:italic}}
.card .pred-bar{{height:24px;border-radius:4px;overflow:hidden;background:#21262d;margin-bottom:4px;position:relative}}
.card .pred-fill{{height:100%;transition:width .3s}}
.card .pred-label{{position:absolute;top:3px;left:8px;font-size:.7em;font-weight:600;color:#fff;text-shadow:0 1px 2px rgba(0,0,0,.5)}}
.card .pred-val{{position:absolute;top:3px;right:8px;font-size:.7em;font-weight:600;color:#fff;text-shadow:0 1px 2px rgba(0,0,0,.5)}}
.card .verdict{{font-size:.8em;font-weight:600;margin-bottom:6px}}
.correct{{color:#3fb950}}
.wrong{{color:#f85149}}
.badge{{display:inline-block;padding:2px 8px;border-radius:10px;font-size:.7em;font-weight:600;margin-bottom:6px}}
.badge-clean{{background:#23863633;color:#3fb950}}
.badge-dist{{background:#d2992233;color:#d29922}}
audio{{width:100%;margin-top:4px}}
.stats{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px;margin-bottom:24px}}
.stats table{{width:100%;border-collapse:collapse;font-size:.85em}}
.stats th{{text-align:left;color:#8b949e;padding:4px 8px;font-weight:normal;text-transform:uppercase;font-size:.75em}}
.stats td{{padding:4px 8px;color:#c9d1d9}}
</style>
</head>
<body>
<h1>Audio Quality / Distortion Classifier (v2)</h1>
<p class="sub">VoiceCLAP-small → MLP (102K params) | Subtle reverb (5-25ms IR, 8-25% wet) | {time.strftime("%Y-%m-%d %H:%M")}</p>

<div class="explainer">
<p><strong>What is this?</strong> A classifier that detects whether audio has been degraded by distortion effects. It works by converting audio to a 768-dimensional "fingerprint" (using VoiceCLAP-small), then running that through a small neural network that outputs P(clean) — the probability the audio is undistorted.</p>

<p><strong>Distortion types applied:</strong></p>
<ul>
<li><strong>Clipping</strong> — audio amplified 2-6x then hard-clamped, simulating a mic that's too loud</li>
<li><strong>Overdrive</strong> — soft saturation via tanh, like a slightly overdriven preamp</li>
<li><strong>Subtle reverb</strong> — very short impulse response (5-25ms) with low wet mix (8-25%), simulating slight room coloration or mic bleed — barely audible but measurable</li>
</ul>

<p><strong>Data:</strong> Clean samples come from both DramaBox (synthetic TTS) and Emilia (real speech). Distorted samples are the same audio with 1-2 random effects applied. The classifier should detect distortion regardless of whether the source is real or synthetic.</p>
</div>
"""

    all_s = clean + distorted
    correct = sum(1 for s in all_s if s["predicted_label"] == s["true_label"])
    html += f"""<div class="stats"><table>
<tr><th>Total samples</th><td>{len(all_s)}</td><th>Accuracy</th><td>{correct}/{len(all_s)} ({correct/len(all_s)*100:.1f}%)</td></tr>
<tr><th>Clean samples</th><td>{len(clean)}</td><th>Distorted samples</th><td>{len(distorted)}</td></tr>
</table></div>"""

    def render_section(items, title, desc):
        nonlocal html
        html += f'<h2>{title}</h2><p class="section-desc">{desc}</p><div class="grid">'
        for i, s in enumerate(items):
            p = s["p_clean"]
            is_correct = s["predicted_label"] == s["true_label"]
            bar_color = "#f85149" if p < 0.3 else "#d29922" if p < 0.6 else "#3fb950"
            verdict_cls = "correct" if is_correct else "wrong"
            badge_cls = "badge-clean" if s["true_label"] == "clean" else "badge-dist"
            badge_text = "CLEAN" if s["true_label"] == "clean" else "DISTORTED"
            effects_str = s.get("effects", "")
            source_str = s.get("source", "")

            html += f"""<div class="card">
<div class="rank">#{i+1}</div>
<div class="name">{s['name']}</div>
<span class="badge {badge_cls}">{badge_text}</span>
<div class="meta">{s['duration']:.1f}s | source: {source_str}</div>"""
            if effects_str:
                html += f'<div class="effects">Effects: {effects_str}</div>'
            html += f"""<div class="pred-bar">
  <div class="pred-fill" style="width:{p*100:.0f}%;background:{bar_color}"></div>
  <div class="pred-label">P(clean)</div>
  <div class="pred-val">{p:.4f}</div>
</div>
<div class="verdict {verdict_cls}">{'Correct' if is_correct else 'WRONG'}: predicted {s['predicted_label']} ({p:.1%} clean)</div>
<audio controls preload="metadata"><source src="{audio_subdir}/{s['filename']}" type="audio/wav"></audio>
</div>"""
        html += '</div>'

    render_section(distorted,
                   "Distorted — sorted by P(clean) descending",
                   "Which distorted samples fool the classifier? These are the hardest to detect.")
    render_section(clean,
                   "Clean — sorted by P(clean) ascending",
                   "Which clean samples look most distorted to the classifier?")

    html += '</body></html>'
    with open(output_html, 'w') as f:
        f.write(html)


def main():
    random.seed(42)
    torch.manual_seed(42)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    os.makedirs(GRID_DIR, exist_ok=True)
    q_audio_dir = os.path.join(GRID_DIR, "quality_audio_v2")
    os.makedirs(q_audio_dir, exist_ok=True)

    # Load CLAP
    log.info("Loading CLAP model...")
    clap_model, _ = load_clap(device)

    # Get file lists
    dramabox_files = sorted([f for f in os.listdir(DRAMABOX_EXTRACT) if f.endswith("_full.mp3")])
    emilia_files = sorted([f for f in os.listdir(EMOLIA_RAW) if f.endswith(".wav")])
    random.shuffle(dramabox_files)
    random.shuffle(emilia_files)

    n_per_class = 1000
    n_half = n_per_class // 2

    # ── Build dataset: 1000 clean + 1000 distorted ──
    log.info("Building quality dataset with subtle reverb...")

    def load_wavs(file_list, base_dir, n):
        wavs = []
        for f in file_list[:n]:
            try:
                wavs.append(load_and_preprocess(os.path.join(base_dir, f)))
            except:
                pass
        return wavs

    # Clean: 500 DB + 500 Emilia
    clean_db_wavs = load_wavs(dramabox_files[:n_half], DRAMABOX_EXTRACT, n_half)
    clean_em_wavs = load_wavs(emilia_files[:n_half], EMOLIA_RAW, n_half)
    log.info(f"Clean: {len(clean_db_wavs)} DramaBox + {len(clean_em_wavs)} Emilia")

    log.info("Embedding clean samples...")
    clean_emb = embed_batch(clean_db_wavs + clean_em_wavs, clap_model, device)
    clean_labels = torch.ones(clean_emb.shape[0])

    # Distorted: 500 DB + 500 Emilia (different files)
    dist_db_wavs = load_wavs(dramabox_files[n_half:n_half*2], DRAMABOX_EXTRACT, n_half)
    dist_em_wavs = load_wavs(emilia_files[n_half:n_half*2], EMOLIA_RAW, n_half)

    log.info("Applying distortions...")
    distorted_wavs = []
    for w in dist_db_wavs + dist_em_wavs:
        dw, _ = apply_distortions(w.clone(), 16000)
        distorted_wavs.append(dw)

    log.info("Embedding distorted samples...")
    dist_emb = embed_batch(distorted_wavs, clap_model, device)
    dist_labels = torch.zeros(dist_emb.shape[0])

    log.info(f"Total: {clean_emb.shape[0]} clean + {dist_emb.shape[0]} distorted")

    # Split: 100 per class for val
    val_clean = clean_emb[:100]
    val_dist = dist_emb[:100]
    train_clean = clean_emb[100:]
    train_dist = dist_emb[100:]

    train_emb = torch.cat([train_clean, train_dist])
    train_labels = torch.cat([clean_labels[100:], dist_labels[100:]])
    val_emb = torch.cat([val_clean, val_dist])
    val_labels = torch.cat([clean_labels[:100], dist_labels[:100]])

    perm = torch.randperm(train_emb.shape[0])
    train_emb = train_emb[perm]
    train_labels = train_labels[perm]

    log.info(f"Train: {train_emb.shape[0]}, Val: {val_emb.shape[0]}")

    # ── Train ──
    log.info("Training quality classifier v2...")
    model, best_acc = train_classifier("Quality_v2", train_emb, train_labels,
                                        val_emb, val_labels, epochs=150, lr=1e-3, device=device)
    log.info(f"Best val accuracy: {best_acc:.4f}")

    # Save
    q_path = os.path.join(CLASSIFIERS_DIR, "quality_classifier.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "val_accuracy": best_acc,
        "n_train": train_emb.shape[0],
        "n_val": val_emb.shape[0],
        "input_dim": 768, "hidden1": 128, "hidden2": 32,
        "label_map": {"0": "distorted", "1": "clean"},
        "distortion_types": ["clipping", "overdrive", "subtle_reverb"],
        "reverb_params": "IR 5-25ms, wet 8-25%",
    }, q_path)
    log.info(f"Saved: {q_path}")

    # ── Generate HTML grid ──
    log.info("Generating HTML grid...")
    model.eval().to(device)

    grid_samples = []

    # 10 clean DramaBox
    for f in dramabox_files[n_half*2:n_half*2+10]:
        src = os.path.join(DRAMABOX_EXTRACT, f)
        wav = load_and_preprocess(src)
        wav_name = f.replace("_full.mp3", "_clean.wav")
        dst = os.path.join(q_audio_dir, wav_name)
        torchaudio.save(dst, wav.unsqueeze(0), 16000)

        emb = embed_batch([wav], clap_model, device)
        with torch.no_grad():
            p = torch.sigmoid(model(emb.to(device))).item()

        grid_samples.append({
            "name": f.replace("_full.mp3", ""), "filename": wav_name,
            "true_label": "clean", "predicted_label": "clean" if p > 0.5 else "distorted",
            "p_clean": p, "duration": get_duration(dst), "source": "DramaBox",
        })

    # 10 clean Emilia
    for f in emilia_files[n_half*2:n_half*2+10]:
        src = os.path.join(EMOLIA_RAW, f)
        wav_name = f.replace(".wav", "_clean.wav")
        dst = os.path.join(q_audio_dir, wav_name)
        shutil.copy2(src, dst)

        wav = load_and_preprocess(src)
        emb = embed_batch([wav], clap_model, device)
        with torch.no_grad():
            p = torch.sigmoid(model(emb.to(device))).item()

        grid_samples.append({
            "name": f.replace(".wav", ""), "filename": wav_name,
            "true_label": "clean", "predicted_label": "clean" if p > 0.5 else "distorted",
            "p_clean": p, "duration": get_duration(dst), "source": "Emilia",
        })

    # 10 distorted DramaBox
    for f in dramabox_files[n_half*2+10:n_half*2+20]:
        src = os.path.join(DRAMABOX_EXTRACT, f)
        wav = load_and_preprocess(src)
        dw, effects = apply_distortions(wav.clone(), 16000)
        wav_name = f.replace("_full.mp3", "_distorted.wav")
        dst = os.path.join(q_audio_dir, wav_name)
        torchaudio.save(dst, dw.unsqueeze(0), 16000)

        emb = embed_batch([dw], clap_model, device)
        with torch.no_grad():
            p = torch.sigmoid(model(emb.to(device))).item()

        grid_samples.append({
            "name": f.replace("_full.mp3", ""), "filename": wav_name,
            "true_label": "distorted", "predicted_label": "clean" if p > 0.5 else "distorted",
            "p_clean": p, "duration": get_duration(dst), "source": "DramaBox",
            "effects": " + ".join(effects),
        })

    # 10 distorted Emilia
    for f in emilia_files[n_half*2+10:n_half*2+20]:
        src = os.path.join(EMOLIA_RAW, f)
        wav = load_and_preprocess(src)
        dw, effects = apply_distortions(wav.clone(), 16000)
        wav_name = f.replace(".wav", "_distorted.wav")
        dst = os.path.join(q_audio_dir, wav_name)
        torchaudio.save(dst, dw.unsqueeze(0), 16000)

        emb = embed_batch([dw], clap_model, device)
        with torch.no_grad():
            p = torch.sigmoid(model(emb.to(device))).item()

        grid_samples.append({
            "name": f.replace(".wav", ""), "filename": wav_name,
            "true_label": "distorted", "predicted_label": "clean" if p > 0.5 else "distorted",
            "p_clean": p, "duration": get_duration(dst), "source": "Emilia",
            "effects": " + ".join(effects),
        })

    html_path = os.path.join(GRID_DIR, "quality_distortion_v2.html")
    generate_quality_grid(grid_samples, html_path, "quality_audio_v2")
    log.info(f"HTML: {html_path}")

    # Summary
    correct = sum(1 for s in grid_samples if s["predicted_label"] == s["true_label"])
    log.info(f"Grid accuracy: {correct}/{len(grid_samples)} ({correct/len(grid_samples)*100:.1f}%)")

    for s in grid_samples:
        tag = "clean" if s["true_label"] == "clean" else "dist "
        eff = s.get("effects", "")
        log.info(f"  [{tag}] {s['name'][:40]:40s} P(clean)={s['p_clean']:.4f}  {eff}")


if __name__ == "__main__":
    main()
