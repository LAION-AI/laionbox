#!/usr/bin/env python3
"""Build a comprehensive validation grid HTML page.

Summarizes all training runs, what worked and what didn't,
training curves, and validation audio samples from the best checkpoints.
"""

import json
import os
import sys
import base64
import glob
from pathlib import Path
from datetime import datetime


def load_metrics(path):
    """Load metrics JSONL file."""
    metrics = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                metrics.append(json.loads(line))
    return metrics


def make_sparkline_svg(values, width=200, height=40, color="#58a6ff", label=""):
    """Generate an inline SVG sparkline."""
    if not values:
        return ""
    mn, mx = min(values), max(values)
    rng = mx - mn if mx != mn else 1.0
    n = len(values)
    points = []
    for i, v in enumerate(values):
        x = (i / max(n - 1, 1)) * width
        y = height - ((v - mn) / rng) * (height - 4) - 2
        points.append(f"{x:.1f},{y:.1f}")
    polyline = " ".join(points)
    return f'''<svg width="{width}" height="{height}" style="vertical-align:middle">
        <polyline points="{polyline}" fill="none" stroke="{color}" stroke-width="1.5"/>
        <text x="2" y="10" font-size="9" fill="#8b949e">{label}</text>
        <text x="{width-2}" y="{height-2}" font-size="9" fill="{color}" text-anchor="end">{values[-1]:.3f}</text>
    </svg>'''


def audio_to_base64(path):
    """Read audio file and return base64 data URI."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        data = f.read()
    ext = path.rsplit(".", 1)[-1].lower()
    mime = {"wav": "audio/wav", "mp3": "audio/mpeg", "flac": "audio/flac"}.get(ext, "audio/wav")
    return f"data:{mime};base64,{base64.b64encode(data).decode()}"


def build_html(output_dir, metrics_5ep, val_dirs):
    """Build the full HTML page."""

    # Parse 5ep metrics by epoch
    epochs = {}
    for m in metrics_5ep:
        ep = (m['step'] - 1) // 41 + 1
        if ep not in epochs:
            epochs[ep] = []
        epochs[ep].append(m)

    # Extract time series
    steps = [m['step'] for m in metrics_5ep]
    flow_losses = [m.get('flow_loss', m.get('loss', 0)) for m in metrics_5ep]
    nat_rewards = [m.get('naturalness_reward', m.get('clap_text_reward', 0)) for m in metrics_5ep]
    cent_scores = [m.get('centroid_score', 0) for m in metrics_5ep]
    spk_sims = [m.get('speaker_sim', 0) for m in metrics_5ep]

    # Per-epoch averages
    epoch_avgs = {}
    for ep in sorted(epochs.keys()):
        g = epochs[ep]
        epoch_avgs[ep] = {
            'flow': sum(m.get('flow_loss', m.get('loss', 0)) for m in g) / len(g),
            'nat': sum(m.get('naturalness_reward', m.get('clap_text_reward', 0)) for m in g) / len(g),
            'cent': sum(m.get('centroid_score', 0) for m in g) / len(g),
            'spk': sum(m.get('speaker_sim', 0) for m in g) / len(g),
        }

    # Collect validation audio samples
    val_samples = {}
    for epoch_num, val_dir in val_dirs.items():
        if not os.path.isdir(val_dir):
            continue
        epoch_dir = os.path.join(val_dir, f"epoch_{epoch_num}")
        if not os.path.isdir(epoch_dir):
            continue
        wavs = sorted(glob.glob(os.path.join(epoch_dir, "*.wav")))
        # Group by sample name
        samples = {}
        for w in wavs:
            fname = os.path.basename(w)
            # Parse sample name and mode
            # e.g., accc_acting_challenge_1_uncond.wav, accc_acting_challenge_1_fwd.wav
            # or accc_acting_challenge_1_ref_Fairy-2.wav
            parts = fname.rsplit(".", 1)[0]
            if "_ref_part" in parts:
                # Reference audio, skip embedding
                continue
            elif "_uncond" in parts:
                sample = parts.replace("_uncond", "")
                mode = "uncond"
            elif "_fwd" in parts:
                sample = parts.replace("_fwd", "")
                mode = "fwd"
            elif "_rev" in parts:
                sample = parts.replace("_rev", "")
                mode = "rev"
            elif "_ref_" in parts:
                idx = parts.index("_ref_")
                sample = parts[:idx]
                mode = "ref_" + parts[idx + 5:]
            else:
                continue
            if sample not in samples:
                samples[sample] = {}
            samples[sample][mode] = w
        val_samples[epoch_num] = samples

    # Load validation metrics
    val_metrics = {}
    for epoch_num, val_dir in val_dirs.items():
        metrics_path = os.path.join(val_dir, f"epoch_{epoch_num}", "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                val_metrics[epoch_num] = json.load(f)

    # Build HTML
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DramaBox LoRA Fine-Tuning: Differentiable Reward Training Results</title>
<style>
:root {{
    --bg: #0d1117;
    --card: #161b22;
    --border: #30363d;
    --text: #c9d1d9;
    --text-dim: #8b949e;
    --accent: #58a6ff;
    --green: #3fb950;
    --red: #f85149;
    --yellow: #d29922;
    --orange: #db6d28;
}}
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.6;
    padding: 24px;
    max-width: 1400px;
    margin: 0 auto;
}}
h1 {{ color: #fff; margin-bottom: 8px; font-size: 28px; }}
h2 {{ color: var(--accent); margin: 32px 0 16px; font-size: 22px; border-bottom: 1px solid var(--border); padding-bottom: 8px; }}
h3 {{ color: #fff; margin: 24px 0 12px; font-size: 18px; }}
.subtitle {{ color: var(--text-dim); font-size: 14px; margin-bottom: 24px; }}
table {{
    border-collapse: collapse;
    width: 100%;
    margin: 16px 0;
    font-size: 14px;
}}
th, td {{
    padding: 8px 12px;
    border: 1px solid var(--border);
    text-align: left;
}}
th {{ background: var(--card); color: var(--accent); font-weight: 600; }}
tr:nth-child(even) {{ background: rgba(22, 27, 34, 0.5); }}
.card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    margin: 12px 0;
}}
.grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 16px;
    margin: 16px 0;
}}
.audio-card {{
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px;
}}
.audio-card h4 {{
    font-size: 13px;
    color: var(--accent);
    margin-bottom: 8px;
}}
.audio-card .mode {{
    font-size: 11px;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
}}
audio {{ width: 100%; height: 32px; }}
.badge {{
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 600;
}}
.badge-green {{ background: rgba(63, 185, 80, 0.2); color: var(--green); }}
.badge-red {{ background: rgba(248, 81, 73, 0.2); color: var(--red); }}
.badge-yellow {{ background: rgba(210, 153, 34, 0.2); color: var(--yellow); }}
.badge-blue {{ background: rgba(88, 166, 255, 0.2); color: var(--accent); }}
.metric-row {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 4px 0;
    border-bottom: 1px solid rgba(48, 54, 61, 0.5);
}}
.metric-row:last-child {{ border-bottom: none; }}
.metric-label {{ color: var(--text-dim); font-size: 13px; }}
.metric-value {{ color: #fff; font-weight: 600; font-size: 14px; }}
.delta-pos {{ color: var(--green); }}
.delta-neg {{ color: var(--red); }}
.delta-neutral {{ color: var(--yellow); }}
code {{
    background: rgba(110, 118, 129, 0.15);
    padding: 2px 6px;
    border-radius: 4px;
    font-size: 13px;
    font-family: 'SFMono-Regular', Consolas, monospace;
}}
.chart-container {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
    margin: 16px 0;
}}
@media (max-width: 768px) {{
    .chart-container {{ grid-template-columns: 1fr; }}
    .grid {{ grid-template-columns: 1fr; }}
}}
.timeline {{
    position: relative;
    padding-left: 24px;
    margin: 16px 0;
}}
.timeline::before {{
    content: '';
    position: absolute;
    left: 8px;
    top: 0;
    bottom: 0;
    width: 2px;
    background: var(--border);
}}
.timeline-item {{
    position: relative;
    margin-bottom: 20px;
    padding: 12px 16px;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
}}
.timeline-item::before {{
    content: '';
    position: absolute;
    left: -20px;
    top: 16px;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: var(--border);
}}
.timeline-item.success::before {{ background: var(--green); }}
.timeline-item.failure::before {{ background: var(--red); }}
.timeline-item.partial::before {{ background: var(--yellow); }}
.timeline-item .run-title {{ font-weight: 600; color: #fff; margin-bottom: 4px; }}
.timeline-item .run-desc {{ font-size: 13px; color: var(--text-dim); }}
.epoch-table th {{ text-align: center; }}
.epoch-table td {{ text-align: center; }}
.sample-section {{
    margin: 24px 0;
    padding: 16px;
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
}}
.sample-section h4 {{
    color: #fff;
    font-size: 15px;
    margin-bottom: 12px;
}}
.prompt-text {{
    font-size: 12px;
    color: var(--text-dim);
    background: rgba(0,0,0,0.3);
    padding: 8px;
    border-radius: 4px;
    max-height: 80px;
    overflow-y: auto;
    margin-bottom: 12px;
    font-family: monospace;
    white-space: pre-wrap;
}}
</style>
</head>
<body>

<h1>DramaBox LoRA Fine-Tuning: Differentiable Reward Training</h1>
<p class="subtitle">
    Multi-auxiliary loss fine-tuning with differentiable backpropagation through CLAP and WavLM.
    <br>Generated {datetime.now().strftime("%Y-%m-%d %H:%M")} | 8x A100 80GB | LTX-2.3 22B + LoRA rank 128
</p>

<h2>Executive Summary</h2>

<div class="card">
<p><strong>Goal:</strong> Improve the naturalness of DramaBox voice cloning LoRA by adding differentiable reward signals
(CLAP naturalness, centroid real/fake, WavLM speaker similarity) that backpropagate through frozen reward models
to provide directional gradient information to the LoRA parameters.</p>

<p style="margin-top:12px"><strong>Result:</strong> After fixing two critical gradient chain bugs (VoiceCLAP's <code>@torch.no_grad()</code> on mel computation,
and WavLM's wrong model class), the naturalness reward improved by <span class="badge badge-green">+33%</span> over 5 epochs
(0.265 &rarr; 0.354). This is the first time in 7 training runs that any reward metric showed sustained improvement.</p>
</div>

<h2>What Didn't Work (Runs 1-5)</h2>

<div class="timeline">
    <div class="timeline-item failure">
        <div class="run-title">Run 1: Low coefficient caps (ratio=1, cap=2)</div>
        <div class="run-desc">3 epochs, 972 steps. Coefficients hit cap of 2.0 immediately. Aux losses too weak to influence training. All rewards flat. Flow loss improved normally (0.50 &rarr; 0.43).</div>
    </div>
    <div class="timeline-item failure">
        <div class="run-title">Run 2: High coefficients (ratio=5, cap=10 vs cap=50)</div>
        <div class="run-desc">2 epochs each, A/B comparison. Despite 5x coefficient increase, rewards remained flat. Option A (cap=10) and C (cap=50) produced identical reward trajectories. <strong>Root cause:</strong> reward-weighted reconstruction loss has the same gradient direction as flow matching &mdash; scaling the coefficient only changes magnitude, not direction.</div>
    </div>
    <div class="timeline-item failure">
        <div class="run-title">Run 3: Sigma threshold + large batch (BS=256)</div>
        <div class="run-desc">Only compute aux losses when &sigma; &lt; 0.4 (low noise, meaningful x0 predictions). 64x gradient accumulation for cleaner signal. Options A and C still produced identical flat reward trajectories. <strong>Root cause confirmed:</strong> the gradient direction problem persists regardless of noise filtering or batch size.</div>
    </div>
    <div class="timeline-item failure">
        <div class="run-title">Run 4: Rejection sampling (top 50%)</div>
        <div class="run-desc">Train only on micro-batches with above-median composite reward. Marginal flow loss improvement (0.4591 &rarr; 0.4487) but rewards still flat. Rejection filters bad examples but provides no directional gradient signal.</div>
    </div>
    <div class="timeline-item failure">
        <div class="run-title">Run 5: Differentiable rewards (broken gradient chain)</div>
        <div class="run-desc">Implemented backprop through decoder/CLAP/WavLM. Rewards STILL flat despite "differentiable" setup. <strong>Root cause found:</strong> VoiceCLAP's <code>compute_log_mel</code> has <code>@torch.no_grad()</code> decorator that silently kills all gradients. Additionally, <code>Wav2Vec2ForXVector</code> loads wrong weights for WavLM (key prefix mismatch).</div>
    </div>
</div>

<h2>What Worked (Runs 6-7)</h2>

<div class="timeline">
    <div class="timeline-item partial">
        <div class="run-title">Run 6: Gradient fix ablation (1 epoch)</div>
        <div class="run-desc">Applied two fixes: (1) <code>encode_clap_waveform_differentiable()</code> bypasses <code>@torch.no_grad()</code> on mel, (2) <code>WavLMForXVector</code> for correct weight loading. Naturalness reward showed first-ever positive trend: +0.013 in 41 steps.</div>
    </div>
    <div class="timeline-item success">
        <div class="run-title">Run 7: Full 5-epoch training with gradient fix</div>
        <div class="run-desc">Sustained naturalness improvement: <strong>+0.089</strong> (0.265 &rarr; 0.354, +33% relative). Best flow loss 0.4386 at step 72. 205 steps in 2h5m on 8x A100. VRAM stable at 67-72 GB with gradient checkpointing.</div>
    </div>
</div>

<h2>The Two Critical Bugs</h2>

<div class="card">
<h3>Bug 1: VoiceCLAP's <code>@torch.no_grad()</code> on mel computation</h3>
<p style="margin:8px 0;font-size:13px;color:var(--text-dim)">File: <code>modeling_voiceclap.py:163</code> in HuggingFace cache</p>
<p>The <code>compute_log_mel</code> method uses <code>@torch.no_grad()</code> as a decorator. Despite <code>torch.stft</code> being fully differentiable internally, this decorator detaches the entire mel computation from the computation graph. The CLAP embedding appears to have a <code>grad_fn</code> (from the projection layers), but <strong>no gradient reaches the input waveform</strong>.</p>
<p style="margin-top:8px"><strong>Fix:</strong> <code>encode_clap_waveform_differentiable()</code> replicates the mel computation without the decorator. Uses <code>torch.stft()</code> + <code>clap_model.mel_filters</code> buffer + <code>clap_model.audio_encoder</code> + <code>clap_model.audio_proj</code>.</p>
<p style="margin-top:8px"><strong>Verification:</strong> <code>wav.grad</code> goes from <code>None</code> (original) to <code>norm=0.05</code> (fixed).</p>
</div>

<div class="card">
<h3>Bug 2: Wrong WavLM model class</h3>
<p style="margin:8px 0;font-size:13px;color:var(--text-dim)">Using <code>Wav2Vec2ForXVector</code> instead of <code>WavLMForXVector</code></p>
<p>The checkpoint <code>microsoft/wavlm-base-plus-sv</code> stores weights with key prefix <code>wavlm.*</code>, but <code>Wav2Vec2ForXVector</code> expects prefix <code>wav2vec2.*</code>. All keys are reported as MISSING/UNEXPECTED, resulting in <strong>randomly initialized weights</strong> and <code>nan</code> gradients.</p>
<p style="margin-top:8px"><strong>Fix:</strong> Use <code>WavLMForXVector</code> from <code>transformers</code>. Gradient norm goes from <code>nan</code> to <code>1.8</code>.</p>
</div>

<h2>Training Curves (Run 7: 5 Epochs)</h2>

<div class="chart-container">
    <div class="card">
        <h3>Flow Matching Loss</h3>
        {make_sparkline_svg(flow_losses, 350, 60, "#58a6ff", "flow")}
        <div class="metric-row">
            <span class="metric-label">Best</span>
            <span class="metric-value">{min(flow_losses):.4f}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">Final</span>
            <span class="metric-value">{flow_losses[-1]:.4f}</span>
        </div>
    </div>
    <div class="card">
        <h3>Naturalness Reward</h3>
        {make_sparkline_svg(nat_rewards, 350, 60, "#3fb950", "nat")}
        <div class="metric-row">
            <span class="metric-label">Start (ep1)</span>
            <span class="metric-value">{epoch_avgs.get(1, {}).get("nat", 0):.4f}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">End (ep5)</span>
            <span class="metric-value delta-pos">{epoch_avgs.get(5, {}).get("nat", 0):.4f} (+{epoch_avgs.get(5, {}).get("nat", 0) - epoch_avgs.get(1, {}).get("nat", 0):.4f})</span>
        </div>
    </div>
    <div class="card">
        <h3>Centroid Score</h3>
        {make_sparkline_svg(cent_scores, 350, 60, "#d29922", "cent")}
        <div class="metric-row">
            <span class="metric-label">Start (ep1)</span>
            <span class="metric-value">{epoch_avgs.get(1, {}).get("cent", 0):.4f}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">End (ep5)</span>
            <span class="metric-value delta-neg">{epoch_avgs.get(5, {}).get("cent", 0):.4f} ({epoch_avgs.get(5, {}).get("cent", 0) - epoch_avgs.get(1, {}).get("cent", 0):+.4f})</span>
        </div>
    </div>
    <div class="card">
        <h3>Speaker Similarity</h3>
        {make_sparkline_svg(spk_sims, 350, 60, "#bc8cff", "spk")}
        <div class="metric-row">
            <span class="metric-label">Start (ep1)</span>
            <span class="metric-value">{epoch_avgs.get(1, {}).get("spk", 0):.4f}</span>
        </div>
        <div class="metric-row">
            <span class="metric-label">End (ep5)</span>
            <span class="metric-value delta-neutral">{epoch_avgs.get(5, {}).get("spk", 0):.4f} ({epoch_avgs.get(5, {}).get("spk", 0) - epoch_avgs.get(1, {}).get("spk", 0):+.4f})</span>
        </div>
    </div>
</div>

<h2>Per-Epoch Metrics</h2>

<table class="epoch-table">
<tr>
    <th>Epoch</th>
    <th>Flow Loss</th>
    <th>Naturalness</th>
    <th>Centroid</th>
    <th>Speaker Sim</th>
    <th>Nat. Delta</th>
</tr>
'''

    prev_nat = None
    for ep in sorted(epoch_avgs.keys()):
        a = epoch_avgs[ep]
        delta_str = ""
        if prev_nat is not None:
            d = a['nat'] - prev_nat
            cls = "delta-pos" if d > 0 else "delta-neg"
            delta_str = f'<span class="{cls}">{d:+.4f}</span>'
        prev_nat = a['nat']
        html += f'''<tr>
    <td>{ep}</td>
    <td>{a["flow"]:.4f}</td>
    <td>{a["nat"]:.4f}</td>
    <td>{a["cent"]:.4f}</td>
    <td>{a["spk"]:.4f}</td>
    <td>{delta_str}</td>
</tr>
'''

    html += '</table>\n'

    # Gradient chain diagram
    html += '''
<h2>Differentiable Gradient Chain</h2>

<div class="card" style="font-family: monospace; font-size: 13px; line-height: 1.8; overflow-x: auto;">
<pre style="color: var(--text);">
LoRA parameters (128-rank, trainable)
  &darr;
LTX-2.3 Transformer (22B, frozen base + LoRA, gradient checkpointed)
  &darr;
velocity prediction (pred_tgt) &#9472;&#9472; has grad &#9472;&#9472;&gt;
  &darr;
x0 recovery: x0 = noisy.detach() - pred_tgt * &sigma;
  &darr;
Patchifier unpatchify &rarr; x0_latent
  &darr;
AudioDecoder + BigVGAN vocoder (frozen, grad flows through)
  &darr;
Predicted waveform (16kHz mono)
  &darr;                                    &darr;
<span style="color:#3fb950">encode_clap_waveform_differentiable()</span>    <span style="color:#bc8cff">WavLMForXVector (frozen)</span>
  &darr;                                    &darr;
<span style="color:#3fb950">CLAP embedding [768-dim]</span>                <span style="color:#bc8cff">Speaker embedding [512-dim]</span>
  &darr;                                    &darr;
<span style="color:#3fb950">Loss 1: -cos(emb, pos) + cos(emb, neg)</span>  <span style="color:#bc8cff">Loss 3: -cos(pred_spk, ref_spk)</span>
<span style="color:#3fb950">Loss 1b: -quality_mlp(emb)</span>
  &darr;
<span style="color:#d29922">Loss 2: -cos(emb, real_centroid) + cos(emb, synth_centroid)</span>
  &darr;
total_loss = flow + c1*aux1 + c2*aux2 + c3*aux3
  &darr;
backward() &rarr; gradients reach LoRA via differentiable chain
</pre>
</div>

<h2>Configuration (Run 7)</h2>

<div class="card">
<table>
<tr><th>Parameter</th><th>Value</th></tr>
<tr><td>Base model</td><td>LTX-2.3-22B (dev variant, audio-only v13 merged)</td></tr>
<tr><td>LoRA rank / alpha</td><td>128 / 128</td></tr>
<tr><td>Resume from</td><td>14-epoch CLAP LoRA (lora_epoch2.safetensors)</td></tr>
<tr><td>Dataset</td><td>3,845 samples (3,247 DramaBox + 598 Emilia top 10%)</td></tr>
<tr><td>3-mode IC-LoRA</td><td>voice_clone_fwd + voice_clone_rev + unconditional</td></tr>
<tr><td>Epochs</td><td>5 (205 optimizer steps, 41/epoch)</td></tr>
<tr><td>Effective batch size</td><td>256 (8 GPUs &times; 1 &times; 32 grad_accum)</td></tr>
<tr><td>Peak LR</td><td>4e-5, cosine schedule, 10 warmup steps</td></tr>
<tr><td>Aux sigma threshold</td><td>&lt; 0.4 (~23% of micro-batches activate aux losses)</td></tr>
<tr><td>Aux target ratio</td><td>5.0 (each aux loss targets 5x flow magnitude)</td></tr>
<tr><td>Coefficient cap</td><td>10.0</td></tr>
<tr><td>Differentiable reward</td><td>Enabled (gradient checkpointing on aux models)</td></tr>
<tr><td>max_ref_tokens</td><td>200 (8 sec at 25fps latent rate)</td></tr>
<tr><td>VRAM per GPU</td><td>67-72 GB (A100 80GB)</td></tr>
<tr><td>Training time</td><td>7,493s (2h 5m)</td></tr>
</table>
</div>
'''

    # Validation audio samples — side-by-side comparison
    html += '\n<h2>Validation Audio Samples</h2>\n'
    html += '<p style="color:var(--text-dim);margin-bottom:16px">Compare baseline (14-epoch CLAP LoRA) vs differentiable reward fine-tuned (epochs 2 and 5). Listen for naturalness, speaker consistency, and overall quality.</p>\n'

    # Collect baseline samples
    baseline_dir = os.path.join(os.path.dirname(output_dir), "combined_14ep_clap", "val")
    baseline_epoch_dir = os.path.join(baseline_dir, "epoch_2")
    baseline_samples = {}
    if os.path.isdir(baseline_epoch_dir):
        for w in sorted(glob.glob(os.path.join(baseline_epoch_dir, "*.wav"))):
            fname = os.path.basename(w)
            parts = fname.rsplit(".", 1)[0]
            if "_ref_part" in parts:
                continue
            elif "_uncond" in parts:
                sample = parts.replace("_uncond", "")
                mode = "uncond"
            elif "_fwd" in parts:
                sample = parts.replace("_fwd", "")
                mode = "fwd"
            elif "_rev" in parts:
                sample = parts.replace("_rev", "")
                mode = "rev"
            elif "_ref_" in parts:
                idx = parts.index("_ref_")
                sample = parts[:idx]
                mode = "ref_" + parts[idx + 5:]
            else:
                continue
            if sample not in baseline_samples:
                baseline_samples[sample] = {}
            baseline_samples[sample][mode] = w

    # Render samples in comparison format
    all_sample_names = set()
    for epoch_num in val_samples:
        all_sample_names.update(val_samples[epoch_num].keys())
    all_sample_names.update(baseline_samples.keys())

    for sample_name in sorted(all_sample_names):
        html += f'<div class="sample-section">\n'
        html += f'<h4>{sample_name}</h4>\n'

        # Get all modes across all checkpoints
        all_modes = set()
        if sample_name in baseline_samples:
            all_modes.update(baseline_samples[sample_name].keys())
        for epoch_num in val_samples:
            if sample_name in val_samples[epoch_num]:
                all_modes.update(val_samples[epoch_num][sample_name].keys())

        mode_order = ['uncond', 'fwd', 'rev'] + sorted([m for m in all_modes if m.startswith('ref_')])

        for mode in mode_order:
            if mode not in all_modes:
                continue

            mode_label = {
                'uncond': 'Unconditional',
                'fwd': 'Forward (Part1 ref)',
                'rev': 'Reverse (Part2 ref)',
            }.get(mode, mode.replace('ref_', 'Ext: '))

            html += f'<div style="margin:12px 0;padding:8px;background:rgba(0,0,0,0.2);border-radius:6px">\n'
            html += f'<div style="font-size:12px;color:var(--accent);font-weight:600;margin-bottom:8px">{mode_label}</div>\n'
            html += '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:12px">\n'

            # Baseline
            if sample_name in baseline_samples and mode in baseline_samples[sample_name]:
                wav_path = baseline_samples[sample_name][mode]
                rel_path = "baseline_val/epoch_2/" + os.path.basename(wav_path)
                html += f'''<div class="audio-card" style="border-left:3px solid var(--text-dim)">
    <div class="mode" style="color:var(--text-dim)">Baseline (14ep CLAP)</div>
    <audio controls preload="none" src="{rel_path}"></audio>
</div>
'''

            # Diff reward epochs
            for epoch_num in sorted(val_samples.keys()):
                if sample_name in val_samples[epoch_num] and mode in val_samples[epoch_num][sample_name]:
                    wav_path = val_samples[epoch_num][sample_name][mode]
                    rel_path = "val/epoch_" + str(epoch_num) + "/" + os.path.basename(wav_path)
                    ep_label = f"Diff Reward Ep{epoch_num}"
                    if epoch_num == 2:
                        ep_label += " (best flow)"
                    elif epoch_num == 5:
                        ep_label += " (best nat.)"
                    border_color = "var(--green)" if epoch_num == 5 else "var(--accent)"
                    html += f'''<div class="audio-card" style="border-left:3px solid {border_color}">
    <div class="mode" style="color:{border_color}">{ep_label}</div>
    <audio controls preload="none" src="{rel_path}"></audio>
</div>
'''

            html += '</div>\n</div>\n'
        html += '</div>\n'

    # All runs summary table
    html += '''
<h2>All Training Runs Summary</h2>

<table>
<tr>
    <th>Run</th>
    <th>Approach</th>
    <th>Epochs</th>
    <th>Reward Trend</th>
    <th>Best Flow Loss</th>
    <th>Verdict</th>
</tr>
<tr>
    <td>1</td>
    <td>Multi-aux, low caps (ratio=1, cap=2)</td>
    <td>3</td>
    <td><span class="badge badge-red">Flat</span></td>
    <td>0.4315</td>
    <td>Coefficients too weak, never compensated magnitude gap</td>
</tr>
<tr>
    <td>2</td>
    <td>High coefficients (ratio=5, cap=10/50)</td>
    <td>2</td>
    <td><span class="badge badge-red">Flat</span></td>
    <td>0.4474</td>
    <td>Same gradient direction as flow matching &mdash; scaling doesn't help</td>
</tr>
<tr>
    <td>3</td>
    <td>Sigma filter (&lt;0.4) + BS=256</td>
    <td>2</td>
    <td><span class="badge badge-red">Flat</span></td>
    <td>0.4591</td>
    <td>Options A and C identical despite 5x cap difference</td>
</tr>
<tr>
    <td>4</td>
    <td>Rejection sampling (top 50%)</td>
    <td>2</td>
    <td><span class="badge badge-red">Flat</span></td>
    <td>0.4487</td>
    <td>Filters bad examples but provides no directional signal</td>
</tr>
<tr>
    <td>5</td>
    <td>Differentiable rewards (broken CLAP)</td>
    <td>2</td>
    <td><span class="badge badge-red">Flat</span></td>
    <td>0.4696</td>
    <td><code>@torch.no_grad()</code> on mel + wrong WavLM class</td>
</tr>
<tr>
    <td>6</td>
    <td>Diff. rewards + gradient fix (ablation)</td>
    <td>1</td>
    <td><span class="badge badge-yellow">+0.013 nat</span></td>
    <td>0.4613</td>
    <td>First-ever reward improvement, confirms fix works</td>
</tr>
<tr>
    <td>7</td>
    <td>Diff. rewards + gradient fix (full)</td>
    <td>5</td>
    <td><span class="badge badge-green">+0.089 nat</span></td>
    <td>0.4386</td>
    <td>Sustained +33% naturalness improvement</td>
</tr>
</table>

<h2>Key Takeaways</h2>

<div class="card">
<ol style="margin-left:20px;line-height:2">
<li><strong>Reward-weighted reconstruction losses don't work</strong> for steering generation quality. The gradient always points toward x0_clean regardless of the reward value &mdash; changing the weight only scales magnitude, never direction.</li>
<li><strong>Differentiable rewards (ReFL-style) DO work</strong> &mdash; but only when the gradient chain is actually intact. Silent gradient killers like <code>@torch.no_grad()</code> decorators can make a differentiable setup appear to work while providing zero gradient signal.</li>
<li><strong>Always verify gradient flow end-to-end</strong> with a simple test: create an input with <code>requires_grad=True</code>, run the full chain, call <code>.backward()</code>, check that <code>input.grad</code> is not None and has non-zero norm.</li>
<li><strong>Model class matters for HuggingFace checkpoints</strong>. <code>Wav2Vec2ForXVector</code> vs <code>WavLMForXVector</code> have different key prefixes. Wrong class = silently random weights.</li>
<li><strong>Naturalness is the most responsive reward signal</strong>. CLAP text similarity improved +33% over 5 epochs. Speaker similarity was already near ceiling (0.90). Centroid score didn't improve (possibly conflicting objective).</li>
</ol>
</div>

<h2>What To Try Next: Paths to Better Results</h2>

<div class="card">
<h3>1. More Epochs / Longer Training</h3>
<p>The naturalness curve hasn't plateaued after 5 epochs (+0.025 in epoch 5, still improving). Running 10-20 epochs with the same setup is the lowest-risk next step. The cosine LR schedule may need adjustment &mdash; consider a linear decay or cosine with longer warmup to avoid the epoch 3 dip (naturalness dropped from 0.321 to 0.285 during the steep part of cosine decay).</p>
<p style="margin-top:8px"><strong>Specific suggestion:</strong> 15 epochs, linear decay from 4e-5 to 1e-6, warmup 20 steps. Keep BS=256. This gives ~615 optimizer steps &mdash; enough for the naturalness trend to either plateau or show divergence.</p>
</div>

<div class="card">
<h3>2. Hyperparameter Tuning</h3>
<ul style="margin-left:20px;line-height:1.8">
<li><strong>Drop or reduce centroid loss</strong>: The centroid coefficient always hits the cap (10.0) and the centroid score got slightly <em>worse</em> (-0.312 &rarr; -0.325). It may be conflicting with naturalness &mdash; the real/synthetic centroid direction in CLAP space may not align with the naturalness text embedding direction. Try <code>aux_target_ratio=0</code> for centroid (disable it) and redirect that gradient budget to naturalness.</li>
<li><strong>Lower coefficient cap for centroid, raise for naturalness</strong>: Instead of all three at ratio=5/cap=10, try naturalness at ratio=10/cap=20, speaker at ratio=5/cap=10, centroid at ratio=1/cap=3.</li>
<li><strong>Sigma threshold</strong>: Currently &sigma;&lt;0.4, hitting ~23% of micro-batches. Try &sigma;&lt;0.3 (only the cleanest predictions) for higher-quality reward signals, or &sigma;&lt;0.5 to increase the hit rate to ~40%.</li>
<li><strong>Increase max_ref_tokens</strong> to 300 (12 sec) &mdash; longer references give WavLM more signal for speaker similarity. May need to lower <code>tgt_T_frames</code> gate from 250 to 200 to keep VRAM in check.</li>
</ul>
</div>

<div class="card">
<h3>3. Better Reward Models</h3>
<ul style="margin-left:20px;line-height:1.8">
<li><strong>Replace VoiceCLAP-small with a larger CLAP</strong>: VoiceCLAP-small has 124M params. <code>laion/voiceclap-large</code> (300M+) may provide richer gradient signal. Need to verify its <code>compute_log_mel</code> doesn't have the same <code>@torch.no_grad()</code> bug (or apply the same fix).</li>
<li><strong>Use UTMOS or DNSMOS as differentiable quality predictors</strong>: These are direct MOS estimators trained on human ratings. If they're differentiable end-to-end, they provide a more direct "quality" signal than CLAP text similarity.</li>
<li><strong>Train a differentiable quality predictor on actual listening test data</strong>: Take 500-1000 DramaBox samples, collect A/B preference data, train a small CNN on raw waveforms to predict preference. This would give the most task-specific gradient signal.</li>
<li><strong>Use a larger speaker verification model</strong>: WavLM-base-plus-sv is 94M params. <code>microsoft/wavlm-large-sv</code> or a TDNN-based model might capture more speaker nuance. Speaker sim is already at 0.90 though, so this may have diminishing returns.</li>
</ul>
</div>

<div class="card">
<h3>4. Better Data</h3>
<ul style="margin-left:20px;line-height:1.8">
<li><strong>More real speech data</strong>: Currently only 598 Emilia samples (top 10% by DNS MOS). Adding more high-quality real speech (LibriTTS, VCTK, more Emilia) would give the model more diverse real-speech targets to learn from during flow matching, making the aux naturalness gradient more effective.</li>
<li><strong>Filter DramaBox by existing quality scores</strong>: Use the CLAP naturalness score to select the top 50% of DramaBox samples for training. Training on better baseline audio means the flow matching loss itself pushes toward more natural audio.</li>
<li><strong>Curate voice reference diversity</strong>: The speaker similarity loss is only useful if references span diverse voice types. If references cluster narrowly, the speaker sim gradient is trivially satisfied.</li>
</ul>
</div>

<div class="card">
<h3>5. Alternative Loss Architectures</h3>
<ul style="margin-left:20px;line-height:1.8">
<li><strong>DPO / RLHF-style training</strong>: Instead of continuous reward maximization, generate paired samples (with and without LoRA changes), score them, and train with a preference objective. This is more stable than reward maximization and avoids reward hacking.</li>
<li><strong>Latent-space reward proxy</strong>: Train a small MLP to predict naturalness/quality directly from x0 latent tokens (no decode needed). This is 100% differentiable by construction, ~100x cheaper per step than decoding through the vocoder, and avoids the CLAP gradient chain entirely. Requires pre-computing (x0_latent, naturalness_score) pairs from the current model.</li>
<li><strong>Multi-step reward</strong>: Instead of scoring a single-step x0 prediction, run a short multi-step denoising (e.g., 4 Euler steps from &sigma;=0.3 to 0) and score the final result. This gives a more accurate x0 estimate at the cost of ~4x compute per aux evaluation.</li>
<li><strong>Adversarial loss</strong>: Train a small discriminator on real vs synthetic CLAP embeddings (or raw waveforms) and use it as a differentiable loss. This is essentially a learned reward function that adapts as the generator improves.</li>
</ul>
</div>

<div class="card">
<h3>6. Addressing the Speaker Similarity Ceiling</h3>
<p>Speaker sim is already at 0.90 and didn't improve with differentiable rewards. This may be because:</p>
<ul style="margin-left:20px;line-height:1.8">
<li>WavLM-SV operates on 16kHz mono audio, losing high-frequency detail that distinguishes voices</li>
<li>The reference is decoded from latent (lossy), not the original wav &mdash; the "target" speaker identity is already degraded</li>
<li>0.90 may be near the achievable ceiling for this model architecture</li>
</ul>
<p style="margin-top:8px"><strong>To push higher:</strong> Use the original raw reference audio (pre-latent-encoding) as the speaker identity target instead of decoding ref_latent. This removes one decode-encode cycle of degradation. Alternatively, compute speaker embeddings in latent space directly (train a small projection head).</p>
</div>

<div class="card">
<h3>7. Addressing the Centroid Score</h3>
<p>The centroid score (cos_real - cos_synth) didn't improve and the coefficient always hit the cap (10.0). Analysis:</p>
<ul style="margin-left:20px;line-height:1.8">
<li>The CLAP real/synthetic centroid direction may be orthogonal to the naturalness text direction &mdash; pushing toward "sounds more like Emilia" doesn't mean "sounds more natural"</li>
<li>The centroid captures dataset-level distribution properties (speaker diversity, recording conditions, accent distribution) rather than per-sample quality</li>
<li>The centroid embeddings were computed from 2597 training samples &mdash; they may be biased by the specific speakers/conditions in the training set</li>
</ul>
<p style="margin-top:8px"><strong>Recommendation:</strong> Drop the centroid loss entirely. Replace it with a direct differentiable quality estimator (UTMOS/DNSMOS) or simply increase the weight on CLAP naturalness.</p>
</div>

<footer style="margin-top:48px;padding-top:16px;border-top:1px solid var(--border);color:var(--text-dim);font-size:12px">
    DramaBox LoRA Fine-Tuning &mdash; Multi-Auxiliary Differentiable Reward Training
    <br>Infrastructure: 8x NVIDIA A100 80GB | LTX-2.3 22B transformer + 128-rank LoRA
    <br>Models: VoiceCLAP-small (laion), WavLM-base-plus-sv (Microsoft), Quality MLP (custom)
    <br>Dataset: 3,845 samples (DramaBox + Emilia top 10%)
</footer>

</body>
</html>'''

    return html


def main():
    base_dir = "/home/deployer/laion/Voice-Acting-Pipeline"
    output_dir = os.path.join(base_dir, "finetune_output/diff_reward_5ep")
    grid_dir = os.path.join(base_dir, "validation_grid")

    # Load metrics
    metrics_path = os.path.join(output_dir, "metrics.jsonl")
    metrics = load_metrics(metrics_path)

    # Validation directories for epochs 2 and 5
    val_dirs = {
        2: os.path.join(output_dir, "val"),
        5: os.path.join(output_dir, "val"),
    }

    html = build_html(output_dir, metrics, val_dirs)

    out_path = os.path.join(grid_dir, "index.html")
    with open(out_path, "w") as f:
        f.write(html)
    print(f"Written: {out_path}")
    print(f"Size: {os.path.getsize(out_path)} bytes")


if __name__ == "__main__":
    main()
