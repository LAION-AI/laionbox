# LaionBox

**LaionBox** is a fine-tuned version of [DramaBox](https://huggingface.co/ResembleAI/Dramabox) (3,3B parameter DiT-based TTS) that produces more natural, emotionally expressive speech with improved voice cloning fidelity.

📄 **[Technical Report](https://laion-ai.github.io/laionbox/)** | 🤗 **[Model on HuggingFace](https://huggingface.co/laion/laionbox-v0.2-wip)** | 🎧 **[Audio Samples](https://laion-ai.github.io/laionbox/)**

## Overview

LaionBox fine-tunes the DramaBox flow-matching transformer using LoRA (rank=128) with **6 differentiable auxiliary losses** that push generated audio toward higher naturalness, quality, and voice cloning fidelity:

1. **CLAP Naturalness** — Maximizes perceptual naturalness via VoiceCLAP text similarity
2. **Quality MLP** — Binary classifier trained to distinguish real from synthetic audio
3. **Centroid Real/Fake** — Distribution matching toward real speech embeddings
4. **Speaker Similarity** — WavLM-SV voice identity preservation
5. **Comb Filter Detector** — Latent-space CNN detecting interference artifacts
6. **Artifact Detector V2** — Residual CNN for general artifact detection

### Key Innovation: Differentiable Reward Chain

The critical breakthrough is a **fully differentiable gradient path** from perceptual quality metrics back to LoRA parameters:

```
LoRA → velocity prediction → x₀ recovery → VAE decoder → waveform → CLAP → loss
```

Non-differentiable reward weighting (scaling loss magnitude) was shown to be ineffective for flow-matching models — it produces the same gradient direction regardless of reward signal.

## Results

| Metric | Vanilla DramaBox | v0.1-wip | v0.2-wip (6-aux) |
|--------|:---:|:---:|:---:|
| CLAP Naturalness | baseline | +5% | +11% |
| Quality Probability | ~0.85 | ~0.89 | 0.99+ |
| Speaker Similarity | baseline | +1% | +2% |
| Comb Artifacts | baseline | reduced | minimal |

## Repository Structure

```
laionbox/
├── README.md
├── docs/                    # Technical report (GitHub Pages)
│   └── index.html
├── scripts/
│   ├── dramabox_finetune_train_multi_aux.py  # Main 6-aux training
│   ├── dramabox_finetune_train.py            # Base training
│   ├── dramabox_finetune_prepare.py          # Data preparation
│   ├── train_artifact_detector_v2.py         # Artifact detector
│   ├── train_comb_filter_detector.py         # Comb filter detector
│   ├── train_binary_classifiers.py           # Quality classifiers
│   ├── fair_eval_worker.py                   # Evaluation worker
│   └── fair_eval_prompts.json                # Eval prompts
├── configs/
│   └── *.yaml                                # Training configurations
└── discriminators/
    ├── quality_classifier.pt
    ├── real_fake_classifier.pt
    ├── best_artifact_detector_v2.pt
    ├── best_comb_detector.pt
    └── best_clap_medium.pt
```

## Quick Start

### Prerequisites
- DramaBox base model: `ResembleAI/Dramabox`
- LaionBox LoRA weights: `laion/laionbox-v0.2-wip` (on HuggingFace)
- 8× A100 80GB (for training) or 1× A100 (for inference)

### Inference
```bash
# Download base model + LoRA
huggingface-cli download ResembleAI/Dramabox
huggingface-cli download laion/laionbox-v0.2-wip

# Run inference server
python scripts/dramabox_server/server.py
```

### Training
```bash
# 1. Prepare training data
python scripts/dramabox_finetune_prepare.py --config configs/finetune_6aux_all_disc.yaml

# 2. Train discriminators (optional, pre-trained ones included)
python scripts/train_binary_classifiers.py --data-dir ./finetune_data
python scripts/train_artifact_detector_v2.py --data-dir ./finetune_data
python scripts/train_comb_filter_detector.py --data-dir ./finetune_data

# 3. Fine-tune with 6 auxiliary losses
accelerate launch --num_processes 8 scripts/dramabox_finetune_train_multi_aux.py \
    --config configs/finetune_6aux_all_disc.yaml
```

## Known Issues

- **Subtle metallic artifacts** in LoRA-modified outputs — caused by BigVGAN v2 vocoder processing slightly out-of-distribution mel spectrograms
- **Not from stereo interference** — L-R correlation >0.999, artifacts are per-channel
- See the [technical report](https://laion-ai.github.io/laionbox/) for detailed analysis

## Future Work

- Fine-tune BigVGAN v2 vocoder on LoRA-modified mels
- GAN-based decoder training (vs L1-only)
- Block-wise streaming for real-time inference
- 8-step distilled diffusion (60-70% compute savings)
- More diverse training data (characters, emotions, languages)

## License

Apache 2.0

## Citation

```bibtex
@misc{laionbox2026,
  title={LaionBox: Fine-tuning DramaBox TTS with Multi-Auxiliary Differentiable Losses},
  author={LAION},
  year={2026},
  url={https://github.com/LAION-AI/laionbox}
}
```
