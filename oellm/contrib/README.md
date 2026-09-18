# Contrib Benchmark Registry

Community-contributed benchmarks integrated into the ELLIOT evaluation platform. Each benchmark runs as a self-contained plugin -- no changes to core scheduling code required.

To add your own benchmark, see the [Contributing Guide](CONTRIBUTING.md).

## Benchmarks

| Benchmark | Task Group | Description | Paper | Code |
|---|---|---|---|---|
| RegionDial-Bench | `regiondial-bench` | Multi-round region grounding and segmentation on RefCOCOg and RefCOCO+. Evaluates robustness to error accumulation across dialogue turns. | [arXiv:2602.03733](https://arxiv.org/abs/2602.03733) | [lmsdss/RegionReasoner](https://github.com/lmsdss/RegionReasoner) |
| AudioBench | `audio-audiobench` (+ `-asr` / `-st` / `-reasoning`) | 27 judge-free audio tasks — ASR (WER), speech translation (BLEU), spoken reasoning, AudioCaps captioning — scored with AudioBench's own normalisers for paper-comparable numbers. | [arXiv:2406.16020](https://arxiv.org/abs/2406.16020) | [AudioLLMs/AudioBench](https://github.com/AudioLLMs/AudioBench) |
| Spurious robustness | `spurious-robustness` (+ `-imagenet` / `-celeba`) | Zero-shot robustness to spurious correlations for OpenCLIP dual encoders: ImageNet (no spurious attribute, top-1) and CelebA (gender, 4 groups, worst-group). | [ACM MM 2026](https://github.com/gsarridis/vlm-spurious-robustness) | [gsarridis/vlm-spurious-robustness](https://github.com/gsarridis/vlm-spurious-robustness) |
| UrbanCars | `spurious-urbancars` | Two-attribute spurious robustness (background + co-occurring object, 8 groups, worst-group). Separate suite so its required data directory is checked before submission. | [ACM MM 2026](https://github.com/gsarridis/vlm-spurious-robustness) | [gsarridis/vlm-spurious-robustness](https://github.com/gsarridis/vlm-spurious-robustness) |

### RegionDial-Bench

**Metrics:** gIoU (primary), cIoU, bbox_AP, pass_rate@0.3/0.5/0.7/0.9, per-round R1–R7

```bash
oellm-eval schedule \
  --models lmsdss/RegionReasoner-7B \
  --task-groups regiondial-bench \
  --venv-path ~/regiondial-venv
```

Requires cluster-specific setup (`REGION_REASONER_DIR`, a dedicated venv, ~30 GB of HF cache). See the full [RegionDial-Bench README](regiondial_bench/README.md) for prerequisites and configuration.

### AudioBench

**Metrics:** `wer` (ASR), `bleu` (speech translation), `accuracy` / `string_match` (reasoning), `meteor` (captioning)

```bash
oellm-eval schedule \
  --models Qwen/Qwen2-Audio-7B-Instruct \
  --task-groups audio-audiobench \
  --venv-path ~/audiobench-venv
```

Requires cluster-specific setup (`AUDIOBENCH_DIR` pointing at an AudioBench clone, a dedicated venv). Only the model families AudioBench itself supports can be evaluated (Qwen2-Audio, SALMONN, Whisper, …). See the full [AudioBench README](audiobench/README.md) for prerequisites and the supported-model table.

### Spurious robustness

**Metrics:** `worst_group_accuracy` (CelebA, UrbanCars), `top1_accuracy` (ImageNet), plus `avg_accuracy`, `accuracy_gap` and per-group accuracies

```bash
oellm-eval schedule \
  --models laion/CLIP-ViT-B-32-laion2B-s34B-b79K \
  --task-groups spurious-robustness,spurious-urbancars \
  --venv-path ~/spurious-venv
```

Evaluates **OpenCLIP-family dual encoders**, not generative VLMs: an image is assigned to the class whose text embedding is nearest. Verified metric-for-metric against the reference implementation.

ImageNet and CelebA are staged automatically from the Hub. ImageNet is gated and CelebA is non-commercial research only — both accepted per researcher under their own account.

UrbanCars ships as the separate `spurious_urbancars` suite because it is the only one needing a cluster-local tree: `CLUSTER_ENV_VARS` applies to every task in a suite, so bundling it would fail ImageNet and CelebA wherever no UrbanCars data exists. Split out, `URBANCARS_DATA_DIR` is validated by the login-node pre-flight before submission. It has no Hub source — see the [UrbanCars README](spurious_urbancars/README.md) for why it must be built.

See the full [spurious-robustness README](spurious_robustness/README.md) for group definitions, the prompting policy, and how to read a worst-group score.
