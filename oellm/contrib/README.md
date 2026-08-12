# Contrib Benchmark Registry

Community-contributed benchmarks integrated into the ELLIOT evaluation platform. Each benchmark runs as a self-contained plugin -- no changes to core scheduling code required.

To add your own benchmark, see the [Contributing Guide](CONTRIBUTING.md).

## Benchmarks

| Benchmark | Task Group | Description | Paper | Code |
|---|---|---|---|---|
| RegionDial-Bench | `regiondial-bench` | Multi-round region grounding and segmentation on RefCOCOg and RefCOCO+. Evaluates robustness to error accumulation across dialogue turns. | [arXiv:2602.03733](https://arxiv.org/abs/2602.03733) | [lmsdss/RegionReasoner](https://github.com/lmsdss/RegionReasoner) |
| AudioBench | `audio-audiobench` (+ `-asr` / `-st` / `-reasoning`) | 27 judge-free audio tasks — ASR (WER), speech translation (BLEU), spoken reasoning, AudioCaps captioning — scored with AudioBench's own normalisers for paper-comparable numbers. | [arXiv:2406.16020](https://arxiv.org/abs/2406.16020) | [AudioLLMs/AudioBench](https://github.com/AudioLLMs/AudioBench) |

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
