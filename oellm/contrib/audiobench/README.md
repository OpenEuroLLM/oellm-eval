# AudioBench

AudioBench (AudioLLMs/AudioBench, [arXiv 2406.16020](https://arxiv.org/abs/2406.16020))
is a broad audio-understanding benchmark covering ASR, speech translation,
spoken reasoning, audio scene QA, and paralinguistics. This contrib plugin
wraps AudioBench as a callable `audiobench` suite inside elliot-cli so WP4
can produce numbers directly comparable with the AudioBench paper and
leaderboard, without the scoring-normalisation drift that would come from
running the same datasets through lmms-eval.

## Scope

**27 judge-free tasks** across ASR (WER), speech translation (BLEU), spoken
reasoning (accuracy / string_match), and AudioCaps (METEOR). Of these:

- **20 tasks are genuinely new** to the platform — not in any of our
  existing lmms-eval `audio-*` groups. Examples: `earnings21_test`,
  `earnings22_test`, GigaSpeech2 (Thai / Indonesian / Vietnamese),
  SEAME code-switch, Spoken-MQA reasoning splits, MMAU mini.
- **7 tasks are dual-registered** duplicates of benchmarks we already run
  through lmms-eval (LibriSpeech test-clean/other, Common Voice 15 EN,
  GigaSpeech, People's Speech, TED-LIUM 3, CoVoST2 en→zh). These use
  AudioBench's own scorer and normaliser so WP4 can report numbers
  aligned with the AudioBench paper.

Every AudioBench task is namespaced with an `audiobench_` prefix so the CSV
`task_path` column unambiguously identifies which scorer produced a number
(e.g. `audiobench_librispeech_test_clean` is AudioBench-scored;
`librispeech_test_clean` remains the lmms-eval version).

Judge-dependent tasks (SLUE-SQA5, Spoken-SQuAD, AudioCaps-QA, IEMOCAP /
MELD / VoxCeleb probes, AudioLLM-InstructionFollowing) are not included
and depend on a vLLM judge service being provisioned on Leonardo.

## Prerequisites

AudioBench is not pip-installable (no upstream build backend, bare imports
in `src/main_evaluate.py`); the plugin invokes it as a subprocess from an
on-cluster clone. A dedicated venv is required: the `[audiobench]` extra
pins `transformers<5` and `jiwer<3`, which conflict with the general eval
venv (see [`docs/VENV.md`](../../../docs/VENV.md) for the framework venvs).

### 1. Clone AudioBench and configure `clusters.yaml`

```bash
git clone https://github.com/AudioLLMs/AudioBench /path/to/AudioBench
```

Add `AUDIOBENCH_DIR` to your cluster block in
`oellm/resources/clusters.yaml`:

```yaml
leonardo:
  ...
  AUDIOBENCH_DIR: "/path/to/AudioBench"
```

### 2. Create the venv

```bash
uv venv --python 3.12 audiobench-venv
source audiobench-venv/bin/activate
uv pip install -e ".[audiobench]"
```

The `[audiobench]` extra pins `transformers>=4.45,<5`, `jiwer<3`,
`sacrebleu`, `pythainlp`, `evaluate`, `soundfile`, `librosa`.

### 3. Install AudioBench's runtime dependencies

```bash
# AudioBench's own requirements (filter vllm; only used by deferred judge tasks)
grep -v -i '^vllm' /path/to/AudioBench/requirements.txt > /tmp/ab-reqs.txt
uv pip install -r /tmp/ab-reqs.txt

# PyTorch for cluster's CUDA driver — PyPI defaults target a newer runtime
# than most HPC drivers (Leonardo / JURECA report CUDA 12.2) and crash with
# `NVIDIA driver too old`.  Use the cu121 index.
uv pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

# rapidfuzz C extension — without this, jiwer's WER scoring hits the
# pure-Python fallback and raises NotImplementedError on Levenshtein.editops.
uv pip install --reinstall rapidfuzz
```

> Verify the venv works:
> ```bash
> python -c "
> from transformers import Qwen2AudioForConditionalGeneration
> from rapidfuzz.distance import Levenshtein
> Levenshtein.editops('a', 'b')   # must not raise
> print('audiobench venv OK')
> "
> ```

### Dataset pre-download

No manual steps required. `schedule` pre-downloads every
`AudioLLMs/*` HF repo referenced by the requested task group on the login
node via `huggingface_hub.snapshot_download(max_workers=2)`, so compute
nodes do not need internet access.

## Running

### Available task groups

| Task group                       | Leaves | What it covers                                                  |
|----------------------------------|--------|-----------------------------------------------------------------|
| `audio-audiobench`               | 27     | Full suite (everything below).                                  |
| `audio-audiobench-asr`           | 15     | WER tasks — 9 new + 6 dual-registered with lmms-eval.           |
| `audio-audiobench-st`            | 6      | BLEU speech-translation — 5 new + 1 dual (en→zh).               |
| `audio-audiobench-reasoning`     | 6      | Spoken-MQA × 4, MMAU mini, AudioCaps METEOR.                    |

### Example

```bash
# Full AudioBench suite on a Qwen2-Audio model:
oellm-eval schedule \
    --models Qwen/Qwen2-Audio-7B-Instruct \
    --task-groups audio-audiobench \
    --venv-path audiobench-venv

# ASR only:
oellm-eval schedule \
    --models Qwen/Qwen2-Audio-7B-Instruct \
    --task-groups audio-audiobench-asr \
    --venv-path audiobench-venv

# Smoke test with --limit:
oellm-eval schedule \
    --models Qwen/Qwen2-Audio-7B-Instruct \
    --task-groups audio-audiobench-asr \
    --limit 100 \
    --venv-path audiobench-venv
```

`--limit N` is forwarded to AudioBench's `--number_of_samples N`. When
unset, the full test split is evaluated.

### Collecting results

```bash
oellm-eval collect /path/to/evals --output-csv audiobench_results.csv
```

The primary metric per task is what's registered in `task_metrics`
(`wer` / `bleu` / `accuracy` / `string_match` / `meteor`). Dual-registered
tasks land in the CSV **alongside** their lmms-eval counterparts, with
different `task_path` values (`audiobench_librispeech_test_clean` vs
`librispeech_test_clean`) and different `eval_suite` values (`audiobench`
vs `lmms_eval`) — no silent averaging.

## Supported model adapters

AudioBench dispatches on a fixed list of literal `model_name` strings
(see `$AUDIOBENCH_DIR/src/model.py`); each loader under `model_src/`
fetches its own HF repo. Arbitrary HF checkpoints are not supported —
only the variants below:

| Model path substring (lowered)                 | AudioBench `model_name` (literal)         |
|------------------------------------------------|-------------------------------------------|
| `qwen2-audio-7b-instruct` / `qwen2_audio_7b_instruct` | `Qwen2-Audio-7B-Instruct`          |
| `qwen-audio-chat` / `qwen_audio_chat`          | `Qwen-Audio-Chat`                         |
| `salmonn`                                      | `SALMONN_7B`                              |
| `meralion-audiollm` / `meralion_audiollm`      | `MERaLiON-AudioLLM-Whisper-SEA-LION`      |
| `whisper-large-v3` / `whisper_large_v3`        | `whisper_large_v3`                        |
| `whisper-large-v2` / `whisper_large_v2`        | `whisper_large_v2`                        |
| `phi-4-multimodal` / `phi_4_multimodal`        | `phi_4_multimodal_instruct`               |
| `seallms-audio-7b` / `seallms_audio_7b`        | `seallms_audio_7b`                        |
| `wavllm`                                       | `WavLLM_fairseq`                          |
| (anything else)                                | error — no generic loader upstream        |

To override detection, pass the literal AudioBench key as a suffix:
`audiobench:Qwen2-Audio-7B-Instruct`. Case is preserved end-to-end
(AudioBench's match is case-sensitive).

