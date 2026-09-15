"""Shared constants and data structures used across the oellm package."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvaluationJob:
    model_path: Path | str
    task_path: str
    n_shot: int
    eval_suite: str


# Mapping of model path patterns to lmms-eval adapter class names.
# Patterns are matched case-insensitively against the model path.
# Order matters: more specific patterns must come before general ones.
# Adapter names must exist in the target venv's lmms-eval; pre-flight checks that.
LMMS_MODEL_ADAPTERS: list[tuple[list[str], str]] = [
    # ── Audio / speech models (must come before generic "qwen" catch-all) ──
    (["qwen2-audio", "qwen2_audio"], "qwen2_audio"),
    (["qwen2.5-omni", "qwen2_5_omni"], "qwen2_5_omni"),
    (["video-salmonn", "video_salmonn"], "video_salmonn_2"),
    (["audio-flamingo-3", "audio_flamingo_3"], "audio_flamingo_3"),
    (["kimi-audio", "kimi_audio"], "kimi_audio"),
    (["whisper"], "whisper"),
    (["phi-4-multimodal", "phi_4_multimodal", "phi4-multimodal"], "phi4_multimodal"),
    (["gemini"], "gemini_api"),
    (["gpt4o-audio", "gpt_4o_audio"], "gpt4o_audio"),
    # ── Vision / video models ──
    (["qwen2.5-vl", "qwen2_5_vl", "qwen2.5vl"], "qwen2_5_vl"),
    (["qwen2-vl", "qwen2_vl"], "qwen2_vl"),
    (["llava-hf"], "llava_hf"),
    (["llava-onevision", "llava_onevision"], "llava_onevision"),
    (["llava-vid", "llava_vid", "llava-video"], "llava_vid"),
    (["video-llava", "video_llava"], "video_llava"),
    (["llava"], "llava_hf"),
    (["internvideo"], "internvideo2"),
    (["internvl"], "internvl2"),
    # No idefics3/SmolVLM entry: the pinned lmms-eval has no adapter that can
    # load them with transformers<4.50 (its generic one loads a headless model).
    (["idefics2"], "idefics2"),
    (["minicpm"], "minicpm_v"),
    (["longva"], "longva"),
    (["videochat2"], "videochat2"),
    (["qwen"], "qwen_vl"),
]

# Fallback metric keys used by collect_results when no task_metrics entry
# is found. Tried in order; first match wins.
METRIC_FALLBACK_KEYS: list[str] = [
    "acc,none",
    "acc",
    "accuracy",
    "acc_norm",
    "f1",
    "exact_match",
    "chrf++",
    "bleu",
]

# Tasks whose primary metric is computed by an LLM judge / extractor and
# requires ``OPENAI_API_KEY`` (or compatible) at evaluation time. Pre-flight
# check refuses to schedule these without the key unless
# ``--allow-missing-judge`` is passed.
#
# NOT included: mmbench_en_dev (regex extraction primary; GPT fallback) and
# mathvista_testmini_* (quick_extract regex primary; GPT fallback) — these
# emit valid numbers without a key, so blocking them would cause friction.
# Add tasks here when the metric is genuinely undefined without a judge call.
JUDGE_REQUIRED_TASKS: frozenset[str] = frozenset(
    {
        # Video
        "activitynetqa",
        # Audio — AudioBench / VoiceBench judge-graded tasks (`gpt_eval` or
        # `llm_as_judge_eval` is the only metric path; no regex fallback).
        "alpaca_audio",
        "openhermes",
        "wavcaps",
        "air_bench_chat_sound",
        "air_bench_chat_music",
        "air_bench_chat_speech",
        "air_bench_chat_mixed",
        "voicebench_commoneval",
    }
)


def detect_lmms_model_type(model_path: str) -> str:
    """Detect the lmms-eval adapter class name from a model path or HF repo name.

    lmms-eval requires --model <adapter_class> (e.g. llava_hf, qwen2_5_vl).
    This is inferred from the model name so users never need to set it manually.

    To add support for a new model family, either add an entry to
    LMMS_MODEL_ADAPTERS above, or export ``LMMS_MODEL_ADAPTERS`` (same shape)
    from a contrib suite module — contrib entries are consulted FIRST, so
    plugins can route new families without touching core files.
    """
    name = str(model_path).lower()

    # Contrib-registered patterns take precedence. Lazy import: the registry
    # walks contrib packages; constants must stay import-light.
    from oellm.registry import get_lmms_adapter_overrides  # noqa: PLC0415

    for patterns, adapter in get_lmms_adapter_overrides():
        if any(p in name for p in patterns):
            return adapter

    for patterns, adapter in LMMS_MODEL_ADAPTERS:
        if any(p in name for p in patterns):
            return adapter
    raise ValueError(
        f"Cannot auto-detect lmms-eval adapter class from model path '{model_path}'. "
        "Add a pattern to LMMS_MODEL_ADAPTERS in constants.py or export "
        "LMMS_MODEL_ADAPTERS from a contrib suite module."
    )
