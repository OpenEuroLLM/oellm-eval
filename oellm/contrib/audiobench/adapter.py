"""AudioBench model adapter: maps a model path to AudioBench's ``--model_name``.

For families in :data:`CHECKPOINT_VARIABLE`, ``launch.py`` loads the given
checkpoint instead of the stock weights; the others run their stock model only.
"""

from __future__ import annotations

from oellm.core.base_model_adapter import BaseModelAdapter

# (audiobench_model_name, substrings_to_match_in_lower(model_path)).
# Order matters — first match wins; put more-specific patterns first.
# Keys MUST be the exact literals AudioBench's model.py dispatch expects.
_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("Qwen2-Audio-7B-Instruct", ("qwen2-audio-7b-instruct", "qwen2_audio_7b_instruct")),
    ("Qwen-Audio-Chat", ("qwen-audio-chat", "qwen_audio_chat")),
    ("SALMONN_7B", ("salmonn",)),
    ("MERaLiON-AudioLLM-Whisper-SEA-LION", ("meralion-audiollm", "meralion_audiollm")),
    ("whisper_large_v3", ("whisper-large-v3", "whisper_large_v3")),
    ("whisper_large_v2", ("whisper-large-v2", "whisper_large_v2")),
    ("phi_4_multimodal_instruct", ("phi-4-multimodal", "phi_4_multimodal")),
    ("seallms_audio_7b", ("seallms-audio-7b", "seallms_audio_7b")),
    ("WavLLM_fairseq", ("wavllm",)),
]


class AudioBenchModelAdapter(BaseModelAdapter):
    """Adapter resolving the ``--model_name`` value for the AudioBench subprocess."""

    def __init__(self, model_path: str) -> None:
        self._path = model_path

    @property
    def model_path(self) -> str:
        return self._path

    def to_lm_eval_args(self) -> str:
        # Unused — AudioBench doesn't route through lm-eval.  Required by
        # BaseModelAdapter.
        return f"pretrained={self._path},trust_remote_code=True"

    def to_lmms_eval_args(self) -> str:
        # Unused — see to_lm_eval_args().
        return f"pretrained={self._path}"

    def to_contrib_flags(self) -> str | None:
        """Return AudioBench's ``model_name`` dispatch key, or ``None`` if no match.

        Returning ``None`` is intentional: AudioBench has no generic loader,
        so an unmatched model path must fail loudly rather than fall through
        to a fictitious ``generic`` key that AudioBench doesn't recognize.
        """
        lowered = self._path.lower()
        for key, needles in _PATTERNS:
            if any(n in lowered for n in needles):
                return key
        return _family_from_config(self._path)


def detect_audiobench_model_type(model_path: str) -> str | None:
    """Convenience wrapper around :meth:`AudioBenchModelAdapter.to_contrib_flags`."""
    return AudioBenchModelAdapter(model_path).to_contrib_flags()


# Family -> (model_src module, variable holding its weights location).
# SeaLLMs-Audio, SALMONN and WavLLM hard-code theirs: stock models only.
CHECKPOINT_VARIABLE: dict[str, tuple[str, str]] = {
    "Qwen2-Audio-7B-Instruct": ("qwen2_audio_7b_instruct", "model_path"),
    "Qwen-Audio-Chat": ("qwen_audio_chat", "model_path"),
    "MERaLiON-AudioLLM-Whisper-SEA-LION": (
        "meralion_audiollm_whisper_sea_lion",
        "repo_id",
    ),
    "whisper_large_v3": ("whisper_large_v3", "whisper_model_path"),
    "whisper_large_v2": ("whisper_large_v2", "whisper_model_path"),
    "phi_4_multimodal_instruct": ("phi_4_multimodal_instruct", "model_path"),
}

# config.json architecture -> family, for checkpoints whose path doesn't name it.
_ARCHITECTURES: dict[str, str] = {
    "Qwen2AudioForConditionalGeneration": "Qwen2-Audio-7B-Instruct",
    "MERaLiONForConditionalGeneration": "MERaLiON-AudioLLM-Whisper-SEA-LION",
    "WhisperForConditionalGeneration": "whisper_large_v3",  # same loader as v2
    "Phi4MMForCausalLM": "phi_4_multimodal_instruct",
}


def _family_from_config(model_path: str) -> str | None:
    import json
    from pathlib import Path

    config = Path(model_path).expanduser() / "config.json"
    try:
        architectures = json.loads(config.read_text()).get("architectures") or []
    except (OSError, ValueError, AttributeError):
        return None
    return next((_ARCHITECTURES[a] for a in architectures if a in _ARCHITECTURES), None)


# Name (last path part, "-" = "_") of each family's stock model.
_STOCK_NAMES: dict[str, tuple[str, ...]] = {
    "Qwen2-Audio-7B-Instruct": ("qwen2-audio-7b-instruct",),
    "Qwen-Audio-Chat": ("qwen-audio-chat",),
    "SALMONN_7B": ("salmonn", "salmonn-7b"),
    "MERaLiON-AudioLLM-Whisper-SEA-LION": ("meralion-audiollm-whisper-sea-lion",),
    "whisper_large_v3": ("whisper-large-v3",),
    "whisper_large_v2": ("whisper-large-v2",),
    "phi_4_multimodal_instruct": ("phi-4-multimodal-instruct",),
    "seallms_audio_7b": ("seallms-audio-7b",),
    "WavLLM_fairseq": ("wavllm", "wavllm-fairseq"),
}


def is_stock_model(model_path: str, key: str) -> bool:
    """*model_path* names the stock model, not a local checkpoint or a fine-tune."""
    from pathlib import Path

    if model_path.startswith(("/", "~", ".")) or Path(model_path).expanduser().exists():
        return False
    name = model_path.rstrip("/").rsplit("/", 1)[-1].lower().replace("_", "-")
    stock = {key.lower().replace("_", "-"), *_STOCK_NAMES.get(key, ())}
    return name in stock


def stock_only_message(model_path: str, key: str) -> str:
    return (
        f"AudioBench cannot load {model_path!r} with its {key} loader: that "
        f"loader only runs the stock model (its name is {_STOCK_NAMES[key][0]}). "
        f"Checkpoints work for: {', '.join(CHECKPOINT_VARIABLE)}."
    )
