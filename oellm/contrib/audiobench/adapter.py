"""AudioBench model adapter.

Maps a HuggingFace model path to AudioBench's literal ``--model_name`` value.

AudioBench's ``Model`` class (in ``$AUDIOBENCH_DIR/src/model.py``) dispatches
on **exact-string** match against a fixed list — there is no family-level
indirection and no fallback.  Each supported model has a hardcoded loader
under ``model_src/`` that loads its own HF repo internally; AudioBench
**cannot evaluate arbitrary HF checkpoints**, only the variants it knows
about.  If we can't map the user's ``model_path`` to one of those literals,
we return ``None`` and ``suite.run`` raises a clear error.
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
        return None


def detect_audiobench_model_type(model_path: str) -> str | None:
    """Convenience wrapper around :meth:`AudioBenchModelAdapter.to_contrib_flags`."""
    return AudioBenchModelAdapter(model_path).to_contrib_flags()
