"""Model adapter for DocVQA 2026."""

from __future__ import annotations

from oellm.core.base_model_adapter import BaseModelAdapter


class DocVQA2026Adapter(BaseModelAdapter):
    def __init__(self, model_path: str) -> None:
        self._model_path = str(model_path)

    @property
    def model_path(self) -> str:
        return self._model_path

    def to_lm_eval_args(self) -> str:
        return f"pretrained={self._model_path}"

    def to_lmms_eval_args(self) -> str:
        return f"pretrained={self._model_path}"

    def to_contrib_flags(self) -> str | None:
        """No flags: one transformers path serves every supported checkpoint."""
        return None
