"""Model adapter for OpenCLIP dual encoders."""

from __future__ import annotations

import os

from oellm.core.base_model_adapter import BaseModelAdapter


class OpenClipAdapter(BaseModelAdapter):
    """Resolves a model path into an OpenCLIP model specifier.

    OpenCLIP loads Hub checkpoints through an ``hf-hub:`` prefix and local
    checkpoints through a filesystem path. A bare repo id such as
    ``laion/CLIP-ViT-B-32-laion2B-s34B-b79K`` is therefore prefixed, while an
    existing directory is passed through untouched.
    """

    def __init__(self, model_path: str) -> None:
        self._path = model_path

    @property
    def model_path(self) -> str:
        return self._path

    def to_open_clip_spec(self) -> str:
        if os.path.isdir(self._path) or self._path.startswith("hf-hub:"):
            return self._path
        return f"hf-hub:{self._path}"

    def to_lm_eval_args(self) -> str:
        # Present only to satisfy the adapter interface: this suite never runs
        # through lm-eval, because scoring is embedding similarity rather than
        # token likelihood.
        return f"pretrained={self._path}"

    def to_lmms_eval_args(self) -> str:
        return f"pretrained={self._path}"

    def to_contrib_flags(self) -> str | None:
        # One backend (open_clip) serves every checkpoint, so there is no
        # model-type routing to do.
        return None
