from abc import ABC, abstractmethod
from pathlib import Path


class BaseModelAdapter(ABC):
    """Abstract base class for model adapters.

    Translates a model path/config into engine-specific argument strings
    passed on the command line. The built-in execution path consumes these
    through :class:`DefaultHFAdapter` (rendered into ``template.sbatch`` by
    the scheduler); contrib suites consume :meth:`to_contrib_flags` via their
    ``detect_model_flags()``.

    Example::

        class HFModelAdapter(BaseModelAdapter):
            def __init__(self, path: str, trust_remote_code: bool = True):
                self._path = path
                self._trust = trust_remote_code

            @property
            def model_path(self) -> str:
                return self._path

            def to_lm_eval_args(self) -> str:
                return f"pretrained={self._path},trust_remote_code={self._trust}"

            def to_lmms_eval_args(self) -> str:
                return f"pretrained={self._path}"
    """

    @property
    @abstractmethod
    def model_path(self) -> str | Path:
        """Path to the model weights or HuggingFace repo ID."""

    @abstractmethod
    def to_lm_eval_args(self) -> str:
        """Return the ``--model_args`` string for lm-eval-harness.

        Example: ``"pretrained=/path/to/model,trust_remote_code=True"``
        """

    @abstractmethod
    def to_lmms_eval_args(self) -> str:
        """Return the ``--model_args`` string for lmms-eval.

        Example: ``"pretrained=/path/to/model"``
        """

    def to_evalchemy_args(self) -> str:
        """Return the ``--model_args`` string for the evalchemy engine.

        Defaults to the lm-eval string (evalchemy is a forked lm-eval).
        """
        return self.to_lm_eval_args()

    def to_contrib_flags(self) -> str | None:
        """Return the model-type flag for contrib suite routing.

        This is the value returned by ``detect_model_flags()`` in
        ``suite.py``.  Override to distinguish between inference backends
        for the same benchmark (e.g. ``"vision_reasoner"`` vs ``"qwen2"``).

        Returns ``None`` by default (no model-type distinction needed).
        """
        return None


class DefaultHFAdapter(BaseModelAdapter):
    """Built-in HuggingFace adapter — the single source of the engine
    ``--model_args`` strings rendered into ``template.sbatch``.

    The scheduler instantiates it once per run with the literal bash
    placeholder ``$model_path`` (each CSV row substitutes its model at
    runtime on the compute node); ``extra_args`` carries run-level additions
    such as quantization (``",load_in_4bit=True"``). Per-model argument
    differentiation arrives with the plugin-protocol ``eval_args`` channel.
    """

    def __init__(
        self,
        model_path: str | Path = "$model_path",
        *,
        trust_remote_code: bool = True,
        extra_args: str = "",
    ) -> None:
        self._path = model_path
        self._trust = trust_remote_code
        self._extra = extra_args

    @property
    def model_path(self) -> str | Path:
        return self._path

    def to_lm_eval_args(self) -> str:
        return f'pretrained="{self._path}",trust_remote_code={self._trust}{self._extra}'

    def to_lmms_eval_args(self) -> str:
        # $_lmms_extra_args is the bash-side per-family hook filled in
        # template.sbatch (llava model_name workaround, qwen frame cap).
        return f"pretrained={self._path},device_map=auto$_lmms_extra_args{self._extra}"

    def to_evalchemy_args(self) -> str:
        return f"trust_remote_code={self._trust},pretrained={self._path}{self._extra}"
