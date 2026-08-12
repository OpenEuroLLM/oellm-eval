"""Core plugin interfaces.

``CORE_API_VERSION`` marks the plugin-API freeze (plan item G4): breaking
changes to :class:`BaseTask` / :class:`BaseMetric` / :class:`BaseModelAdapter`
after 1.0 require a deprecation cycle. The v2 ``BaseMetric`` contract
(per-sample records instead of parallel prediction/reference string lists)
landed *before* this freeze, precisely so external contributions never build
against the narrower signature.
"""

from oellm.core.base_metric import BaseMetric
from oellm.core.base_model_adapter import BaseModelAdapter, DefaultHFAdapter
from oellm.core.base_task import BaseTask

CORE_API_VERSION = "1.0"

__all__ = [
    "CORE_API_VERSION",
    "BaseMetric",
    "BaseModelAdapter",
    "BaseTask",
    "DefaultHFAdapter",
]
