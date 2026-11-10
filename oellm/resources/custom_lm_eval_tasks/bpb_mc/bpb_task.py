"""Gold-answer bits-per-byte (BPB) for multiple-choice tasks.

For each question, BPB scores only the correct answer: its log-likelihood and
the UTF-8 bytes of the text lm-eval scored for it (the target delimiter plus
the answer). lm-eval's `bits_per_byte` aggregation then takes the
micro-average -sum(ll) / (ln 2 * sum(bytes)) over all questions.

Accuracy only moves when the gold answer overtakes the others, so it sits at
chance for very small models. Gold-answer BPB still changes when the model
gives the correct answer more probability, which separates small models.

A task is declared by a YAML that names the lm-eval task to wrap:

    task: arc_easy_bpb
    class: !function bpb_task.BPBTask
    base_task: arc_easy

The wrapped task's config is looked up by name at run time, because lm-eval's
`include:` only takes file paths and the built-in task files live in different
places in the venv and the container. Only use this on cloze-style tasks,
whose answers are text: on tasks that score letters ("A", "B", ...) BPB
measures how likely a single letter is, which says nothing about the model.

Not for runs with lm-eval's --apply_chat_template: lm-eval then drops the
target delimiter from the scored text, but process_results cannot see that and
still counts it, one byte too many per question.
"""

import functools
import logging
import numbers
from copy import deepcopy
from pathlib import Path

from lm_eval.api.task import ConfigurableTask
from lm_eval.tasks import TaskManager

try:
    from lm_eval.tasks._yaml_loader import load_yaml
except ImportError as e:  # the task index was rewritten in lm-eval 0.4.12
    raise ImportError(
        "The *_bpb tasks need lm-eval >= 0.4.12 (e.g. the DCLM venv pins 0.4.9.2)"
    ) from e

eval_logger = logging.getLogger(__name__)

BPB_METRIC = [
    {"metric": "bits_per_byte", "aggregation": "bits_per_byte", "higher_is_better": False}
]

# Keys that name or group the wrapped task, or set its metrics, which the
# wrapper replaces with its own.
_WRAPPER_OWNED_KEYS = ("task", "task_alias", "tag", "group", "metric_list")


@functools.cache
def _task_index() -> dict:
    """Index of every task lm-eval knows, built once per process because
    walking all task files takes a while. Includes the bundled custom tasks so
    they can be wrapped too."""
    return TaskManager(
        include_path=str(Path(__file__).resolve().parent.parent)
    ).task_index


def base_task_config(name: str) -> dict:
    """The full config of the lm-eval task called *name*, with its includes
    merged and its `!function` references resolved."""
    entry = _task_index().get(name)
    if entry is None:
        raise ValueError(f"BPB wrapper: unknown base_task {name!r}")
    if entry.yaml_path:
        cfg = load_yaml(entry.yaml_path, resolve_func=True)
    elif entry.cfg:
        cfg = entry.cfg
    else:
        raise ValueError(f"BPB wrapper: base_task {name!r} has no config to wrap")
    return deepcopy(cfg)


def _as_index(value) -> int | None:
    """A gold index as a plain int. Accepts numpy integers, which datasets and
    process_docs hooks can return, but not bools."""
    if isinstance(value, numbers.Integral) and not isinstance(value, bool):
        return int(value)
    return None


class BPBTask(ConfigurableTask):
    def __init__(self, config: dict | None = None, **kwargs) -> None:
        config = dict(config or {})
        config.pop("class", None)
        base_name = config.pop("base_task", None)
        if not base_name:
            raise ValueError(f"BPB wrapper {config.get('task')!r} needs a base_task")
        base = base_task_config(base_name)
        if base.get("output_type") != "multiple_choice":
            raise ValueError(
                f"BPB wrapper: base_task {base_name!r} is {base.get('output_type')!r}, "
                "not multiple_choice"
            )
        for key in _WRAPPER_OWNED_KEYS:
            base.pop(key, None)
        # Record what was wrapped, so the results show which version of the
        # base task's prompt and data produced them.
        config["metadata"] = {
            **(config.get("metadata") or {}),
            "base_task": base_name,
            "base_task_version": (base.get("metadata") or {}).get("version"),
        }
        super().__init__(config={**base, **config, "metric_list": BPB_METRIC}, **kwargs)

    def process_results(self, doc: dict, results) -> dict:
        lls = [ll for ll, _is_greedy in results]
        if self.multiple_input:
            # Partial evaluation (e.g. winogrande): each choice is a context,
            # all of them continue with the same answer text, and the gold
            # index comes from doc_to_text.
            n_choices = len(self.doc_to_choice(doc))
            gold = _as_index(self.doc_to_text(doc))
            answer = self.doc_to_target(doc)
        else:
            choices = self.doc_to_choice(doc)
            n_choices = len(choices)
            gold = self.doc_to_target(doc)
            if isinstance(gold, list):  # several correct answers: score the first
                gold = gold[0] if gold else None
            if isinstance(gold, str):
                gold = choices.index(gold) if gold in choices else None
            gold = _as_index(gold)
            answer = choices[gold] if gold is not None and 0 <= gold < n_choices else None

        if (
            gold is None
            or not 0 <= gold < min(n_choices, len(lls))
            or not isinstance(answer, str)
        ):
            eval_logger.warning(f"BPB: no valid gold answer for doc, skipping it: {doc}")
            # Zero log-likelihood over zero bytes leaves the micro-average unchanged.
            return {"bits_per_byte": (0.0, 0)}

        scored = f"{self.config.target_delimiter}{answer}"
        return {"bits_per_byte": (lls[gold], len(scored.encode("utf-8")))}
