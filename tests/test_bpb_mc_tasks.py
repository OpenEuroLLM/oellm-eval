"""Gold-answer bits-per-byte tasks in custom_lm_eval_tasks/bpb_mc."""

import importlib.util
from importlib.resources import files
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from oellm.task_groups import _select_tasks

TASK_DIR = files("oellm.resources") / "custom_lm_eval_tasks" / "bpb_mc"


class _Loader(yaml.SafeLoader):
    """Reads `!function` tags as plain strings, so the task files load without lm-eval."""


_Loader.add_constructor(
    "!function", lambda loader, node: f"!function {loader.construct_scalar(node)}"
)


def test_task_files_match_the_group():
    """Every task the group schedules has a wrapper that names the lm-eval task
    it wraps, and no wrapper exists that the group never schedules."""
    tasks = {t.name for _suite, t in _select_tasks(["bpb-mc-en"])}
    task_files = sorted(p for p in TASK_DIR.iterdir() if p.name.endswith(".yaml"))
    assert {p.name.removesuffix(".yaml") for p in task_files} == tasks
    for path in task_files:
        config = yaml.load(path.read_text(), Loader=_Loader)
        name = path.name.removesuffix(".yaml")
        assert config["task"] == name
        assert config["class"] == "!function bpb_task.BPBTask"
        assert config["base_task"] == name.removesuffix("_bpb")


@pytest.fixture(scope="module")
def bpb_task():
    pytest.importorskip("lm_eval")
    spec = importlib.util.spec_from_file_location(
        "bpb_task", str(TASK_DIR / "bpb_task.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _task(module, *, choices, target, text=None, multiple_input=False):
    """A BPBTask with its document accessors stubbed, so process_results runs
    without loading a dataset."""
    task = object.__new__(module.BPBTask)
    task._config = SimpleNamespace(target_delimiter=" ")
    task.multiple_input = multiple_input
    task.doc_to_choice = lambda doc: choices
    task.doc_to_target = lambda doc: target
    task.doc_to_text = lambda doc: text
    return task


LLS = [(-2.0, False), (-3.0, False), (-4.0, False)]


@pytest.mark.parametrize(
    "target, expected",
    [
        (1, (-3.0, len(" blue"))),  # gold as an index
        (np.int64(1), (-3.0, len(" blue"))),  # as a numpy integer from a dataset
        ("red", (-2.0, len(" red"))),  # gold as the answer text
        ([2, 0], (-4.0, len(" grün".encode()))),  # several golds: the first one
        (7, (0.0, 0)),  # out of range: skipped without moving the average
        (True, (0.0, 0)),  # a bool is not an index
    ],
)
def test_scores_the_gold_answer_and_its_bytes(bpb_task, target, expected):
    """The byte count is the UTF-8 length of what lm-eval scored, delimiter
    included, so "grün" counts its "ü" as two bytes."""
    task = _task(bpb_task, choices=["red", "blue", "grün"], target=target)
    assert task.process_results({}, LLS) == {"bits_per_byte": expected}


def test_partial_evaluation_scores_the_shared_answer(bpb_task):
    """Winogrande-style tasks put the choices in the context and continue every
    one with the same answer text; the gold index comes from doc_to_text."""
    task = _task(
        bpb_task,
        choices=["Anna said", "Ben said"],
        target="that she would come.",
        text=1,
        multiple_input=2,
    )
    assert task.process_results({}, LLS[:2]) == {
        "bits_per_byte": (-3.0, len(" that she would come."))
    }


def test_base_task_config_comes_from_lm_eval(bpb_task):
    config = bpb_task.base_task_config("arc_easy")
    assert config["output_type"] == "multiple_choice"
    assert config["dataset_name"] == "ARC-Easy"


def test_group_pre_downloads_the_dataset_each_task_reads(bpb_task):
    """Compute nodes are offline, so each wrapped task must read exactly the
    dataset and subset its task-group entry pre-downloads. A wrapper's own
    keys, such as social_iqa_bpb's pinned dataset_path, override the base."""
    for _suite, t in _select_tasks(["bpb-mc-en"]):
        wrapper = yaml.load((TASK_DIR / f"{t.name}.yaml").read_text(), Loader=_Loader)
        config = {**bpb_task.base_task_config(wrapper["base_task"]), **wrapper}
        assert (config["dataset_path"], config.get("dataset_name")) == (
            t.dataset,
            t.subset,
        ), t.name


def test_only_multiple_choice_tasks_can_be_wrapped(bpb_task):
    with pytest.raises(ValueError, match="not multiple_choice"):
        bpb_task.BPBTask(config={"task": "wikitext_bpb", "base_task": "wikitext"})
