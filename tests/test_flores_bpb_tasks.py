"""The FLORES-200 bits-per-byte tasks in custom_lm_eval_tasks/flores_bpb."""

import importlib.util
from importlib.resources import files

import datasets
import pytest
import yaml

from oellm.task_groups import _select_tasks

TASK_DIR = files("oellm.resources") / "custom_lm_eval_tasks" / "flores_bpb"


@pytest.fixture(scope="module")
def utils():
    path = TASK_DIR / "utils.py"
    spec = importlib.util.spec_from_file_location("flores_bpb_utils", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_task_files_match_the_group():
    """lm-eval finds tasks by name under --include_path, so every task the group
    schedules needs a task file that loads the matching FLORES language config,
    and no task file should exist that the group never schedules."""
    tasks = {t.name: t.subset for _suite, t in _select_tasks(["bpb-flores-eu"])}
    task_files = sorted(p for p in TASK_DIR.iterdir() if p.name.endswith(".yaml"))
    assert {p.name.removesuffix(".yaml") for p in task_files} == set(tasks)
    for path in task_files:
        config = yaml.safe_load(path.read_text())
        name = path.name.removesuffix(".yaml")
        assert config == {
            "dataset_name": tasks[name],
            "include": "_default_template_yaml",
            "task": name,
        }


def test_process_docs_joins_the_sentences_of_each_article(utils):
    """Articles keep the order they first appear in, and sentences their order
    within the article, even when an article's sentences are not adjacent."""
    dataset = datasets.Dataset.from_dict(
        {
            "URL": ["a", "a", "b", "a", "c"],
            "sentence": ["One.", "Two. ", "Uno.", "Three.", "Eins."],
        }
    )
    docs = utils.process_docs(dataset)
    assert list(docs["URL"]) == ["a", "b", "c"]
    assert list(docs["text"]) == ["One. Two. Three.", "Uno.", "Eins."]
