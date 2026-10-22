"""The bundled custom lighteval tasks must match their task-groups.yaml entries.

Compute nodes are offline, so lighteval can only load what the scheduler
pre-downloaded from each task's ``dataset``/``subset``. A task whose lighteval
config reads a different repo or config than the registry declares fails at
eval time, not at schedule time.
"""

import importlib.util
from importlib.resources import files

import pytest

from oellm.task_groups import _select_tasks

pytest.importorskip("lighteval")

CUSTOM_GROUPS = ["gpqa-diamond"]


@pytest.fixture(scope="module")
def custom_tasks() -> dict:
    path = (
        files("oellm.resources") / "custom_lighteval_tasks" / "reasoning" / "reasoning.py"
    )
    spec = importlib.util.spec_from_file_location("reasoning", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {task.name: task for task in module.TASKS_TABLE}


@pytest.mark.parametrize("group", CUSTOM_GROUPS)
def test_registry_tasks_match_custom_task_datasets(custom_tasks, group):
    selected = _select_tasks([group])
    assert selected
    for suite, task in selected:
        assert suite == "lighteval"
        assert task.name in custom_tasks, f"{task.name} is not defined in reasoning.py"
        config = custom_tasks[task.name]
        assert (config.hf_repo, config.hf_subset) == (task.dataset, task.subset)


def test_custom_tasks_load_alongside_builtin_registry(custom_tasks):
    """Mirrors the template's `--load-tasks-multilingual --custom-tasks FILE`:
    lighteval rejects custom names that collide with built-in tasks."""
    from lighteval.tasks.registry import Registry

    path = (
        files("oellm.resources") / "custom_lighteval_tasks" / "reasoning" / "reasoning.py"
    )
    loaded = Registry.load_all_task_configs(
        custom_tasks=str(path), load_multilingual=True
    )
    assert set(custom_tasks) <= set(loaded)


RECORD = {
    "Question": "  Which particle mediates the strong force?\n",
    "Correct Answer": " Gluon ",
    "Incorrect Answer 1": "Photon",
    "Incorrect Answer 2": "W boson\n",
    "Incorrect Answer 3": "Graviton",
}


def _prompt(custom_tasks, name, record=RECORD):
    return custom_tasks[name].prompt_function(record, task_name=name)


@pytest.mark.parametrize("position", [0, 1, 2, 3])
def test_english_prompt_matches_lighteval_gpqa_diamond(
    custom_tasks, monkeypatch, position
):
    """Same query bytes as lighteval's built-in gpqa:diamond for every answer
    position; only the way that position is chosen differs."""
    import random

    from lighteval.tasks.tasks.gpqa import gpqa_instruct_prompt

    prompt_fn = custom_tasks["gpqa_diamond:en"].prompt_function
    monkeypatch.setitem(prompt_fn.__globals__, "_gold_index", lambda question: position)
    ours = prompt_fn(RECORD, task_name="gpqa_diamond:en")
    monkeypatch.setattr(random, "randint", lambda a, b: position)
    builtin = gpqa_instruct_prompt(RECORD, task_name="gpqa:diamond")

    assert ours.gold_index == builtin.gold_index == position
    assert ours.query == builtin.query
    assert ours.choices == builtin.choices
    assert ours.instruction == builtin.instruction


def test_finnish_prompt_matches_lumiopen_fork(custom_tasks):
    """The fork's gpqa_fi_instruct_prompt (LumiOpen/lighteval, gpqa_fi.py)."""
    doc = _prompt(custom_tasks, "gpqa_diamond:fi")
    choices = ["Photon", "W boson", "Graviton"]
    choices.insert(doc.gold_index, "Gluon")

    assert doc.query == (
        "Vastaa seuraavaan monivalintakysymykseen. Vastauksesi viimeisen rivin tulee "
        "olla muotoa: 'Vastaus: $KIRJAIN' (ilman lainausmerkkejä), jossa KIRJAIN on A, "
        "B, C tai D. Ajattele vaihe vaiheelta ennen vastaamista.\n\n"
        "Which particle mediates the strong force?\n\n"
        + "\n".join(
            f"{letter}) {choice}" for letter, choice in zip("ABCD", choices, strict=True)
        )
    )
    assert doc.choices == ["A", "B", "C", "D"]


def test_answer_order_is_fixed_per_question(custom_tasks):
    """The position of the correct answer depends only on the question: not on the
    global RNG, which lighteval seeds once per run and every task draws from."""
    import random

    state = random.getstate()
    try:
        first = _prompt(custom_tasks, "gpqa_diamond:en")
        random.seed(0)
        random.random()
        again = _prompt(custom_tasks, "gpqa_diamond:en")
        finnish = _prompt(custom_tasks, "gpqa_diamond:fi")
    finally:
        random.setstate(state)

    assert again.query == first.query
    assert again.gold_index == first.gold_index == finnish.gold_index
    positions = {
        _prompt(
            custom_tasks, "gpqa_diamond:en", {**RECORD, "Question": f"Q{i}"}
        ).gold_index
        for i in range(40)
    }
    assert positions == {0, 1, 2, 3}


@pytest.mark.parametrize(
    ("name", "response", "correct"),
    [
        ("gpqa_diamond:en", "Reasoning...\nAnswer: {gold}", True),
        ("gpqa_diamond:en", "Reasoning...\nAnswer: {wrong}", False),
        ("gpqa_diamond:fi", "Päättelyä...\nVastaus: {gold}", True),
        ("gpqa_diamond:fi", "Päättelyä...\nVastaus: {wrong}", False),
        ("gpqa_diamond:fi", "No answer given.", False),
        # Each language reads its own anchor when both appear.
        ("gpqa_diamond:fi", "Vastaus: {gold}. Answer: {wrong}", True),
        ("gpqa_diamond:en", "Vastaus: {gold}. Answer: {wrong}", False),
    ],
)
def test_metric_reads_the_answer_line(custom_tasks, name, response, correct):
    from lighteval.models.model_output import ModelResponse

    doc = _prompt(custom_tasks, name)
    gold = "ABCD"[doc.gold_index]
    wrong = "ABCD"[(doc.gold_index + 1) % 4]
    (metric,) = custom_tasks[name].metrics
    text = response.format(gold=gold, wrong=wrong)

    scores = metric.compute_sample(doc=doc, model_response=ModelResponse(text=[text]))

    assert scores == {"gpqa_pass@k:k=1": 1.0 if correct else 0.0}
