"""Task groups that leave out the test documents a decontamination run marked.

``resources/decontaminated/<name>.json`` lists, for every lm-eval task holding
a marked row in any language, its number of documents (``documents``), the
positions to leave out (``excluded``) and :func:`oellm.sample_exclusion.item_key`
of the kept documents' answers (``kept_answers``). A task group with
``decontaminated: <name>`` runs the other documents through lm-eval's
``--samples``, and ``collect`` reports them with ``subset``
``decontaminated:<name>``. ``scripts/build_decontaminated_samples.py`` builds
the list from the marked rows.
"""

import json
from functools import cache
from importlib.resources import files
from pathlib import Path

from oellm.sample_exclusion import item_key

_LISTS = "decontaminated"


@cache
def load(name: str) -> dict:
    """The list of documents to leave out named *name* (``agenttrove-0.0``)."""
    path = files("oellm.resources") / _LISTS / f"{name}.json"
    if not path.is_file():
        raise ValueError(f"No list of decontaminated samples named {name!r}")
    return json.loads(path.read_text())


def samples(name: str) -> dict[str, list[int]]:
    """lm-eval's ``--samples``: the documents each task runs on."""
    return {task: _kept(entry) for task, entry in sorted(load(name)["tasks"].items())}


def _kept(entry: dict) -> list[int]:
    excluded = set(entry["excluded"])
    return [i for i in range(entry["documents"]) if i not in excluded]


def names() -> list[str]:
    directory = files("oellm.resources") / _LISTS
    if not directory.is_dir():
        return []
    return sorted(
        entry.name.removesuffix(".json")
        for entry in directory.iterdir()
        if entry.name.endswith(".json")
    )


# The job template names such results <run id>--decontaminated--<name>_<date>.json,
# as lm-eval does not record --samples.
MARK = "--decontaminated--"


def run_of(results_file: Path, data: dict) -> str | None:
    """The list a run left documents out by, from its file name, or None.

    Raises ValueError unless each task ran on the documents the list keeps
    (n-samples, and the saved samples with their answers).
    """
    stem = results_file.stem.rsplit("_", 1)[0]
    if MARK not in stem:
        return None
    name = stem.split(MARK, 1)[1]
    tasks = load(name)["tasks"]
    if not data.get("n-samples"):
        raise ValueError(
            f"{results_file} records no number of documents its tasks ran on "
            "(n-samples), so it cannot be checked against the decontaminated "
            f"list {name}."
        )
    for task, counts in data["n-samples"].items():
        entry = tasks.get(task)
        original = counts.get("original")
        expected = (
            (entry["documents"], entry["documents"] - len(entry["excluded"]))
            if entry
            else (original, original)
        )
        if (original, counts.get("effective")) != expected:
            raise ValueError(
                f"{task}: {results_file} ran on {counts.get('effective')} of "
                f"{original} documents, but the decontaminated list {name} keeps "
                f"{expected[1]} of {expected[0]}."
            )
        if entry:
            _check_samples(results_file, task, entry, name)
    return name


def _check_samples(results_file: Path, task: str, entry: dict, name: str) -> None:
    # The date has no underscore.
    timestamp = results_file.stem.rsplit("_", 1)[-1]
    path = results_file.with_name(f"samples_{task}_{timestamp}.jsonl")
    if not path.exists():
        raise ValueError(
            f"{task}: no saved samples next to {results_file} (expected {path.name}), "
            f"so its documents cannot be checked against the decontaminated list {name}."
        )
    answers = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                sample = json.loads(line)
                answers.setdefault(sample["doc_id"], str(sample.get("target")))
    if (
        sorted(answers) != _kept(entry)
        or item_key([answers[i] for i in sorted(answers)]) != entry["kept_answers"]
    ):
        raise ValueError(
            f"{task}: the documents {results_file} ran on are not the ones the "
            f"decontaminated list {name} keeps."
        )
