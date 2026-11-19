"""The template for bits-per-byte on your own text, custom_lm_eval_tasks/corpus_bpb."""

import json
from importlib.resources import files

import pytest
import yaml

TEMPLATE = (
    files("oellm.resources")
    / "custom_lm_eval_tasks"
    / "corpus_bpb"
    / "_corpus_bpb_template"
)


def test_template_is_not_loaded_as_a_task():
    """lm-eval only loads *.yaml files, so the unfilled template never shows up
    as a task with a placeholder path."""
    assert not TEMPLATE.name.endswith((".yaml", ".yml"))
    config = yaml.safe_load(TEMPLATE.read_text())
    assert config["output_type"] == "loglikelihood_rolling"
    assert config["metric_list"] == [
        {
            "metric": "bits_per_byte",
            "aggregation": "bits_per_byte",
            "higher_is_better": False,
        }
    ]


def test_filled_in_template_scores_a_local_file(tmp_path, monkeypatch):
    """Following docs/TASKS.md: copy the template next to a JSONL file, point
    it at the file, and lm-eval builds a task with one document per row."""
    pytest.importorskip("lm_eval")
    from lm_eval.tasks import TaskManager

    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "cache"))
    rows = ["Erste Zeile.", "Ο ήλιος είναι ένα αστέρι."]
    data = tmp_path / "heldout.jsonl"
    data.write_text("".join(json.dumps({"text": r}) + "\n" for r in rows))
    (tmp_path / "bpb_my_corpus.yaml").write_text(
        TEMPLATE.read_text().replace("/path/to/heldout.jsonl", str(data))
    )

    loaded = TaskManager(include_path=str(tmp_path)).load(["bpb_my_corpus"])
    task = loaded["tasks"]["bpb_my_corpus"]
    docs = list(task.test_docs())
    assert [task.doc_to_target(d) for d in docs] == rows
    # A document's BPB is weighted by its UTF-8 bytes: 2 per Greek letter.
    assert task.process_results(docs[1], [-10.0])["bits_per_byte"] == (
        -10.0,
        len(rows[1].encode()),
    )
