"""Tests for the multilingual BPB tasks (the ``oellm-multilingual-bpb`` super group).

The registry checks guard against drift between task-groups.yaml and the task
YAMLs in custom_lm_eval_tasks/bpb/; the scoring checks pin the byte convention
documented in bpb/utils.py on synthetic documents (no datasets or models).
"""

import importlib.util
import math
import re
from importlib.resources import files

import pytest
import yaml

from oellm.task_groups import _expand_task_groups

BPB_DIR = files("oellm.resources") / "custom_lm_eval_tasks" / "bpb"


def _load_utils():
    spec = importlib.util.spec_from_file_location("bpb_utils", str(BPB_DIR / "utils.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


utils = _load_utils()


def _defined_bpb_tasks() -> set[str]:
    names = set()
    for path in BPB_DIR.rglob("*.yaml"):
        match = re.search(r"^task: (\S+)$", path.read_text(encoding="utf-8"), re.M)
        if match:
            names.add(match.group(1))
    return names


# --- registry -----------------------------------------------------------------


def test_super_group_expands_to_every_benchmark():
    jobs = _expand_task_groups(["oellm-multilingual-bpb"])
    assert len(jobs) == 169
    assert {j.suite for j in jobs} == {"lm-eval-harness"}
    assert {j.n_shot for j in jobs} == {0}


def test_every_scheduled_task_is_defined_and_reports_bpb():
    task_metrics = yaml.safe_load(
        (files("oellm.resources") / "task-groups.yaml").read_text()
    )["task_metrics"]
    defined = _defined_bpb_tasks()
    for job in _expand_task_groups(["oellm-multilingual-bpb"]):
        assert job.task in defined, f"{job.task} has no YAML in custom_lm_eval_tasks/bpb"
        assert task_metrics.get(job.task) == "bits_per_byte", job.task


def test_super_group_bracket_mirrors_oellm_multilingual():
    """Scoping to one language selects that language's BPB counterpart of each
    oellm-multilingual benchmark (the non-BPB test expects the same set)."""
    tasks = {j.task for j in _expand_task_groups(["oellm-multilingual-bpb[deu_Latn]"])}
    assert tasks == {
        "belebele_deu_Latn_bpb",
        "flores200_deu_Latn-eng_Latn_bpb",
        "flores200_eng_Latn-deu_Latn_bpb",
        "global_mmlu_de_bpb",
        "include_base_44_german_bpb",
        "mgsm_direct_de_bpb",
    }


def test_template_functions_exist_in_utils():
    for path in BPB_DIR.glob("_*_template_yaml"):
        for fn in re.findall(r"!function utils\.(\w+)", path.read_text(encoding="utf-8")):
            assert callable(getattr(utils, fn, None)), f"{path.name}: utils.{fn}"


def test_task_yamls_in_subdirectories_use_no_functions():
    """`!function utils.x` resolves next to the YAML that declares it, so only
    the templates beside bpb/utils.py may use it."""
    for path in BPB_DIR.glob("*/*.yaml"):
        assert "!function" not in path.read_text(encoding="utf-8"), path


# --- scoring ------------------------------------------------------------------


def _mc(*lls):
    return [(ll, False) for ll in lls]


def test_belebele_scores_gold_answer_text():
    doc = {
        "mc_answer1": "a",
        "mc_answer2": "é",
        "mc_answer3": "猫",
        "mc_answer4": "d",
        "correct_answer_num": "3",
    }
    out = utils.process_results_belebele(doc, _mc(-4, -3, -1, -5))
    assert out == {"acc": 1.0, "acc_norm": 1.0, "bits_per_byte": (-1.0, 3)}


def test_global_mmlu_letter_answer():
    doc = {"option_a": "x", "option_b": "réponse", "option_c": "y", "option_d": "z"}
    out = utils.process_results_global_mmlu({**doc, "answer": "B"}, _mc(-4, -1, -2, -3))
    assert out["bits_per_byte"] == (-1.0, len("réponse".encode()))
    malformed = utils.process_results_global_mmlu(
        {**doc, "answer": ""}, _mc(-4, -1, -2, -3)
    )
    assert malformed["bits_per_byte"] == (0.0, 1)


def test_include_integer_answer():
    doc = {"option_a": "x", "option_b": "y", "option_c": "Antwort", "option_d": "z"}
    out = utils.process_results_include({**doc, "answer": 2}, _mc(-4, -3, -1, -5))
    assert out["bits_per_byte"] == (-1.0, 7)


def test_xcopa_lowercases_choice_like_upstream():
    doc = {"choice1": "See oli õrn.", "choice2": "Ärge.", "label": 1}
    assert utils.doc_to_choice_xcopa(doc) == ["see oli õrn.", "ärge."]
    out = utils.process_results_xcopa(doc, _mc(-2, -1))
    assert out["bits_per_byte"] == (-1.0, len("ärge.".encode()))


def test_xstorycloze_one_indexed_ending():
    doc = {"sentence_quiz1": "a", "sentence_quiz2": "猫", "answer_right_ending": 2}
    assert utils.process_results_xstorycloze(doc, _mc(-2, -1))["bits_per_byte"] == (
        -1.0,
        3,
    )


def test_xwinograd_counts_shared_continuation():
    doc = {
        "sentence": "_ était ici",
        "option1": "Elle",
        "option2": "La personne",
        "answer": "1",
    }
    out = utils.process_results_winogrande(doc, _mc(-1, -2))
    assert out["bits_per_byte"] == (-1.0, len("était ici".encode()))


def test_mgsm_excludes_the_space_it_prepends():
    doc = {"answer_number": 18.0}
    assert utils.doc_to_target_mgsm(doc) == " 18"
    assert utils.process_results_mgsm(doc, [(-2.0, False)]) == {
        "bits_per_byte": (-2.0, 2)
    }


def test_flores_scores_reference_in_both_directions():
    doc = {"sentence_deu_Latn": "Grüße", "sentence_eng_Latn": "Greetings"}
    assert utils.doc_to_target_flores_xx_eng(doc) == " Greetings"
    assert utils.process_results_flores_xx_eng(doc, [(-3.0, False)]) == {
        "bits_per_byte": (-3.0, 9)
    }
    assert utils.doc_to_target_flores_eng_xx(doc) == " Grüße"
    assert utils.process_results_flores_eng_xx(doc, [(-3.0, False)]) == {
        "bits_per_byte": (-3.0, len("Grüße".encode()))
    }


def test_pairs_aggregate_to_corpus_level_bpb():
    """lm-eval's bits_per_byte aggregation is byte-weighted: -sum(ll)/sum(bytes)/ln 2."""
    pairs = [
        utils.process_results_flores_xx_eng(
            {"sentence_deu_Latn": "x", "sentence_eng_Latn": "a"}, [(-math.log(2), False)]
        )["bits_per_byte"],
        utils.process_results_flores_xx_eng(
            {"sentence_deu_Latn": "x", "sentence_eng_Latn": "猫"},
            [(-6 * math.log(2), False)],
        )["bits_per_byte"],
    ]
    lls, n_bytes = zip(*pairs, strict=True)
    assert -sum(lls) / sum(n_bytes) / math.log(2) == pytest.approx(1.75)
