"""AudioBench task registry.

Single source of truth for the task set.  Consumed by
:mod:`oellm.contrib.audiobench.suite` to build ``TASK_GROUPS`` and to look up
per-task metadata (HF repo, upstream task name, metric) at dispatch time.

Every canonical task name is prefixed ``audiobench_`` so the CSV ``task_path``
column uniquely identifies the scorer and doesn't collide with lmms-eval's
names for the same benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass

SUITE_NAME = "audiobench"
_TASK_NAME_PREFIX = "audiobench_"


@dataclass(frozen=True)
class AudioBenchTaskSpec:
    """Metadata for a single AudioBench task.

    ``upstream_name`` is the literal string AudioBench's ``--dataset_name``
    expects (matched exactly against ``$AUDIOBENCH_DIR/src/dataset.py``'s
    dispatch table).  ``upstream_metric`` is what ``--metrics`` expects
    (usually identical to our canonical ``metric``).
    """

    name: str
    upstream_name: str
    hf_repo: str
    metric: str
    upstream_metric: str
    family: str

    @property
    def task_group(self) -> str:
        return f"audio-audiobench-{self.family}"


def _t(
    upstream_name: str,
    hf_repo: str,
    metric: str,
    family: str,
    *,
    upstream_metric: str | None = None,
    name: str | None = None,
) -> AudioBenchTaskSpec:
    """Build a spec with ``name = audiobench_<upstream_name>`` by default."""
    return AudioBenchTaskSpec(
        name=name if name is not None else _TASK_NAME_PREFIX + upstream_name,
        upstream_name=upstream_name,
        hf_repo=hf_repo,
        metric=metric,
        upstream_metric=upstream_metric or metric,
        family=family,
    )


# Tasks not covered by our lmms-eval task groups.
_NEW_ASR = [
    _t("aishell_asr_zh_test", "AudioLLMs/aishell_1_zh_test", "wer", "asr"),
    _t("earnings21_test", "AudioLLMs/earnings21_test", "wer", "asr"),
    _t("earnings22_test", "AudioLLMs/earnings22_test", "wer", "asr"),
    _t("tedlium3_long_form_test", "AudioLLMs/tedlium3_long_form_test", "wer", "asr"),
    # GigaSpeech2 — 3 languages share one HF repo.  AudioBench dispatches via
    # the dataset_name string itself (gigaspeech2_thai/indo/viet), not via a
    # --data_dir flag (which doesn't exist upstream).
    _t(
        "gigaspeech2_thai",
        "AudioLLMs/gigaspeech2-test",
        "wer",
        "asr",
        name="audiobench_gigaspeech2_thai",
    ),
    _t(
        "gigaspeech2_indo",
        "AudioLLMs/gigaspeech2-test",
        "wer",
        "asr",
        name="audiobench_gigaspeech2_indo",
    ),
    _t(
        "gigaspeech2_viet",
        "AudioLLMs/gigaspeech2-test",
        "wer",
        "asr",
        name="audiobench_gigaspeech2_viet",
    ),
    _t("seame_dev_man", "AudioLLMs/seame_dev_man", "wer", "asr"),
    _t("seame_dev_sge", "AudioLLMs/seame_dev_sge", "wer", "asr"),
]

_NEW_ST = [
    _t("covost2_en_id_test", "AudioLLMs/covost2_en_id_test", "bleu", "st"),
    _t("covost2_en_ta_test", "AudioLLMs/covost2_en_ta_test", "bleu", "st"),
    _t("covost2_id_en_test", "AudioLLMs/covost2_id_en_test", "bleu", "st"),
    _t("covost2_zh_en_test", "AudioLLMs/covost2_zh_en_test", "bleu", "st"),
    _t("covost2_ta_en_test", "AudioLLMs/covost2_ta_en_test", "bleu", "st"),
]

_NEW_REASONING = [
    # Spoken-MQA — 4 splits share one HF repo.  AudioBench dispatches via
    # the hyphen-prefixed dataset_name (spoken-mqa_<split>), not --data_dir.
    _t(
        "spoken-mqa_short_digit",
        "amao0o0/spoken-mqa",
        "accuracy",
        "reasoning",
        upstream_metric="acc",
        name="audiobench_spoken_mqa_short_digit",
    ),
    _t(
        "spoken-mqa_long_digit",
        "amao0o0/spoken-mqa",
        "accuracy",
        "reasoning",
        upstream_metric="acc",
        name="audiobench_spoken_mqa_long_digit",
    ),
    _t(
        "spoken-mqa_single_step_reasoning",
        "amao0o0/spoken-mqa",
        "accuracy",
        "reasoning",
        upstream_metric="acc",
        name="audiobench_spoken_mqa_single_step_reasoning",
    ),
    _t(
        "spoken-mqa_multi_step_reasoning",
        "amao0o0/spoken-mqa",
        "accuracy",
        "reasoning",
        upstream_metric="acc",
        name="audiobench_spoken_mqa_multi_step_reasoning",
    ),
    _t("mmau_mini", "AudioLLMs/MMAU-mini", "string_match", "reasoning"),
    _t("audiocaps_test", "AudioLLMs/audiocaps_test", "meteor", "reasoning"),
]

# Dual-registered duplicates of benchmarks also in lmms-eval.  These use
# AudioBench's scorer/normaliser for paper-comparable numbers; the lmms-eval
# versions stay in place.  HF repos differ (AudioLLMs/* vs lmms-lab/*) so
# snapshot_download does not collide.
_DUAL = [
    _t("librispeech_test_clean", "AudioLLMs/librispeech_test_clean", "wer", "asr"),
    _t("librispeech_test_other", "AudioLLMs/librispeech_test_other", "wer", "asr"),
    _t("common_voice_15_en_test", "AudioLLMs/common_voice_15_en_test", "wer", "asr"),
    _t("gigaspeech_test", "AudioLLMs/gigaspeech_test", "wer", "asr"),
    _t("peoples_speech_test", "AudioLLMs/peoples_speech_test", "wer", "asr"),
    _t("tedlium3_test", "AudioLLMs/tedlium3_test", "wer", "asr"),
    _t("covost2_en_zh_test", "AudioLLMs/covost2_en_zh_test", "bleu", "st"),
]


AUDIOBENCH_TASKS: list[AudioBenchTaskSpec] = [
    *_NEW_ASR,
    *_NEW_ST,
    *_NEW_REASONING,
    *_DUAL,
]


def _validate() -> None:
    seen_names: set[str] = set()
    for t in AUDIOBENCH_TASKS:
        if t.name in seen_names:
            raise RuntimeError(f"Duplicate AudioBench task name {t.name!r} in registry")
        seen_names.add(t.name)
        if not t.name.startswith(_TASK_NAME_PREFIX):
            raise RuntimeError(
                f"AudioBench task {t.name!r} missing required prefix "
                f"{_TASK_NAME_PREFIX!r}"
            )
        if t.family not in {"asr", "st", "reasoning"}:
            raise RuntimeError(
                f"AudioBench task {t.name!r} has unknown family {t.family!r}"
            )


_validate()


def get_task_spec(name: str) -> AudioBenchTaskSpec:
    """Look up a spec by canonical task name; raises ``KeyError`` if missing."""
    for t in AUDIOBENCH_TASKS:
        if t.name == name:
            return t
    known = sorted(t.name for t in AUDIOBENCH_TASKS)
    raise KeyError(f"Unknown AudioBench task {name!r}.  Known tasks: {', '.join(known)}")
