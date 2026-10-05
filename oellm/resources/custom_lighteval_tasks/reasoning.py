"""Multilingual GPQA Diamond, contributed by LumiOpen (AMD Silo AI).

Loaded by lighteval through ``--custom-tasks``; every task name ends in
``:<iso639-1>`` so oellm-eval can resolve its language.

- ``gpqa_diamond:{lang}``: GPQA Diamond, generative with letter extraction.
  English is ``Idavidrein/gpqa`` (gated), Finnish is the professionally
  post-edited ``LumiOpen/GPQA-FI``.

The prompt and metric are lighteval's ``gpqa:diamond`` for English and the
LumiOpen lighteval fork's ``gpqa-fi:diamond`` (``gpqa_fi.py``) for Finnish,
with two deliberate changes:

- Only the native (``lighteval accelerate``) backend is wired; oellm-eval never
  runs the inspect-ai backend.
- The answer order is derived from a hash of the question instead of the
  process-global RNG. lighteval seeds that RNG once per run and every prompt
  draws from it in turn, so a question's letter depends on which other tasks
  and samples were built before it in the same run. With the hash, a question
  gets the same letter in every run.
"""

import hashlib
import random
from string import ascii_uppercase

import numpy as np
from lighteval.metrics.dynamic_metrics import MultilingualExtractiveMatchMetric
from lighteval.metrics.metrics_sample import PassAtK
from lighteval.metrics.utils.extractive_match_utils import IndicesExtractionConfig
from lighteval.metrics.utils.metric_utils import SampleLevelMetric, SamplingMethod
from lighteval.tasks.lighteval_task import LightevalTaskConfig
from lighteval.tasks.requests import Doc
from lighteval.utils.language import Language

MAX_NEW_TOKENS = 32768

# --- GPQA Diamond -----------------------------------------------------------

# en is verbatim from lighteval's gpqa:diamond; fi from the LumiOpen fork.
GPQA_INSTRUCTIONS = {
    "en": "Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.",
    "fi": "Vastaa seuraavaan monivalintakysymykseen. Vastauksesi viimeisen rivin tulee olla muotoa: 'Vastaus: $KIRJAIN' (ilman lainausmerkkejä), jossa KIRJAIN on A, B, C tai D. Ajattele vaihe vaiheelta ennen vastaamista.",
}
_GPQA_SOURCE = {
    "en": ("Idavidrein/gpqa", "gpqa_diamond"),
    "fi": ("LumiOpen/GPQA-FI", "default"),
}
_GPQA_LANGUAGE = {"en": Language.ENGLISH, "fi": Language.FINNISH}


def _gold_index(question: str) -> int:
    """Position of the correct answer, fixed per question across runs and processes."""
    seed = int.from_bytes(hashlib.sha256(question.strip().encode()).digest()[:8], "big")
    return random.Random(seed).randint(0, 3)


def _gpqa_choices(record) -> tuple[list[str], int]:
    gold_index = _gold_index(record["Question"])
    choices = [
        record["Incorrect Answer 1"],
        record["Incorrect Answer 2"],
        record["Incorrect Answer 3"],
    ]
    choices.insert(gold_index, record["Correct Answer"])
    return [choice.strip() for choice in choices], gold_index


def _gpqa_prompt_fn(lang: str):
    instruction = GPQA_INSTRUCTIONS[lang]

    def gpqa_prompt(line, task_name: str = None):
        choices, gold_index = _gpqa_choices(line)
        query = f"{instruction}\n\n{line['Question'].strip()}\n\n" + "\n".join(
            f"{letter}) {choice}"
            for letter, choice in zip(
                ascii_uppercase[: len(choices)], choices, strict=True
            )
        )
        return Doc(
            task_name=task_name,
            query=query,
            choices=list(ascii_uppercase[: len(choices)]),
            gold_index=gold_index,
            instruction=instruction,
        )

    return gpqa_prompt


def _gpqa_pass_at_1(language: Language) -> SampleLevelMetric:
    """lighteval's gpqa_instruct_pass_at_k with language-specific answer anchors."""
    letters = IndicesExtractionConfig(
        prefix_for_extraction="NativeLetters", try_extract_without_anchor=True
    )
    return SampleLevelMetric(
        metric_name="gpqa_pass@k",
        sample_level_fn=PassAtK(
            sample_scoring_function=MultilingualExtractiveMatchMetric(
                language=language,
                gold_extraction_target=[letters],
                pred_extraction_target=[letters],
                precision=6,
            ),
        ),
        category=SamplingMethod.GENERATIVE,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )(sample_params={"k": 1})


def _gpqa_tasks() -> list[LightevalTaskConfig]:
    tasks = []
    for lang, (repo, subset) in _GPQA_SOURCE.items():
        language = _GPQA_LANGUAGE[lang]
        tasks.append(
            LightevalTaskConfig(
                name=f"gpqa_diamond:{lang}",
                prompt_function=_gpqa_prompt_fn(lang),
                hf_repo=repo,
                hf_subset=subset,
                hf_avail_splits=["train"],
                evaluation_splits=["train"],
                few_shots_split=None,
                few_shots_select=None,
                generation_size=MAX_NEW_TOKENS,
                metrics=[_gpqa_pass_at_1(language)],
                stop_sequence=[],
                version=1,
            )
        )
    return tasks


TASKS_TABLE = _gpqa_tasks()
