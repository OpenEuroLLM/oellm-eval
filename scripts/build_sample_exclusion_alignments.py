"""Build the alignment tables of ``oellm-eval collect --exclude_samples``.

For benchmarks whose translations keep no field identifying an item (XNLI), or
that lighteval runs, a table in oellm/resources/sample-exclusion-alignments/
gives each item's position in every task (see
oellm.sample_exclusion.load_alignments). The tables are built from the
evaluated datasets, checking that each language's documents are translations
of the same items, in the same order with the same answers or by a shared id.

Run in the eval container, with lm-eval's Python for the lm-eval tables and
lighteval's for the lighteval ones:
    python scripts/build_sample_exclusion_alignments.py lm-eval [--check] [benchmark ...]
    /opt/uv-tools/lighteval/bin/python scripts/build_sample_exclusion_alignments.py \\
        lighteval [--check] [benchmark ...]
With --check, the rebuilt tables are compared with the repository's, not written.
"""

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from oellm.sample_exclusion import _canonical, _normalized, item_key  # noqa: E402

TABLES = REPO / "oellm/resources/sample-exclusion-alignments"


class AlignmentError(Exception):
    """Documents that are not translations of each other as a table assumes."""


class CannotBuild(Exception):
    """A task the harness cannot build its documents for, so cannot run."""


def items_of(
    key: list[str], docs: list[dict], benchmark: str, repeats=False
) -> list[str]:
    """The item keys of *docs*, which must identify one document each, unless
    the benchmark *repeats* some (MMLU)."""
    items = [item_key([doc[f] for f in key]) for doc in docs]
    if not repeats and len(set(items)) != len(items):
        raise AlignmentError(f"{benchmark}: {key} does not identify one document each")
    return items


def entry(positions: list[int | None], gold: list[str], shared=False) -> dict:
    """A task's table entry, from the position of each item among its documents
    (None where it lacks the item) and the answer of each document; items can
    be the same document only where they are *shared* (WSC and XWinograd)."""
    held = [(i, p) for i, p in enumerate(positions) if p is not None]
    taken = [p for _, p in held]
    if (
        not held
        or (not shared and len(set(taken)) != len(taken))
        or not all(0 <= p < len(gold) for p in taken)
    ):
        raise AlignmentError("positions must be distinct documents of the task")
    table = {"documents": len(gold), "answers": item_key(gold)}
    first = held[0][0]
    if held == [(first + p, p) for p in range(len(gold))]:
        return table | {"first": first}
    return table | {"positions": positions}


def by_position(
    benchmark: str,
    anchor: list[dict],
    task: str,
    docs: list[dict],
    answer,
    tolerate: float = 0.0,
) -> list[int]:
    """Identity positions, after checking that *docs* (a translation of
    *anchor*) are in its order: as many, each with the same answer, but for
    the fraction *tolerate* of them (MMMLU mislabels a few of its translations'
    answers); documents out of order would differ in most answers."""
    if len(docs) != len(anchor):
        raise AlignmentError(
            f"{benchmark}: {task} has {len(docs)} documents, not {len(anchor)}"
        )
    differ = [
        i
        for i, (a, d) in enumerate(zip(anchor, docs, strict=True))
        if answer(a) != answer(d)
    ]
    if len(differ) > tolerate * len(docs):
        raise AlignmentError(
            f"{benchmark}: {task}'s documents {differ[:10]} have another answer than "
            f"the same position of the English ones ({len(differ)} in all)"
        )
    if differ:
        print(
            f"note: {task}'s documents {differ} have another answer than the "
            f"English ones, {len(differ)} of {len(docs)}",
            flush=True,
        )
    return list(range(len(docs)))


def by_field(
    benchmark: str,
    anchor: list[dict],
    field: str,
    task: str,
    docs: list[dict],
    answer=None,
) -> list[int | None]:
    """Position in *docs* of each *anchor* item, by its identifying *field*."""
    where = {}
    for position, doc in enumerate(docs):
        if doc[field] in where:
            raise AlignmentError(f"{benchmark}: {task} repeats {field} {doc[field]!r}")
        where[doc[field]] = position
    positions = [where.get(a[field]) for a in anchor]
    if answer:
        differ = [
            a[field]
            for a, p in zip(anchor, positions, strict=True)
            if p is not None and answer(a) != answer(docs[p])
        ]
        if differ:
            raise AlignmentError(
                f"{benchmark}: {task}'s documents {differ[:10]} have another answer "
                f"than the English ones of the same {field} ({len(differ)} in all)"
            )
    return positions


def number(text) -> str:
    """An answer as a number, the way it reads in every language."""
    text = re.sub(r"(?<=\d)[,\s](?=\d{3}\b)", "", str(text).strip())
    found = re.fullmatch(r"-?\d+(?:\.\d+)?", text)
    if not found:
        raise AlignmentError(f"not a number: {text!r}")
    return str(float(text))


# --- lm-eval -----------------------------------------------------------------


class LmEval:
    def __init__(self):
        from lm_eval.tasks import TaskManager

        self.manager = TaskManager(
            include_path=str(REPO / "oellm/resources/custom_lm_eval_tasks")
        )
        self.cache = {}
        # The languages to build (--languages), of a table built in parts.
        self.languages = None

    def docs(self, name: str) -> tuple[list[dict], list[str]]:
        """A task's documents as lm-eval evaluates them (doc_id is the
        position), and their answers as it saves them (str of the target)."""
        if name not in self.cache:
            task = self.manager.load_task_or_group([name])[name]
            docs = [json.loads(json.dumps(d, default=str)) for d in task.eval_docs]
            gold = [str(task.doc_to_target(d)) for d in task.eval_docs]
            self.cache[name] = docs, gold
        return self.cache[name]


def xnli(lm: LmEval) -> dict:
    # facebook/xnli's languages are one table in another language each.
    langs = ["bul_Cyrl", "deu_Latn", "ell_Grek", "eng_Latn", "spa_Latn", "fra_Latn"]
    key = ["premise", "hypothesis"]
    anchor, _ = lm.docs("xnli_eng_Latn")
    tasks = {}
    for lang in langs:
        docs, gold = lm.docs(f"xnli_{lang}")
        positions = by_position(
            "XNLI", anchor, f"xnli_{lang}", docs, lambda d: d["label"]
        )
        tasks[f"xnli_{lang}"] = entry(positions, gold)
    return {"key": key, "items": items_of(key, anchor, "XNLI"), "tasks": tasks}


MGSM_LANGS = ["en", "de", "es", "fr"]
GLOBAL_MGSM_LANGS = ["de", "fr", "es", "el", "cs", "hu", "en", "ca", "eu", "gl", "sr"]
MGSM_TASKS = [f"mgsm_native_cot_{lang}" for lang in MGSM_LANGS] + [
    f"global_mgsm_{lang}" for lang in GLOBAL_MGSM_LANGS
]
POLYMATH_LANGS = ["de", "en", "es", "fr", "it", "pt"]
POLYMATH_LEVELS = ["low", "medium", "high", "top"]


def mgsm_answer(doc: dict) -> str:
    # MGSM gives the answer as answer_number, Global-MGSM and PolyMath as
    # answer, GSM8K at the end of its answer, after "####".
    answer = doc["answer_number"] if "answer_number" in doc else doc["answer"]
    return number(str(answer).rsplit("####", 1)[-1])


def first_items(
    benchmark: str, anchor: list[dict], task: str, docs: list[dict], n: int
) -> list[int | None]:
    """Positions of the first *n* items of *anchor* among *docs*, which are
    other copies of GSM8K's first test questions, in the same order: checked
    by their answers."""
    if len(docs) < n:
        raise AlignmentError(f"{benchmark}: {task} has {len(docs)} documents, not {n}")
    differ = [i for i in range(n) if mgsm_answer(anchor[i]) != mgsm_answer(docs[i])]
    if differ:
        raise AlignmentError(
            f"{benchmark}: {task}'s documents {differ[:10]} have another answer "
            f"than the same position of the anchor's ({len(differ)} in all)"
        )
    return list(range(n)) + [None] * (len(anchor) - n)


def _mgsm_tasks(lm: LmEval, benchmark: str, anchor: list[dict]) -> dict:
    # MGSM translates GSM8K's first 250 test questions, and Global-MGSM MGSM's,
    # in the same order; PolyMath's low level is the first 125 of them.
    tasks = {}
    for task in MGSM_TASKS:
        docs, gold = lm.docs(task)
        tasks[task] = entry(by_position(benchmark, anchor, task, docs, mgsm_answer), gold)
    for task in [f"polymath_{lang}_low" for lang in POLYMATH_LANGS]:
        docs, gold = lm.docs(task)
        tasks[task] = entry(first_items(benchmark, anchor, task, docs, len(docs)), gold)
    return tasks


def gsm8k(lm: LmEval) -> dict:
    key = ["question"]
    anchor, _ = lm.docs("mgsm_native_cot_en")
    questions = {d["question"] for d in lm.docs("gsm8k")[0]}
    missing = [i for i, d in enumerate(anchor) if d["question"] not in questions]
    if missing:
        raise AlignmentError(
            f"GSM8K: MGSM's English questions {missing[:10]} are not in gsm8k"
        )
    tasks = _mgsm_tasks(lm, "GSM8K", anchor)
    return {"key": key, "items": items_of(key, anchor, "GSM8K"), "tasks": tasks}


def global_mgsm(lm: LmEval) -> dict:
    key = ["question"]
    anchor, _ = lm.docs("global_mgsm_en")
    tasks = _mgsm_tasks(lm, "GlobalMGSM", anchor)
    docs, gold = lm.docs("gsm8k")
    positions = first_items("GlobalMGSM", anchor, "gsm8k", docs, len(anchor))
    tasks["gsm8k"] = entry(positions, gold)
    return {"key": key, "items": items_of(key, anchor, "GlobalMGSM"), "tasks": tasks}


def polymath(lm: LmEval) -> dict:
    # Qwen/PolyMath's ids name the language (low-en-50), so languages are
    # aligned by position. Its low level is GSM8K's first 125 test questions.
    key = ["id"]
    anchors = {level: lm.docs(f"polymath_en_{level}")[0] for level in POLYMATH_LEVELS}
    every = [d for level in POLYMATH_LEVELS for d in anchors[level]]
    items = items_of(key, every, "PolyMath")
    tasks = {}
    start = 0
    for level in POLYMATH_LEVELS:
        anchor = anchors[level]
        for lang in POLYMATH_LANGS:
            task = f"polymath_{lang}_{level}"
            docs, gold = lm.docs(task)
            within = by_position("PolyMath", anchor, task, docs, lambda d: d["answer"])
            positions = [None] * len(items)
            positions[start : start + len(anchor)] = within
            tasks[task] = entry(positions, gold)
        start += len(anchor)
    for task in ["gsm8k", *MGSM_TASKS]:
        docs, gold = lm.docs(task)
        positions = first_items("PolyMath", every, task, docs, len(anchors["low"]))
        tasks[task] = entry(positions, gold)
    return {"key": key, "items": items, "tasks": tasks}


XCSQA_LANGS = [
    "deu_Latn",
    "eng_Latn",
    "spa_Latn",
    "fra_Latn",
    "ita_Latn",
    "nld_Latn",
    "pol_Latn",
    "por_Latn",
]


def commonsense_qa(lm: LmEval) -> dict:
    # X-CSQA translates some of CommonsenseQA's validation questions, under
    # ids of its own that every language shares, with the choices shuffled.
    key = ["id"]
    anchor, _ = lm.docs("commonsense_qa")
    english, _ = lm.docs("xcsqa_eng_Latn")

    def answer_text(choices, answer_key):
        return choices["text"][choices["label"].index(answer_key)]

    stems = {}
    for d in english:
        stems.setdefault(" ".join(d["question"]["stem"].split()), []).append(d)
    x_ids = []
    for a in anchor:
        same = [
            d
            for d in stems.get(" ".join(a["question"].split()), [])
            if sorted(d["question"]["choices"]["text"]) == sorted(a["choices"]["text"])
            and answer_text(d["question"]["choices"], d["answerKey"])
            == answer_text(a["choices"], a["answerKey"])
        ]
        if len(same) > 1:
            raise AlignmentError(
                f"CommonsenseQA: {a['id']} is in X-CSQA {len(same)} times"
            )
        x_ids.append(same[0]["id"] if same else None)
    tasks = {}
    for lang in XCSQA_LANGS:
        docs, gold = lm.docs(f"xcsqa_{lang}")
        where = {d["id"]: p for p, d in enumerate(docs)}
        positions = [where.get(x) if x is not None else None for x in x_ids]
        if any(x is not None and where.get(x) is None for x in x_ids):
            raise AlignmentError(
                f"CommonsenseQA: xcsqa_{lang} lacks some of X-CSQA's ids"
            )
        tasks[f"xcsqa_{lang}"] = entry(positions, gold)
    return {"key": key, "items": items_of(key, anchor, "CommonsenseQA"), "tasks": tasks}


def xcsqa(lm: LmEval) -> dict:
    # The pipeline's X-CSQA rows are its test split (no answers), which oellm
    # does not evaluate; some are CommonsenseQA validation questions.
    key = ["id"]
    task = lm.manager.load_task_or_group(["xcsqa_eng_Latn"])["xcsqa_eng_Latn"]
    test = [json.loads(json.dumps(d, default=str)) for d in task.dataset["test"]]
    anchor, gold = lm.docs("commonsense_qa")

    def question(stem: str, choices: list[str]) -> tuple:
        return " ".join(stem.split()), tuple(sorted(choices))

    where = {}
    for p, a in enumerate(anchor):
        where.setdefault(question(a["question"], a["choices"]["text"]), []).append(p)
    positions = []
    for d in test:
        found = where.get(
            question(d["question"]["stem"], d["question"]["choices"]["text"]), []
        )
        if len(found) > 1:
            raise AlignmentError(f"Xcsqa: {d['id']} is commonsense_qa {found}")
        positions.append(found[0] if found else None)
    tasks = {"commonsense_qa": entry(positions, gold)}
    return {"key": key, "items": items_of(key, test, "Xcsqa"), "tasks": tasks}


def flores200(lm: LmEval) -> dict:
    # SIB-200 labels FLORES-200 sentences with a topic, every language at the
    # same position (index_id, not FLORES' id): a FLORES row is one of its
    # items when it is the same English sentence.
    key = ["sentence"]
    anchor, _ = lm.docs("sib200_eng_Latn")
    tasks = {}
    sib200 = REPO / "oellm/resources/custom_lm_eval_tasks/sib200"
    for path in sorted(sib200.glob("*.yaml")):
        task = path.stem
        docs, gold = lm.docs(task)
        positions = by_field(
            "Flores200", anchor, "index_id", task, docs, lambda d: d["category"]
        )
        tasks[task] = entry(positions, gold)
    sentences = [{"sentence": d["text"]} for d in anchor]
    return {"key": key, "items": items_of(key, sentences, "Flores200"), "tasks": tasks}


def winograd_pairs(lm: LmEval) -> list[tuple[int, int]]:
    """(wsc273 position, xwinograd_en position) of each XWinograd English
    sentence that is a WSC one with one of its words blanked. XWinograd holds
    most WSC items, some twice with the options swapped, and sometimes blanks
    another pronoun of the sentence (one WSC text can be two items)."""
    wsc, _ = lm.docs("wsc273")
    english, _ = lm.docs("xwinograd_en")
    texts = [_normalized(d["text"]) for d in wsc]
    words = [{_normalized(w) for w in d["text"].split()} for d in wsc]
    pairs = []
    for x, doc in enumerate(english):
        before, _, after = (_normalized(s) for s in doc["sentence"].partition("_"))
        found = [
            w
            for w, text in enumerate(texts)
            if len(text) > len(before) + len(after)
            and text.startswith(before)
            and text.endswith(after)
            and text[len(before) : len(text) - len(after)] in words[w]
        ]
        if len({texts[w] for w in found}) > 1:
            raise AlignmentError(f"XWinograd: xwinograd_en {x} is several WSC sentences")
        pairs += [(w, x) for w in found]
    return pairs


def winograd_schema_challenge(lm: LmEval) -> dict:
    # Rows keep only the fields they are matched on (sample-exclusion.yaml).
    key = ["pronoun", "pronoun_loc", "quote", "quote_loc", "label", "source"]
    wsc, _ = lm.docs("wsc273")
    _, gold = lm.docs("xwinograd_en")
    pairs = winograd_pairs(lm)
    items = items_of(
        key, [wsc[w] for w, _ in pairs], "WinogradSchemaChallenge", repeats=True
    )
    tasks = {"xwinograd_en": entry([x for _, x in pairs], gold, shared=True)}
    return {"key": key, "items": items, "tasks": tasks}


def xwinograd(lm: LmEval) -> dict:
    key = ["sentence", "option1", "option2", "answer"]
    english, _ = lm.docs("xwinograd_en")
    _, gold = lm.docs("wsc273")
    pairs = winograd_pairs(lm)
    items = items_of(key, [english[x] for _, x in pairs], "XWinograd", repeats=True)
    tasks = {"wsc273": entry([w for w, _ in pairs], gold, shared=True)}
    return {"key": key, "items": items, "tasks": tasks}


GLOBAL_MMLU_LANGS = [
    "cs",
    "de",
    "el",
    "en",
    "es",
    "fr",
    "it",
    "lt",
    "nl",
    "pl",
    "pt",
    "ro",
    "sr",
    "sv",
    "tr",
    "uk",
]
MMMLU_LANGS = ["de_de", "es_la", "fr_fr", "it_it", "pt_br"]


def mmlu(lm: LmEval) -> dict:
    # Global-MMLU's English is MMLU's test split, and its languages share
    # sample_id; MMMLU translates the same questions in the same order. lm-eval
    # runs each as a task per subject.
    key = ["subject", "question", "choices"]
    loaded = lm.manager.load_task_or_group(["mmlu"])
    subjects = sorted(
        name.removeprefix("mmlu_")
        for name in _leaf_names(loaded)
        if name.startswith("mmlu_")
    )
    anchor, starts = [], {}
    for subject in subjects:
        starts[subject] = len(anchor)
        anchor += lm.docs(f"mmlu_{subject}")[0]
    items = items_of(key, anchor, "MMLU", repeats=True)

    def full(subject: str, within: list[int | None]) -> list[int | None]:
        positions = [None] * len(items)
        positions[starts[subject] : starts[subject] + len(within)] = within
        return positions

    tasks = {}
    global_langs = [
        x for x in GLOBAL_MMLU_LANGS if lm.languages is None or x in lm.languages
    ]
    mmmlu_langs = [x for x in MMMLU_LANGS if lm.languages is None or x in lm.languages]
    for subject in subjects:
        english = lm.docs(f"mmlu_{subject}")[0]
        global_en = lm.docs(f"global_mmlu_full_en_{subject}")[0]
        same = len(global_en) == len(english) and all(
            g["question"] == e["question"]
            and [g["option_a"], g["option_b"], g["option_c"], g["option_d"]]
            == e["choices"]
            for g, e in zip(global_en, english, strict=True)
        )
        if not same:
            raise AlignmentError(f"MMLU: Global-MMLU's English {subject} is not MMLU's")
        for lang in global_langs:
            task = f"global_mmlu_full_{lang}_{subject}"
            docs, gold = lm.docs(task)
            within = by_field("MMLU", global_en, "sample_id", task, docs)
            if None in within:
                raise AlignmentError(f"MMLU: {task} lacks some of {subject}'s questions")
            tasks[task] = entry(full(subject, within), gold)
        for lang in mmmlu_langs:
            task = f"mmmlu_{lang}_{subject}"
            docs, gold = lm.docs(task)
            within = by_position(
                "MMLU",
                english,
                task,
                docs,
                lambda d: d["Answer"] if "Answer" in d else "ABCD"[d["answer"]],
                tolerate=0.02,
            )
            tasks[task] = entry(full(subject, within), gold)
    return {"key": key, "items": items, "tasks": tasks}


def _leaf_names(loaded: dict) -> list[str]:
    names, stack = [], [loaded]
    while stack:
        for name, task in stack.pop().items():
            if isinstance(task, dict):
                stack.append(task)
            else:
                names.append(name)
    return names


LM_EVAL_TABLES = {
    "MMLU": mmlu,
    "XNLI": xnli,
    "GSM8K": gsm8k,
    "GlobalMGSM": global_mgsm,
    "PolyMath": polymath,
    "CommonsenseQA": commonsense_qa,
    "Xcsqa": xcsqa,
    "Flores200": flores200,
    "WinogradSchemaChallenge": winograd_schema_challenge,
    "XWinograd": xwinograd,
}


# --- lighteval ---------------------------------------------------------------


class Lighteval:
    def __init__(self):
        self.custom = (
            REPO / "oellm/resources/custom_lighteval_tasks/reasoning/reasoning.py"
        )

    def docs(self, name: str) -> tuple[list[dict], list[str]]:
        """A task's dataset rows by position (lighteval's doc id), and the
        answer of the document built from each, as its details save it."""
        from lighteval.tasks.lighteval_task import LightevalTask
        from lighteval.tasks.registry import Registry

        registry = Registry(
            tasks=f"{name}|0", load_multilingual=True, custom_tasks=str(self.custom)
        )
        tasks = registry.load_tasks()
        LightevalTask.load_datasets(tasks)
        (task,) = tasks.values()
        (split,) = task.evaluation_split
        rows = [dict(r) for r in task.dataset[split]]
        gold = [None] * len(rows)
        try:
            docs = task.eval_docs()
        except AttributeError as e:
            # Its prompt template lacks the language's words (Latvian's).
            if "Translation for" not in str(e):
                raise
            raise CannotBuild(f"{name}: {' '.join(str(e).split())}") from e
        for doc in docs:
            gold[int(doc.id)] = _canonical(
                doc.gold_index
                if isinstance(doc.gold_index, int)
                else list(doc.gold_index)
            )
        if None in gold:
            raise AlignmentError(f"{name}: lighteval skips some of its rows")
        return rows, gold


BELEBELE_CF_LANGS = [
    "bul_Cyrl",
    "hrv_Latn",
    "ces_Latn",
    "dan_Latn",
    "nld_Latn",
    "eng_Latn",
    "est_Latn",
    "fin_Latn",
    "fra_Latn",
    "deu_Latn",
    "ell_Grek",
    "hun_Latn",
    "ita_Latn",
    "lvs_Latn",
    "lit_Latn",
    "mlt_Latn",
    "pol_Latn",
    "por_Latn",
    "ron_Latn",
    "slk_Latn",
    "slv_Latn",
    "spa_Latn",
    "swe_Latn",
    "nob_Latn",
    "eus_Latn",
    "cat_Latn",
]


def belebele(le: Lighteval) -> dict:
    # facebook/belebele keeps each language's questions in an order of its own;
    # a question is its passage's link and its number there.
    key = ["link", "question_number"]
    anchor, _ = le.docs("belebele_eng_Latn_cf")
    for row in anchor:
        row["item"] = json.dumps([row[f] for f in key])
    tasks = {}
    for lang in BELEBELE_CF_LANGS:
        try:
            rows, gold = le.docs(f"belebele_{lang}_cf")
        except CannotBuild as e:
            # Never run, so never filtered: left out of the table.
            print(f"skipped {e}", flush=True)
            continue
        for row in rows:
            row["item"] = json.dumps([row[f] for f in key])
        positions = by_field(
            "Belebele",
            anchor,
            "item",
            f"belebele_{lang}_cf",
            rows,
            lambda r: r["correct_answer_num"],
        )
        if None in positions:
            raise AlignmentError(f"Belebele: belebele_{lang}_cf lacks some questions")
        tasks[f"belebele_{lang}_cf"] = entry(positions, gold)
    return {"key": key, "items": items_of(key, anchor, "Belebele"), "tasks": tasks}


def gpqa(le: Lighteval) -> dict:
    # LumiOpen/GPQA-FI translates GPQA Diamond in its order, keeping its
    # (English) domain and subdomain; numeric answers read the same.
    key = ["Question", "Correct Answer"]
    anchor, _ = le.docs("gpqa_diamond:en")
    rows, gold = le.docs("gpqa_diamond:fi")

    def as_number(row):
        try:
            return number(row["Correct Answer"].replace("\u2013", "-"))
        except AlignmentError:
            return None

    positions = by_position(
        "GPQA",
        anchor,
        "gpqa_diamond:fi",
        rows,
        lambda r: (r["High-level domain"], r["Subdomain"]),
    )
    differ = [
        i
        for i, (a, r) in enumerate(zip(anchor, rows, strict=True))
        if None not in (as_number(a), as_number(r)) and as_number(a) != as_number(r)
    ]
    if differ:
        raise AlignmentError(
            f"GPQA: gpqa_diamond:fi's numeric answers {differ[:10]} differ"
        )
    return {
        "key": key,
        "items": items_of(key, anchor, "GPQA"),
        "tasks": {"gpqa_diamond:fi": entry(positions, gold)},
    }


LIGHTEVAL_TABLES = {"Belebele": belebele, "GPQA": gpqa}


# Tables that can be built a few languages at a time (--languages).
IN_PARTS = {"MMLU"}


def merged(path: Path, part: dict) -> dict:
    """The table at *path* with the tasks of *part*, built from the same items."""
    if not path.exists():
        return part
    table = json.loads(path.read_text())
    if (table["key"], table["items"]) != (part["key"], part["items"]):
        raise AlignmentError(f"{path.name} was built from other items; rebuild it whole")
    return table | {"tasks": table["tasks"] | part["tasks"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("harness", choices=["lm-eval", "lighteval"])
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--languages",
        help="comma-separated languages of the tasks to build (de,de_de), merged "
        f"into the table: for {', '.join(sorted(IN_PARTS))}, too slow to build whole "
        "in one 30-minute job",
    )
    parser.add_argument("benchmarks", nargs="*")
    # Options may come before or after the benchmarks.
    args = parser.parse_intermixed_args()
    builders, harness = (
        (LM_EVAL_TABLES, LmEval)
        if args.harness == "lm-eval"
        else (LIGHTEVAL_TABLES, Lighteval)
    )
    unknown = set(args.benchmarks) - builders.keys()
    if unknown:
        parser.error(f"no {args.harness} table for {sorted(unknown)}")
    if args.languages and not (args.benchmarks and set(args.benchmarks) <= IN_PARTS):
        parser.error(f"--languages builds part of {', '.join(sorted(IN_PARTS))} only")
    languages = set(args.languages.split(",")) if args.languages else None
    if languages and languages - set(GLOBAL_MMLU_LANGS) - set(MMMLU_LANGS):
        parser.error(
            f"no MMLU tasks in {sorted(languages - set(GLOBAL_MMLU_LANGS) - set(MMMLU_LANGS))}"
        )
    loader = harness()
    loader.languages = languages
    failed = []
    for benchmark, build in builders.items():
        if args.benchmarks and benchmark not in args.benchmarks:
            continue
        path = TABLES / f"{benchmark}.json"
        try:
            part = build(loader)
            table = merged(path, part) if args.languages else part
        except AlignmentError as e:
            print(f"FAILED {benchmark}: {e}", flush=True)
            failed.append(benchmark)
            continue
        text = json.dumps(table, indent=1, ensure_ascii=False) + "\n"
        if args.check:
            same = path.exists() and path.read_text() == text
            print(f"{'ok' if same else 'DIFFERS'} {path.name}", flush=True)
            if not same:
                failed.append(benchmark)
        else:
            TABLES.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            print(f"wrote {path.name}", flush=True)
        for task, e in part["tasks"].items():
            held = sum(p is not None for p in e.get("positions") or [0] * e["documents"])
            print(
                f"    {task}: {held} of {len(table['items'])} items, {e['documents']} documents",
                flush=True,
            )
    if failed:
        sys.exit(f"failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
