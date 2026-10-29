"""Build the list of documents a decontaminated task group leaves out.

Finds the documents matching the marked rows (agenttrove-0.0's dropped/) with
the matching of ``collect --exclude_samples`` and writes
oellm/resources/decontaminated/<name>.json; see oellm/decontaminated.py.

Run in the eval container, with lm-eval's Python and HF_ALLOW_CODE_EVAL=1
(loading lm-eval's mbpp checks HF's code_eval metric):
    python scripts/build_decontaminated_samples.py <name> <marked rows dir> [--check]
With --check, the rebuilt list is compared with the repository's, not written.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from oellm import sample_exclusion as se  # noqa: E402
from oellm.sample_exclusion import item_key  # noqa: E402

LISTS = REPO / "oellm/resources/decontaminated"


def held_by(table: dict, task: str, lm_eval_tasks) -> list[str]:
    """The lm-eval tasks of alignment *table* that are *task* or its subtasks;
    not lighteval's (belebele_bul_Cyrl_cf is no subtask of belebele_bul_Cyrl)."""
    return [
        t
        for t in table["tasks"]
        if t in lm_eval_tasks and (t == task or t.startswith(f"{task}_"))
    ]


def leaves(manager, task: str) -> dict:
    """A task's leaf tasks, by name, as lm-eval runs it."""
    out, stack = {}, [manager.load([task])["tasks"]]
    while stack:
        for name, t in stack.pop().items():
            if isinstance(t, dict):
                stack.append(t)
            else:
                out[name] = t
    return out


def documents(task) -> tuple[list[dict], list[str]]:
    """A task's documents (doc_id is the position) and their answers, as lm-eval
    saves them (str of the target)."""
    docs = list(task.eval_docs)
    return docs, [str(task.doc_to_target(d)) for d in docs]


def build(name: str, rows_dir: Path) -> tuple[dict, list[str]]:
    from lm_eval.tasks import TaskManager

    manager = TaskManager(include_path=str(REPO / "oellm/resources/custom_lm_eval_tasks"))
    match_fields = se.load_match_fields()
    blocklist = se.load_blocklist(rows_dir, match_fields)
    task_map = se.load_task_map()
    alignments = se.load_alignments()
    normalized = se.load_normalized()

    excluded: dict[str, set[int]] = defaultdict(set)
    loaded: dict[str, object] = {}
    notes = []
    for benchmark, rows in sorted(blocklist.items()):
        mapped = [t for t in task_map.get(benchmark, []) if t in manager.task_index]
        if not mapped:
            notes.append(f"{benchmark}: {len(rows)} rows, no lm-eval task; left out")
            continue
        table = alignments.get(benchmark, {"tasks": {}})
        matched = [False] * len(rows)

        aligned = [leaf for t in mapped for leaf in held_by(table, t, manager.task_index)]
        items = se.table_items(table, rows) if aligned else []
        for leaf in aligned:
            held = se.held_items(table["tasks"][leaf])
            for r, indices in enumerate(items):
                for i in indices:
                    if i in held:
                        excluded[leaf].add(held[i])
                        matched[r] = True

        direct = {}
        for t in mapped:
            if not held_by(table, t, manager.task_index):
                direct |= leaves(manager, t)
        loaded |= direct
        if direct:
            samples = {
                leaf: [
                    {"doc_id": i, "doc": json.loads(json.dumps(d, default=str))}
                    for i, d in enumerate(task.eval_docs)
                ]
                for leaf, task in direct.items()
            }
            found, flags = se._excluded_docs(
                rows,
                samples,
                benchmark,
                benchmark in normalized,
                benchmark in match_fields,
            )
            for leaf, doc_ids in found.items():
                excluded[leaf] |= doc_ids
            matched = [a or b for a, b in zip(matched, flags, strict=True)]
        notes.append(
            f"{benchmark}: {len(rows)} rows, {matched.count(False)} match no document"
        )

    tasks = {}
    for leaf in sorted(excluded):
        if not excluded[leaf]:
            continue
        task = loaded.get(leaf) or leaves(manager, leaf)[leaf]
        docs, answers = documents(task)
        for benchmark, table in alignments.items():
            entry = table["tasks"].get(leaf)
            if entry and (len(answers), item_key(answers)) != (
                entry["documents"],
                entry["answers"],
            ):
                raise SystemExit(
                    f"{leaf}: its documents are not the ones the {benchmark} "
                    "alignment table was built from; rebuild it first"
                )
        kept = [i for i in range(len(docs)) if i not in excluded[leaf]]
        tasks[leaf] = {
            "documents": len(docs),
            "excluded": sorted(excluded[leaf]),
            "kept_answers": item_key([answers[i] for i in kept]),
        }
        # Few-shot examples from the scored split can be marked documents.
        config = task.config
        scored = config.test_split or config.validation_split
        if config.fewshot_split == scored:
            notes.append(
                f"{leaf}: few-shot examples come from the documents it scores "
                f"({scored}), so they can be marked ones"
            )
    return {"name": name, "tasks": tasks}, notes


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("name")
    parser.add_argument("rows_dir", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    table, notes = build(args.name, args.rows_dir)
    for note in notes:
        print(note, flush=True)
    for leaf, entry in table["tasks"].items():
        print(f"    {leaf}: {len(entry['excluded'])} of {entry['documents']} left out")
    path = LISTS / f"{args.name}.json"
    text = json.dumps(table, indent=1) + "\n"
    if args.check:
        same = path.exists() and path.read_text() == text
        print(f"{'ok' if same else 'DIFFERS'} {path.name}")
        if not same:
            sys.exit(1)
    else:
        LISTS.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        print(f"wrote {path.name}")


if __name__ == "__main__":
    main()
