"""INCLUDE Estonian: its validation split repeats items (21 distinct of 25) and all but two distinct
ones are also test items. The examples are the first five distinct validation items (4 of them are
test items), and test items identical to an example (question and all four options; some Estonian
questions are only a generic instruction) are not scored: 220 of 224. For the other FLAG languages
the first five validation items are distinct and none is a test item (checked 2026-11-18)."""
import datasets


def _key(d):
    return (d["question"].strip(), d["option_a"], d["option_b"], d["option_c"], d["option_d"])


def estonian_shots():
    val = datasets.load_dataset("CohereForAI/include-base-44", "Estonian", split="validation")
    shots, seen = [], set()
    for d in val:
        if _key(d) not in seen:
            seen.add(_key(d))
            shots.append(dict(d))
        if len(shots) == 5:
            break
    return shots


def drop_estonian_shots(dataset):
    shots = {_key(d) for d in estonian_shots()}
    return dataset.filter(lambda d: _key(d) not in shots)
