"""Calibrated zero-shot GPQA prompt and seeded shuffled options; no exclusions."""
import random
try:
    from evaluation_policies import choice
except ImportError:
    from oellm.evaluation_policies import choice


def docs(dataset):
    rng = random.Random(42)
    def process(doc):
        answers = [doc["Correct Answer"]] + [doc[f"Incorrect Answer {i}"] for i in (1, 2, 3)]
        rng.shuffle(answers)
        return dict(choices=answers, answer="ABCD"[answers.index(doc["Correct Answer"])])
    return dataset.map(process, load_from_cache_file=False)


def text(doc):
    return ("Question: " + doc["Question"] + "\nChoices:\n" +
            "\n".join(f"({letter}) {answer}" for letter, answer in zip("ABCD", doc["choices"], strict=True)) +
            '\nGive step by step reasoning before you answer, and when you\'re ready to answer, please use the format "The correct answer is (insert answer here)":\n')


def results(doc, results):
    import re
    letters = re.findall(r"\b(A|B|C|D)\b", results[0].upper())
    return dict(exact_match=float(choice(results[0]) == doc["answer"]),
                legacy_match=float(bool(letters) and letters[-1] == doc["answer"]))
