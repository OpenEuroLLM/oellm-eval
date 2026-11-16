import os
import sys

# noreval_corpus.py lives at the repository root, one level up
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from noreval_corpus import bleu_stderr, chrf_stderr, mean_bleu, mean_chrf  # noqa: F401


def process_results(doc, results):
    # results[0] holds the K sampled generations; pair each with the reference
    pairs = [(doc["response"], prediction) for prediction in results[0]]
    return {"bleu": pairs, "chrf": pairs, "bleu_stderr": pairs, "chrf_stderr": pairs}
