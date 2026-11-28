import os
import sys

# noreval_metrics.py lives at the repository root, two levels up
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from noreval_metrics import make_process_results


def choices(doc):
    return [doc["correct_text"], doc["incorrect_text"]]


# doc_to_choice is [correct_text, incorrect_text], so the correct choice is always index 0
process_results = make_process_results(choices, lambda doc: [0])
