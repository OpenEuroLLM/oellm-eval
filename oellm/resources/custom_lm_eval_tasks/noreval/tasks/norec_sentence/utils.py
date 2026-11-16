import os
import sys

# noreval_corpus.py lives at the repository root, one level up
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from noreval_corpus import multi_f1, multi_f1_stderr  # noqa: F401
