# lm-eval 0.4.12's few-shot builder calls doc_to_text(doc, cfg) and doc_to_target(doc, cfg), but SQuAD2
# takes only doc, so every num_fewshot > 0 run fails with a TypeError. Prompt, targets and metric unchanged.
from lm_eval.tasks.squadv2.task import SQuAD2 as _SQuAD2


class SQuAD2(_SQuAD2):
    def doc_to_text(self, doc, doc_to_text=None):
        return super().doc_to_text(doc)

    def doc_to_target(self, doc, doc_to_target=None):
        return super().doc_to_target(doc)
