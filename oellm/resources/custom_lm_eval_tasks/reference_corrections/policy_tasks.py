"""Opt-in tasks using installed harness prompts and versioned answer policies."""
from pathlib import Path
import lm_eval
from lm_eval.api.task import ConfigurableTask
from lm_eval.tasks.squadv2.task import SQuAD2 as OriginalSQuAD2
import yaml

try:
    from evaluation_policies import gsm8k_results, squad_class
except ModuleNotFoundError:
    from oellm.evaluation_policies import gsm8k_results, squad_class

SQuAD2 = squad_class(OriginalSQuAD2)


class GSM8KCoTNumeric(ConfigurableTask):
    VERSION = 1

    def __init__(self, config=None):
        path = Path(lm_eval.__file__).parent/'tasks/gsm8k/gsm8k-cot.yaml'
        definition = yaml.safe_load(path.read_text())
        if definition['task'] != 'gsm8k_cot' or len(definition['fewshot_config']['samples']) < 4:
            raise ValueError('Installed GSM8K CoT prompt source changed')
        definition.update(task='gsm8k_cot_numeric_v1', num_fewshot=4,
                          process_results=gsm8k_results,
                          metadata=dict(version=1.0, protocol='reference-corrections-v1'))
        definition['generation_kwargs'].update(temperature=0, max_gen_toks=1024)
        definition['metric_list'] = [dict(metric=k,aggregation='mean',higher_is_better=True)
                                     for k in ('exact_match','numeric_match')]
        super().__init__(config=definition)
