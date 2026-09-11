"""Actual-library CPU checks for both installed backends and EOS policies."""
import json
import os
os.environ['OMP_NUM_THREADS']='1'
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast, GPT2Config, GPT2LMHeadModel, GenerationConfig, LogitsProcessor
from lm_eval.models.huggingface import HFLM
from lm_eval.models.vllm_causallms import VLLM
from vllm import SamplingParams
import oellm_ifeval

torch.set_num_threads(1)
tokenizer=PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({'<unk>':0,'<bos>':1,'<eos>':2,'A':3},unk_token='<unk>')),
                                  unk_token='<unk>',bos_token='<bos>',eos_token='<eos>',pad_token='<eos>')
lm=HFLM.__new__(HFLM); lm.tokenizer=tokenizer; lm._device=torch.device('cpu'); lm.mixed_precision_dtype=None
lm._model=GPT2LMHeadModel(GPT2Config(vocab_size=4,n_positions=1300,n_embd=8,n_layer=1,n_head=1,bos_token_id=1,eos_token_id=2)).eval()
lm._model.generation_config=GenerationConfig(max_new_tokens=1)
class Forced(LogitsProcessor):
    def __call__(self,ids,scores):
        scores.fill_(-float('inf')); first=ids.shape[1]==2
        scores[0,2 if first else 3]=0
        scores[1,3 if first else 2]=0
        return scores
ids=torch.tensor([[1,3],[1,3]])
kwargs=dict(do_sample=False,attention_mask=torch.ones_like(ids),logits_processor=[Forced()])
stopped=lm._model_generate(ids,max_length=6,stop=['<eos>'],**kwargs)
assert stopped[0,2:].tolist()==[2,2],stopped
assert lm._model.generation_config.max_new_tokens==1
# Disable EOS explicitly: the request's four-token budget must override the
# checkpoint's one-token default, while explicit request overrides still win.
longer=lm._model_generate(ids,max_length=6,stop=[],eos_token_id=None,**kwargs)
assert longer.shape==(2,6)
explicit=lm._model_generate(ids,max_length=6,stop=[],eos_token_id=None,max_new_tokens=2,**kwargs)
assert explicit.shape==(2,4)
# Same actual generate implementation; the explicit policy disables token and
# text EOS, rather than depending on when another batch row happens to finish.
oellm_ifeval.install('hf')
continued=lm._model_generate(ids,max_length=1282,stop=['<eos>'],**kwargs)
assert continued.shape==(2,1282)
assert continued[0,2:5].tolist()==[2,3,3]
normal,stops,limit=VLLM.modify_gen_kwargs({'do_sample':False},'<eos>',1280)
assert SamplingParams(**normal,stop=stops,max_tokens=limit).temperature==0
cont,stops,limit=oellm_ifeval.vllm_kwargs(normal,stops,'<eos>',limit)
p=SamplingParams(**cont,stop=stops,max_tokens=limit)
assert p.ignore_eos and not p.stop and not p.stop_token_ids and p.skip_special_tokens and p.max_tokens==1280
print(json.dumps(dict(hf_eos=True,hf_continuation_tokens=1280,vllm_greedy=True,vllm_continuation=True,hf_task_budget_precedence=True)))
