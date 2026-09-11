"""Exercise actual HF generation with two rows that finish at different steps."""
import argparse
import json
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel, GenerationConfig, LogitsProcessor
from lm_eval.models.huggingface import HFLM

p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--expect-fixed',action='store_true');a=p.parse_args()
tokenizer=AutoTokenizer.from_pretrained(a.model,trust_remote_code=True)
letter=tokenizer.encode('A',add_special_tokens=False)[0]
lm=HFLM.__new__(HFLM);lm.tokenizer=tokenizer;lm._device=torch.device('cpu');lm.mixed_precision_dtype=None
tiny=GPT2LMHeadModel(GPT2Config(vocab_size=len(tokenizer),n_positions=16,n_embd=8,n_layer=1,n_head=1,bos_token_id=1,eos_token_id=2)).eval()
tiny.generation_config=GenerationConfig()  # Reproduce exported file with missing EOS.
lm._model=tiny


class ForcedTokens(LogitsProcessor):
    def __call__(self,input_ids,scores):
        scores.fill_(-float('inf'))
        first=input_ids.shape[1]==2
        scores[0,2 if first else letter]=0
        scores[1,letter if first else 2]=0
        return scores


inputs=torch.tensor([[1,letter],[1,letter]])
tokens=lm._model_generate(inputs,max_length=6,stop=[tokenizer.eos_token],do_sample=False,
                          attention_mask=torch.ones_like(inputs),logits_processor=[ForcedTokens()])
first=tokens[0,2:].tolist()
assert first==([2,2] if a.expect_fixed else [2,letter]),first
if a.expect_fixed:
    for configured,explicit,expected in [(None,{},2),([2,3],{},'absent'),(None,{'eos_token_id':7},7),(None,{'eos_token_id':None},None)]:
        model=SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=configured),generate=Mock(return_value=inputs))
        lm._model=model
        lm._model_generate(inputs,max_length=6,stop=[tokenizer.eos_token],do_sample=False,**explicit)
        assert model.generate.call_args.kwargs.get('eos_token_id','absent')==expected
print(json.dumps(dict(fixed=a.expect_fixed,first_row_tokens=first,text=tokenizer.decode(first,skip_special_tokens=True),passed=True)))
