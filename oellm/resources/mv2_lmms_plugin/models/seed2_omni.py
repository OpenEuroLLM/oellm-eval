import os
import sys

import torch
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from tqdm import tqdm


@register_model("seed2_omni")
class Seed2Omni(lmms):
    def __init__(self, pretrained, device="cuda", batch_size=1, **kwargs):
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"
        assert int(batch_size) == 1, "seed2_omni only supports batch_size=1"

        mv2_dir = os.environ.get(
            "MV2_MULTIMODAL_DIR", "/e/project1/jureap59/raj3/mixturevitae2"
        )
        if mv2_dir not in sys.path:
            sys.path.insert(0, mv2_dir)
        from multimodal_processing.model_backends.seed2 import Backend, seed2_block

        self._seed2_block = seed2_block
        self._seed2 = Backend()
        self._seed2.load()

        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._device = torch.device(device)
        self._tokenizer = AutoTokenizer.from_pretrained(
            pretrained, trust_remote_code=True
        )
        self._model = (
            AutoModelForCausalLM.from_pretrained(
                pretrained, dtype=torch.bfloat16, trust_remote_code=True
            )
            .to(self._device)
            .eval()
        )
        im_end_id = self._tokenizer.convert_tokens_to_ids("<|im_end|>")
        self._eos_token_ids = [self._tokenizer.eos_token_id, im_end_id]
        self.batch_size_per_gpu = 1

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    def loglikelihood(self, requests):
        raise NotImplementedError("seed2_omni only supports generate_until tasks")

    def generate_until_multi_round(self, requests):
        raise NotImplementedError("seed2_omni does not support multi-round generation")

    def generate_until(self, requests):
        res = []
        pbar = tqdm(
            total=len(requests), disable=(self.rank != 0), desc="seed2_omni responding"
        )
        for request in requests:
            context, gen_kwargs, doc_to_visual, doc_id, task, split = request.args
            visuals = doc_to_visual(self.task_dict[task][split][doc_id])
            if not isinstance(visuals, list):
                visuals = [visuals]

            blocks = []
            for v in visuals:
                ids = self._seed2.encode(v.convert("RGB"))
                if ids:
                    blocks.append(self._seed2_block(ids))
            prompt = f"{''.join(blocks)} {context}".strip() if blocks else context
            chat = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n</think>\n"

            inputs = self._tokenizer(chat, return_tensors="pt").to(self._device)
            do_sample = gen_kwargs.get("temperature", 0) > 0
            with torch.no_grad():
                out = self._model.generate(
                    **inputs,
                    max_new_tokens=gen_kwargs.get("max_new_tokens", 256),
                    do_sample=do_sample,
                    temperature=gen_kwargs.get("temperature") if do_sample else None,
                    pad_token_id=self._tokenizer.eos_token_id,
                    eos_token_id=self._eos_token_ids,
                )
            new_tokens = out[0][inputs["input_ids"].shape[1] :]
            res.append(
                self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            )
            self.cache_hook.add_partial("generate_until", (context, gen_kwargs), res[-1])
            pbar.update(1)
        pbar.close()
        return res
