# HF stopping with incomplete generation configuration

Some checkpoint exports contain a `generation_config.json` that omits `eos_token_id`, although the model configuration and tokenizer define EOS. Transformers4.57.1 can retain the missing value. The harness's text stopping criterion waits for the whole batch; decoding strips special tokens before text-stop postprocessing. An individual response can therefore contain ordinary generated text after EOS, and grading can credit that text. A higher historical score is not evidence of better model responses under the intended EOS-stopping protocol.

The HF fallback patch supplies tokenizer EOS only when the model's generation configuration omits EOS and no explicit per-call EOS setting exists. It preserves configured EOS lists and explicit overrides, including `None`. vLLM already stops on EOS; this correction is in the HF adapter. The task, prompts, examples, generation token limit and grading code are unchanged.

New source reconstruction and image recipes include `patches/harness-hf-eos-fallback.patch`. For an existing image, prepare an isolated replacement file, then bind it read-only at that image's `lm_eval/models/huggingface.py` path:

```bash
python containers/hf_eos/prepare.py --source /path/to/original/huggingface.py --out /path/to/overlay/huggingface.py
```

`containers/hf_eos/runtime_checks.py --model /path/to/checkpoint --expect-fixed` runs inside the evaluation image. It uses a tiny CPU HF model with forced token sequences to test two rows that finish at different steps, and verifies configured/explicit EOS settings are preserved. The unpatched control omits `--expect-fixed` and reproduces text after EOS. Five dependency-light tests execute the fallback shipped in the source patch.

Both complete 64k IFEval GPU controls passed: legacy HF job1752585 and EOS-fixed HF job1752602, each541 prompts/834 instructions with retained responses and token IDs, using the same checkpoint, original HF image and four GPUs. Legacy HF satisfies300 strict instructions, corrected HF255, and vLLM256. Ordinary text after EOS occurs in239 legacy responses and zero corrected responses. Cutting the same legacy token sequences at their first EOS reduces the diagnostic count from300 to250, accounting for the apparent historical score advantage. Every saved official HF response exactly matches its decoded token trace; saved instruction verdicts reproduce both official aggregates. Diagnostic regrading uses fixed Python/langdetect seeds, while official scores preserve the unchanged benchmark protocol.

Five focused fallback tests, actual tiny-HF generation and override tests in both Transformers4.57.1 and5.15.1 images, and full source reconstruction pass. The reconstructed harness tree is `9b7bbc80fbec2a891efd0fa496707cdab6c96c10`; its HF adapter is byte-identical to the evaluated overlay. Installing this source recipe does not modify an already-built image: use the explicit overlay above or rebuild.
