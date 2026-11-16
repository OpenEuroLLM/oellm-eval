### Evaluation example

ERRANT F0.5 is computed by the harness itself, as a corpus-level metric
(`utils.errant_agg`). ERRANT must be importable from the same environment that
runs `lm_eval`; see the installation instructions
[here](https://github.com/chrisjbryant/errant/tree/main), and note that it also
needs the `en_core_web_sm` spaCy model:

```bash
pip install "errant==3.0.0"
python -m spacy download en_core_web_sm
```

Then run the task normally — no `--predict_only`:

```bash
lm_eval \
  --model hf \
  --model_args pretrained=AI-Sweden-Models/Llama-3-8B \
  --include_path path/to/noreval/tasks \
  --tasks ask_gec_nob \
  --output results/ask_gec_nob/0-shot/ \
  --log_samples \
  --show_config \
  --write_out \
  --batch_size auto \
  --num_fewshot 0
```

`errant_f05` is reported in `results.json` alongside every other task, with its
standard error under `errant_f05_stderr`: the expensive ERRANT corpus pass runs
exactly once (cached on the item content), per-sentence TP/FP/FN counts are
recovered from the M2 files and verified against `errant_compare`'s corpus
score, and the stderr is a seeded bootstrap over documents — each document's K
sampled corrections are resampled together.

If ERRANT or the spaCy model is missing, the metric is reported as `NaN` and the
reason is logged, so an incomplete environment costs this one metric rather than
the whole run's results.

**Scoring the predictions separately**

The two-step workflow still works if you prefer to score outside the eval job —
pass `--predict_only` above, then:

```bash
python3 ask_gec/errant.py --fpath results/ask_gec_nob/0-shot/AI-Sweden-Models__Llama-3-8B/samples_ask_gec_nob_p0_2025-01-28T01-08-13.454441.jsonl --out_fdir results/ask_gec_nob/0-shot/AI-Sweden-Models__Llama-3-8B/
```

The results are saved as `..._errant.json`.

Note: utils.py` and this `errant.py` normalise predictions identically
(`.replace("\n\n", "\n")`), so an empty prediction is a blank
line in both and is scored as deleting the sentence.

They differ in one respect. ERRANT reads the parallel files with `zip` over
file handles, which never yields a trailing empty line, so `errant.py` loses
the last sentence whenever the final prediction is empty and `errant_compare`
then fails on `assert len(hyp_m2) == len(ref_m2)`. `utils.py` terminates every
line instead of joining with newlines, so it scores those files correctly.
