# HumanEval and LiveCodeBench runtime acceptance

The portable recipe below prepares the validated grading corrections in a new Evalchemy directory. It preserves the original benchmark test bodies, examples, dataset release and default repeats. It addresses concurrent fork/filelock failures, the historical LiveCodeBench cache configuration name, and temporary-storage failures being scored as wrong answers. Existing images and source checkouts are not overwritten.

Start with Evalchemy reconstructed by `containers/prepare_vllm_sources.py` (the pinned `b321416135050aa6919b430dbadbd4cc43cc8c15` source). In the activated control environment:

```bash
python containers/code_grading/prepare.py \
  --source /path/to/sources/evalchemy \
  --out /path/to/new/evalchemy-code-grading
export EVALCHEMY_DIR=/path/to/new/evalchemy-code-grading
```

Use that directory as the evaluator source in your explicit scheduling profile and ensure it is mounted in the container. The scheduler supports a host Evalchemy checkout. `code_grading_manifest.json` records the eight corrected source files and recipe hashes. The default image still contains its pinned base Evalchemy; merely updating this repository does not retrofit the image. Preserve the manifest with each submission.

The container launcher now creates a private temporary directory on the compute host, binds it at `/tmp` and `/var/tmp`, sets temporary-directory environment variables, logs capacity and removes its owned directory on normal or error exit. `OELLM_EVAL_TMP_ROOT` may select a site-approved temporary-storage parent. JUPITER's contained default `/tmp` was a separate 16 MiB filesystem; a large shared filesystem did not increase that limit. The private bind fixes the tested failure without raising global container settings.

CPU runtime fixtures run inside the selected evaluation image. Set `HUMANEVAL_BENCH` to the prepared `eval/chat_benchmarks/HumanEval` directory and `HUMANEVAL_ORIGINAL` to the base Evalchemy source before running `containers/code_grading/runtime_checks_human.py`. For `runtime_checks_lcb.py`, set `LCB_ROOT` to the prepared source and `LCB_ORIGINAL` to the base source. Use the same private temporary bind as evaluation. On a JUPITER image with the reproduced 16 MiB default mount, `LCB_EXPECT_SMALL_TMP=1` checks that a real 32 MiB request fails as an infrastructure error; the regular fixture verifies the request succeeds with sufficient storage. These fixtures do not need GPUs.

Complete production-64k GPU acceptance passed. HumanEval retained and independently reproduced verdicts for all164 Python and158 shell problems. LiveCodeBench job1752517 completed all511 questions × six repeats in4592 seconds on four GPUs, with zero infrastructure-error verdicts. Correct counts were33,37,37,37,34,38; mean accuracy is7.04501%. All generated responses and six verdict files were retained, and their counts and aggregate score were independently verified. All eight reconstructed grading files match the evaluated source byte for byte. An earlier run whose grading hit ENOSPC is invalid even though Slurm reported successful completion.

The historical HF LiveCodeBench reference used one repeat and a different token limit, while this run uses six standard repeats. Its elapsed-time ratio is not a comparison of identical work. The full current reference campaign passed77 coverage/result audits; coverage checks on other generative benchmarks do not claim independent semantic regrading of every task.
