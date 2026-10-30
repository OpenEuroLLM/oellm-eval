# Changelog

Notable changes to oellm-eval, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[PEP 440](https://peps.python.org/pep-0440/) as described in
[docs/RELEASING.md](docs/RELEASING.md). Changes before 0.3.0b1 are recorded only in the
git history.

"Changed results" lists everything that can make a task's numbers differ from the
previous version: tasks, languages, shots, metrics, prompts, datasets and scoring. Check
it before comparing runs made with different versions.

## [Unreleased]

<!-- One line per change, under the matching heading. Empty headings are removed at release. -->

### Changed results

- SIB-200 (`sib200-eu`) runs 10-shot instead of 0-shot, with few-shot examples taken from the start of the train split. Few-shot runs previously drew them from the label-sorted test split, so every example was science/technology and scores sat at 0.25, that label's share (#137).
- LUMI container: `lm-eval[api,tasks]` 0.4.13 with `datasets>=4` and `transformers<5.20`, PolyMath's scoring dependencies, and patches so `dclm-core-22`'s `agieval_lsat_ar`, `squadv2` and `coqa` run at n_shot>0. Applies once the image is rebuilt (#130).

### Added

- `bpb-corpus` task group: bits-per-byte on WikiText-2 for comparing small models.
- `bpb-flores-eu` task group: bits-per-byte on FLORES-200 devtest in the 36 OpenEuroLLM target languages.
- `bpb-mc-en` task group: gold-answer bits-per-byte on seven English multiple-choice tasks.
- Template and docs for bits-per-byte on your own text (`custom_lm_eval_tasks/corpus_bpb`).
- lm-eval runs pass `--write_out` by default, and `--log_samples` saves every sample when requested.
- `collect --exclude_samples` also reports scores without blocklisted (e.g. contaminated) test rows, recomputed from the saved samples, in every language and in benchmarks holding copies of them (lm-eval, lighteval and evalchemy's MATH500).
- Task group `decontaminated-agenttrove-0.0` runs the lm-eval tasks holding rows the decontamination pipeline marked, without those rows (`--samples`); `collect` reports them with `subset` `decontaminated:agenttrove-0.0`.

### Changed

### Fixed

- `collect` no longer reports lm-eval's `sample_len` document count as a metric for tasks outside the task groups or with `--fetch_all_metrics`.
- `--limit` now shortens evalchemy's benchmarks (`reasoning`: MATH500, GPQADiamond, AIME24, ...). They ignored it and ran in full; they now get `--debug`, which runs the first 2 questions (10 for LiveCodeBench) whatever the limit, and `schedule` warns about it (#154).

### Removed

## [0.3.0] - 2026-10-20

### Added

- `oellm-eval --version` prints the version, the git commit a `git+` install came from,
  and the versions of the evaluation harnesses and the libraries they depend on.
- Versioned releases: `main` carries the next release as a beta (starting at 0.3.0b1),
  releases are git tags, and the `stable` branch follows the latest release. See
  [docs/RELEASING.md](docs/RELEASING.md).
- Issue forms. A bug report asks for the output of `oellm-eval --version`, the cluster,
  the runtime, the command and the logs. Blank issues are disabled.
- This changelog, and a pull request template that asks for an entry in it.
