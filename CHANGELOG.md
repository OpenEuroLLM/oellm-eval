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

### Added

### Changed

### Fixed

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
