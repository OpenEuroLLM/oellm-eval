# Releasing

This is the reference for how oellm-eval is versioned and released: which number to bump,
how to bump it, and every step of cutting a release.

Versions follow [PEP 440](https://peps.python.org/pep-0440/). `main` always carries the
next release as a beta (`0.4.0b1`, `0.4.0b2`, ...). Releases are tags on `main`
(`v0.3.0`), and the `stable` branch points at the latest release. Nobody commits to
`stable`; it only moves forward to the next release tag.

The old `v0.01` and `v0.02` tags are evaluation-protocol snapshots on a side branch, not
releases. PEP 440 reads them as 0.1 and 0.2, which is why the release line starts at 0.3.

## Channels

| Ref | Version | Use it for |
|---|---|---|
| `main` | next release as a beta, e.g. `0.4.0b1` | Development, and fixes or benchmarks not yet released. Any commit may change results. |
| `stable` | latest release, e.g. `0.3.0` | The default install in the README. |
| `vX.Y.Z` tags | fixed | Evaluation campaigns and papers: pin one so every run measures the same thing. |

Pull requests always target `main`. Releasing never holds `main` back; it marks a commit
on `main` and moves `stable` to it.

## Which number goes up

Until 1.0, the minor number goes up when a change can alter results or break a command
that used to work:

- a task group gains or loses tasks or languages, or changes shots, metric, prompt or dataset;
- a fix to scoring or answer extraction;
- a CLI option, output file or result column is renamed or removed.

New task groups, new clusters, fixes that leave every number unchanged, and docs only
need a patch release.

Pull requests do not edit the version. `oellm-eval --version` reports the commit a beta
was installed from, so a per-PR bump would add nothing, and every PR editing the same line
would conflict.

The version lives only in `pyproject.toml`. Change it with `uv version`, never by hand:

| Command | Effect |
|---|---|
| `uv version --bump beta --frozen` | `0.4.0b1` → `0.4.0b2` |
| `uv version --bump stable --frozen` | `0.4.0b2` → `0.4.0` |
| `uv version --bump minor --bump beta --frozen` | `0.4.0` → `0.5.0b1` |
| `uv version 0.3.1 --frozen` | sets an exact version (used for patch releases, below) |

## When to release

Release when `## [Unreleased]` holds changes people should get by default, in particular
anything under "Changed results" that has been tested. Release often, roughly every two
weeks while there are changes, so `stable` never lags far behind `main`.

Before merging a batch of pull requests that change results, release what `main` has now.
Results made before and after the change then come from different versions instead of the
same version at different times.

## Changelog

Every pull request a user could notice adds one line to `CHANGELOG.md` under
`## [Unreleased]`, under the matching heading. Anything that can change results goes
under "Changed results", even when it is also a fix. CI, test and refactoring changes need
no entry. Reviewers check the entry like any other part of the diff.

`.gitattributes` marks the changelog for git's union merge, so two branches that each add
a line rebase onto each other without a conflict.

## Cutting a beta

When testers need a fixed point to pin, open a PR that only runs:

```bash
uv version --bump beta --frozen        # 0.4.0b1 -> 0.4.0b2
```

After it merges, tag the merge commit:

```bash
git tag v0.4.0b2 <merge commit> && git push origin v0.4.0b2
```

The changelog stays as it is: at the beta's tag, `## [Unreleased]` lists exactly what the
beta contains.

## Releasing

1. **Release PR.** Open a PR that only does the following:
   - `uv version --bump stable --frozen` (`0.4.0b2` becomes `0.4.0`). For a patch release
     (nothing under "Changed results" and no broken commands since the last release), set
     the version instead: `uv version 0.3.1 --frozen`.
   - In `CHANGELOG.md`, rename `## [Unreleased]` to `## [0.4.0] - YYYY-MM-DD`, remove its
     empty headings, and add a new `## [Unreleased]` with the empty headings above it.

   PRs still open at that point will add their line under the old section when they
   rebase; move it up before merging them.
2. **Tag and move `stable`.** After the release PR merges:

   ```bash
   git fetch origin
   git tag v0.4.0 <merge commit>
   git push origin v0.4.0
   git push origin 'v0.4.0^{commit}:refs/heads/stable'
   ```

3. **GitHub release.** Create it from the tag, with the version's changelog section as its
   description:

   ```bash
   awk '/^## \[0.4.0\]/{f=1;next} /^## \[/{f=0} f' CHANGELOG.md > /tmp/notes.md
   gh release create v0.4.0 --title v0.4.0 --notes-file /tmp/notes.md
   ```

4. **Next beta.** Open a PR that only runs `uv version --bump minor --bump beta --frozen`
   (`0.4.0` becomes `0.5.0b1`).
5. **Announce** the release on Slack with a link to it, and say explicitly if it has
   "Changed results".

## Checklist

- [ ] Changelog entries under `## [Unreleased]` are complete, results-changing ones under "Changed results"
- [ ] Release PR merged: version bumped, changelog section dated, new empty `## [Unreleased]`
- [ ] Tag `vX.Y.Z` pushed on the release PR's merge commit
- [ ] `stable` moved to the tag
- [ ] GitHub release created from the changelog section
- [ ] Next-beta PR merged
- [ ] Release announced
