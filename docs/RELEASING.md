# Releasing

Versions follow [PEP 440](https://peps.python.org/pep-0440/). `main` always carries the
next release as a beta (`0.3.0b1`, `0.3.0b2`, ...). Releases are tags on `main`
(`v0.3.0`), and the `stable` branch points at the latest release. Nobody commits to
`stable`; it only moves forward to the next release tag.

The old `v0.01` and `v0.02` tags are evaluation-protocol snapshots on a side branch, not
releases. PEP 440 reads them as 0.1 and 0.2, which is why the release line starts at 0.3.

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
uv version --bump beta --frozen        # 0.3.0b1 -> 0.3.0b2
```

After it merges, tag the merge commit:

```bash
git tag v0.3.0b2 <merge commit> && git push origin v0.3.0b2
```

The changelog stays as it is: at the beta's tag, `## [Unreleased]` lists exactly what the
beta contains.

## Releasing

1. Open a PR that runs `uv version --bump stable --frozen` (`0.3.0b2` becomes `0.3.0`)
   and, in `CHANGELOG.md`, renames `## [Unreleased]` to `## [0.3.0] - YYYY-MM-DD`, removes
   its empty headings, and adds a new `## [Unreleased]` with the empty headings above it.
   PRs still open at that point will add their line under the old section when they
   rebase; move it up before merging them.
2. After it merges, tag it and move `stable`:

   ```bash
   git tag v0.3.0 <merge commit>
   git push origin v0.3.0
   git push origin 'v0.3.0^{commit}:refs/heads/stable'
   ```

3. On GitHub, create a release from the tag and paste the version's section of
   `CHANGELOG.md` as its description.
4. Open a PR that starts the next beta: `uv version --bump minor --bump beta --frozen`
   (`0.3.0` becomes `0.4.0b1`).

For the first release only, also change the install command in the README to
`git+https://github.com/OpenEuroLLM/oellm-eval.git@stable`.
