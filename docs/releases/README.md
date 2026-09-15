# Evaluation protocol releases

These tags identify benchmark procedures, independently of the Python distribution version. `v0.01` freezes the existing standard checkpoint evaluation. Its release commit adds provenance only; executable files are identical to `70fa2a1f7288cf70b8d29f9bd08fb07ed3bcd013`. The installed control revision is older but its standard runtime was audited byte for byte; the difference is standalone HumanEval sampling and installation checks, not ordinary benchmark scoring.

[v0.01.json](v0.01.json) records the 77 task/shot entries, installed runtime hashes, source identities and historical JUPITER image SHA-256. Reconstruct from this tag using [the JUPITER from-scratch guide](../machines/jupiter/from-scratch.md), retaining the environment/source locks, dataset revisions, model revision, prompt view and resolved launch settings with each run. GPU generation need not be bitwise identical across batch layouts. A tag alone does not reconstruct a particular model, dataset cache or campaign's context budget.

Existing planned and running campaigns finish using their frozen v0.01 installations and manifests. Do not update these source trees, images or queued scripts in place. Store future v0.02 runs under separate output roots and report protocol version alongside every score. Do not relabel historic scores or splice different protocols into an unlabeled curve. Standalone calibration and sampled HumanEval runs remain separately identified.

## Calibrated release

See [v0.02](v0.02.md) for the six calibrated families, explicit selection, immutable per-job runtime snapshots, context requirements and version-separated results. Existing campaigns keep their frozen v0.01 inputs.
