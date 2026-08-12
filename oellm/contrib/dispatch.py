"""SLURM-side entry point for contrib suite evaluation.

Called from template.sbatch's ``*)`` catch-all case as::

    python -m oellm.contrib.dispatch \\
        --suite      "regiondial_bench:vision_reasoner" \\
        --model_path "/path/to/model" \\
        --task       "regiondial_refcocog" \\
        --n_shot     0 \\
        --output_path "/evals/dir/abc123.json"

The suite name may include a model-flags suffix separated by ``:``, e.g.
``regiondial_bench:vision_reasoner``.  The suffix is passed to ``suite.run()``
as ``model_flags``.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Dispatch evaluation to a registered contrib suite."
    )
    p.add_argument("--suite", required=True, help="Suite name (optionally :model_flags)")
    p.add_argument("--model_path", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--n_shot", required=True, type=int)
    p.add_argument("--output_path", required=True, type=Path)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    args = _parse_args(argv)

    if ":" in args.suite:
        suite_name, model_flags = args.suite.split(":", 1)
    else:
        suite_name, model_flags = args.suite, None

    from oellm import registry

    try:
        mod = registry.get_suite(suite_name)
    except KeyError as exc:
        logging.error(str(exc))
        sys.exit(1)

    missing_vars = []
    for var in getattr(mod, "CLUSTER_ENV_VARS", []):
        if not os.environ.get(var):
            missing_vars.append(var)
    if missing_vars:
        logging.error(
            "Contrib suite %r requires environment variables that are not set: %s\n"
            "Add them to clusters.yaml for this cluster.",
            suite_name,
            ", ".join(missing_vars),
        )
        sys.exit(1)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info(
        "Dispatching to suite %r (model_flags=%r): %s | %s | n_shot=%d",
        suite_name,
        model_flags,
        args.model_path,
        args.task,
        args.n_shot,
    )

    mod.run(
        model_path=args.model_path,
        task=args.task,
        n_shot=args.n_shot,
        output_path=args.output_path,
        model_flags=model_flags,
        env=dict(os.environ),
    )


if __name__ == "__main__":
    main()
