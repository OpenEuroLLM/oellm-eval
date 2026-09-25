"""Run one AudioBench evaluation, optionally on your own checkpoint.

Points the loader's weights variable at ``--checkpoint`` and writes AudioBench's
files to ``--log-dir``. Run from the AudioBench clone: some loaders use relative paths.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--audiobench-dir", required=True)
    p.add_argument("--log-dir", required=True)
    p.add_argument("--dataset-name", required=True)
    p.add_argument("--model-name", required=True, help="AudioBench dispatch key")
    p.add_argument("--metrics", required=True)
    p.add_argument("--number-of-samples", type=int, default=-1)
    p.add_argument("--module", help="model_src module whose weights to replace")
    p.add_argument("--variable", help="module variable holding the weights location")
    p.add_argument("--checkpoint", help="checkpoint to load instead")
    args = p.parse_args(argv)

    src = Path(args.audiobench_dir).resolve() / "src"
    # AudioBench imports its modules by bare name; keep this folder off the path.
    here = str(Path(__file__).resolve().parent)
    sys.path[:] = [str(src)] + [entry for entry in sys.path if entry != here]
    os.chdir(args.audiobench_dir)

    import main_evaluate

    main_evaluate.file_save_folder = str(Path(args.log_dir).resolve())

    if args.checkpoint:
        module = importlib.import_module(f"model_src.{args.module}")
        stock = getattr(module, args.variable)
        setattr(module, args.variable, args.checkpoint)
        print(
            f"AudioBench {args.model_name}: loading {args.checkpoint} instead of {stock}"
        )

    main_evaluate.main(
        dataset_name=args.dataset_name,
        model_name=args.model_name,
        metrics=args.metrics,
        overwrite=True,
        number_of_samples=args.number_of_samples,
    )


if __name__ == "__main__":
    main()
