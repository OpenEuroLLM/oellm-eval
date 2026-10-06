import argparse
import sys

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pivot eval CSVs into a model × task leaderboard."
    )
    parser.add_argument("csvs", nargs="+", help="One or more result CSV files")
    parser.add_argument(
        "-o", "--output", default="leaderboard.csv", help="Output CSV path"
    )
    parser.add_argument(
        "--no-average", action="store_true", help="Do not append an average column"
    )
    args = parser.parse_args()

    frames = []
    for path in args.csvs:
        try:
            frames.append(pd.read_csv(path))
        except Exception as e:
            print(f"Warning: skipping {path}: {e}", file=sys.stderr)

    if not frames:
        print("Error: no valid CSV files provided.", file=sys.stderr)
        sys.exit(1)

    df = pd.concat(frames, ignore_index=True)

    required = {"model_name", "task", "n_shot", "metric_name", "performance"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        print(f"Error: input CSVs missing columns: {missing}", file=sys.stderr)
        sys.exit(1)

    df["task_label"] = df.apply(
        lambda r: f"{r['task']} ({int(r['n_shot'])}-shot) [{r['metric_name']}]", axis=1
    )
    df = df.drop_duplicates(subset=["model_name", "task_label"], keep="last")

    pivot = df.pivot(index="model_name", columns="task_label", values="performance")

    pivot = pivot[sorted(pivot.columns)]

    if not args.no_average:
        pivot["average"] = pivot.mean(axis=1)

    pivot = pivot.reset_index()
    pivot.to_csv(args.output, index=False)
    print(
        f"Wrote {args.output}  ({len(pivot)} models × {len(pivot.columns) - 1} columns)"
    )


if __name__ == "__main__":
    main()
