import argparse
import sys

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Pivot eval CSVs into a model × task leaderboard. Cell values are "
            "the 0-100 normalized scores (performance_normalized) when "
            "available; rows without one fall back to the raw performance and "
            "their column is labelled ', raw'."
        )
    )
    parser.add_argument("csvs", nargs="+", help="One or more result CSV files")
    parser.add_argument(
        "-o", "--output", default="leaderboard.csv", help="Output CSV path"
    )
    parser.add_argument(
        "--average",
        action="store_true",
        help=(
            "Append a mean over the NORMALIZED columns. Caveat: the mean "
            "skips NaNs, so models missing tasks average over different "
            "subsets — only comparable when every model covers every column."
        ),
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

    required = {"model_name", "task", "n_shot", "performance"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        print(f"Error: input CSVs missing columns: {missing}", file=sys.stderr)
        sys.exit(1)

    if "metric_name" not in df.columns:
        df["metric_name"] = ""
    has_norm = "performance_normalized" in df.columns

    # Metric identity is part of the column label so rows that legitimately
    # differ only in metric_name don't collapse into one cell; the n_shot
    # label is not int()-coerced because collect legitimately emits "unknown".
    def _label(r) -> str:
        metric = str(r["metric_name"]).split(",")[0] or "metric"
        norm_ok = has_norm and pd.notna(r.get("performance_normalized"))
        suffix = "" if norm_ok else ", raw"
        return f"{r['task']} ({r['n_shot']}-shot, {metric}{suffix})"

    df["task_label"] = df.apply(_label, axis=1)
    if has_norm:
        df["value"] = df["performance_normalized"].where(
            df["performance_normalized"].notna(), df["performance"]
        )
    else:
        df["value"] = df["performance"]

    # keep="last" matches collect's newest-wins duplicate semantics.
    df = df.drop_duplicates(subset=["model_name", "task_label"], keep="last")

    pivot = df.pivot(index="model_name", columns="task_label", values="value")
    pivot = pivot[sorted(pivot.columns)]

    if args.average:
        norm_cols = [c for c in pivot.columns if ", raw" not in c]
        if norm_cols:
            pivot["average (normalized cols)"] = pivot[norm_cols].mean(axis=1)
            print(
                "Note: the average covers normalized columns only and skips "
                "NaNs — models missing tasks average over different subsets.",
                file=sys.stderr,
            )

    pivot = pivot.reset_index()
    pivot.to_csv(args.output, index=False)
    print(
        f"Wrote {args.output}  ({len(pivot)} models × {len(pivot.columns) - 1} columns)"
    )


if __name__ == "__main__":
    main()
