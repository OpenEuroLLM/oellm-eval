#!/usr/bin/env python3
"""Quick script to inspect lighteval details (sample-level predictions) from parquet files.

Usage:
    python scripts/inspect_details.py <eval_result_dir> [--n 10] [--task flores200]

The details parquet files are saved when lighteval is run with --save-details.
Each row has: doc.query (prompt), model_response.text (raw generated), metric (per-sample scores).
"""
import argparse
import sys
from pathlib import Path


def find_detail_files(result_dir: Path, task_filter: str | None):
    files = list(result_dir.glob("**/details*.parquet"))
    if task_filter:
        files = [f for f in files if task_filter in f.name]
    return sorted(files)


def inspect_file(parquet_path: Path, n: int):
    try:
        import pandas as pd
    except ImportError:
        print("pandas not installed, trying pyarrow directly")
        import pyarrow.parquet as pq

        table = pq.read_table(parquet_path)
        df = table.to_pandas()
    else:
        df = pd.read_parquet(parquet_path)

    print(f"\n{'='*80}")
    print(f"File: {parquet_path.name}")
    print(f"Columns: {list(df.columns)}")
    print(f"Total samples: {len(df)}")
    print("="*80)

    for i, row in df.head(n).iterrows():
        print(f"\n--- Sample {i} ---")
        # prompt
        if "doc" in df.columns and isinstance(row["doc"], dict):
            prompt = row["doc"].get("query", "")
        elif "query" in df.columns:
            prompt = row["query"]
        else:
            prompt = str(row.get("doc", ""))
        print(f"PROMPT:\n{prompt!r}")

        # model output
        if "model_response" in df.columns and isinstance(row["model_response"], dict):
            raw = row["model_response"].get("text", "")
            processed = row["model_response"].get("text_post_processed", raw)
        elif "text" in df.columns:
            raw = row.get("text", "")
            processed = row.get("text_post_processed", raw)
        else:
            raw = processed = ""
        print(f"OUTPUT (raw):       {raw!r}")
        if raw != processed:
            print(f"OUTPUT (processed): {processed!r}")

        # metric scores
        if "metric" in df.columns:
            metric = row["metric"]
            if isinstance(metric, dict):
                print(f"METRICS: {metric}")

        # gold/reference
        if "doc" in df.columns and isinstance(row["doc"], dict):
            choices = row["doc"].get("choices", [])
            gold_idx = row["doc"].get("gold_index", 0)
            if choices:
                if isinstance(gold_idx, (list,)):
                    golds = [choices[g] for g in gold_idx if g < len(choices)]
                else:
                    golds = [choices[gold_idx]] if gold_idx < len(choices) else []
                print(f"REFERENCE: {golds}")


def main():
    parser = argparse.ArgumentParser(description="Inspect lighteval detail parquet files")
    parser.add_argument("result_dir", help="Path to lighteval result dir (contains details*.parquet)")
    parser.add_argument("--n", type=int, default=10, help="Number of samples to print per file")
    parser.add_argument("--task", default=None, help="Filter by task name substring (e.g. flores200)")
    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    if not result_dir.exists():
        print(f"Directory not found: {result_dir}", file=sys.stderr)
        sys.exit(1)

    files = find_detail_files(result_dir, args.task)
    if not files:
        print(f"No details parquet files found in {result_dir}")
        if args.task:
            print(f"(filtered by task: {args.task!r})")
        print("Make sure the eval was run with --save-details")
        sys.exit(1)

    for f in files:
        inspect_file(f, args.n)


if __name__ == "__main__":
    main()
