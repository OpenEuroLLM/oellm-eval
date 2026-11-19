# Running BBQ

The `bbq` task group runs zero-shot evaluation through Inspect AI. Create an
[Inspect virtual environment](VENV.md#inspect-ai) before running the commands below.

## Prepare metadata

Fairness calculation requires BBQ metadata. On a network-enabled host, run this
once using the Python environment where `oellm-eval` is installed:

```bash
python -m oellm.bbq
```

The file is cached under `$XDG_CACHE_HOME/oellm/bbq/` (default:
`~/.cache/oellm/bbq/`). To use a shared location, set `OELLM_BBQ_METADATA` to the
same file path during preparation and collection. Collection does not download
metadata automatically.

## Run

Run a local smoke test with 10 samples:

```bash
oellm-eval schedule \
    --models "HuggingFaceTB/SmolLM2-135M-Instruct" \
    --task_groups "bbq" \
    --venv_path /path/to/inspect-venv \
    --local true \
    --limit 10
```

Remove `--limit 10` for a full evaluation. Omit `--local true` to submit through
SLURM on a supported cluster.
