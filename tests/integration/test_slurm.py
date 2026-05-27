import csv
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

import pytest

from oellm.task_groups import _expand_task_groups, get_all_task_group_names


def get_first_task_per_group() -> list[tuple[str, str, int, str]]:
    results = []
    for group_name in get_all_task_group_names():
        expanded = _expand_task_groups([group_name])
        if expanded:
            first = expanded[0]
            results.append((group_name, first.task, first.n_shot, first.suite))
    return results


def find_eval_dir(base_dir: Path) -> Path | None:
    user = os.environ.get("USER", "runner")
    output_dir = base_dir / user

    if not output_dir.exists():
        return None

    dirs = sorted(output_dir.iterdir(), reverse=True)
    for d in dirs:
        if d.is_dir():
            return d

    return None


def wait_for_slurm_jobs(timeout: int = 600, poll_interval: int = 10) -> bool:
    start_time = time.time()

    while time.time() - start_time < timeout:
        result = subprocess.run(
            ["squeue", "-h", "-o", "%i"],
            capture_output=True,
            text=True,
        )

        jobs = [j.strip() for j in result.stdout.strip().split("\n") if j.strip()]

        if not jobs:
            return True

        print(
            f"Waiting for {len(jobs)} job(s) to complete... "
            f"(elapsed: {int(time.time() - start_time)}s)"
        )
        time.sleep(poll_interval)

    return False


def run_schedule_eval(
    task_groups: str,
    limit: int = 5,
    dry_run: bool = False,
    skip_checks: bool = False,
    venv_path: str | None = None,
):
    cmd = [
        "uv",
        "run",
        "oellm-eval",
        "schedule",
        "--models",
        "HuggingFaceTB/SmolLM2-135M-Instruct",
        "--task_groups",
        task_groups,
        "--limit",
        str(limit),
    ]
    if venv_path:
        cmd.extend(["--venv_path", venv_path])
    if dry_run:
        cmd.extend(["--dry_run", "true"])
    if skip_checks:
        cmd.extend(["--skip_checks", "true"])

    return subprocess.run(cmd, capture_output=True, text=True)


def run_schedule_eval_with_csv(
    csv_path: str,
    limit: int = 1,
    dry_run: bool = False,
    skip_checks: bool = False,
    venv_path: str | None = None,
):
    cmd = [
        "uv",
        "run",
        "oellm-eval",
        "schedule",
        "--eval_csv_path",
        csv_path,
        "--limit",
        str(limit),
    ]
    if venv_path:
        cmd.extend(["--venv_path", venv_path])
    if dry_run:
        cmd.extend(["--dry_run", "true"])
    if skip_checks:
        cmd.extend(["--skip_checks", "true"])

    return subprocess.run(cmd, capture_output=True, text=True)


@pytest.mark.usefixtures("slurm_available")
class TestSlurmAvailability:
    def test_sinfo_works(self):
        result = subprocess.run(["sinfo"], capture_output=True, text=True)
        assert result.returncode == 0
        assert "PARTITION" in result.stdout or "STATE" in result.stdout


@pytest.mark.dry_run
@pytest.mark.usefixtures("slurm_available")
class TestScheduleEvalDryRun:
    @pytest.fixture(autouse=True)
    def setup(self, slurm_env, venv_path):
        all_task_groups = ",".join(get_all_task_group_names())

        os.environ["SINGULARITY_ARGS"] = ""
        os.environ["GPUS_PER_NODE"] = "0"

        result = run_schedule_eval(
            all_task_groups,
            limit=1,
            dry_run=True,
            skip_checks=True,
            venv_path=venv_path,
        )

        assert result.returncode == 0, (
            f"schedule failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )

        self.eval_dir = find_eval_dir(slurm_env)
        if self.eval_dir is None:
            user = os.environ.get("USER", "runner")
            output_dir = slurm_env / user
            print(f"DEBUG: Looking for eval dir in {output_dir}")
            print(f"DEBUG: output_dir exists: {output_dir.exists()}")
            if output_dir.exists():
                print(f"DEBUG: contents: {list(output_dir.iterdir())}")
            print(f"DEBUG: EVAL_OUTPUT_DIR={os.environ.get('EVAL_OUTPUT_DIR')}")
            print(f"DEBUG: stdout: {result.stdout}")
        assert self.eval_dir is not None, f"Could not find eval dir under {slurm_env}"

        self.sbatch_path = self.eval_dir / "submit_evals.sbatch"
        self.csv_path = self.eval_dir / "jobs.csv"
        self.venv_path = venv_path

    @pytest.mark.parametrize(
        "pattern,description",
        [
            (r"#SBATCH --job-name=", "job-name"),
            (r"#SBATCH --time=", "time"),
            (r"#SBATCH --output=", "output"),
            (r"#SBATCH --partition=", "partition"),
            (r"#SBATCH --array=", "array"),
            (r"CSV_PATH=", "CSV_PATH"),
            (r"lm_eval", "lm_eval"),
            (r"LIMIT=", "LIMIT-var"),
            (r"\$\{LIMIT:\+--limit \$LIMIT\}", "LIMIT-expansion"),
            (r"VENV_PATH=", "VENV_PATH-var"),
        ],
        ids=lambda x: x[1],
    )
    def test_sbatch_contains(self, pattern, description):
        content = self.sbatch_path.read_text()
        assert re.search(pattern, content), f"sbatch missing {description}"

    def test_sbatch_execution_mode_patterns(self):
        content = self.sbatch_path.read_text()
        if self.venv_path:
            assert self.venv_path in content, (
                "venv mode should include venv_path in script"
            )
        else:
            assert re.search(r"singularity exec", content), (
                "container mode should use singularity exec"
            )

    def test_sbatch_bash_syntax_valid(self):
        result = subprocess.run(
            ["bash", "-n", str(self.sbatch_path)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Bash syntax error: {result.stderr}"

    def test_jobs_csv_has_header(self):
        content = self.csv_path.read_text()
        header = content.split("\n")[0]

        assert "model_path" in header
        assert "task_path" in header
        assert "n_shot" in header
        assert "eval_suite" in header

    def test_jobs_csv_has_jobs(self):
        content = self.csv_path.read_text()
        lines = [line for line in content.strip().split("\n") if line]

        assert len(lines) > 1, "No jobs in CSV (only header found)"


def _get_dataset_specs():
    from oellm.task_groups import _collect_dataset_specs

    specs = _collect_dataset_specs(get_all_task_group_names())
    return [(spec.repo_id, spec.subset) for spec in specs]


def _dataset_id(val):
    repo_id, subset = val
    if subset:
        return f"{repo_id.split('/')[-1]}/{subset}"
    return repo_id.split("/")[-1]


@pytest.mark.usefixtures("slurm_available")
class TestDatasetDownloads:
    @pytest.mark.parametrize("dataset_spec", _get_dataset_specs(), ids=_dataset_id)
    def test_dataset_downloads(self, slurm_env, dataset_spec):
        from oellm.task_groups import DatasetSpec
        from oellm.utils import _pre_download_datasets_from_specs

        repo_id, subset = dataset_spec
        spec = DatasetSpec(repo_id=repo_id, subset=subset)
        _pre_download_datasets_from_specs([spec])


def _first_task_id(val):
    group_name, task_name, n_shot, suite = val
    return f"{group_name}:{task_name}"


@pytest.mark.slow
@pytest.mark.usefixtures("slurm_available")
class TestFullEvaluationPipeline:
    @pytest.mark.parametrize("task_info", get_first_task_per_group(), ids=_first_task_id)
    def test_task_group_evaluation(self, slurm_env, task_info, venv_path):
        group_name, task_name, n_shot, suite = task_info

        mode_desc = "venv" if venv_path else "container"
        print(f"\n{'=' * 60}")
        print(
            f"Testing: {group_name} -> {task_name} (n_shot={n_shot}, suite={suite}, mode={mode_desc})"
        )
        print(f"{'=' * 60}")

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["model_path", "task_path", "n_shot", "eval_suite"])
            writer.writerow(
                ["HuggingFaceTB/SmolLM2-135M-Instruct", task_name, n_shot, suite]
            )
            csv_path = csv_file.name

        result = run_schedule_eval_with_csv(
            csv_path, limit=1, dry_run=False, venv_path=venv_path
        )
        os.unlink(csv_path)

        if result.returncode != 0:
            print(f"schedule stdout:\n{result.stdout}")
            print(f"schedule stderr:\n{result.stderr}")
        assert result.returncode == 0, (
            f"schedule failed for {group_name}:\nSTDERR: {result.stderr}"
        )

        print("Waiting for job to complete...")
        jobs_completed = wait_for_slurm_jobs(timeout=600, poll_interval=10)

        eval_dir = find_eval_dir(slurm_env)

        if not jobs_completed or eval_dir is None:
            user = os.environ.get("USER", "runner")
            for log in (slurm_env / user).rglob("*.out"):
                print(f"\n--- {log} ---")
                print(log.read_text()[-2000:])
            for log in (slurm_env / user).rglob("*.err"):
                print(f"\n--- {log} ---")
                print(log.read_text()[-2000:])

        assert jobs_completed, f"Job for {group_name} did not complete within timeout"
        assert eval_dir is not None, f"Could not find eval directory for {group_name}"

        results_dir = eval_dir / "results"
        if not results_dir.exists():
            slurm_logs_dir = eval_dir / "slurm_logs"
            if slurm_logs_dir.exists():
                for log in slurm_logs_dir.glob("*.out"):
                    print(f"\n--- {log.name} (stdout) ---")
                    print(log.read_text()[-3000:])
                for log in slurm_logs_dir.glob("*.err"):
                    print(f"\n--- {log.name} (stderr) ---")
                    print(log.read_text()[-3000:])
        assert results_dir.exists(), (
            f"Results directory not found for {group_name}: {results_dir}"
        )

        json_files = list(results_dir.glob("**/*.json"))
        assert len(json_files) > 0, f"No result JSON files for {group_name}"

        for json_file in json_files:
            with open(json_file) as f:
                data = json.load(f)

            assert "results" in data, f"Missing 'results' in {json_file.name}"

            results = data.get("results", {})
            assert results, f"Empty results for {group_name}"

            for _, task_results in results.items():
                if "acc,none" in task_results:
                    acc = task_results["acc,none"]
                    assert 0.0 <= acc <= 1.0, f"Accuracy {acc} out of range"

        print(f"{group_name}: PASSED ({task_name}, {len(json_files)} result files)")
