"""Tests for the `lmms-eval` suite in `template.sbatch`.

The job script picks the lmms-eval model class from the suite
(`lmms-eval:<class>`) or from the model name. The tests render a real script
through `schedule_evals` and run it against a stub venv.
"""

import json
import logging
import os
import stat
import subprocess
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from oellm.main import collect_results, schedule_evals

# stub for `python -m lmms_eval`: records its arguments and writes a result file
HARNESS = """#!/bin/bash
printf '%s\\n' "$@" > "$LAUNCH_LOG"
for arg in "$@"; do
    if [ "$prev" = "--output_path" ] && [ -z "$HARNESS_WRITES_NOTHING" ]; then
        mkdir -p "$arg/model"
        touch "$arg/model/20260101_000000_results.json"
    fi
    prev="$arg"
done
exit 0
"""
# stub `uname`, so the tests do not depend on the machine
UNAME = """#!/bin/bash
echo "${FAKE_UNAME:-Linux}"
"""


@pytest.fixture(autouse=True)
def _restore_root_logger():
    # both functions reconfigure the root logger; restore it for later tests
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    root.handlers[:] = handlers
    root.setLevel(level)


def _stub(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def _launch(
    tmp_path: Path,
    model: str,
    *,
    suite: str = "lmms-eval",
    env: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess, dict[str, str]]:
    """Run one lmms-eval row; return the job result and the launched options."""
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "bin" / "activate").write_text(f'export PATH="{venv}/bin:$PATH"\n')
    _stub(venv / "bin" / "python", HARNESS)
    tools = tmp_path / "tools"
    tools.mkdir()
    _stub(tools / "uname", UNAME)

    jobs = tmp_path / "jobs_in.csv"
    pd.DataFrame(
        [{"model_path": model, "task_path": "mme", "n_shot": 0, "eval_suite": suite}]
    ).to_csv(jobs, index=False)
    with (
        patch("oellm.main._load_cluster_env"),
        patch("oellm.main._num_jobs_in_queue", return_value=0),
        patch.dict(
            os.environ,
            {
                "EVAL_OUTPUT_DIR": str(tmp_path),
                "HF_HOME": str(tmp_path / "hf_home"),
                "GPUS_PER_NODE": "1",
            },
        ),
    ):
        schedule_evals(
            eval_csv_path=str(jobs),
            skip_checks=True,
            venv_path=str(venv),
            dry_run=True,
        )
    script = next(iter(tmp_path.glob("**/submit_evals.sbatch")))
    launch_log = tmp_path / "launch.log"
    result = subprocess.run(
        ["bash", str(script)],
        env={
            **os.environ,
            "PATH": f"{tools}:{os.environ['PATH']}",
            "SLURM_ARRAY_TASK_ID": "0",
            "SLURM_ARRAY_JOB_ID": "1",
            "SLURM_JOB_ID": "1",
            "LAUNCH_LOG": str(launch_log),
            **(env or {}),
        },
        capture_output=True,
        text=True,
    )
    args = launch_log.read_text().splitlines() if launch_log.exists() else []
    options = {
        flag: value
        for flag, value in zip(args, args[1:], strict=False)
        if flag.startswith("--")
    }
    return result, options


@pytest.mark.parametrize(
    "model, model_class",
    [
        # audio
        ("Qwen/Qwen2-Audio-7B-Instruct", "qwen2_audio"),
        ("Qwen/Qwen2.5-Omni-7B", "qwen2_5_omni"),
        ("tsinghua-ee/video-SALMONN-2", "video_salmonn_2"),
        ("nvidia/audio-flamingo-3", "audio_flamingo_3"),
        ("moonshotai/Kimi-Audio-7B-Instruct", "kimi_audio"),
        ("openai/whisper-large-v3", "whisper"),
        ("microsoft/Phi-4-multimodal-instruct", "phi4_multimodal"),
        # image and video
        ("Qwen/Qwen2.5-VL-7B-Instruct", "qwen2_5_vl"),
        ("Qwen/Qwen2-VL-7B-Instruct", "qwen2_vl"),
        ("llava-hf/llava-onevision-qwen2-7b-ov-hf", "llava_hf"),
        ("llava-hf/llava-interleave-qwen-0.5b-hf", "llava_hf"),
        ("lmms-lab/llava-onevision-qwen2-7b-ov", "llava_onevision"),
        ("lmms-lab/LLaVA-Video-7B-Qwen2", "llava_vid"),
        ("LanguageBind/Video-LLaVA-7B-hf", "video_llava"),
        ("liuhaotian/llava-v1.5-7b", "llava_hf"),
        ("OpenGVLab/InternVideo2-Chat-8B", "internvideo2"),
        ("OpenGVLab/InternVL2-8B", "internvl2"),
        ("HuggingFaceM4/idefics2-8b", "idefics2"),
        ("openbmb/MiniCPM-V-2_6", "minicpm_v"),
        ("lmms-lab/LongVA-7B", "longva"),
        ("OpenGVLab/VideoChat2_stage3_Mistral_7B", "videochat2"),
        ("Qwen/Qwen-VL-Chat", "qwen_vl"),
    ],
)
def test_model_class_from_model_name(tmp_path, model, model_class):
    """The model name picks the lmms-eval model class."""
    result, options = _launch(tmp_path, model)

    assert result.returncode == 0, result.stdout + result.stderr
    assert options["--model"] == model_class
    assert options["--tasks"] == "mme"


def test_model_class_named_in_the_suite_is_used_as_is(tmp_path):
    """`lmms-eval:<class>` in a jobs CSV skips the lookup by name."""
    result, options = _launch(
        tmp_path, "/checkpoints/whisper-run-7", suite="lmms-eval:internvl2"
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert options["--model"] == "internvl2"


@pytest.mark.parametrize(
    "model",
    ["/checkpoints/run-7", "Qwen/Qwen2.5-7B-Instruct"],
    ids=["unknown-name", "text-only-model"],
)
def test_unknown_model_fails_the_evaluation(tmp_path, model):
    """A model no class is known for is an error, and nothing is launched."""
    result, options = _launch(tmp_path, model)

    assert result.returncode == 1
    assert "no known model class" in result.stdout
    assert options == {}


def test_a_run_that_writes_no_result_fails_the_job(tmp_path):
    """lmms-eval can exit 0 after a failed evaluation; no result file is the tell."""
    result, _ = _launch(
        tmp_path, "openai/whisper-large-v3", env={"HARNESS_WRITES_NOTHING": "1"}
    )

    assert result.returncode == 1
    assert "wrote no result" in result.stdout


@pytest.mark.parametrize(
    "model, platform, model_args",
    [
        ("openai/whisper-large-v3", "Linux", "device_map=auto"),
        ("openai/whisper-large-v3", "Darwin", "device=mps,device_map=mps"),
        # these classes take no device_map argument
        ("moonshotai/Kimi-Audio-7B-Instruct", "Linux", ""),
        ("openbmb/MiniCPM-V-2_6", "Linux", ""),
        ("Qwen/Qwen-VL-Chat", "Linux", ""),
    ],
)
def test_where_the_model_is_loaded(tmp_path, model, platform, model_args):
    """Spread over the node's GPUs by default, MPS for a local run on macOS."""
    _, options = _launch(tmp_path, model, env={"FAKE_UNAME": platform})

    expected = f"pretrained={model}" + (f",{model_args}" if model_args else "")
    assert options["--model_args"] == expected


@pytest.mark.parametrize(
    "extra, model_args",
    [
        ("dtype=float16", "device_map=auto,dtype=float16"),
        ("device_map=cuda:0", "device_map=cuda:0"),
    ],
    ids=["appended", "device-map-given"],
)
def test_extra_model_args(tmp_path, extra, model_args):
    """LMMS_MODEL_ARGS is passed on, and a device_map in it replaces the default."""
    _, options = _launch(
        tmp_path, "openai/whisper-large-v3", env={"LMMS_MODEL_ARGS": extra}
    )

    assert options["--model_args"] == f"pretrained=openai/whisper-large-v3,{model_args}"


@pytest.mark.parametrize(
    "env, frames", [({}, "8"), ({"MAX_NUM_FRAMES": "16"}, "16")], ids=["default", "set"]
)
def test_qwen_vl_gets_a_video_frame_cap(tmp_path, env, frames):
    """Qwen VL samples at most MAX_NUM_FRAMES frames per video (default 8)."""
    _, options = _launch(tmp_path, "Qwen/Qwen2.5-VL-7B-Instruct", env=env)

    assert options["--model_args"] == (
        f"pretrained=Qwen/Qwen2.5-VL-7B-Instruct,device_map=auto,max_num_frames={frames}"
    )


def test_llava_onevision_is_loaded_by_name(tmp_path):
    """The original-layout LLaVA-OneVision class needs the model name spelled out."""
    _, options = _launch(tmp_path, "lmms-lab/llava-onevision-qwen2-7b-ov")

    assert options["--model_args"] == (
        "pretrained=lmms-lab/llava-onevision-qwen2-7b-ov,device_map=auto,"
        "model_name=llava-onevision-qwen2-7b-ov"
    )


MODEL = "llava-hf/llava-interleave-qwen-0.5b-hf"


def _lmms_eval_run(tmp_path: Path) -> Path:
    """A finished run as lmms-eval leaves it: one directory per evaluation."""
    run = tmp_path / "run"
    result_dir = (
        run / "results" / "d1789852cc" / "llava-hf__llava-interleave-qwen-0.5b-hf"
    )
    result_dir.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "model_path": MODEL,
                "task_path": "realworldqa",
                "n_shot": 0,
                "eval_suite": "lmms-eval:llava_hf",
            }
        ]
    ).to_csv(run / "jobs.csv", index=False)
    (result_dir / "20260922_165022_results.json").write_text(
        json.dumps(
            {
                "results": {
                    "realworldqa": {
                        "alias": "realworldqa",
                        "exact_match,none": 0.25,
                        "exact_match_stderr,none": 0.25,
                        "exact_match_stderr_clustered,none": "N/A",
                    }
                },
                "configs": {"realworldqa": {}},
                "n-shot": {"realworldqa": 0},
                "model_name": MODEL,
                "lmms_eval_version": "0.7.2",
            }
        )
    )
    return run


def test_collect_reads_lmms_eval_results(tmp_path):
    """collect finds the result under <id>/<model>/ and reports its metrics."""
    run = _lmms_eval_run(tmp_path)
    out = run / "eval_results.csv"

    collect_results(str(run), str(out), check=True)

    df = pd.read_csv(out)
    assert set(df["model_name"]) == {MODEL}
    assert set(df["task"]) == {"realworldqa"}
    scores = dict(zip(df["metric_name"], df["performance"], strict=True))
    assert scores == {"exact_match": 0.25, "exact_match_stderr": 0.25}
    # ... and --check does not ask for the evaluation to be run again
    assert not (run / "eval_results_missing.csv").exists()
