"""Tests for the lm_eval / evalchemy ``--batch_size`` selection."""

from unittest.mock import patch

import pytest

from oellm.scheduler import _resolve_lm_eval_batch_size


def _schedule(tmp_path, monkeypatch, **kw):
    from oellm.scheduler import schedule_evals

    monkeypatch.setenv("EVAL_OUTPUT_DIR", str(tmp_path))
    with (
        patch("oellm.scheduler._load_cluster_env"),
        patch("oellm.scheduler._num_jobs_in_queue", return_value=0),
    ):
        schedule_evals(dry_run=True, skip_checks=True, **kw)
    return next(tmp_path.glob("**/submit_evals.sbatch")).read_text()


class TestResolveBatchSize:
    def test_local_is_explicit_and_cluster_is_auto(self, monkeypatch):
        monkeypatch.delenv("BATCH_SIZE", raising=False)
        assert _resolve_lm_eval_batch_size(local=True) == "8"
        assert _resolve_lm_eval_batch_size(local=False) == "auto"

    @pytest.mark.parametrize("value", ["1", "16", "64"])
    def test_env_override_wins_everywhere(self, monkeypatch, value):
        monkeypatch.setenv("BATCH_SIZE", value)
        assert _resolve_lm_eval_batch_size(local=True) == value
        assert _resolve_lm_eval_batch_size(local=False) == value

    @pytest.mark.parametrize("value", ["0", "-4", "auto", "big", " "])
    def test_invalid_override_falls_back_to_the_default(self, monkeypatch, value):
        monkeypatch.setenv("BATCH_SIZE", value)
        assert _resolve_lm_eval_batch_size(local=True) == "8"
        assert _resolve_lm_eval_batch_size(local=False) == "auto"


class TestBatchSizeReachesTheScript:
    def test_both_engines_read_the_variable(self, tmp_path, monkeypatch):
        monkeypatch.delenv("BATCH_SIZE", raising=False)
        sbatch = _schedule(
            tmp_path, monkeypatch, models="org/m", tasks="hellaswag", n_shot=0
        )
        # lm_eval and evalchemy both defer to it, and the literal is gone
        assert sbatch.count('--batch_size "${LM_EVAL_BATCH_SIZE:-auto}"') == 2
        assert "--batch_size auto" not in sbatch

    def test_cluster_run_keeps_auto(self, tmp_path, monkeypatch):
        monkeypatch.delenv("BATCH_SIZE", raising=False)
        sbatch = _schedule(
            tmp_path, monkeypatch, models="org/m", tasks="hellaswag", n_shot=0
        )
        assert 'LM_EVAL_BATCH_SIZE="auto"' in sbatch

    def test_local_run_pins_a_small_batch(self, tmp_path, monkeypatch):
        monkeypatch.delenv("BATCH_SIZE", raising=False)
        sbatch = _schedule(
            tmp_path,
            monkeypatch,
            models="org/m",
            tasks="hellaswag",
            n_shot=0,
            local=True,
            venv_path="/tmp/venv",
        )
        assert 'LM_EVAL_BATCH_SIZE="8"' in sbatch

    def test_override_is_rendered(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BATCH_SIZE", "4")
        sbatch = _schedule(
            tmp_path, monkeypatch, models="org/m", tasks="hellaswag", n_shot=0
        )
        assert 'LM_EVAL_BATCH_SIZE="4"' in sbatch
