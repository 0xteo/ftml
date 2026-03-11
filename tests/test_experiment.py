"""Tests for the experiment loop: ExperimentLog, TimeBudgetCallback, ExperimentRunner."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from ftml.experiment import (
    ExperimentLog,
    ExperimentRepo,
    ExperimentResult,
    ExperimentRunner,
    _current_hyperparams,
    _validate_param,
)
from ftml.train import TimeBudgetCallback

if TYPE_CHECKING:
    from pathlib import Path

    from ftml.settings import Settings


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_result(**overrides) -> ExperimentResult:
    defaults = {
        "experiment_id": 1,
        "timestamp": "2026-03-11T00:00:00+00:00",
        "eval_loss": 2.5,
        "train_loss": 2.3,
        "status": "baseline",
        "description": "initial baseline",
        "wall_seconds": 300.0,
        "peak_vram_gb": 8.5,
        "commit_sha": "abc1234",
        "lora_r": 16,
        "lora_alpha": 32,
        "learning_rate": 2e-4,
        "batch_size": 4,
        "gradient_accumulation_steps": 4,
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "cosine",
        "lora_dropout": 0.05,
    }
    defaults.update(overrides)
    return ExperimentResult(**defaults)


# ---------------------------------------------------------------------------
# ExperimentLog tests
# ---------------------------------------------------------------------------


class TestExperimentLog:
    def test_append_and_load_roundtrip(self, tmp_path: Path) -> None:
        log = ExperimentLog(tmp_path / "experiments.tsv")
        r1 = _make_result(experiment_id=1, eval_loss=2.5)
        r2 = _make_result(experiment_id=2, eval_loss=2.3, status="keep", description="lr change")

        log.append(r1)
        log.append(r2)

        loaded = log.load_all()
        assert len(loaded) == 2
        assert loaded[0].experiment_id == 1
        assert loaded[0].eval_loss == 2.5
        assert loaded[1].experiment_id == 2
        assert loaded[1].eval_loss == 2.3
        assert loaded[1].status == "keep"

    def test_best_returns_lowest_eval_loss(self, tmp_path: Path) -> None:
        log = ExperimentLog(tmp_path / "experiments.tsv")
        log.append(_make_result(experiment_id=1, eval_loss=2.5, status="baseline"))
        log.append(_make_result(experiment_id=2, eval_loss=2.3, status="keep"))
        log.append(_make_result(experiment_id=3, eval_loss=2.8, status="discard"))

        best = log.best()
        assert best is not None
        assert best.experiment_id == 2

    def test_best_ignores_discard_and_crash(self, tmp_path: Path) -> None:
        log = ExperimentLog(tmp_path / "experiments.tsv")
        log.append(_make_result(experiment_id=1, eval_loss=2.5, status="baseline"))
        log.append(_make_result(experiment_id=2, eval_loss=1.0, status="discard"))
        log.append(_make_result(experiment_id=3, eval_loss=0.5, status="crash"))

        best = log.best()
        assert best is not None
        assert best.experiment_id == 1

    def test_best_returns_none_when_empty(self, tmp_path: Path) -> None:
        log = ExperimentLog(tmp_path / "experiments.tsv")
        assert log.best() is None

    def test_load_all_returns_empty_when_no_file(self, tmp_path: Path) -> None:
        log = ExperimentLog(tmp_path / "nonexistent.tsv")
        assert log.load_all() == []

    def test_as_table_str(self, tmp_path: Path) -> None:
        log = ExperimentLog(tmp_path / "experiments.tsv")
        log.append(_make_result(experiment_id=1))
        text = log.as_table_str()
        assert "experiment_id" in text  # header
        assert "baseline" in text

    def test_tsv_preserves_float_precision(self, tmp_path: Path) -> None:
        log = ExperimentLog(tmp_path / "experiments.tsv")
        log.append(_make_result(learning_rate=1.5e-5))
        loaded = log.load_all()
        assert loaded[0].learning_rate == pytest.approx(1.5e-5)


# ---------------------------------------------------------------------------
# TimeBudgetCallback tests
# ---------------------------------------------------------------------------


class TestTimeBudgetCallback:
    def test_stops_after_budget(self) -> None:
        cb = TimeBudgetCallback(budget_seconds=0)
        state = MagicMock()
        control = MagicMock()
        control.should_training_stop = False
        args = MagicMock()

        cb.on_train_begin(args, state, control)
        # Budget is 0s so any step should trigger stop
        time.sleep(0.01)
        cb.on_step_end(args, state, control)

        assert control.should_training_stop is True

    def test_does_not_stop_before_budget(self) -> None:
        cb = TimeBudgetCallback(budget_seconds=9999)
        state = MagicMock()
        control = MagicMock()
        control.should_training_stop = False
        args = MagicMock()

        cb.on_train_begin(args, state, control)
        cb.on_step_end(args, state, control)

        assert control.should_training_stop is not True

    def test_no_error_before_train_begin(self) -> None:
        cb = TimeBudgetCallback(budget_seconds=0)
        state = MagicMock()
        control = MagicMock()
        args = MagicMock()

        # on_step_end before on_train_begin should not crash
        cb.on_step_end(args, state, control)


# ---------------------------------------------------------------------------
# _validate_param tests
# ---------------------------------------------------------------------------


class TestValidateParam:
    def test_valid_lora_r(self) -> None:
        assert _validate_param("lora_r", "32") == 32

    def test_invalid_lora_r_not_in_choices(self) -> None:
        with pytest.raises(ValueError, match="not in allowed choices"):
            _validate_param("lora_r", "17")

    def test_valid_learning_rate(self) -> None:
        assert _validate_param("learning_rate", "1e-4") == pytest.approx(1e-4)

    def test_learning_rate_below_min(self) -> None:
        with pytest.raises(ValueError, match="below minimum"):
            _validate_param("learning_rate", "1e-8")

    def test_learning_rate_above_max(self) -> None:
        with pytest.raises(ValueError, match="above maximum"):
            _validate_param("learning_rate", "1.0")

    def test_valid_scheduler(self) -> None:
        assert _validate_param("lr_scheduler_type", "linear") == "linear"

    def test_invalid_scheduler(self) -> None:
        with pytest.raises(ValueError, match="not in allowed choices"):
            _validate_param("lr_scheduler_type", "polynomial")

    def test_unknown_parameter(self) -> None:
        with pytest.raises(ValueError, match="Unknown tunable parameter"):
            _validate_param("nonexistent", "42")


# ---------------------------------------------------------------------------
# _current_hyperparams tests
# ---------------------------------------------------------------------------


class TestCurrentHyperparams:
    def test_defaults_from_settings(self, mock_settings: Settings) -> None:
        params = _current_hyperparams(mock_settings, {})
        assert params["lora_r"] == mock_settings.lora_r
        assert params["learning_rate"] == mock_settings.learning_rate

    def test_overrides_applied(self, mock_settings: Settings) -> None:
        params = _current_hyperparams(mock_settings, {"lora_r": 64})
        assert params["lora_r"] == 64


# ---------------------------------------------------------------------------
# ExperimentRepo tests
# ---------------------------------------------------------------------------


class TestExperimentRepo:
    def test_begin_creates_branch(self, tmp_path: Path) -> None:
        from ftml.experiment import _git

        _git("init", cwd=tmp_path)
        _git("commit", "--allow-empty", "-m", "init", cwd=tmp_path)

        repo = ExperimentRepo(tmp_path)
        branch = repo.begin("test-tag")

        assert branch == "experiment/test-tag"
        cp = _git("rev-parse", "--abbrev-ref", "HEAD", cwd=tmp_path)
        assert cp.stdout.strip() == "experiment/test-tag"

    def test_commit_keep_returns_sha(self, tmp_path: Path) -> None:
        from ftml.experiment import _git

        _git("init", cwd=tmp_path)
        _git("commit", "--allow-empty", "-m", "init", cwd=tmp_path)

        tsv = tmp_path / "experiments.tsv"
        tsv.write_text("header\ndata\n")

        repo = ExperimentRepo(tmp_path)
        result = _make_result()
        sha = repo.commit_keep(result, tsv)

        assert len(sha) >= 7

    def test_finish_returns_branch_name(self, tmp_path: Path) -> None:
        from ftml.experiment import _git

        _git("init", cwd=tmp_path)
        _git("commit", "--allow-empty", "-m", "init", cwd=tmp_path)
        _git("checkout", "-b", "experiment/test", cwd=tmp_path)

        repo = ExperimentRepo(tmp_path)
        name = repo.finish("best=#1 eval_loss=2.3")

        assert name == "experiment/test"


# ---------------------------------------------------------------------------
# ExperimentRunner tests (with mocked trainer)
# ---------------------------------------------------------------------------


class TestExperimentRunner:
    @pytest.fixture
    def runner_deps(self, mock_settings: Settings) -> dict:
        """Create mocked dependencies for ExperimentRunner."""
        mock_settings.experiment_time_budget = 60
        mock_settings.experiment_max_runs = 3
        mock_settings.experiment_min_improvement = 0.01

        orch = MagicMock()
        model = MagicMock()
        tokenizer = MagicMock()
        train_ds = MagicMock()
        eval_ds = MagicMock()

        return {
            "settings": mock_settings,
            "orchestrator": orch,
            "model": model,
            "tokenizer": tokenizer,
            "train_dataset": train_ds,
            "eval_dataset": eval_ds,
            "task_description": "test task",
        }

    @patch("ftml.experiment.train_and_evaluate")
    @patch("ftml.experiment.ExperimentRepo")
    @patch("ftml.experiment.torch")
    def test_run_single_experiment_returns_result(
        self,
        mock_torch: MagicMock,
        mock_repo_cls: MagicMock,
        mock_train_eval: MagicMock,
        runner_deps: dict,
    ) -> None:
        mock_torch.cuda.is_available.return_value = False
        mock_torch.cuda.reset_peak_memory_stats = MagicMock()
        mock_torch.cuda.max_memory_allocated.return_value = 0

        mock_trainer = MagicMock()
        mock_trainer.state.log_history = [{"loss": 2.1}]
        mock_trainer.save_model = MagicMock()
        mock_train_eval.return_value = (mock_trainer, {"eval_loss": 2.3, "train_loss": 2.1})

        runner = ExperimentRunner(**runner_deps)
        result = runner.run_single_experiment({}, "test run")

        assert result.experiment_id == 1
        assert result.eval_loss == 2.3
        assert result.status == "pending"

    @patch("ftml.experiment.train_and_evaluate")
    @patch("ftml.experiment.ExperimentRepo")
    @patch("ftml.experiment.torch")
    def test_run_single_experiment_handles_oom(
        self,
        mock_torch: MagicMock,
        mock_repo_cls: MagicMock,
        mock_train_eval: MagicMock,
        runner_deps: dict,
    ) -> None:
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.empty_cache = MagicMock()
        mock_torch.cuda.reset_peak_memory_stats = MagicMock()
        mock_torch.cuda.OutOfMemoryError = type("OutOfMemoryError", (RuntimeError,), {})
        mock_train_eval.side_effect = mock_torch.cuda.OutOfMemoryError("OOM")

        runner = ExperimentRunner(**runner_deps)
        result = runner.run_single_experiment({}, "oom run")

        assert result.status == "crash"
        assert "CRASH" in result.description
        mock_torch.cuda.empty_cache.assert_called_once()
