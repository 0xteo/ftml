"""Autonomous experiment loop: tracking, git integration, and runner."""

from __future__ import annotations

import csv
import subprocess
import time
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from peft import LoraConfig, TaskType
from trl.trainer.sft_config import SFTConfig

from ftml.train import TimeBudgetCallback, train_and_evaluate

if TYPE_CHECKING:
    from collections.abc import Callable

    from datasets import Dataset

    from ftml.agent.orchestrator import Orchestrator
    from ftml.settings import Settings


# ---------------------------------------------------------------------------
# ExperimentResult
# ---------------------------------------------------------------------------


@dataclass
class ExperimentResult:
    experiment_id: int
    timestamp: str
    eval_loss: float
    train_loss: float
    status: str  # baseline / keep / discard / crash
    description: str
    wall_seconds: float
    peak_vram_gb: float
    commit_sha: str
    # Hyperparameter snapshot
    lora_r: int
    lora_alpha: int
    learning_rate: float
    batch_size: int
    gradient_accumulation_steps: int
    warmup_ratio: float
    lr_scheduler_type: str
    lora_dropout: float


# ---------------------------------------------------------------------------
# ExperimentLog — TSV-backed experiment journal
# ---------------------------------------------------------------------------

_TSV_FIELDS = [f.name for f in fields(ExperimentResult)]


class ExperimentLog:
    """Append-only TSV experiment journal."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, result: ExperimentResult) -> None:
        write_header = not self.path.exists() or self.path.stat().st_size == 0
        with self.path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=_TSV_FIELDS, delimiter="\t")
            if write_header:
                writer.writeheader()
            writer.writerow(asdict(result))

    def load_all(self) -> list[ExperimentResult]:
        if not self.path.exists():
            return []
        with self.path.open(newline="") as f:
            reader = csv.DictReader(f, delimiter="\t")
            results: list[ExperimentResult] = []
            for row in reader:
                results.append(
                    ExperimentResult(
                        experiment_id=int(row["experiment_id"]),
                        timestamp=row["timestamp"],
                        eval_loss=float(row["eval_loss"]),
                        train_loss=float(row["train_loss"]),
                        status=row["status"],
                        description=row["description"],
                        wall_seconds=float(row["wall_seconds"]),
                        peak_vram_gb=float(row["peak_vram_gb"]),
                        commit_sha=row["commit_sha"],
                        lora_r=int(row["lora_r"]),
                        lora_alpha=int(row["lora_alpha"]),
                        learning_rate=float(row["learning_rate"]),
                        batch_size=int(row["batch_size"]),
                        gradient_accumulation_steps=int(row["gradient_accumulation_steps"]),
                        warmup_ratio=float(row["warmup_ratio"]),
                        lr_scheduler_type=row["lr_scheduler_type"],
                        lora_dropout=float(row["lora_dropout"]),
                    ),
                )
            return results

    def best(self) -> ExperimentResult | None:
        results = [r for r in self.load_all() if r.status in {"baseline", "keep"}]
        if not results:
            return None
        return min(results, key=lambda r: r.eval_loss)

    def as_table_str(self) -> str:
        """Return TSV content as a string (for feeding to LLM context)."""
        if not self.path.exists():
            return ""
        return self.path.read_text()


# ---------------------------------------------------------------------------
# ExperimentRepo — git autocommit layer
# ---------------------------------------------------------------------------


def _git(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


class ExperimentRepo:
    """Manages an experiment branch with autocommit on keep / reset on discard."""

    def __init__(self, repo_dir: Path) -> None:
        self.repo_dir = repo_dir

    def begin(self, tag: str) -> str:
        """Create and checkout experiment branch. Returns branch name."""
        branch = f"experiment/{tag}"
        _git("checkout", "-b", branch, cwd=self.repo_dir)
        return branch

    def commit_keep(
        self,
        result: ExperimentResult,
        tsv_path: Path,
        adapter_dir: Path | None = None,
    ) -> str:
        """Stage TSV + adapter, commit. Returns short SHA."""
        _git("add", str(tsv_path), cwd=self.repo_dir)
        if adapter_dir and adapter_dir.exists():
            _git("add", str(adapter_dir), cwd=self.repo_dir)

        msg = (
            f"experiment(keep): #{result.experiment_id} "
            f"eval_loss={result.eval_loss:.4f} — {result.description}\n\n"
            f"eval_loss: {result.eval_loss:.4f}, "
            f"train_loss: {result.train_loss:.4f}, "
            f"wall: {result.wall_seconds:.0f}s"
        )
        _git("commit", "-m", msg, cwd=self.repo_dir)
        cp = _git("rev-parse", "--short", "HEAD", cwd=self.repo_dir)
        return cp.stdout.strip()

    def reset_discard(self) -> None:
        """Hard-reset to HEAD to discard unstaged artifacts from a failed/discarded run."""
        _git("reset", "--hard", "HEAD", cwd=self.repo_dir)
        _git("clean", "-fd", cwd=self.repo_dir)

    def finish(self, summary: str) -> str:
        """Final summary commit. Returns branch name."""
        _git("add", "-A", cwd=self.repo_dir)
        _git("commit", "-m", f"experiment(finish): {summary}", "--allow-empty", cwd=self.repo_dir)
        cp = _git("rev-parse", "--abbrev-ref", "HEAD", cwd=self.repo_dir)
        return cp.stdout.strip()


# ---------------------------------------------------------------------------
# ExperimentRunner — core autonomous loop
# ---------------------------------------------------------------------------

# Allowed parameter ranges for validation
_PARAM_RANGES: dict[str, dict[str, Any]] = {
    "lora_r": {"type": int, "choices": [4, 8, 16, 32, 64, 128]},
    "lora_alpha": {"type": int, "min": 4, "max": 256},
    "learning_rate": {"type": float, "min": 1e-6, "max": 1e-2},
    "batch_size": {"type": int, "choices": [1, 2, 4, 8, 16]},
    "gradient_accumulation_steps": {"type": int, "min": 1, "max": 64},
    "warmup_ratio": {"type": float, "min": 0.0, "max": 0.3},
    "lr_scheduler_type": {
        "type": str,
        "choices": ["cosine", "linear", "constant", "constant_with_warmup"],
    },
    "lora_dropout": {"type": float, "min": 0.0, "max": 0.5},
}


def _validate_param(name: str, raw_value: str) -> Any:
    """Validate and cast a proposed parameter value. Raises ValueError on invalid."""
    spec = _PARAM_RANGES.get(name)
    if spec is None:
        msg = f"Unknown tunable parameter: {name}"
        raise ValueError(msg)

    typ = spec["type"]
    value = typ(raw_value)

    if "choices" in spec and value not in spec["choices"]:
        msg = f"{name}={value} not in allowed choices {spec['choices']}"
        raise ValueError(msg)
    if "min" in spec and value < spec["min"]:
        msg = f"{name}={value} below minimum {spec['min']}"
        raise ValueError(msg)
    if "max" in spec and value > spec["max"]:
        msg = f"{name}={value} above maximum {spec['max']}"
        raise ValueError(msg)

    return value


def _current_hyperparams(settings: Settings, overrides: dict[str, Any]) -> dict[str, Any]:
    """Build current hyperparameter dict from settings + overrides."""
    base = {
        "lora_r": settings.lora_r,
        "lora_alpha": settings.lora_alpha,
        "learning_rate": settings.learning_rate,
        "batch_size": settings.batch_size,
        "gradient_accumulation_steps": settings.gradient_accumulation_steps,
        "warmup_ratio": settings.warmup_ratio,
        "lr_scheduler_type": settings.lr_scheduler_type,
        "lora_dropout": settings.lora_dropout,
    }
    base.update(overrides)
    return base


class ExperimentRunner:
    """Runs the autonomous experiment loop."""

    def __init__(
        self,
        settings: Settings,
        orchestrator: Orchestrator,
        model: Any,
        tokenizer: Any,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        task_description: str,
        initial_overrides: dict[str, Any] | None = None,
    ) -> None:
        self.settings = settings
        self.orchestrator = orchestrator
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.task_description = task_description
        self.overrides: dict[str, Any] = initial_overrides or {}
        self.output_dir = Path(settings.output_dir)
        self.log = ExperimentLog(self.output_dir / "experiments.tsv")
        self.repo = ExperimentRepo(self.output_dir)
        self._next_id = 1

    def _build_lora_config(self, params: dict[str, Any]) -> LoraConfig:
        from ftml.train import _parse_target_modules  # pyright: ignore[reportPrivateUsage]

        return LoraConfig(
            r=params["lora_r"],
            lora_alpha=params["lora_alpha"],
            lora_dropout=params["lora_dropout"],
            target_modules=_parse_target_modules(self.settings.target_modules),
            task_type=TaskType.CAUSAL_LM,
            bias="none",
            use_rslora=self.settings.use_rslora,
            use_dora=self.settings.use_dora,
        )

    def _build_sft_config(self, params: dict[str, Any]) -> SFTConfig:
        kwargs: dict[str, Any] = {
            "output_dir": str(self.output_dir / "current_run"),
            "num_train_epochs": 999,  # effectively infinite — callback controls stopping
            "per_device_train_batch_size": params["batch_size"],
            "gradient_accumulation_steps": params["gradient_accumulation_steps"],
            "learning_rate": params["learning_rate"],
            "bf16": True,
            "logging_steps": 10,
            "save_strategy": "no",
            "max_length": self.settings.max_seq_length,
            "dataset_text_field": "text",
            "report_to": "none",
            "optim": "adamw_8bit",
            "weight_decay": 0.01,
            "lr_scheduler_type": params["lr_scheduler_type"],
            "warmup_ratio": params["warmup_ratio"],
            "tf32": self.settings.tf32,
            "packing": self.settings.use_packing,
            "seed": 3407,
            "eval_strategy": "no",
            "gradient_checkpointing": True,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
        }
        return SFTConfig(**kwargs)

    def _ensure_base_model(self) -> None:
        """Unwrap PeftModel back to base model if needed."""
        from peft import PeftModel as _PeftModel

        if isinstance(self.model, _PeftModel):
            self.model = self.model.unload()  # pyright: ignore[reportCallIssue]

    def run_single_experiment(
        self,
        overrides: dict[str, Any],
        description: str,
    ) -> ExperimentResult:
        """Run one experiment. Returns result (status may be 'crash' on failure)."""
        params = _current_hyperparams(self.settings, overrides)
        exp_id = self._next_id
        self._next_id += 1

        try:
            self._ensure_base_model()
            torch.cuda.reset_peak_memory_stats()

            lora_config = self._build_lora_config(params)
            sft_config = self._build_sft_config(params)
            callback = TimeBudgetCallback(self.settings.experiment_time_budget)

            start = time.monotonic()
            trainer, metrics = train_and_evaluate(
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                training_args=sft_config,
                peft_config=lora_config,
                callbacks=[callback],
            )
            wall = time.monotonic() - start

            peak_vram = (
                torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
            )

            # Save adapter for potential keeping
            adapter_dir = self.output_dir / f"adapter_exp{exp_id}"
            trainer.save_model(str(adapter_dir))

            return ExperimentResult(
                experiment_id=exp_id,
                timestamp=datetime.now(tz=UTC).isoformat(timespec="seconds"),
                eval_loss=metrics.get("eval_loss", float("inf")),
                train_loss=metrics.get(
                    "train_loss",
                    trainer.state.log_history[-1].get("loss", float("inf")),
                ),
                status="pending",
                description=description,
                wall_seconds=round(wall, 1),
                peak_vram_gb=round(peak_vram, 2),
                commit_sha="",
                **{k: params[k] for k in _PARAM_RANGES},
            )

        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return ExperimentResult(
                experiment_id=exp_id,
                timestamp=datetime.now(tz=UTC).isoformat(timespec="seconds"),
                eval_loss=float("inf"),
                train_loss=float("inf"),
                status="crash",
                description=f"{description} [CRASH: {exc}]",
                wall_seconds=0.0,
                peak_vram_gb=0.0,
                commit_sha="",
                **{k: params[k] for k in _PARAM_RANGES},
            )

    def run_loop(
        self,
        on_result: Callable[[ExperimentResult], None] | None = None,
    ) -> ExperimentResult:
        """Run the full autonomous experiment loop. Returns the best result."""
        tag = self.settings.experiment_branch_tag or datetime.now(tz=UTC).strftime("%b%d").lower()
        self.repo.begin(tag)

        # Baseline experiment
        baseline = self.run_single_experiment(self.overrides, "initial baseline")
        baseline.status = "baseline"
        sha = self.repo.commit_keep(
            baseline,
            self.log.path,
            self.output_dir / f"adapter_exp{baseline.experiment_id}",
        )
        baseline.commit_sha = sha
        self.log.append(baseline)
        if on_result:
            on_result(baseline)

        best = baseline

        for _ in range(self.settings.experiment_max_runs - 1):
            # Ask LLM to propose next change
            best_detail = (
                f"#{best.experiment_id}: eval_loss={best.eval_loss:.4f}, "
                f"lora_r={best.lora_r}, lora_alpha={best.lora_alpha}, "
                f"lr={best.learning_rate}, batch={best.batch_size}, "
                f"grad_accum={best.gradient_accumulation_steps}, "
                f"warmup={best.warmup_ratio}, scheduler={best.lr_scheduler_type}, "
                f"dropout={best.lora_dropout}"
            )

            proposal = self.orchestrator.propose_next_experiment(
                task_description=self.task_description,
                experiment_history=self.log.as_table_str(),
                best_experiment=best_detail,
            )

            if proposal.should_stop:
                break

            # Validate proposed change
            param_name = str(proposal.parameter_to_change).strip()
            raw_value = str(proposal.new_value).strip()
            try:
                validated = _validate_param(param_name, raw_value)
            except ValueError:
                continue  # skip invalid proposals

            # Build overrides with the one change
            new_overrides = dict(self.overrides)
            new_overrides[param_name] = validated
            old_value = _current_hyperparams(self.settings, self.overrides).get(param_name)
            description = f"{param_name} {old_value} -> {validated}"

            result = self.run_single_experiment(new_overrides, description)

            if result.status == "crash":
                self.log.append(result)
                self.repo.reset_discard()
                if on_result:
                    on_result(result)
                continue

            # Compare to best
            if result.eval_loss < best.eval_loss - self.settings.experiment_min_improvement:
                result.status = "keep"
                adapter_dir = self.output_dir / f"adapter_exp{result.experiment_id}"
                sha = self.repo.commit_keep(result, self.log.path, adapter_dir)
                result.commit_sha = sha
                self.overrides = new_overrides
                best = result
            else:
                result.status = "discard"
                self.repo.reset_discard()

            self.log.append(result)
            if on_result:
                on_result(result)

        summary = f"best=#{best.experiment_id} eval_loss={best.eval_loss:.4f}"
        self.repo.finish(summary)

        return best
