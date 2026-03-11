from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from peft import LoraConfig, TaskType

from ftml.train import (
    _parse_target_modules,
    build_lora_config,
    build_training_args,
    save_adapter,
)

if TYPE_CHECKING:
    from pathlib import Path

    from ftml.settings import Settings


class TestParseTargetModules:
    def test_all_linear_passthrough(self) -> None:
        assert _parse_target_modules("all-linear") == "all-linear"

    def test_comma_separated(self) -> None:
        result = _parse_target_modules("q_proj,v_proj,k_proj")
        assert result == ["q_proj", "v_proj", "k_proj"]

    def test_comma_separated_with_spaces(self) -> None:
        result = _parse_target_modules("q_proj, v_proj, k_proj")
        assert result == ["q_proj", "v_proj", "k_proj"]

    def test_single_module(self) -> None:
        result = _parse_target_modules("q_proj")
        assert result == ["q_proj"]


class TestBuildLoraConfig:
    def test_default_config(self, mock_settings: Settings) -> None:
        config = build_lora_config(mock_settings)

        assert config is not None
        assert isinstance(config, LoraConfig)
        assert config.r == 8
        assert config.lora_alpha == 16
        assert config.lora_dropout == 0.1
        assert config.task_type == TaskType.CAUSAL_LM
        assert config.bias == "none"

    def test_all_linear_target_modules(self, mock_settings: Settings) -> None:
        config = build_lora_config(mock_settings)

        assert config is not None
        assert config.target_modules == "all-linear"

    def test_custom_target_modules(self, mock_settings: Settings) -> None:
        mock_settings.target_modules = "q_proj,v_proj"  # type: ignore[misc]
        config = build_lora_config(mock_settings)

        assert config is not None
        assert config.target_modules == {"q_proj", "v_proj"}

    def test_rslora_enabled(self, mock_settings: Settings) -> None:
        mock_settings.use_rslora = True  # type: ignore[misc]
        config = build_lora_config(mock_settings)

        assert config is not None
        assert config.use_rslora is True

    def test_dora_enabled(self, mock_settings: Settings) -> None:
        mock_settings.use_dora = True  # type: ignore[misc]
        config = build_lora_config(mock_settings)

        assert config is not None
        assert config.use_dora is True

    def test_returns_none_for_unsloth(self, mock_settings: Settings) -> None:
        mock_settings.use_unsloth = True  # type: ignore[misc]

        assert build_lora_config(mock_settings) is None


class TestBuildTrainingArgs:
    def test_training_args_from_settings(self, mock_settings: Settings) -> None:
        args = build_training_args(mock_settings)

        assert args.num_train_epochs == 1
        assert args.per_device_train_batch_size == 2
        assert args.gradient_accumulation_steps == 1
        assert args.learning_rate == 1e-4
        assert args.bf16 is True
        assert args.gradient_checkpointing is True
        assert args.logging_steps == 10
        assert args.save_strategy == "epoch"
        assert args.max_length == 512
        assert args.report_to == []
        assert args.lr_scheduler_type == "cosine"
        assert args.warmup_ratio == 0.03
        assert args.tf32 is True
        assert args.packing is False

    def test_unsloth_skips_gradient_checkpointing_kwarg(
        self,
        mock_settings: Settings,
    ) -> None:
        mock_settings.use_unsloth = True  # type: ignore[misc]
        args = build_training_args(mock_settings)

        assert args.gradient_checkpointing_kwargs is None

    def test_packing_enabled(self, mock_settings: Settings) -> None:
        mock_settings.use_packing = True  # type: ignore[misc]
        args = build_training_args(mock_settings)

        assert args.packing is True

    def test_linear_scheduler_override(self, mock_settings: Settings) -> None:
        mock_settings.lr_scheduler_type = "linear"  # type: ignore[misc]
        args = build_training_args(mock_settings)

        assert args.lr_scheduler_type == "linear"


class TestSaveAdapter:
    def test_save_adapter_path(self, tmp_output_dir: Path) -> None:
        mock_trainer = MagicMock()

        result = save_adapter(mock_trainer, tmp_output_dir)

        assert result == tmp_output_dir / "adapter"
        mock_trainer.save_model.assert_called_once_with(str(tmp_output_dir / "adapter"))
