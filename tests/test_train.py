from pathlib import Path
from unittest.mock import MagicMock

from peft import TaskType

from ftml.settings import Settings
from ftml.train import build_lora_config, build_training_args, save_adapter


class TestBuildLoraConfig:
    def test_lora_params(self, mock_settings: Settings) -> None:
        config = build_lora_config(mock_settings)

        assert config.r == 8
        assert config.lora_alpha == 16
        assert config.lora_dropout == 0.1
        assert config.task_type == TaskType.CAUSAL_LM
        assert config.target_modules is not None
        assert "q_proj" in config.target_modules
        assert "v_proj" in config.target_modules
        assert config.bias == "none"

    def test_target_modules_complete(self, mock_settings: Settings) -> None:
        config = build_lora_config(mock_settings)
        expected = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
        assert config.target_modules is not None
        assert set(config.target_modules) == expected


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


class TestSaveAdapter:
    def test_save_adapter_path(self, tmp_output_dir: Path) -> None:
        mock_trainer = MagicMock()

        result = save_adapter(mock_trainer, tmp_output_dir)

        assert result == tmp_output_dir / "adapter"
        mock_trainer.save_model.assert_called_once_with(str(tmp_output_dir / "adapter"))
