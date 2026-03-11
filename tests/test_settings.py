from pathlib import Path
from typing import TYPE_CHECKING

from ftml.settings import Settings

if TYPE_CHECKING:
    import pytest


class TestSettings:
    def test_default_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir("/tmp")
        monkeypatch.delenv("MODEL_NAME", raising=False)
        s = Settings(hf_token="")
        assert s.model_name == "openai/gpt-oss-20b"
        assert s.dataset_name == "llm-bg/Tucan-BG-v1.0"
        assert s.lora_r == 16
        assert s.lora_alpha == 32
        assert s.lora_dropout == 0.05
        assert s.learning_rate == 2e-4
        assert s.num_epochs == 3
        assert s.batch_size == 4
        assert s.gradient_accumulation_steps == 4
        assert s.max_seq_length == 2048
        assert s.use_4bit is True
        assert s.use_unsloth is False
        assert s.use_flash_attention is False
        assert s.use_rslora is False
        assert s.use_dora is False
        assert s.use_packing is False
        assert s.lr_scheduler_type == "cosine"
        assert s.target_modules == "all-linear"
        assert s.tf32 is True
        assert s.warmup_ratio == 0.03

    def test_env_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MODEL_NAME", "custom/model")
        monkeypatch.setenv("LORA_R", "32")
        monkeypatch.setenv("USE_4BIT", "false")
        monkeypatch.setenv("USE_RSLORA", "true")
        monkeypatch.setenv("LR_SCHEDULER_TYPE", "linear")
        monkeypatch.setenv("TARGET_MODULES", "q_proj,v_proj")
        monkeypatch.setenv("WARMUP_RATIO", "0.1")
        s = Settings()
        assert s.model_name == "custom/model"
        assert s.lora_r == 32
        assert s.use_4bit is False
        assert s.use_rslora is True
        assert s.lr_scheduler_type == "linear"
        assert s.target_modules == "q_proj,v_proj"
        assert s.warmup_ratio == 0.1

    def test_output_dir_path_coercion(self) -> None:
        s = Settings(output_dir=Path("/tmp/test-outputs"))
        assert isinstance(s.output_dir, Path)
        assert s.output_dir == Path("/tmp/test-outputs")
