from pathlib import Path
from typing import TYPE_CHECKING

from ftml.settings import Settings

if TYPE_CHECKING:
    import pytest


class TestSettings:
    def test_default_values(self) -> None:
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

    def test_env_overrides(self, monkeypatch: "pytest.MonkeyPatch") -> None:
        monkeypatch.setenv("MODEL_NAME", "custom/model")
        monkeypatch.setenv("LORA_R", "32")
        monkeypatch.setenv("USE_4BIT", "false")
        s = Settings()
        assert s.model_name == "custom/model"
        assert s.lora_r == 32
        assert s.use_4bit is False

    def test_output_dir_path_coercion(self) -> None:
        s = Settings(output_dir=Path("/tmp/test-outputs"))
        assert isinstance(s.output_dir, Path)
        assert s.output_dir == Path("/tmp/test-outputs")
