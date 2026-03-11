from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from ftml.settings import Settings

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def mock_settings(tmp_path: Path) -> Settings:
    return Settings(
        hf_token="hf_test_token",
        model_name="test-org/test-model",
        dataset_name="test-org/test-dataset",
        lora_r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        learning_rate=1e-4,
        num_epochs=1,
        batch_size=2,
        gradient_accumulation_steps=1,
        max_seq_length=512,
        use_4bit=True,
        use_unsloth=False,
        use_flash_attention=False,
        use_rslora=False,
        use_dora=False,
        use_packing=False,
        lr_scheduler_type="cosine",
        target_modules="all-linear",
        tf32=True,
        warmup_ratio=0.03,
        output_dir=tmp_path / "outputs",
    )


@pytest.fixture
def tmp_output_dir(tmp_path: Path) -> Path:
    output = tmp_path / "outputs"
    output.mkdir()
    return output


@pytest.fixture
def mock_tokenizer() -> MagicMock:
    tokenizer = MagicMock()
    tokenizer.pad_token = None
    tokenizer.eos_token = "</s>"
    tokenizer.eos_token_id = 2
    tokenizer.apply_chat_template.return_value = "<s>User: Hello\nAssistant: Hi</s>"
    return tokenizer
