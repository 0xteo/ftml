from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    hf_token: str = ""
    model_name: str = "openai/gpt-oss-20b"
    dataset_name: str = "llm-bg/Tucan-BG-v1.0"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048
    use_4bit: bool = True
    use_unsloth: bool = False
    use_flash_attention: bool = False
    use_rslora: bool = False
    use_dora: bool = False
    use_packing: bool = False
    lr_scheduler_type: str = "cosine"
    target_modules: str = "all-linear"
    tf32: bool = True
    warmup_ratio: float = 0.03
    output_dir: Path = Path("./outputs")

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
