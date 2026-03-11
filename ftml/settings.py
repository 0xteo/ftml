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

    # Experiment loop settings
    experiment_time_budget: int = 300
    experiment_max_runs: int = 10
    experiment_min_improvement: float = 0.01
    experiment_branch_tag: str = ""

    # Evaluation settings
    eval_num_samples: int = 5

    # Slack settings
    slack_bot_token: str = ""
    slack_signing_secret: str = ""
    slack_app_port: int = 3000

    # Agent settings
    agent_model_id: str = "Qwen/Qwen2.5-72B-Instruct"
    agent_provider: str = "together"
    agent_api_key: str = ""
    agent_max_steps: int = 15
    agent_verbosity: int = 1
    gpu_vram_gb: float = 24.0

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
