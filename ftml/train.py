from __future__ import annotations

from typing import TYPE_CHECKING, Any

from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

if TYPE_CHECKING:
    from pathlib import Path

    from datasets import Dataset
    from peft import LoraConfig

    from ftml.settings import Settings


def _parse_target_modules(raw: str) -> str | list[str]:
    if raw == "all-linear":
        return "all-linear"
    return [m.strip() for m in raw.split(",")]


def build_lora_config(settings: Settings) -> LoraConfig | None:
    if settings.use_unsloth:
        return None

    from peft import LoraConfig as _LoraConfig
    from peft import TaskType

    return _LoraConfig(
        r=settings.lora_r,
        lora_alpha=settings.lora_alpha,
        lora_dropout=settings.lora_dropout,
        target_modules=_parse_target_modules(settings.target_modules),
        task_type=TaskType.CAUSAL_LM,
        bias="none",
        use_rslora=settings.use_rslora,
        use_dora=settings.use_dora,
    )


def apply_lora_unsloth(model: Any, settings: Settings) -> Any:
    from unsloth import FastLanguageModel

    return FastLanguageModel.get_peft_model(
        model,
        r=settings.lora_r,
        lora_alpha=settings.lora_alpha,
        lora_dropout=settings.lora_dropout,
        target_modules=_parse_target_modules(settings.target_modules),
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        bias="none",
    )


def build_training_args(settings: Settings) -> SFTConfig:
    kwargs: dict = {
        "output_dir": str(settings.output_dir),
        "num_train_epochs": settings.num_epochs,
        "per_device_train_batch_size": settings.batch_size,
        "gradient_accumulation_steps": settings.gradient_accumulation_steps,
        "learning_rate": settings.learning_rate,
        "bf16": True,
        "logging_steps": 10,
        "save_strategy": "epoch",
        "max_length": settings.max_seq_length,
        "dataset_text_field": "text",
        "report_to": "none",
        "optim": "adamw_8bit",
        "weight_decay": 0.01,
        "lr_scheduler_type": settings.lr_scheduler_type,
        "warmup_ratio": settings.warmup_ratio,
        "tf32": settings.tf32,
        "packing": settings.use_packing,
        "seed": 3407,
    }

    if not settings.use_unsloth:
        kwargs["gradient_checkpointing"] = True
        kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}

    return SFTConfig(**kwargs)


def train(
    model: Any,
    tokenizer: Any,
    dataset: Dataset,
    training_args: SFTConfig,
    peft_config: LoraConfig | None = None,
) -> SFTTrainer:
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train()
    return trainer


def save_adapter(trainer: SFTTrainer, output_dir: Path) -> Path:
    adapter_path = output_dir / "adapter"
    trainer.save_model(str(adapter_path))
    return adapter_path
