from pathlib import Path

from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from trl.trainer.sft_config import SFTConfig
from trl.trainer.sft_trainer import SFTTrainer

from ftml.settings import Settings


def build_lora_config(settings: Settings) -> LoraConfig:
    return LoraConfig(
        r=settings.lora_r,
        lora_alpha=settings.lora_alpha,
        lora_dropout=settings.lora_dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )


def build_training_args(settings: Settings) -> SFTConfig:
    return SFTConfig(
        output_dir=str(settings.output_dir),
        num_train_epochs=settings.num_epochs,
        per_device_train_batch_size=settings.batch_size,
        gradient_accumulation_steps=settings.gradient_accumulation_steps,
        learning_rate=settings.learning_rate,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_strategy="epoch",
        max_length=settings.max_seq_length,
        dataset_text_field="text",
        report_to="none",
    )


def train(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dataset,
    lora_config: LoraConfig,
    training_args: SFTConfig,
) -> SFTTrainer:
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )
    trainer.train()
    return trainer


def save_adapter(trainer: SFTTrainer, output_dir: Path) -> Path:
    adapter_path = output_dir / "adapter"
    trainer.save_model(str(adapter_path))
    return adapter_path
