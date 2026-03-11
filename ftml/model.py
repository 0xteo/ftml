from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


def load_model_and_tokenizer(
    model_name: str,
    hf_token: str,
    use_4bit: bool = True,
    max_seq_length: int = 2048,
    use_unsloth: bool = False,
    use_flash_attention: bool = False,
) -> tuple[Any, Any]:
    token = hf_token or None

    if use_unsloth:
        return _load_unsloth(model_name, use_4bit, max_seq_length)
    return _load_transformers(model_name, token, use_4bit, use_flash_attention)


def _load_unsloth(
    model_name: str,
    use_4bit: bool,
    max_seq_length: int,
) -> tuple[Any, Any]:
    from unsloth import FastLanguageModel

    print("Using unsloth backend")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        dtype=None,
        max_seq_length=max_seq_length,
        load_in_4bit=use_4bit,
        full_finetuning=False,
    )
    return model, tokenizer


def _load_transformers(
    model_name: str,
    token: str | None,
    use_4bit: bool,
    use_flash_attention: bool = False,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print("Using transformers backend")

    quantization_config: BitsAndBytesConfig | None = None
    if use_4bit:
        config = AutoConfig.from_pretrained(model_name, token=token)
        if getattr(config, "quantization_config", None) is not None:
            print(f"Model {model_name} already quantized, skipping BitsAndBytes")
        else:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

    kwargs: dict = {
        "device_map": "auto",
        "dtype": torch.bfloat16,
        "token": token,
        "offload_folder": "offload",
    }
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    if use_flash_attention:
        kwargs["attn_implementation"] = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer
