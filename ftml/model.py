import torch
from peft import prepare_model_for_kbit_training
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


def build_quantization_config(use_4bit: bool) -> BitsAndBytesConfig | None:
    if not use_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def _model_already_quantized(model_name: str, token: str | None) -> bool:
    config = AutoConfig.from_pretrained(model_name, token=token)
    return getattr(config, "quantization_config", None) is not None


def load_model_and_tokenizer(
    model_name: str,
    hf_token: str,
    quantization_config: BitsAndBytesConfig | None = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    token = hf_token or None

    # Skip BnB quantization if model already has built-in quantization (e.g. MXFP4)
    if quantization_config is not None and _model_already_quantized(model_name, token):
        print(f"Model {model_name} already quantized, skipping BitsAndBytes config")
        quantization_config = None

    kwargs: dict = {
        "device_map": "auto",
        "dtype": torch.bfloat16,
        "token": token,
        "offload_folder": "offload",
    }
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model)

    return model, tokenizer
