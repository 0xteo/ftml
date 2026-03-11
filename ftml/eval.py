"""Post-training evaluation: sample generation and perplexity computation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    from ftml.settings import Settings


def generate_samples(
    settings: Settings,
    adapter_path: str,
    prompts: list[str],
    max_new_tokens: int = 256,
) -> list[dict[str, str]]:
    """Load model + adapter, generate responses for each prompt.

    Returns list of {"prompt": ..., "response": ...} dicts.
    """
    from peft import PeftModel

    from ftml.model import load_model_and_tokenizer

    model, tokenizer = load_model_and_tokenizer(
        settings.model_name,
        settings.hf_token,
        use_4bit=settings.use_4bit,
        max_seq_length=settings.max_seq_length,
        use_unsloth=settings.use_unsloth,
        use_flash_attention=settings.use_flash_attention,
    )

    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    results = []
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        # Decode only the generated tokens (skip the input)
        generated = output_ids[0][inputs["input_ids"].shape[1] :]
        response = tokenizer.decode(generated, skip_special_tokens=True)
        results.append({"prompt": prompt, "response": response})

    return results


def compute_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    texts: list[str],
    max_length: int = 2048,
) -> float:
    """Compute perplexity on a list of texts via cross-entropy loss."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])

        num_tokens = inputs["input_ids"].shape[1]
        total_loss += outputs.loss.item() * num_tokens
        total_tokens += num_tokens

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    return torch.exp(torch.tensor(avg_loss)).item()
