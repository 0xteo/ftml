from __future__ import annotations

from typing import TYPE_CHECKING, Any

from datasets import Dataset, DatasetDict
from datasets import load_dataset as _load_dataset

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


def load_dataset_from_hf(
    dataset_name: str,
    hf_token: str,
) -> DatasetDict:
    token = hf_token or None
    ds: Any = _load_dataset(dataset_name, token=token)

    if isinstance(ds, DatasetDict):
        if "validation" not in ds and "test" not in ds:
            split = ds["train"].train_test_split(test_size=0.05, seed=42)
            return DatasetDict({"train": split["train"], "validation": split["test"]})
        return ds

    if isinstance(ds, Dataset):
        split = ds.train_test_split(test_size=0.05, seed=42)
        return DatasetDict({"train": split["train"], "validation": split["test"]})

    raise TypeError(f"Unexpected dataset type: {type(ds)}")


def format_for_sft(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
) -> Dataset:
    columns = dataset.column_names

    if "messages" in columns:
        return dataset

    if "conversations" in columns:
        return dataset.rename_column("conversations", "messages")

    if "text" in columns:
        return dataset

    if "instruction" in columns:

        def format_instruction(example: dict) -> dict:
            output = example.get("output", example.get("response", ""))
            inp = example.get("input", "")
            if inp:
                text = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{inp}\n\n### Response:\n{output}"
            else:
                text = f"### Instruction:\n{example['instruction']}\n\n### Response:\n{output}"
            return {"text": text}

        return dataset.map(format_instruction, remove_columns=columns)

    msg = f"Unsupported dataset format. Columns: {columns}. Expected one of: conversations, messages, text, instruction."
    raise ValueError(msg)
