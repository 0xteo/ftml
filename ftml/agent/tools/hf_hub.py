"""HuggingFace Hub tools for model and dataset research."""

from smolagents import tool


@tool
def search_models(query: str, task: str = "text-generation", limit: int = 10) -> str:
    """Search HuggingFace Hub for models. Returns a list of models with key metadata.

    Args:
        query: Search query (e.g., "Bulgarian", "multilingual chat", "Mistral 7B").
        task: Pipeline task filter (e.g., "text-generation", "text2text-generation").
        limit: Maximum number of results to return.
    """
    from huggingface_hub import list_models

    models = list_models(
        search=query,
        filter=task,
        sort="downloads",
        direction=-1,
        limit=limit,
    )
    results = []
    for m in models:
        safetensors = m.safetensors
        params = None
        if safetensors and hasattr(safetensors, "total"):
            params = safetensors.total
        params_str = f"{params / 1e9:.1f}B" if params else "unknown"
        results.append(
            f"- {m.id} | downloads: {m.downloads:,} | likes: {m.likes} "
            f"| params: {params_str} | tags: {', '.join(m.tags or [])}",
        )
    if not results:
        return "No models found matching the query."
    return f"Found {len(results)} models:\n" + "\n".join(results)


@tool
def get_model_info(model_id: str) -> str:
    """Get detailed information about a specific model on HuggingFace Hub.

    Args:
        model_id: The model identifier (e.g., "mistralai/Mistral-7B-v0.3").
    """
    from huggingface_hub import model_info

    info = model_info(model_id)

    safetensors = info.safetensors
    params = None
    if safetensors and hasattr(safetensors, "total"):
        params = safetensors.total
    params_str = f"{params / 1e9:.2f}B" if params else "unknown"

    card_text = ""
    if info.card_data:
        card = info.card_data
        card_text = (
            f"Language: {getattr(card, 'language', 'not specified')}\n"
            f"License: {getattr(card, 'license', 'not specified')}\n"
            f"Base model: {getattr(card, 'base_model', 'none')}\n"
        )

    tags = ", ".join(info.tags or [])

    return (
        f"Model: {info.id}\n"
        f"Parameters: {params_str}\n"
        f"Pipeline: {info.pipeline_tag}\n"
        f"Library: {info.library_name}\n"
        f"Downloads: {info.downloads:,}\n"
        f"Likes: {info.likes}\n"
        f"Last modified: {info.last_modified}\n"
        f"Tags: {tags}\n"
        f"{card_text}"
    )


@tool
def search_datasets(query: str, limit: int = 10) -> str:
    """Search HuggingFace Hub for datasets. Returns a list of datasets with metadata.

    Args:
        query: Search query (e.g., "Bulgarian instruction", "bg chat").
        limit: Maximum number of results to return.
    """
    from huggingface_hub import list_datasets

    datasets = list_datasets(
        search=query,
        sort="downloads",
        direction=-1,
        limit=limit,
    )
    results = []
    for d in datasets:
        tags = ", ".join(d.tags or []) if d.tags else ""
        results.append(f"- {d.id} | downloads: {d.downloads:,} | likes: {d.likes} | tags: {tags}")
    if not results:
        return "No datasets found matching the query."
    return f"Found {len(results)} datasets:\n" + "\n".join(results)


@tool
def get_dataset_info(dataset_id: str) -> str:
    """Get detailed information about a specific dataset on HuggingFace Hub.

    Args:
        dataset_id: The dataset identifier (e.g., "llm-bg/Tucan-BG-v1.0").
    """
    from huggingface_hub import dataset_info

    info = dataset_info(dataset_id)

    card_text = ""
    if info.card_data:
        card = info.card_data
        card_text = (
            f"Language: {getattr(card, 'language', 'not specified')}\n"
            f"License: {getattr(card, 'license', 'not specified')}\n"
            f"Task categories: {getattr(card, 'task_categories', 'not specified')}\n"
            f"Size category: {getattr(card, 'size_categories', 'not specified')}\n"
        )

    tags = ", ".join(info.tags or []) if info.tags else ""

    return (
        f"Dataset: {info.id}\n"
        f"Downloads: {info.downloads:,}\n"
        f"Likes: {info.likes}\n"
        f"Last modified: {info.last_modified}\n"
        f"Tags: {tags}\n"
        f"{card_text}"
    )


@tool
def preview_dataset(dataset_id: str, split: str = "train", num_rows: int = 3) -> str:
    """Preview a few rows from a dataset to understand its format and content.

    Args:
        dataset_id: The dataset identifier (e.g., "llm-bg/Tucan-BG-v1.0").
        split: Which split to preview (e.g., "train").
        num_rows: Number of rows to show.
    """
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split=split, streaming=True)
    rows = list(ds.take(num_rows))

    if not rows:
        return "Dataset is empty or split not found."

    columns = list(rows[0].keys())
    output = f"Dataset: {dataset_id} (split={split})\nColumns: {', '.join(columns)}\n\n"

    for i, row in enumerate(rows):
        output += f"--- Row {i + 1} ---\n"
        for col in columns:
            value = str(row[col])
            if len(value) > 300:
                value = value[:300] + "..."
            output += f"  {col}: {value}\n"

    return output
