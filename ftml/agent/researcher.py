"""Research agents for model and dataset discovery on HuggingFace Hub."""

from smolagents import CodeAgent, LiteLLMModel

from ftml.agent.tools.hardware import estimate_vram
from ftml.agent.tools.hf_hub import (
    get_dataset_info,
    get_model_info,
    preview_dataset,
    search_datasets,
    search_models,
)

MODEL_RESEARCH_PROMPT = """\
You are a model research specialist for LLM fine-tuning. Your job is to find the best
base model on HuggingFace Hub for a given fine-tuning task.

Research methodology:
1. Search for models using multiple relevant queries (try different keywords)
2. Get detailed info on the most promising candidates (top 3-5)
3. Check VRAM requirements — the model MUST fit on the available GPU
4. Prefer models that:
   - Have strong multilingual or target-language support
   - Are well-maintained (recent updates, high downloads)
   - Have a compatible architecture (decoder-only for chat/instruction tasks)
   - Have a permissive license for fine-tuning
   - Are in the right size range for the hardware

Return your findings as a structured summary with scores for each candidate.
"""

DATASET_RESEARCH_PROMPT = """\
You are a dataset research specialist for LLM fine-tuning. Your job is to find the best
fine-tuning dataset on HuggingFace Hub for a given task.

Research methodology:
1. Search for datasets using multiple relevant queries (language, task type, domain)
2. Get detailed info on promising candidates
3. Preview dataset rows to check format (messages, instruction, or text format)
4. Prefer datasets that:
   - Match the target language
   - Have a clean format compatible with SFTTrainer (messages > instruction > text)
   - Have sufficient size (1K+ samples minimum, 10K+ preferred)
   - Are well-maintained and documented
   - Cover the target domain

Return your findings as a structured summary with format details for each candidate.
"""


def build_model_researcher(model: LiteLLMModel, vram_gb: float = 24.0) -> CodeAgent:
    """Create a research agent specialized in finding HF models."""
    return CodeAgent(
        tools=[search_models, get_model_info, estimate_vram],
        model=model,
        name="model_researcher",
        description=(
            "Searches HuggingFace Hub for the best base model for fine-tuning. "
            "Give it the task description, target language, and hardware constraints."
        ),
        system_prompt=MODEL_RESEARCH_PROMPT,
        max_steps=10,
        additional_authorized_imports=["json"],
    )


def build_dataset_researcher(model: LiteLLMModel) -> CodeAgent:
    """Create a research agent specialized in finding HF datasets."""
    return CodeAgent(
        tools=[search_datasets, get_dataset_info, preview_dataset],
        model=model,
        name="dataset_researcher",
        description=(
            "Searches HuggingFace Hub for the best fine-tuning dataset. "
            "Give it the task description, target language, and domain."
        ),
        system_prompt=DATASET_RESEARCH_PROMPT,
        max_steps=10,
        additional_authorized_imports=["json"],
    )
