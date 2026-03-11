"""DSPy signatures for structured agent reasoning."""

import dspy


class UnderstandTask(dspy.Signature):
    """Extract fine-tuning requirements from a user's natural language description.
    Be specific about the language, task type, and any constraints mentioned."""

    user_request: str = dspy.InputField(
        desc="The user's natural language description of what they need",
    )
    task_type: str = dspy.OutputField(
        desc="The LLM task type: chat, instruction-following, classification, summarization, translation, or other",
    )
    language: str = dspy.OutputField(
        desc="ISO 639-1 language code (e.g., bg, en, de). Use 'multi' for multilingual",
    )
    domain: str = dspy.OutputField(
        desc="Application domain (e.g., customer-support, medical, legal, general)",
    )
    model_size_hint: str = dspy.OutputField(
        desc="Suggested model size range based on task complexity: small (1-3B), medium (7-9B), large (13B+)",
    )
    search_queries: str = dspy.OutputField(
        desc="3-5 HuggingFace Hub search queries to find relevant models and datasets, one per line",
    )


class EvaluateCandidate(dspy.Signature):
    """Evaluate a model or dataset candidate for a fine-tuning task. Score based on
    relevance, quality, community adoption, and technical fit."""

    task_description: str = dspy.InputField(desc="What the user wants to achieve")
    candidate_info: str = dspy.InputField(desc="Detailed info about the model or dataset")
    hardware_constraints: str = dspy.InputField(desc="Available VRAM and hardware info")
    score: float = dspy.OutputField(desc="Suitability score from 0.0 to 10.0")
    pros: str = dspy.OutputField(desc="Key advantages, one per line")
    cons: str = dspy.OutputField(desc="Key disadvantages or risks, one per line")
    verdict: str = dspy.OutputField(desc="One sentence recommendation: use, consider, or skip")


class GenerateProposal(dspy.Signature):
    """Generate a fine-tuning proposal based on research findings. The proposal should
    recommend the best model + dataset combination and suggest training hyperparameters."""

    task_description: str = dspy.InputField()
    model_findings: str = dspy.InputField(
        desc="Research findings about candidate models with scores",
    )
    dataset_findings: str = dspy.InputField(
        desc="Research findings about candidate datasets with scores",
    )
    hardware_constraints: str = dspy.InputField()

    recommended_model: str = dspy.OutputField(
        desc="HuggingFace model ID (e.g., mistralai/Mistral-7B-v0.3)",
    )
    recommended_dataset: str = dspy.OutputField(desc="HuggingFace dataset ID")
    model_rationale: str = dspy.OutputField(
        desc="Why this model is the best choice (2-3 sentences)",
    )
    dataset_rationale: str = dspy.OutputField(
        desc="Why this dataset is the best choice (2-3 sentences)",
    )
    suggested_lora_r: int = dspy.OutputField(desc="Suggested LoRA rank (8, 16, 32, or 64)")
    suggested_lora_alpha: int = dspy.OutputField(
        desc="Suggested LoRA alpha (typically 2x the rank)",
    )
    suggested_num_epochs: int = dspy.OutputField(desc="Suggested number of training epochs (1-5)")
    suggested_learning_rate: float = dspy.OutputField(desc="Suggested learning rate (e.g., 2e-4)")
    alternatives: str = dspy.OutputField(
        desc="1-2 alternative model+dataset combos with brief rationale",
    )


class ProposeNextExperiment(dspy.Signature):
    """Propose the next hyperparameter change for an experiment loop. Analyze the
    experiment history to identify which changes helped or hurt, then suggest exactly
    one parameter change. Signal should_stop when converged or no further gains likely."""

    task_description: str = dspy.InputField(desc="What the model is being fine-tuned for")
    experiment_history: str = dspy.InputField(
        desc="TSV table of all experiments so far with eval_loss, status, and descriptions",
    )
    best_experiment: str = dspy.InputField(
        desc="Details of the best experiment so far (parameters, metrics)",
    )
    hardware_constraints: str = dspy.InputField(desc="Available VRAM and hardware info")

    parameter_to_change: str = dspy.OutputField(
        desc="Exactly one of: lora_r, lora_alpha, learning_rate, batch_size, gradient_accumulation_steps, warmup_ratio, lr_scheduler_type, lora_dropout",
    )
    new_value: str = dspy.OutputField(
        desc="The new value for the parameter (as a string, e.g. '32', '1e-4', 'linear')",
    )
    rationale: str = dspy.OutputField(
        desc="Brief explanation of why this change should improve results",
    )
    should_stop: bool = dspy.OutputField(
        desc="True if the experiments have converged and no further gains are likely",
    )


class JudgeResponse(dspy.Signature):
    """Judge quality of a fine-tuned model's response. Evaluate relevance to task,
    language fluency, and factual accuracy."""

    task_description: str = dspy.InputField(desc="What the model was fine-tuned for")
    prompt: str = dspy.InputField(desc="The input prompt given to the model")
    response: str = dspy.InputField(desc="The model's generated response")
    relevance: float = dspy.OutputField(desc="0.0-1.0 how relevant the response is to the prompt")
    fluency: float = dspy.OutputField(desc="0.0-1.0 language fluency and grammatical correctness")
    accuracy: float = dspy.OutputField(desc="0.0-1.0 factual accuracy and correctness")
    verdict: str = dspy.OutputField(desc="EXCELLENT / SUFFICIENT / PARTIAL / INSUFFICIENT")
    explanation: str = dspy.OutputField(desc="Brief explanation of the judgment")
