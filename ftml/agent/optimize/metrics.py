"""Metric functions for DSPy signature optimization."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import dspy

# Valid values for categorical fields
VALID_TASK_TYPES = {
    "chat",
    "instruction-following",
    "classification",
    "summarization",
    "translation",
    "other",
}
VALID_MODEL_SIZES = {"small (1-3B)", "medium (7-9B)", "large (13B+)"}
VALID_VERDICTS = {"EXCELLENT", "SUFFICIENT", "PARTIAL", "INSUFFICIENT"}


def understand_task_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
    """Metric for UnderstandTask: checks field validity and exact matches."""
    score = 0.0
    total = 5.0

    # task_type: must be valid
    if str(prediction.task_type).strip().lower() in {t.lower() for t in VALID_TASK_TYPES}:
        score += 1.0
    if hasattr(example, "task_type") and prediction.task_type == example.task_type:
        score += 0.5

    # language: must be 2-5 char ISO code or 'multi'
    lang = str(prediction.language).strip().lower()
    if 2 <= len(lang) <= 5 or lang == "multi":
        score += 1.0
    if hasattr(example, "language") and prediction.language == example.language:
        score += 0.5

    # domain: non-empty
    if len(str(prediction.domain).strip()) > 0:
        score += 1.0

    # model_size_hint: must be valid
    if str(prediction.model_size_hint).strip() in VALID_MODEL_SIZES:
        score += 1.0

    # search_queries: at least 2 queries
    queries = [q.strip() for q in str(prediction.search_queries).split("\n") if q.strip()]
    if len(queries) >= 2:
        score += 1.0

    return score / (total + 1.0)  # +1.0 for the bonus exact-match points


def generate_proposal_metric(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace=None,
) -> float:
    """Metric for GenerateProposal: checks structural validity."""
    score = 0.0
    total = 6.0

    # recommended_model: looks like a HF model ID (org/name)
    model_id = str(prediction.recommended_model).strip()
    if "/" in model_id and len(model_id) > 3:
        score += 1.0

    # recommended_dataset: looks like a HF dataset ID
    dataset_id = str(prediction.recommended_dataset).strip()
    if "/" in dataset_id and len(dataset_id) > 3:
        score += 1.0

    # rationales: non-empty and substantive (>20 chars)
    if len(str(prediction.model_rationale).strip()) > 20:
        score += 1.0
    if len(str(prediction.dataset_rationale).strip()) > 20:
        score += 1.0

    # LoRA params: reasonable ranges
    try:
        lora_r = int(prediction.suggested_lora_r)
        if lora_r in {4, 8, 16, 32, 64, 128}:
            score += 0.5
    except ValueError, TypeError:
        pass

    try:
        epochs = int(prediction.suggested_num_epochs)
        if 1 <= epochs <= 10:
            score += 0.5
    except ValueError, TypeError:
        pass

    try:
        lr = float(prediction.suggested_learning_rate)
        if 1e-6 <= lr <= 1e-2:
            score += 0.5
    except ValueError, TypeError:
        pass

    # alternatives: non-empty
    if len(str(prediction.alternatives).strip()) > 10:
        score += 0.5

    return score / total


def judge_response_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
    """Metric for JudgeResponse: checks score ranges and verdict validity."""
    score = 0.0
    total = 5.0

    # Score fields: must be 0.0-1.0
    for field in ("relevance", "fluency", "accuracy"):
        try:
            val = float(getattr(prediction, field))
            if 0.0 <= val <= 1.0:
                score += 1.0
                # Bonus for being close to example
                if hasattr(example, field):
                    expected = float(getattr(example, field))
                    if abs(val - expected) < 0.2:
                        score += 0.25
        except ValueError, TypeError, AttributeError:
            pass

    # Verdict: must be valid
    verdict = str(prediction.verdict).strip().upper()
    if verdict in VALID_VERDICTS:
        score += 1.0
        if hasattr(example, "verdict") and verdict == str(example.verdict).strip().upper():
            score += 0.5

    # Explanation: non-empty and substantive
    if len(str(prediction.explanation).strip()) > 10:
        score += 1.0

    return score / (total + 1.25)  # +1.25 for bonus points
