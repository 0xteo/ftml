"""DSPy MIPROv2 optimization runner for agent signatures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import dspy

from ftml.agent.optimize.metrics import (
    generate_proposal_metric,
    judge_response_metric,
    understand_task_metric,
)
from ftml.agent.signatures import GenerateProposal, JudgeResponse, UnderstandTask

if TYPE_CHECKING:
    from ftml.settings import Settings

EXAMPLES_PATH = Path(__file__).parent / "examples.json"
OPTIMIZED_DIR = Path(__file__).parent / "optimized"


def _load_examples() -> dict[str, list[dspy.Example]]:
    """Load curated examples from JSON and convert to DSPy Examples."""
    with EXAMPLES_PATH.open() as f:
        raw = json.load(f)

    result = {}
    for key, items in raw.items():
        examples = []
        for item in items:
            ex = dspy.Example(**item)
            # Mark input fields based on signature
            if key == "understand_task":
                ex = ex.with_inputs("user_request")
            elif key == "generate_proposal":
                ex = ex.with_inputs(
                    "task_description",
                    "model_findings",
                    "dataset_findings",
                    "hardware_constraints",
                )
            elif key == "judge_response":
                ex = ex.with_inputs("task_description", "prompt", "response")
            examples.append(ex)
        result[key] = examples

    return result


def optimize_signature(
    name: str,
    signature_cls: type[dspy.Signature],
    metric,
    examples: list[dspy.Example],
    num_candidates: int = 7,
    max_bootstrapped_demos: int = 3,
    max_labeled_demos: int = 4,
) -> dspy.Module:
    """Optimize a single signature using MIPROv2."""
    module = dspy.ChainOfThought(signature_cls)

    optimizer = dspy.MIPROv2(
        metric=metric,
        num_candidates=num_candidates,
        max_bootstrapped_demos=max_bootstrapped_demos,
        max_labeled_demos=max_labeled_demos,
        auto="light",
    )

    return optimizer.compile(
        module,
        trainset=examples,
    )


def save_optimized(module: dspy.Module, name: str) -> Path:
    """Save an optimized module to disk."""
    OPTIMIZED_DIR.mkdir(parents=True, exist_ok=True)
    path = OPTIMIZED_DIR / f"{name}.json"
    module.save(str(path))
    return path


def load_optimized(name: str) -> dspy.Module | None:
    """Load an optimized module from disk, or return None if not found."""
    path = OPTIMIZED_DIR / f"{name}.json"
    if not path.exists():
        return None
    module = dspy.ChainOfThought(
        {
            "understand_task": UnderstandTask,
            "generate_proposal": GenerateProposal,
            "judge_response": JudgeResponse,
        }[name],
    )
    module.load(str(path))
    return module


def run_optimization(settings: Settings) -> dict[str, Path]:
    """Run full optimization for all signatures. Returns paths to saved modules."""
    # Configure DSPy LM
    litellm_model_id = f"{settings.agent_provider}/{settings.agent_model_id}"
    lm = dspy.LM(litellm_model_id, api_key=settings.agent_api_key or None)
    dspy.configure(lm=lm)

    examples = _load_examples()

    configs = [
        ("understand_task", UnderstandTask, understand_task_metric),
        ("generate_proposal", GenerateProposal, generate_proposal_metric),
        ("judge_response", JudgeResponse, judge_response_metric),
    ]

    results = {}
    for name, sig_cls, metric in configs:
        print(f"Optimizing {name}...")
        optimized = optimize_signature(name, sig_cls, metric, examples[name])
        path = save_optimized(optimized, name)
        results[name] = path
        print(f"  Saved to {path}")

    return results
