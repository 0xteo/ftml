"""LLM-as-judge evaluator for post-training quality assessment."""

from __future__ import annotations

from typing import TYPE_CHECKING

import dspy

from ftml.agent.signatures import JudgeResponse

if TYPE_CHECKING:
    from ftml.settings import Settings


class Evaluator:
    """Evaluates fine-tuned model outputs using LLM-as-judge via DSPy."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.judge = dspy.ChainOfThought(JudgeResponse)

    def judge_samples(
        self,
        task_description: str,
        samples: list[dict[str, str]],
    ) -> list[dspy.Prediction]:
        """Judge a list of {"prompt": ..., "response": ...} samples."""
        judgments = []
        for sample in samples:
            judgment = self.judge(
                task_description=task_description,
                prompt=sample["prompt"],
                response=sample["response"],
            )
            judgments.append(judgment)
        return judgments

    def summarize(self, judgments: list[dspy.Prediction]) -> dict:
        """Summarize judgments into aggregate scores and verdict distribution."""
        if not judgments:
            return {"avg_relevance": 0, "avg_fluency": 0, "avg_accuracy": 0, "verdicts": {}}

        relevance = [float(j.relevance) for j in judgments]
        fluency = [float(j.fluency) for j in judgments]
        accuracy = [float(j.accuracy) for j in judgments]

        verdicts: dict[str, int] = {}
        for j in judgments:
            v = str(j.verdict).strip().upper()
            verdicts[v] = verdicts.get(v, 0) + 1

        return {
            "avg_relevance": sum(relevance) / len(relevance),
            "avg_fluency": sum(fluency) / len(fluency),
            "avg_accuracy": sum(accuracy) / len(accuracy),
            "verdicts": verdicts,
            "num_samples": len(judgments),
        }
