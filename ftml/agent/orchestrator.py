"""Main orchestrator agent that coordinates research and training."""

from __future__ import annotations

from typing import TYPE_CHECKING

import dspy
from smolagents import LiteLLMModel

from ftml.agent.researcher import build_dataset_researcher, build_model_researcher
from ftml.agent.signatures import (
    EvaluateCandidate,
    GenerateProposal,
    ProposeNextExperiment,
    UnderstandTask,
)

if TYPE_CHECKING:
    from ftml.settings import Settings


class Orchestrator:
    """Coordinates the research and training pipeline."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._setup_llm(settings)

    def _setup_llm(self, settings: Settings) -> None:
        """Initialize LLM backends for smolagents and DSPy."""
        litellm_model_id = f"{settings.agent_provider}/{settings.agent_model_id}"

        # smolagents model
        self.sm_model = LiteLLMModel(
            model_id=litellm_model_id,
            api_key=settings.agent_api_key or None,
        )

        # DSPy LM (shares same litellm backend)
        self.dspy_lm = dspy.LM(
            litellm_model_id,
            api_key=settings.agent_api_key or None,
        )
        dspy.configure(lm=self.dspy_lm)

        # Build sub-agents
        self.model_researcher = build_model_researcher(self.sm_model, settings.gpu_vram_gb)
        self.dataset_researcher = build_dataset_researcher(self.sm_model)

        # DSPy modules (load optimized if available, otherwise default)
        from ftml.agent.optimize.run import load_optimized

        self.understand = load_optimized("understand_task") or dspy.ChainOfThought(UnderstandTask)
        self.evaluate = dspy.ChainOfThought(EvaluateCandidate)
        self.propose = load_optimized("generate_proposal") or dspy.ChainOfThought(GenerateProposal)
        self.propose_experiment = dspy.ChainOfThought(ProposeNextExperiment)

    def understand_task(self, user_request: str) -> dspy.Prediction:
        """Parse user request into structured task requirements using DSPy."""
        return self.understand(user_request=user_request)

    def research_models(self, task_description: str) -> str:
        """Research models via the model researcher sub-agent."""
        return self.model_researcher.run(
            f"Find best model for: {task_description}. "
            f"Hardware: {self.settings.gpu_vram_gb}GB VRAM, QLoRA.",
        )

    def research_datasets(self, task_description: str) -> str:
        """Research datasets via the dataset researcher sub-agent."""
        return self.dataset_researcher.run(f"Find best dataset for: {task_description}.")

    def research(self, task_description: str) -> tuple[str, str]:
        """Run the full research pipeline. Returns (model_findings, dataset_findings)."""
        return self.research_models(task_description), self.research_datasets(task_description)

    def generate_proposal(
        self,
        task_description: str,
        model_findings: str,
        dataset_findings: str,
    ) -> dspy.Prediction:
        """Generate a structured training proposal using DSPy."""
        return self.propose(
            task_description=task_description,
            model_findings=model_findings,
            dataset_findings=dataset_findings,
            hardware_constraints=f"GPU with {self.settings.gpu_vram_gb}GB VRAM, bf16, QLoRA 4-bit",
        )

    def regenerate_proposal(
        self,
        task_description: str,
        model_findings: str,
        dataset_findings: str,
        modification: str,
    ) -> dspy.Prediction:
        """Re-generate a proposal with user modifications applied."""
        augmented = (
            f"{task_description}\n\n"
            f"USER MODIFICATION: {modification}\n"
            f"Adjust the proposal accordingly."
        )
        return self.propose(
            task_description=augmented,
            model_findings=model_findings,
            dataset_findings=dataset_findings,
            hardware_constraints=f"GPU with {self.settings.gpu_vram_gb}GB VRAM, bf16, QLoRA 4-bit",
        )

    def propose_next_experiment(
        self,
        task_description: str,
        experiment_history: str,
        best_experiment: str,
    ) -> dspy.Prediction:
        """Propose the next hyperparameter change based on experiment history."""
        return self.propose_experiment(
            task_description=task_description,
            experiment_history=experiment_history,
            best_experiment=best_experiment,
            hardware_constraints=f"GPU with {self.settings.gpu_vram_gb}GB VRAM, bf16, QLoRA 4-bit",
        )

    def evaluate_candidate(
        self,
        task_description: str,
        candidate_info: str,
    ) -> dspy.Prediction:
        """Evaluate a single model or dataset candidate using DSPy."""
        return self.evaluate(
            task_description=task_description,
            candidate_info=candidate_info,
            hardware_constraints=f"GPU with {self.settings.gpu_vram_gb}GB VRAM",
        )
