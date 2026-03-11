"""Tests for the Orchestrator agent."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from ftml.settings import Settings


@pytest.fixture
def _mock_llm():
    """Patch all external LLM/agent dependencies."""
    with (
        patch("ftml.agent.orchestrator.LiteLLMModel") as mock_litellm,
        patch("ftml.agent.orchestrator.dspy") as mock_dspy,
        patch("ftml.agent.orchestrator.build_model_researcher") as mock_build_model,
        patch("ftml.agent.orchestrator.build_dataset_researcher") as mock_build_dataset,
    ):
        mock_litellm.return_value = MagicMock()
        mock_dspy.LM.return_value = MagicMock()
        mock_dspy.ChainOfThought.return_value = MagicMock()
        mock_build_model.return_value = MagicMock(name="model_researcher")
        mock_build_dataset.return_value = MagicMock(name="dataset_researcher")
        yield {
            "litellm": mock_litellm,
            "dspy": mock_dspy,
            "build_model": mock_build_model,
            "build_dataset": mock_build_dataset,
        }


class TestOrchestratorInit:
    @pytest.mark.usefixtures("_mock_llm")
    def test_creates_sub_agents(self, mock_agent_settings: Settings, _mock_llm):
        from ftml.agent.orchestrator import Orchestrator

        Orchestrator(mock_agent_settings)

        _mock_llm["build_model"].assert_called_once()
        _mock_llm["build_dataset"].assert_called_once()


class TestUnderstandTask:
    @pytest.mark.usefixtures("_mock_llm")
    def test_calls_dspy(self, mock_agent_settings: Settings, _mock_llm):
        from ftml.agent.orchestrator import Orchestrator

        orch = Orchestrator(mock_agent_settings)
        orch.understand_task("I need a Bulgarian chatbot")

        orch.understand.assert_called_once_with(user_request="I need a Bulgarian chatbot")


class TestResearch:
    @pytest.mark.usefixtures("_mock_llm")
    def test_returns_model_and_dataset_findings(self, mock_agent_settings: Settings, _mock_llm):
        from ftml.agent.orchestrator import Orchestrator

        orch = Orchestrator(mock_agent_settings)
        orch.model_researcher.run.return_value = "Found model X"
        orch.dataset_researcher.run.return_value = "Found dataset Y"

        model_findings, dataset_findings = orch.research("Bulgarian chatbot")

        orch.model_researcher.run.assert_called_once()
        orch.dataset_researcher.run.assert_called_once()
        assert model_findings == "Found model X"
        assert dataset_findings == "Found dataset Y"


class TestGenerateProposal:
    @pytest.mark.usefixtures("_mock_llm")
    def test_calls_propose(self, mock_agent_settings: Settings, _mock_llm):
        from ftml.agent.orchestrator import Orchestrator

        orch = Orchestrator(mock_agent_settings)
        orch.generate_proposal("task desc", "model info", "dataset info")

        orch.propose.assert_called_once_with(
            task_description="task desc",
            model_findings="model info",
            dataset_findings="dataset info",
            hardware_constraints=f"GPU with {mock_agent_settings.gpu_vram_gb}GB VRAM, bf16, QLoRA 4-bit",
        )


class TestRegenerateProposal:
    @pytest.mark.usefixtures("_mock_llm")
    def test_prepends_modification(self, mock_agent_settings: Settings, _mock_llm):
        from ftml.agent.orchestrator import Orchestrator

        orch = Orchestrator(mock_agent_settings)
        orch.regenerate_proposal(
            "task desc",
            "model info",
            "dataset info",
            modification="use Llama instead",
        )

        call_args = orch.propose.call_args
        assert "use Llama instead" in call_args.kwargs["task_description"]
        assert call_args.kwargs["model_findings"] == "model info"
