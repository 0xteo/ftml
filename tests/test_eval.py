"""Tests for evaluation module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestGenerateSamples:
    @patch("peft.PeftModel")
    @patch("ftml.model.load_model_and_tokenizer")
    def test_generates_responses(self, mock_load, mock_peft, mock_settings):
        import torch

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_load.return_value = (mock_model, mock_tokenizer)

        # PeftModel wrapping
        mock_peft_model = MagicMock()
        mock_peft.from_pretrained.return_value = mock_peft_model

        # Tokenizer returns input IDs
        mock_input_ids = torch.tensor([[1, 2, 3]])
        mock_encoded = MagicMock()
        mock_encoded.__getitem__ = lambda self, key: mock_input_ids
        mock_encoded.to.return_value = mock_encoded
        mock_tokenizer.return_value = mock_encoded

        # Model generates tokens
        mock_peft_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6]])
        mock_peft_model.device = "cpu"
        mock_tokenizer.decode.return_value = "Generated response"

        from ftml.eval import generate_samples

        results = generate_samples(mock_settings, "/fake/adapter", ["Hello"])

        assert len(results) == 1
        assert results[0]["prompt"] == "Hello"
        assert results[0]["response"] == "Generated response"


class TestComputePerplexity:
    def test_computes_perplexity(self):
        import torch

        from ftml.eval import compute_perplexity

        mock_model = MagicMock()
        mock_model.device = "cpu"

        mock_outputs = MagicMock()
        mock_outputs.loss = MagicMock()
        mock_outputs.loss.item.return_value = 2.0  # cross-entropy loss
        mock_model.return_value = mock_outputs

        mock_tokenizer = MagicMock()
        mock_inputs = MagicMock()
        mock_inputs.__getitem__ = lambda self, key: torch.tensor([[1, 2, 3, 4, 5]])
        mock_inputs.to.return_value = mock_inputs
        mock_tokenizer.return_value = mock_inputs

        result = compute_perplexity(mock_model, mock_tokenizer, ["test text"])

        assert result > 0
        assert result == pytest.approx(torch.exp(torch.tensor(2.0)).item())

    def test_empty_texts(self):
        import math

        from ftml.eval import compute_perplexity

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        result = compute_perplexity(mock_model, mock_tokenizer, [])

        assert math.isinf(result)


class TestEvaluator:
    @patch("ftml.agent.evaluator.dspy")
    def test_judge_samples(self, mock_dspy, mock_settings):
        from ftml.agent.evaluator import Evaluator

        mock_judgment = MagicMock()
        mock_judgment.relevance = 0.9
        mock_judgment.fluency = 0.8
        mock_judgment.accuracy = 0.85
        mock_judgment.verdict = "EXCELLENT"
        mock_judgment.explanation = "Good response"

        mock_dspy.ChainOfThought.return_value = MagicMock(return_value=mock_judgment)

        evaluator = Evaluator(mock_settings)
        samples = [{"prompt": "Hello", "response": "Hi there"}]
        judgments = evaluator.judge_samples("chatbot task", samples)

        assert len(judgments) == 1
        assert judgments[0].relevance == 0.9

    @patch("ftml.agent.evaluator.dspy")
    def test_summarize(self, mock_dspy, mock_settings):
        from ftml.agent.evaluator import Evaluator

        evaluator = Evaluator(mock_settings)

        j1 = MagicMock(relevance=0.9, fluency=0.8, accuracy=0.7, verdict="EXCELLENT")
        j2 = MagicMock(relevance=0.6, fluency=0.7, accuracy=0.5, verdict="SUFFICIENT")

        summary = evaluator.summarize([j1, j2])

        assert summary["avg_relevance"] == pytest.approx(0.75)
        assert summary["avg_fluency"] == pytest.approx(0.75)
        assert summary["avg_accuracy"] == pytest.approx(0.6)
        assert summary["verdicts"]["EXCELLENT"] == 1
        assert summary["verdicts"]["SUFFICIENT"] == 1
        assert summary["num_samples"] == 2

    @patch("ftml.agent.evaluator.dspy")
    def test_summarize_empty(self, mock_dspy, mock_settings):
        from ftml.agent.evaluator import Evaluator

        evaluator = Evaluator(mock_settings)
        summary = evaluator.summarize([])

        assert summary["avg_relevance"] == 0
        assert summary["verdicts"] == {}
