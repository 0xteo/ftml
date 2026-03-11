"""Tests for Slack Block Kit formatters."""

from __future__ import annotations

from unittest.mock import MagicMock


class TestFormatTaskUnderstanding:
    def test_contains_fields(self):
        from ftml.slack.formatters import format_task_understanding

        task_info = MagicMock()
        task_info.task_type = "chat"
        task_info.language = "bg"
        task_info.domain = "customer-support"
        task_info.model_size_hint = "medium (7-9B)"

        blocks = format_task_understanding(task_info)

        assert len(blocks) == 1
        text = blocks[0]["text"]["text"]
        assert "chat" in text
        assert "bg" in text
        assert "customer-support" in text


class TestFormatResearchFindings:
    def test_contains_model_and_dataset(self):
        from ftml.slack.formatters import format_research_findings

        blocks = format_research_findings("Found model X", "Found dataset Y")

        texts = [b["text"]["text"] for b in blocks if b["type"] == "section"]
        combined = " ".join(texts)
        assert "Found model X" in combined
        assert "Found dataset Y" in combined

    def test_truncates_long_text(self):
        from ftml.slack.formatters import format_research_findings

        long_text = "x" * 2000
        blocks = format_research_findings(long_text, "short")

        model_text = blocks[0]["text"]["text"]
        assert len(model_text) < 2000
        assert model_text.endswith("...")


class TestFormatProposal:
    def test_contains_proposal_fields(self):
        from ftml.slack.formatters import format_proposal

        proposal = MagicMock()
        proposal.recommended_model = "org/model-7b"
        proposal.recommended_dataset = "org/dataset-bg"
        proposal.model_rationale = "Good multilingual support"
        proposal.dataset_rationale = "Clean Bulgarian data"
        proposal.suggested_lora_r = 16
        proposal.suggested_lora_alpha = 32
        proposal.suggested_num_epochs = 3
        proposal.suggested_learning_rate = 2e-4
        proposal.alternatives = "Alternative combo"

        blocks = format_proposal(proposal)

        all_text = str(blocks)
        assert "org/model-7b" in all_text
        assert "org/dataset-bg" in all_text
        assert "white_check_mark" in all_text


class TestFormatTrainingComplete:
    def test_contains_adapter_path(self):
        from ftml.slack.formatters import format_training_complete

        blocks = format_training_complete("/path/to/adapter")

        text = blocks[0]["text"]["text"]
        assert "/path/to/adapter" in text
        assert "Training complete" in text


class TestFormatEvalResults:
    def test_contains_scores(self):
        from ftml.slack.formatters import format_eval_results

        summary = {
            "avg_relevance": 0.85,
            "avg_fluency": 0.90,
            "avg_accuracy": 0.80,
            "num_samples": 5,
            "verdicts": {"EXCELLENT": 3, "SUFFICIENT": 2},
        }

        blocks = format_eval_results(summary)

        all_text = str(blocks)
        assert "0.85" in all_text
        assert "0.90" in all_text
        assert "EXCELLENT" in all_text
