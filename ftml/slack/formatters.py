"""Slack Block Kit formatters for pipeline messages."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import dspy


def format_task_understanding(task_info: dspy.Prediction) -> list[dict[str, Any]]:
    """Format task understanding results as Block Kit blocks."""
    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"*Task understood:*\n"
                    f"- Type: `{task_info.task_type}`\n"
                    f"- Language: `{task_info.language}`\n"
                    f"- Domain: `{task_info.domain}`\n"
                    f"- Size hint: `{task_info.model_size_hint}`"
                ),
            },
        },
    ]


def format_research_findings(model_findings: str, dataset_findings: str) -> list[dict[str, Any]]:
    """Format research findings as Block Kit blocks."""
    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Model Research:*\n{_truncate(model_findings, 1500)}",
            },
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Dataset Research:*\n{_truncate(dataset_findings, 1500)}",
            },
        },
    ]


def format_proposal(proposal: dspy.Prediction) -> list[dict[str, Any]]:
    """Format a training proposal as Block Kit blocks."""
    return [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "Training Proposal"},
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*Model:*\n`{proposal.recommended_model}`"},
                {"type": "mrkdwn", "text": f"*Dataset:*\n`{proposal.recommended_dataset}`"},
            ],
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"*Model rationale:* {proposal.model_rationale}\n"
                    f"*Dataset rationale:* {proposal.dataset_rationale}"
                ),
            },
        },
        {
            "type": "section",
            "fields": [
                {"type": "mrkdwn", "text": f"*LoRA rank:* {proposal.suggested_lora_r}"},
                {"type": "mrkdwn", "text": f"*LoRA alpha:* {proposal.suggested_lora_alpha}"},
                {"type": "mrkdwn", "text": f"*Epochs:* {proposal.suggested_num_epochs}"},
                {"type": "mrkdwn", "text": f"*LR:* {proposal.suggested_learning_rate}"},
            ],
        },
        {"type": "divider"},
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"*Alternatives:* {_truncate(proposal.alternatives, 500)}",
                },
            ],
        },
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": "React with :white_check_mark: to approve and start training.",
                },
            ],
        },
    ]


def format_training_complete(adapter_path: str) -> list[dict[str, Any]]:
    """Format training completion message."""
    return [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f":tada: *Training complete!*\nAdapter saved to: `{adapter_path}`",
            },
        },
    ]


def format_eval_results(summary: dict) -> list[dict[str, Any]]:
    """Format evaluation results as Block Kit blocks."""
    verdicts_str = ", ".join(f"{k}: {v}" for k, v in summary.get("verdicts", {}).items())
    return [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": "Evaluation Results"},
        },
        {
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": f"*Relevance:* {summary.get('avg_relevance', 0):.2f}",
                },
                {"type": "mrkdwn", "text": f"*Fluency:* {summary.get('avg_fluency', 0):.2f}"},
                {"type": "mrkdwn", "text": f"*Accuracy:* {summary.get('avg_accuracy', 0):.2f}"},
                {
                    "type": "mrkdwn",
                    "text": f"*Samples:* {summary.get('num_samples', 0)}",
                },
            ],
        },
        {
            "type": "context",
            "elements": [{"type": "mrkdwn", "text": f"*Verdicts:* {verdicts_str}"}],
        },
    ]


def _truncate(text: str, max_len: int) -> str:
    """Truncate text to max length, adding ellipsis if needed."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."
