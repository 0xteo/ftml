"""Tests for Slack event handlers."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ftml.settings import Settings


@pytest.fixture
def slack_settings(tmp_path):
    return Settings(
        hf_token="hf_test",
        model_name="test/model",
        dataset_name="test/dataset",
        output_dir=tmp_path / "outputs",
        slack_bot_token="xoxb-test",
        slack_signing_secret="test-secret",
        agent_model_id="test-model",
        agent_provider="test",
        agent_api_key="test-key",
        agent_max_steps=3,
    )


class TestHandleMessage:
    @pytest.mark.asyncio
    async def test_creates_background_task(self, slack_settings):
        from ftml.slack.handlers import handle_message

        mock_client = MagicMock()

        with patch(
            "ftml.slack.handlers._process_pipeline",
            new_callable=AsyncMock,
        ) as mock_pipeline:
            await handle_message(
                client=mock_client,
                settings=slack_settings,
                channel="C123",
                thread_ts="1234.5678",
                text="Build a Bulgarian chatbot",
            )

            # Give the task a chance to start
            await asyncio.sleep(0.1)

            mock_pipeline.assert_called_once_with(
                mock_client,
                slack_settings,
                "C123",
                "1234.5678",
                "Build a Bulgarian chatbot",
            )


class TestHandleReaction:
    @pytest.mark.asyncio
    async def test_ignores_non_checkmark(self, slack_settings):
        from ftml.slack.handlers import handle_reaction

        mock_client = MagicMock()

        with patch("ftml.slack.handlers._run_training_task", new_callable=AsyncMock) as mock_train:
            await handle_reaction(
                client=mock_client,
                settings=slack_settings,
                reaction="thumbsup",
                item_ts="1234.5678",
            )

            mock_train.assert_not_called()

    @pytest.mark.asyncio
    async def test_approves_pending_proposal(self, slack_settings):
        from ftml.slack.handlers import _pending_proposals, handle_reaction

        mock_client = MagicMock()
        mock_proposal = MagicMock()

        _pending_proposals["1234.5678"] = {
            "settings": slack_settings,
            "proposal": mock_proposal,
            "task_desc": "test task",
            "channel": "C123",
            "thread_ts": "1234.0000",
        }

        with patch("ftml.slack.handlers._run_training_task", new_callable=AsyncMock) as mock_train:
            await handle_reaction(
                client=mock_client,
                settings=slack_settings,
                reaction="white_check_mark",
                item_ts="1234.5678",
            )

            await asyncio.sleep(0.1)
            mock_train.assert_called_once()

        # Proposal should be consumed
        assert "1234.5678" not in _pending_proposals
