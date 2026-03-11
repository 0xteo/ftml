"""Throttled progress reporter for Slack message updates."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from slack_sdk import WebClient


class ProgressReporter:
    """Throttled progress updates via Slack message edits."""

    def __init__(
        self,
        client: WebClient,
        channel: str,
        thread_ts: str,
        min_interval: float = 3.0,
    ) -> None:
        self.client = client
        self.channel = channel
        self.thread_ts = thread_ts
        self.min_interval = min_interval
        self._message_ts: str | None = None
        self._last_update: float = 0

    def update(self, text: str, force: bool = False) -> None:
        """Post or update a progress message, throttled to min_interval."""
        now = time.monotonic()
        if not force and (now - self._last_update) < self.min_interval:
            return

        if self._message_ts:
            self.client.chat_update(
                channel=self.channel,
                ts=self._message_ts,
                text=text,
            )
        else:
            resp = self.client.chat_postMessage(
                channel=self.channel,
                thread_ts=self.thread_ts,
                text=text,
            )
            self._message_ts = resp["ts"]

        self._last_update = now

    def finish(self, text: str) -> None:
        """Send a final update (always posts)."""
        self.update(text, force=True)
