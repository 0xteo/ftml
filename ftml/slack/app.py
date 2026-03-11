"""FastAPI app with Slack webhook endpoint."""

from __future__ import annotations

import hashlib
import hmac
import logging
import time

from fastapi import FastAPI, Header, Request, Response

from ftml.settings import Settings
from ftml.slack.handlers import handle_message, handle_reaction

logger = logging.getLogger(__name__)

app = FastAPI(title="ftml Slack Bot")

_settings: Settings | None = None
_client = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def get_slack_client():
    global _client
    if _client is None:
        from slack_sdk import WebClient

        _client = WebClient(token=get_settings().slack_bot_token)
    return _client


def verify_slack_signature(
    body: bytes,
    timestamp: str,
    signature: str,
    signing_secret: str,
) -> bool:
    """Verify that the request came from Slack."""
    if abs(time.time() - float(timestamp)) > 300:
        return False
    base = f"v0:{timestamp}:{body.decode()}"
    expected = "v0=" + hmac.new(signing_secret.encode(), base.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


@app.post("/slack/events")
async def slack_events(
    request: Request,
    x_slack_request_timestamp: str = Header(""),
    x_slack_signature: str = Header(""),
) -> Response:
    """Handle Slack Events API webhook."""
    body = await request.body()
    settings = get_settings()

    # Verify signature
    if settings.slack_signing_secret and not verify_slack_signature(
        body,
        x_slack_request_timestamp,
        x_slack_signature,
        settings.slack_signing_secret,
    ):
        return Response(status_code=401, content="Invalid signature")

    payload = await request.json()

    # URL verification challenge
    if payload.get("type") == "url_verification":
        return Response(content=payload["challenge"], media_type="text/plain")

    # Event callback
    if payload.get("type") == "event_callback":
        event = payload.get("event", {})
        event_type = event.get("type")

        # Ignore bot messages
        if event.get("bot_id"):
            return Response(status_code=200)

        if event_type == "message" and "subtype" not in event:
            client = get_slack_client()
            await handle_message(
                client=client,
                settings=settings,
                channel=event["channel"],
                thread_ts=event.get("thread_ts", event["ts"]),
                text=event.get("text", ""),
            )

        elif event_type == "reaction_added":
            client = get_slack_client()
            await handle_reaction(
                client=client,
                settings=settings,
                reaction=event.get("reaction", ""),
                item_ts=event.get("item", {}).get("ts", ""),
            )

    return Response(status_code=200)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create the FastAPI app with optional settings override."""
    global _settings
    if settings is not None:
        _settings = settings
    return app
