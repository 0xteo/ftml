"""Slack event handlers and background task processing."""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from ftml.slack.formatters import (
    format_proposal,
    format_research_findings,
    format_task_understanding,
    format_training_complete,
)
from ftml.slack.progress import ProgressReporter

if TYPE_CHECKING:
    from slack_sdk import WebClient

    from ftml.settings import Settings

logger = logging.getLogger(__name__)

# Track background tasks to prevent GC
_background_tasks: set[asyncio.Task] = set()


async def handle_message(
    client: WebClient,
    settings: Settings,
    channel: str,
    thread_ts: str,
    text: str,
) -> None:
    """Process a user message: research, propose, and await approval."""
    task = asyncio.create_task(
        _process_pipeline(client, settings, channel, thread_ts, text),
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def _process_pipeline(
    client: WebClient,
    settings: Settings,
    channel: str,
    thread_ts: str,
    text: str,
) -> None:
    """Run the full research + proposal pipeline in a background task."""
    from ftml.agent.orchestrator import Orchestrator

    progress = ProgressReporter(client, channel, thread_ts)

    try:
        progress.update(":brain: Initializing agents...")
        orch = await asyncio.to_thread(Orchestrator, settings)

        # Step 1: Understand task
        progress.update(":mag: Understanding your request...")
        task_info = await asyncio.to_thread(orch.understand_task, text)

        client.chat_postMessage(
            channel=channel,
            thread_ts=thread_ts,
            blocks=format_task_understanding(task_info),
            text=f"Task: {task_info.task_type}, Language: {task_info.language}",
        )

        # Step 2: Research
        progress.update(":mag_right: Researching models and datasets...")
        task_desc = (
            f"Task: {task_info.task_type}, Language: {task_info.language}, "
            f"Domain: {task_info.domain}, Size: {task_info.model_size_hint}. "
            f"Original request: {text}"
        )

        model_findings, dataset_findings = await asyncio.to_thread(orch.research, task_desc)

        client.chat_postMessage(
            channel=channel,
            thread_ts=thread_ts,
            blocks=format_research_findings(model_findings, dataset_findings),
            text="Research complete",
        )

        # Step 3: Generate proposal
        progress.update(":memo: Generating training proposal...")
        proposal = await asyncio.to_thread(
            orch.generate_proposal,
            task_desc,
            model_findings,
            dataset_findings,
        )

        # Post proposal with approval instructions
        resp = client.chat_postMessage(
            channel=channel,
            thread_ts=thread_ts,
            blocks=format_proposal(proposal),
            text=f"Proposal: {proposal.recommended_model} + {proposal.recommended_dataset}",
        )

        progress.finish(
            ":white_check_mark: Proposal ready! React with :white_check_mark: to approve.",
        )

        # Store proposal context for reaction handler
        _pending_proposals[resp["ts"]] = {
            "settings": settings,
            "proposal": proposal,
            "task_desc": task_desc,
            "channel": channel,
            "thread_ts": thread_ts,
        }

    except Exception:
        logger.exception("Pipeline failed")
        client.chat_postMessage(
            channel=channel,
            thread_ts=thread_ts,
            text=":x: Pipeline failed. Check server logs for details.",
        )


# Pending proposals awaiting approval (message_ts -> context)
_pending_proposals: dict[str, dict] = {}


async def handle_reaction(
    client: WebClient,
    settings: Settings,
    reaction: str,
    item_ts: str,
) -> None:
    """Handle a reaction_added event — approve training if checkmark on a proposal."""
    if reaction != "white_check_mark":
        return

    context = _pending_proposals.pop(item_ts, None)
    if not context:
        return

    task = asyncio.create_task(
        _run_training_task(client, context),
    )
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


async def _run_training_task(client: WebClient, context: dict) -> None:
    """Run training in a background thread."""
    from ftml.data import format_for_sft, load_dataset_from_hf
    from ftml.model import load_model_and_tokenizer
    from ftml.train import build_lora_config, build_training_args, save_adapter, train

    channel = context["channel"]
    thread_ts = context["thread_ts"]
    settings: Settings = context["settings"]
    proposal = context["proposal"]

    progress = ProgressReporter(client, channel, thread_ts)

    try:
        from ftml.settings import Settings as SettingsCls

        train_settings = SettingsCls(
            model_name=proposal.recommended_model,
            dataset_name=proposal.recommended_dataset,
            lora_r=int(proposal.suggested_lora_r),
            lora_alpha=int(proposal.suggested_lora_alpha),
            num_epochs=int(proposal.suggested_num_epochs),
            learning_rate=float(proposal.suggested_learning_rate),
            hf_token=settings.hf_token,
            use_4bit=settings.use_4bit,
            max_seq_length=settings.max_seq_length,
            batch_size=settings.batch_size,
            output_dir=settings.output_dir,
        )

        progress.update(":hourglass: Loading model...")
        model, tokenizer = await asyncio.to_thread(
            load_model_and_tokenizer,
            train_settings.model_name,
            train_settings.hf_token,
            use_4bit=train_settings.use_4bit,
            max_seq_length=train_settings.max_seq_length,
            use_unsloth=train_settings.use_unsloth,
            use_flash_attention=train_settings.use_flash_attention,
        )

        progress.update(":hourglass: Loading dataset...")
        ds = await asyncio.to_thread(
            load_dataset_from_hf,
            train_settings.dataset_name,
            train_settings.hf_token,
        )
        train_ds = format_for_sft(ds["train"], tokenizer)

        progress.update(f":runner: Training on {len(train_ds):,} samples...")
        lora_config = build_lora_config(train_settings)
        training_args = build_training_args(train_settings)
        trainer = await asyncio.to_thread(
            train,
            model,
            tokenizer,
            train_ds,
            training_args,
            lora_config,
        )

        adapter_path = save_adapter(trainer, train_settings.output_dir)

        client.chat_postMessage(
            channel=channel,
            thread_ts=thread_ts,
            blocks=format_training_complete(str(adapter_path)),
            text=f"Training complete! Adapter: {adapter_path}",
        )
        progress.finish(":tada: Training complete!")

    except Exception:
        logger.exception("Training failed")
        client.chat_postMessage(
            channel=channel,
            thread_ts=thread_ts,
            text=":x: Training failed. Check server logs.",
        )
