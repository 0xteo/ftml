"""Interactive CLI for the agentic fine-tuning pipeline."""

import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.theme import Theme

from ftml.settings import Settings

theme = Theme(
    {
        "info": "cyan",
        "success": "green",
        "warning": "yellow",
        "error": "red bold",
        "step": "dim cyan",
    },
)
console = Console(theme=theme)


def print_banner() -> None:
    console.print(
        Panel(
            "[bold]ftml[/bold] — agentic fine-tuning pipeline\n"
            "[dim]Describe what you need. I'll find the best model, dataset, and train it.[/dim]",
            border_style="cyan",
        ),
    )


def print_proposal(proposal) -> None:
    """Pretty-print a training proposal."""
    text = (
        f"**Model:** `{proposal.recommended_model}`\n"
        f"{proposal.model_rationale}\n\n"
        f"**Dataset:** `{proposal.recommended_dataset}`\n"
        f"{proposal.dataset_rationale}\n\n"
        f"**Config:**\n"
        f"- LoRA rank: {proposal.suggested_lora_r}\n"
        f"- LoRA alpha: {proposal.suggested_lora_alpha}\n"
        f"- Epochs: {proposal.suggested_num_epochs}\n"
        f"- Learning rate: {proposal.suggested_learning_rate}\n\n"
        f"**Alternatives:**\n{proposal.alternatives}"
    )
    console.print(Panel(Markdown(text), title="Proposal", border_style="green"))


def run_interactive() -> None:
    """Main interactive loop."""
    settings = Settings()
    print_banner()

    console.print(f"[dim]Agent: {settings.agent_model_id} via {settings.agent_provider}[/dim]")
    console.print(f"[dim]GPU: {settings.gpu_vram_gb}GB VRAM[/dim]\n")

    # Lazy import to avoid loading heavy deps at startup
    from ftml.agent.orchestrator import Orchestrator

    with console.status("[info]Initializing agents...[/info]"):
        orch = Orchestrator(settings)

    while True:
        try:
            user_input = console.input("[bold cyan]ftml>[/bold cyan] ").strip()
        except EOFError, KeyboardInterrupt:
            console.print("\n[dim]Bye![/dim]")
            sys.exit(0)

        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            console.print("[dim]Bye![/dim]")
            break
        if user_input.lower() == "help":
            console.print(
                "[info]Describe what you want to fine-tune an LLM for.\n"
                "Examples:\n"
                '  "I need a Bulgarian chatbot for customer support"\n'
                '  "Build a code assistant that understands Python and Go"\n'
                '  "Fine-tune a model for medical Q&A in German"\n\n'
                "Commands: help, train, quit[/info]",
            )
            continue

        # Step 1: Understand the task via DSPy
        console.print("\n[step]Understanding your request...[/step]")
        task_info = orch.understand_task(user_input)
        console.print(
            f"  Task: [bold]{task_info.task_type}[/bold] | "
            f"Language: [bold]{task_info.language}[/bold] | "
            f"Domain: [bold]{task_info.domain}[/bold] | "
            f"Size: [bold]{task_info.model_size_hint}[/bold]",
        )

        # Step 2: Research via smolagents
        console.print("\n[step]Researching models and datasets...[/step]")
        task_desc = (
            f"Task: {task_info.task_type}, Language: {task_info.language}, "
            f"Domain: {task_info.domain}, Size: {task_info.model_size_hint}. "
            f"Original request: {user_input}"
        )

        model_findings, dataset_findings = orch.research(task_desc)
        console.print(Panel(str(model_findings), title="Model Research", border_style="blue"))
        console.print(Panel(str(dataset_findings), title="Dataset Research", border_style="blue"))

        # Step 3-4: Proposal loop (generate, approve/modify/reject)
        modification = None
        while True:
            console.print("\n[step]Generating training proposal...[/step]")
            if modification:
                proposal = orch.regenerate_proposal(
                    task_desc,
                    model_findings,
                    dataset_findings,
                    modification,
                )
            else:
                proposal = orch.generate_proposal(
                    task_description=task_desc,
                    model_findings=model_findings,
                    dataset_findings=dataset_findings,
                )
            print_proposal(proposal)

            action = (
                console.input(
                    "\n[bold]approve[/bold] / [bold]experiment[/bold] / [bold]modify[/bold] / [bold]reject[/bold]: ",
                )
                .strip()
                .lower()
            )

            if action == "reject":
                console.print("[warning]Rejected. Describe what you'd like differently.[/warning]")
                break

            if action == "modify":
                console.print(
                    "[info]Tell me what to change "
                    "(e.g., 'use Llama instead', 'increase epochs'):[/info]",
                )
                modification = console.input("[bold cyan]ftml>[/bold cyan] ").strip()
                console.print(f"[warning]Modification noted: {modification}[/warning]")
                continue

            if action in {"approve", "yes", "y", ""}:
                adapter_path = _run_training(settings, proposal)
                if adapter_path:
                    _maybe_evaluate(settings, proposal, task_desc, adapter_path)
                break

            if action in {"experiment", "exp", "e"}:
                best = _run_experiment_loop(settings, proposal, task_desc, orch)
                if best:
                    _maybe_evaluate_experiment(settings, proposal, task_desc, best)
                break

            console.print(
                "[warning]Unknown action. Try: approve, experiment, modify, reject[/warning]",
            )
            break


def _run_training(settings: Settings, proposal) -> str | None:
    """Execute fine-tuning with the proposed configuration. Returns adapter path."""
    console.print("\n[success]Starting fine-tuning...[/success]")

    # Override settings with proposal values
    train_settings = Settings(
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

    from ftml.data import format_for_sft, load_dataset_from_hf
    from ftml.model import load_model_and_tokenizer
    from ftml.train import build_lora_config, build_training_args, save_adapter, train

    with console.status(f"[info]Loading model: {train_settings.model_name}[/info]"):
        model, tokenizer = load_model_and_tokenizer(
            train_settings.model_name,
            train_settings.hf_token,
            use_4bit=train_settings.use_4bit,
            max_seq_length=train_settings.max_seq_length,
            use_unsloth=train_settings.use_unsloth,
            use_flash_attention=train_settings.use_flash_attention,
        )

    with console.status(f"[info]Loading dataset: {train_settings.dataset_name}[/info]"):
        ds = load_dataset_from_hf(train_settings.dataset_name, train_settings.hf_token)
        train_ds = format_for_sft(ds["train"], tokenizer)

    console.print(f"[info]Train samples: {len(train_ds):,}[/info]")

    lora_config = build_lora_config(train_settings)
    training_args = build_training_args(train_settings)

    console.print("[info]Training...[/info]")
    trainer = train(model, tokenizer, train_ds, training_args, peft_config=lora_config)

    adapter_path = save_adapter(trainer, train_settings.output_dir)
    console.print(f"\n[success]Done! Adapter saved to: {adapter_path}[/success]")
    return str(adapter_path)


def _maybe_evaluate(settings: Settings, proposal, task_desc: str, adapter_path: str) -> None:
    """Prompt user for post-training evaluation, run if accepted."""
    try:
        run_eval = console.input("\n[bold]Run evaluation? [y/n]:[/bold] ").strip().lower()
    except EOFError, KeyboardInterrupt:
        return

    if run_eval not in {"y", "yes", ""}:
        return

    from rich.table import Table

    from ftml.agent.evaluator import Evaluator
    from ftml.eval import generate_samples

    eval_settings = Settings(
        model_name=proposal.recommended_model,
        hf_token=settings.hf_token,
        use_4bit=settings.use_4bit,
        max_seq_length=settings.max_seq_length,
    )

    # Generate sample prompts based on the task
    prompts = [
        f"You are a helpful assistant. {task_desc}. Respond in the target language.",
        "Hello, how can I help you today?",
        "Explain the concept of fine-tuning in simple terms.",
        "What are the advantages of using LoRA?",
        "Summarize the following: Machine learning is a subset of AI.",
    ][: settings.eval_num_samples]

    with console.status("[info]Generating samples from fine-tuned model...[/info]"):
        samples = generate_samples(eval_settings, adapter_path, prompts)

    with console.status("[info]Evaluating with LLM judge...[/info]"):
        evaluator = Evaluator(settings)
        judgments = evaluator.judge_samples(task_desc, samples)
        summary = evaluator.summarize(judgments)

    # Display results
    table = Table(title="Evaluation Results")
    table.add_column("Prompt", style="cyan", max_width=40)
    table.add_column("Relevance", justify="center")
    table.add_column("Fluency", justify="center")
    table.add_column("Accuracy", justify="center")
    table.add_column("Verdict", justify="center")

    for sample, judgment in zip(samples, judgments, strict=True):
        prompt_short = sample["prompt"][:40] + ("..." if len(sample["prompt"]) > 40 else "")
        table.add_row(
            prompt_short,
            f"{float(judgment.relevance):.2f}",
            f"{float(judgment.fluency):.2f}",
            f"{float(judgment.accuracy):.2f}",
            str(judgment.verdict),
        )

    console.print(table)
    console.print(
        f"\n[info]Average scores — "
        f"Relevance: {summary['avg_relevance']:.2f}, "
        f"Fluency: {summary['avg_fluency']:.2f}, "
        f"Accuracy: {summary['avg_accuracy']:.2f}[/info]",
    )
    console.print(f"[info]Verdicts: {summary['verdicts']}[/info]")


def _run_experiment_loop(
    settings: Settings,
    proposal: object,
    task_desc: str,
    orch: object,
) -> object | None:
    """Launch the autonomous experiment loop. Returns best ExperimentResult or None."""
    from rich.live import Live
    from rich.table import Table

    from ftml.data import format_for_sft, load_dataset_from_hf
    from ftml.experiment import ExperimentResult, ExperimentRunner
    from ftml.model import load_model_and_tokenizer

    train_settings = Settings(
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
        experiment_time_budget=settings.experiment_time_budget,
        experiment_max_runs=settings.experiment_max_runs,
        experiment_min_improvement=settings.experiment_min_improvement,
        experiment_branch_tag=settings.experiment_branch_tag,
    )

    with console.status(f"[info]Loading model: {train_settings.model_name}[/info]"):
        model, tokenizer = load_model_and_tokenizer(
            train_settings.model_name,
            train_settings.hf_token,
            use_4bit=train_settings.use_4bit,
            max_seq_length=train_settings.max_seq_length,
            use_unsloth=train_settings.use_unsloth,
            use_flash_attention=train_settings.use_flash_attention,
        )

    with console.status(f"[info]Loading dataset: {train_settings.dataset_name}[/info]"):
        ds = load_dataset_from_hf(train_settings.dataset_name, train_settings.hf_token)
        train_ds = format_for_sft(ds["train"], tokenizer)
        eval_key = "validation" if "validation" in ds else "test"
        eval_ds = format_for_sft(ds[eval_key], tokenizer)

    console.print(f"[info]Train: {len(train_ds):,} | Eval: {len(eval_ds):,}[/info]")
    console.print(
        f"[info]Budget: {train_settings.experiment_time_budget}s/run, "
        f"max {train_settings.experiment_max_runs} runs[/info]\n",
    )

    initial_overrides = {
        "lora_r": int(proposal.suggested_lora_r),
        "lora_alpha": int(proposal.suggested_lora_alpha),
        "learning_rate": float(proposal.suggested_learning_rate),
    }

    runner = ExperimentRunner(
        settings=train_settings,
        orchestrator=orch,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        task_description=task_desc,
        initial_overrides=initial_overrides,
    )

    tag = train_settings.experiment_branch_tag or "auto"
    results: list[ExperimentResult] = []

    def _build_table() -> Table:
        table = Table(title=f"Experiment Loop — branch: experiment/{tag}")
        table.add_column("#", justify="right", style="bold")
        table.add_column("eval_loss", justify="center")
        table.add_column("status", justify="center")
        table.add_column("time(s)", justify="right")
        table.add_column("commit", justify="center", style="dim")
        table.add_column("description")
        for r in results:
            status_style = {
                "baseline": "cyan",
                "keep": "green",
                "discard": "yellow",
                "crash": "red",
            }.get(r.status, "")
            table.add_row(
                str(r.experiment_id),
                f"{r.eval_loss:.4f}" if r.eval_loss < float("inf") else "---",
                f"[{status_style}]{r.status}[/{status_style}]",
                f"{r.wall_seconds:.0f}",
                r.commit_sha or "---",
                r.description[:50],
            )
        return table

    def on_result(result: ExperimentResult) -> None:
        results.append(result)
        live.update(_build_table())

    with Live(_build_table(), console=console, refresh_per_second=1) as live:
        best = runner.run_loop(on_result=on_result)
        live.update(_build_table())

    console.print(
        f"\n[success]Best: #{best.experiment_id} "
        f"(eval_loss={best.eval_loss:.4f}, commit={best.commit_sha})[/success]",
    )
    return best


def _maybe_evaluate_experiment(
    settings: Settings,
    proposal: object,
    task_desc: str,
    best: object,
) -> None:
    """Offer LLM-judge eval on the best experiment's adapter."""
    try:
        run_eval = (
            console.input("\n[bold]Run LLM-judge evaluation on best adapter? [y/n]:[/bold] ")
            .strip()
            .lower()
        )
    except EOFError, KeyboardInterrupt:
        return

    if run_eval not in {"y", "yes", ""}:
        return

    adapter_path = str(settings.output_dir / f"adapter_exp{best.experiment_id}")
    _maybe_evaluate(settings, proposal, task_desc, adapter_path)
