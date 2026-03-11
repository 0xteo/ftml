import argparse
import sys
from pathlib import Path

from ftml.settings import Settings


def train_command(args: argparse.Namespace) -> None:
    from ftml.data import format_for_sft, load_dataset_from_hf
    from ftml.model import load_model_and_tokenizer
    from ftml.train import (
        apply_lora_unsloth,
        build_lora_config,
        build_training_args,
        save_adapter,
        train,
    )

    settings = Settings(
        **{k: v for k, v in vars(args).items() if v is not None and k != "command"},
    )

    print(f"Loading model: {settings.model_name}")
    model, tokenizer = load_model_and_tokenizer(
        settings.model_name,
        settings.hf_token,
        use_4bit=settings.use_4bit,
        max_seq_length=settings.max_seq_length,
        use_unsloth=settings.use_unsloth,
        use_flash_attention=settings.use_flash_attention,
    )

    print(f"Loading dataset: {settings.dataset_name}")
    ds = load_dataset_from_hf(settings.dataset_name, settings.hf_token)
    train_ds = format_for_sft(ds["train"], tokenizer)

    lora_config = build_lora_config(settings)
    if settings.use_unsloth:
        print("Applying LoRA (unsloth)...")
        model = apply_lora_unsloth(model, settings)

    print("Starting fine-tuning...")
    training_args = build_training_args(settings)
    trainer = train(model, tokenizer, train_ds, training_args, peft_config=lora_config)

    adapter_path = save_adapter(trainer, settings.output_dir)
    print(f"Adapter saved to: {adapter_path}")


def experiment_command(args: argparse.Namespace) -> None:
    from ftml.agent.orchestrator import Orchestrator
    from ftml.data import format_for_sft, load_dataset_from_hf
    from ftml.experiment import ExperimentResult, ExperimentRunner
    from ftml.model import load_model_and_tokenizer

    overrides = {k: v for k, v in vars(args).items() if v is not None and k != "command"}
    settings = Settings(**overrides)

    print(f"Loading model: {settings.model_name}")
    model, tokenizer = load_model_and_tokenizer(
        settings.model_name,
        settings.hf_token,
        use_4bit=settings.use_4bit,
        max_seq_length=settings.max_seq_length,
        use_unsloth=settings.use_unsloth,
        use_flash_attention=settings.use_flash_attention,
    )

    print(f"Loading dataset: {settings.dataset_name}")
    ds = load_dataset_from_hf(settings.dataset_name, settings.hf_token)
    train_ds = format_for_sft(ds["train"], tokenizer)
    eval_key = "validation" if "validation" in ds else "test"
    eval_ds = format_for_sft(ds[eval_key], tokenizer)

    print(f"Train: {len(train_ds):,} | Eval: {len(eval_ds):,}")
    print(
        f"Budget: {settings.experiment_time_budget}s/run, max {settings.experiment_max_runs} runs",
    )

    print("Initializing orchestrator...")
    orch = Orchestrator(settings)

    runner = ExperimentRunner(
        settings=settings,
        orchestrator=orch,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        task_description=f"Fine-tune {settings.model_name} on {settings.dataset_name}",
    )

    def on_result(result: ExperimentResult) -> None:
        status = result.status
        loss = f"{result.eval_loss:.4f}" if result.eval_loss < float("inf") else "---"
        print(f"  #{result.experiment_id}: eval_loss={loss} status={status} — {result.description}")

    best = runner.run_loop(on_result=on_result)
    print(
        f"\nBest: #{best.experiment_id} (eval_loss={best.eval_loss:.4f}, commit={best.commit_sha})",
    )


def merge_command(args: argparse.Namespace) -> None:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name: str = args.model_name
    adapter_path: str = args.adapter
    output_dir_str: str = args.output_dir
    token = args.hf_token or None

    print(f"Loading base model: {model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(model_name, token=token)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)

    print(f"Loading adapter: {adapter_path}")
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)
    merged = peft_model.merge_and_unload()  # pyright: ignore[reportCallIssue]

    output = Path(output_dir_str)
    output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))
    print(f"Merged model saved to: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="ftml", description="Bulgarian LLM fine-tuning pipeline")
    subparsers = parser.add_subparsers(dest="command")

    # train subcommand
    train_parser = subparsers.add_parser("train", help="Fine-tune a model with LoRA/QLoRA")
    train_parser.add_argument("--model-name", dest="model_name", type=str)
    train_parser.add_argument("--dataset-name", dest="dataset_name", type=str)
    train_parser.add_argument("--hf-token", dest="hf_token", type=str)
    train_parser.add_argument("--lora-r", dest="lora_r", type=int)
    train_parser.add_argument("--lora-alpha", dest="lora_alpha", type=int)
    train_parser.add_argument("--lora-dropout", dest="lora_dropout", type=float)
    train_parser.add_argument("--learning-rate", dest="learning_rate", type=float)
    train_parser.add_argument("--num-epochs", dest="num_epochs", type=int)
    train_parser.add_argument("--batch-size", dest="batch_size", type=int)
    train_parser.add_argument(
        "--gradient-accumulation-steps",
        dest="gradient_accumulation_steps",
        type=int,
    )
    train_parser.add_argument("--max-seq-length", dest="max_seq_length", type=int)
    train_parser.add_argument("--use-4bit", dest="use_4bit", type=bool)
    train_parser.add_argument("--use-unsloth", dest="use_unsloth", action="store_true")
    train_parser.add_argument(
        "--use-flash-attention",
        dest="use_flash_attention",
        action="store_true",
    )
    train_parser.add_argument("--use-rslora", dest="use_rslora", action="store_true")
    train_parser.add_argument("--use-dora", dest="use_dora", action="store_true")
    train_parser.add_argument("--use-packing", dest="use_packing", action="store_true")
    train_parser.add_argument("--lr-scheduler-type", dest="lr_scheduler_type", type=str)
    train_parser.add_argument("--target-modules", dest="target_modules", type=str)
    train_parser.add_argument("--tf32", dest="tf32", type=bool)
    train_parser.add_argument("--warmup-ratio", dest="warmup_ratio", type=float)
    train_parser.add_argument("--output-dir", dest="output_dir", type=Path)

    # auto subcommand (interactive agentic mode)
    subparsers.add_parser("auto", help="Interactive agentic fine-tuning (research + train)")

    # optimize subcommand
    subparsers.add_parser("optimize", help="Optimize DSPy prompts via MIPROv2")

    # serve subcommand (Slack bot)
    serve_parser = subparsers.add_parser("serve", help="Start Slack bot server")
    serve_parser.add_argument("--port", type=int, default=None)

    # experiment subcommand
    experiment_parser = subparsers.add_parser(
        "experiment",
        help="Run autonomous experiment loop (headless)",
    )
    experiment_parser.add_argument("--model-name", dest="model_name", type=str)
    experiment_parser.add_argument("--dataset-name", dest="dataset_name", type=str)
    experiment_parser.add_argument("--hf-token", dest="hf_token", type=str)
    experiment_parser.add_argument("--time-budget", dest="experiment_time_budget", type=int)
    experiment_parser.add_argument("--max-runs", dest="experiment_max_runs", type=int)
    experiment_parser.add_argument(
        "--min-improvement",
        dest="experiment_min_improvement",
        type=float,
    )
    experiment_parser.add_argument("--branch-tag", dest="experiment_branch_tag", type=str)
    experiment_parser.add_argument("--output-dir", dest="output_dir", type=Path)

    # merge subcommand
    merge_parser = subparsers.add_parser("merge", help="Merge LoRA adapter into base model")
    merge_parser.add_argument("--model-name", dest="model_name", type=str, required=True)
    merge_parser.add_argument("--adapter", type=str, required=True)
    merge_parser.add_argument("--output-dir", dest="output_dir", type=str, required=True)
    merge_parser.add_argument("--hf-token", dest="hf_token", type=str, default="")

    parsed = parser.parse_args()
    if parsed.command == "train":
        train_command(parsed)
    elif parsed.command == "merge":
        merge_command(parsed)
    elif parsed.command == "optimize":
        from ftml.agent.optimize.run import run_optimization

        settings = Settings()
        run_optimization(settings)
    elif parsed.command == "experiment":
        experiment_command(parsed)
    elif parsed.command == "serve":
        import uvicorn

        from ftml.slack.app import create_app

        settings = Settings()
        port = parsed.port or settings.slack_app_port
        slack_app = create_app(settings)
        uvicorn.run(slack_app, host="0.0.0.0", port=port)
    elif parsed.command == "auto" or parsed.command is None:
        from ftml.cli.app import run_interactive

        run_interactive()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
