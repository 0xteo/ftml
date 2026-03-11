# ftml — Bulgarian LLM Fine-Tuning Pipeline

## Project Overview
Fine-tunes OSS LLMs from HuggingFace on Bulgarian language corpus using LoRA/QLoRA via SFTTrainer.

## Architecture
Flat 5-module layout under `ftml/`:
- `settings.py` — Pydantic BaseSettings (env-driven config)
- `model.py` — HF model loading + BitsAndBytes quantization
- `data.py` — Bulgarian dataset loading + chat template formatting
- `train.py` — LoRA/QLoRA fine-tuning via SFTTrainer
- `__main__.py` — CLI entry point (`python -m ftml`)

## Conventions
- **Python**: 3.14
- **Type checking**: basedpyright (recommended mode)
- **Linting/formatting**: ruff
- **Pre-commit**: ruff, pyupgrade, trailing-comma, standard hooks
- **Testing**: pytest with pytest-mock; all HF API calls must be mocked
- **Commits**: conventional commits — `feat/fix/refactor/test/docs/chore`
  - Scopes: `model`, `data`, `train`, `cli`, `config`
  - Examples: `feat(model): add quantization config`, `fix(data): handle missing text column`

## Dev Commands
```bash
workon ftml                          # activate virtualenv
basedpyright                         # type check
ruff check ftml/ tests/              # lint
ruff format ftml/ tests/             # format
pytest                               # run tests
pytest -m "not slow and not gpu"     # skip slow/GPU tests
python -m ftml train                 # run training
```

## Target Hardware
RTX 3090 24GB VRAM (Ampere, bf16 support). QLoRA on 7B models (~6-8GB), 13B (~12-14GB).
