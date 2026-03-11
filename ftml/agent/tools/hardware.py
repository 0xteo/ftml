"""Hardware estimation tools for fine-tuning feasibility checks."""

from smolagents import tool


@tool
def estimate_vram(
    num_params_billions: float,
    use_4bit: bool = True,
    lora_r: int = 16,
    max_seq_length: int = 2048,
    batch_size: int = 4,
    available_vram_gb: float = 24.0,
) -> str:
    """Estimate VRAM usage for QLoRA/LoRA fine-tuning and check if it fits on the GPU.

    Args:
        num_params_billions: Number of model parameters in billions (e.g., 7.0 for a 7B model).
        use_4bit: Whether to use 4-bit quantization (QLoRA). True = ~0.5 bytes/param.
        lora_r: LoRA rank. Higher rank = more trainable params = more VRAM.
        max_seq_length: Maximum sequence length for training.
        batch_size: Per-device training batch size.
        available_vram_gb: Available GPU VRAM in GB.
    """
    # Base model memory
    bytes_per_param = 0.5 if use_4bit else 2.0  # NF4 vs bf16

    model_gb = (num_params_billions * 1e9 * bytes_per_param) / (1024**3)

    # LoRA adapter memory (trainable params in bf16)
    # Rough estimate: 2 * rank * hidden_dim * num_layers, typical hidden_dim ~ 4096 for 7B
    hidden_dim_estimate = int(4096 * (num_params_billions / 7.0) ** 0.5)
    num_layers_estimate = int(32 * (num_params_billions / 7.0) ** 0.5)
    lora_params = 2 * lora_r * hidden_dim_estimate * num_layers_estimate * 2  # up + down proj
    lora_gb = (lora_params * 2) / (1024**3)  # bf16

    # Optimizer states (AdamW 8-bit: ~1 byte per trainable param)
    optimizer_gb = (lora_params * 1) / (1024**3)

    # Activations and KV cache (rough estimate)
    activation_gb = 0.5 * batch_size * (max_seq_length / 2048)

    # Gradient memory
    gradient_gb = lora_gb  # Same size as trainable params

    total_gb = model_gb + lora_gb + optimizer_gb + activation_gb + gradient_gb
    fits = total_gb < available_vram_gb * 0.9  # 90% threshold for safety

    return (
        f"VRAM Estimate for {num_params_billions:.1f}B model:\n"
        f"  Base model ({'4-bit' if use_4bit else 'bf16'}): {model_gb:.1f} GB\n"
        f"  LoRA adapters (r={lora_r}): {lora_gb:.1f} GB\n"
        f"  Optimizer states: {optimizer_gb:.1f} GB\n"
        f"  Activations (batch={batch_size}, seq={max_seq_length}): {activation_gb:.1f} GB\n"
        f"  Gradients: {gradient_gb:.1f} GB\n"
        f"  ──────────────────\n"
        f"  Total: {total_gb:.1f} GB\n"
        f"  Available: {available_vram_gb:.1f} GB\n"
        f"  Fits: {'YES' if fits else 'NO — reduce batch size, LoRA rank, or use smaller model'}\n"
    )
