# %% Inspect last batch of DPO training data
# This simulates the shuffle order used in DPO training to find
# which examples end up in the last training batch.

import json
import torch
from datasets import Dataset


def load_dpo_jsonl(path: str) -> Dataset:
    """Load DPO dataset from JSONL file."""
    with open(path, "r") as f:
        data = [json.loads(line) for line in f]
    return Dataset.from_list(data)


def simulate_training_order(
    dataset: Dataset,
    seed: int = 42,
    num_processes: int = 8,
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
) -> list[int]:
    """
    Simulate the exact order of examples as seen during DPO training.

    The training flow:
    1. HF datasets shuffle (seed=42)
    2. SeedableRandomSampler shuffles again (initial_seed=42, epoch=0)
    3. BatchSamplerShard distributes to processes

    Returns indices in the order they are processed.
    """
    # Step 1: HF datasets shuffle
    shuffled_ds = dataset.shuffle(seed=seed)
    n = len(shuffled_ds)

    # Step 2: SeedableRandomSampler creates another permutation
    # At epoch 0, seed = 0 + initial_seed = 42
    generator = torch.Generator()
    generator.manual_seed(seed)  # epoch 0 + initial_seed
    sampler_indices = torch.randperm(n, generator=generator).tolist()

    # Step 3: Distribute across processes with even batches (padding if needed)
    # Each process gets indices at positions: process_idx, process_idx+8, process_idx+16, ...
    # With even_batches=True (default), dataset is padded so all processes get same # of batches

    batches_per_process = (n + num_processes - 1) // num_processes
    total_padded = batches_per_process * num_processes

    # Pad sampler indices by wrapping around
    padded_indices = sampler_indices.copy()
    for i in range(total_padded - n):
        padded_indices.append(sampler_indices[i % n])

    # Reconstruct the order: interleave across processes
    # Training processes examples in order: [p0_b0, p1_b0, ..., p7_b0, p0_b1, p1_b1, ...]
    training_order = []
    for batch_idx in range(batches_per_process):
        for process_idx in range(num_processes):
            global_idx = batch_idx * num_processes + process_idx
            training_order.append(padded_indices[global_idx])

    return training_order


def get_last_step_examples(
    dataset: Dataset,
    seed: int = 42,
    num_processes: int = 8,
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int = 16,
) -> list[dict]:
    """
    Get the examples that appear in the last training step.

    One step = num_processes * per_device_batch_size * gradient_accumulation_steps
    With 8 GPUs, batch_size=1, grad_accum=16: one step = 128 examples
    """
    training_order = simulate_training_order(
        dataset, seed, num_processes, per_device_batch_size, gradient_accumulation_steps
    )

    step_size = num_processes * per_device_batch_size * gradient_accumulation_steps
    n = len(dataset)
    total_steps = (n + step_size - 1) // step_size

    # Last step starts at index (total_steps - 1) * step_size
    last_step_start = (total_steps - 1) * step_size
    last_step_indices = training_order[last_step_start : last_step_start + step_size]

    # Filter out padded indices (those that repeat from early in the dataset)
    # Actually, keep them all but mark which are padding
    examples = []
    for i, idx in enumerate(last_step_indices):
        ex = dataset[idx]
        # Add metadata about position in last step
        ex["_last_step_position"] = i
        ex["_original_idx"] = idx
        ex["_is_padding"] = i >= (n - last_step_start)
        examples.append(ex)

    return examples


def format_example(ex: dict, include_full_content: bool = False) -> str:
    """Format a single example for display."""
    lines = []
    lines.append(f"Index: {ex.get('_original_idx', '?')}, Position in step: {ex.get('_last_step_position', '?')}")

    # Extract the prompt from chosen conversation
    if "chosen" in ex and ex["chosen"]:
        user_msg = ex["chosen"][0] if isinstance(ex["chosen"], list) else ex["chosen"]
        if isinstance(user_msg, dict) and "content" in user_msg:
            prompt = user_msg["content"]
            if include_full_content:
                lines.append(f"Prompt: {prompt}")
            else:
                lines.append(f"Prompt: {prompt[:200]}..." if len(prompt) > 200 else f"Prompt: {prompt}")

    if ex.get("_is_padding"):
        lines.append("[PADDING - repeated example]")

    return "\n".join(lines)


if __name__ == "__main__":
    # Load the DPO filtered dataset used in training
    DATA_PATH = "/workspace/olmo/dpo_filter_data/33K-5pct.jsonl"

    print(f"Loading dataset from {DATA_PATH}...")
    dataset = load_dpo_jsonl(DATA_PATH)
    print(f"Dataset size: {len(dataset)}")

    # Training config from dpo_filter_sweep.sh
    NUM_GPUS = 8
    BATCH_SIZE = 1
    GRAD_ACCUM = 16
    SEED = 42

    step_size = NUM_GPUS * BATCH_SIZE * GRAD_ACCUM
    total_steps = (len(dataset) + step_size - 1) // step_size
    last_step_examples = len(dataset) - (total_steps - 1) * step_size

    print(f"\nTraining config:")
    print(f"  Processes: {NUM_GPUS}")
    print(f"  Per-device batch size: {BATCH_SIZE}")
    print(f"  Gradient accumulation: {GRAD_ACCUM}")
    print(f"  Step size: {step_size}")
    print(f"  Total steps: {total_steps}")
    print(f"  Examples in last step: {last_step_examples}")

    print("\n" + "="*80)
    print("LAST STEP EXAMPLES")
    print("="*80 + "\n")

    examples = get_last_step_examples(
        dataset, SEED, NUM_GPUS, BATCH_SIZE, GRAD_ACCUM
    )

    # Show all real examples (not padding)
    real_examples = [ex for ex in examples if not ex.get("_is_padding")]
    print(f"Showing {len(real_examples)} real examples from the last step:\n")

    for ex in real_examples:
        print(format_example(ex))
        print("-" * 40)

# %%