"""Compute a sycophancy vector from feedback eval samples.

The vector captures the direction in activation space corresponding to
"agreeing with user's stated opinion" vs "responding neutrally".
"""

import argparse
import json
from pathlib import Path

import torch
from loguru import logger

from dpo_embedding_analysis import (
    compute_embedding_diffs,
    load_model,
    load_tokenizer,
)


def load_sycophancy_samples(
    samples_dir: Path,
    bias_types: list[str] = ["positive_bias", "negative_bias"],
) -> tuple[list[list[dict]], list[list[dict]]]:
    """Load sycophancy eval samples and format for embedding analysis.

    Args:
        samples_dir: Path to samples directory containing *_epoch_1.json files
        bias_types: Which bias types to include

    Returns:
        (chosen_chats, rejected_chats) where:
        - chosen_chats: [[user msg, assistant msg with biased_response], ...]
        - rejected_chats: [[user msg, assistant msg with baseline_response], ...]
    """
    chosen_chats = []
    rejected_chats = []

    sample_files = sorted(samples_dir.glob("*_epoch_1.json"))
    logger.info(f"Found {len(sample_files)} sample files in {samples_dir}")

    for sample_file in sample_files:
        with open(sample_file) as f:
            data = json.load(f)

        metadata = data["metadata"]
        bias_type = metadata["bias_type"]

        if bias_type not in bias_types:
            continue

        biased_prompt = metadata["biased_prompt"]
        biased_response = metadata["biased_response"]
        baseline_response = metadata["baseline_response"]

        # Chosen: sycophantic response (agrees with user's opinion)
        chosen_chat = [
            {"role": "user", "content": biased_prompt},
            {"role": "assistant", "content": biased_response},
        ]

        # Rejected: neutral response (doesn't sycophantically agree)
        rejected_chat = [
            {"role": "user", "content": biased_prompt},
            {"role": "assistant", "content": baseline_response},
        ]

        chosen_chats.append(chosen_chat)
        rejected_chats.append(rejected_chat)

    logger.info(f"Loaded {len(chosen_chats)} samples with bias types: {bias_types}")
    return chosen_chats, rejected_chats


def compute_sycophancy_vector(
    model,
    tokenizer,
    chosen_chats: list[list[dict]],
    rejected_chats: list[list[dict]],
    layer: int,
    batch_size: int = 8,
) -> torch.Tensor:
    """Compute sycophancy vector by averaging activation diffs."""
    activation_diffs = compute_embedding_diffs(
        model=model,
        tokenizer=tokenizer,
        chosen_chats=chosen_chats,
        rejected_chats=rejected_chats,
        layers=[layer],
        batch_size=batch_size,
    )

    logger.info(f"Computed {len(activation_diffs)} activation diffs")

    # Average across all samples
    vectors = [diff[layer] for diff in activation_diffs]
    sycophancy_vector = torch.stack(vectors).mean(dim=0)

    logger.info(f"Sycophancy vector shape: {sycophancy_vector.shape}")
    return sycophancy_vector


def main():
    p = argparse.ArgumentParser(description="Compute sycophancy vector from feedback eval samples")
    p.add_argument("--samples-dir", type=Path, default=Path("sycophancy_eval/results/20260112_064751/olmo-3-7b-instruct-sft/feedback/samples"))
    p.add_argument("--model-name", default="allenai/Olmo-3-7B-Instruct-SFT")
    p.add_argument("--layer", type=int, default=23)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    args = p.parse_args()

    # Set default output path
    if args.output is None:
        args.output = Path(f"sycophancy_eval/vectors/feedback_L{args.layer}.pt")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Load samples
    bias_types = ["positive_bias", "negative_bias"]
    chosen_chats, rejected_chats = load_sycophancy_samples(
        samples_dir=args.samples_dir,
        bias_types=bias_types,
    )

    # Optionally limit samples for testing
    if args.max_samples is not None:
        chosen_chats = chosen_chats[:args.max_samples]
        rejected_chats = rejected_chats[:args.max_samples]
        logger.info(f"Limited to {len(chosen_chats)} samples")

    # Load model and tokenizer
    model = load_model(
        model_name=args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    tokenizer = load_tokenizer(model_name=args.model_name)

    # Compute sycophancy vector
    sycophancy_vector = compute_sycophancy_vector(
        model=model,
        tokenizer=tokenizer,
        chosen_chats=chosen_chats,
        rejected_chats=rejected_chats,
        layer=args.layer,
        batch_size=args.batch_size,
    )

    # Save with metadata
    output_data = {
        "vector": sycophancy_vector,
        "layer": args.layer,
        "model": args.model_name,
        "num_samples": len(chosen_chats),
        "bias_types": bias_types,
        "samples_dir": str(args.samples_dir),
    }
    torch.save(output_data, args.output)
    logger.success(f"Saved sycophancy vector to {args.output}")

if __name__ == "__main__":
    main()
