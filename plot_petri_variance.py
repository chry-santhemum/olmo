#!/usr/bin/env python3
"""Plot needs_attention scores across model stages for each instruction."""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("/workspace/olmo/petri-repeat-results")
INSTRUCTIONS_FILE = Path("/workspace/olmo/small_subset.txt")

# Model directories in training order
MODELS = [
    ("olmo-3-7b-think-sft", "SFT"),
    ("olmo-3-7b-instruct-sft", "Instruct SFT"),
    ("olmo-3-7b-instruct-dpo", "Instruct DPO"),
]


def load_instructions() -> dict[int, str]:
    """Load instructions from file, returning dict of index -> instruction text."""
    instructions = {}
    with open(INSTRUCTIONS_FILE) as f:
        for i, line in enumerate(f, start=1):
            instructions[i] = line.strip()
    return instructions


def extract_instruction_index(filename: str) -> int | None:
    """Extract instruction index from transcript filename (e.g., 'transcript_..._4.json' -> 4)."""
    match = re.search(r"_(\d+)\.json$", filename)
    if match:
        return int(match.group(1))
    return None


def load_scores() -> dict[int, dict[str, list[float]]]:
    """
    Load needs_attention scores from all transcript files.

    Returns:
        Dict mapping instruction_index -> {model_dir -> [scores for each run]}
    """
    scores: dict[int, dict[str, list[float]]] = {}

    for model_dir, _ in MODELS:
        model_path = RESULTS_DIR / model_dir
        if not model_path.exists():
            print(f"Warning: {model_path} does not exist")
            continue

        for run_dir in sorted(model_path.glob("run-*")):
            for transcript_file in run_dir.glob("transcript_*.json"):
                instr_idx = extract_instruction_index(transcript_file.name)
                if instr_idx is None:
                    continue

                try:
                    with open(transcript_file) as f:
                        data = json.load(f)
                    score = data["metadata"]["judge_output"]["scores"].get(
                        "needs_attention"
                    )
                    if score is not None:
                        if instr_idx not in scores:
                            scores[instr_idx] = {m[0]: [] for m in MODELS}
                        scores[instr_idx][model_dir].append(score)
                except (KeyError, json.JSONDecodeError) as e:
                    print(f"Warning: Could not parse {transcript_file}: {e}")

    return scores


def truncate_instruction(text: str, max_len: int = 60) -> str:
    """Truncate instruction text for plot title."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def plot_scores(
    scores: dict[int, dict[str, list[float]]], instructions: dict[int, str]
) -> None:
    """Create subplot grid with needs_attention scores for each instruction."""
    n_instructions = len(scores)
    ncols = 3
    nrows = (n_instructions + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = np.atleast_2d(axes)

    x_positions = np.arange(len(MODELS))
    x_labels = [label for _, label in MODELS]
    jitter_width = 0.15

    for idx, (instr_idx, model_scores) in enumerate(sorted(scores.items())):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        for x_pos, (model_dir, _) in enumerate(MODELS):
            run_scores = model_scores.get(model_dir, [])
            if run_scores:
                # Add jitter to x positions for visibility
                jittered_x = x_pos + np.random.uniform(
                    -jitter_width, jitter_width, len(run_scores)
                )
                ax.scatter(jittered_x, run_scores, alpha=0.7, s=50, c="steelblue")
                # Plot mean as horizontal line
                mean_score = np.mean(run_scores)
                ax.hlines(
                    mean_score,
                    x_pos - 0.3,
                    x_pos + 0.3,
                    colors="red",
                    linewidths=2,
                    alpha=0.8,
                )

        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, fontsize=9)
        ax.set_ylim(0, 10.5)
        ax.set_ylabel("needs_attention")
        ax.set_title(
            f"Instr {instr_idx}: {truncate_instruction(instructions.get(instr_idx, 'Unknown'))}",
            fontsize=9,
        )
        ax.grid(axis="y", alpha=0.3)

    # Hide empty subplots
    for idx in range(n_instructions, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    plt.tight_layout()
    output_path = RESULTS_DIR / "needs_attention_variance.png"
    plt.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")
    plt.close()


def main():
    instructions = load_instructions()
    scores = load_scores()

    if not scores:
        print("No scores found!")
        return

    print(f"Loaded scores for {len(scores)} instructions")
    for instr_idx, model_scores in sorted(scores.items()):
        for model_dir, s in model_scores.items():
            print(f"  Instruction {instr_idx}, {model_dir}: {len(s)} runs")

    plot_scores(scores, instructions)


if __name__ == "__main__":
    main()
