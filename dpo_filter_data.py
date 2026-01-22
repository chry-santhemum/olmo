import json
from collections import defaultdict
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset, Dataset
from loguru import logger
from torch import Tensor
from tqdm import tqdm

from dpo_embedding_analysis import cache_embedding_diffs_multi, Manifest
from persona_vectors.eval.eval_persona import main as eval_persona_main
from persona_vectors.generate_vec import save_persona_vector



NUM_PROC = 16

def load_chunk(cache_dir: Path, chunk_idx: int, manifest: Manifest) -> dict:
    """Load a single chunk from the cache directory."""
    num_digits = len(str(manifest.total_chunks - 1))
    chunk_path = cache_dir / f"chunk_{chunk_idx:0{num_digits}d}.pt"
    if not chunk_path.exists():
        raise FileNotFoundError(f"Chunk {chunk_idx} not found at {chunk_path}")
    return torch.load(chunk_path)

def filter_dataset(
    cache_dir: Path,
    save_dir: Path,
    vector: Tensor,  # [hidden_dim]
    layer: int,
    top_pct: float | None = None,
    bottom_pct: float | None = None,
    action: Literal["prune", "flip"] = "prune",
    method: Literal["dot", "cosine"] = "cosine",
    print_percentiles: list[float] | None = None,
    print_num_examples: int = 10,
):
    """
    Compute dot product between (chosen - rejected) activation diff and persona vector,
    then filter or flip samples based on similarity.

    Saves a HF Dataset with 'chosen', 'rejected', 'flipped', 'cache_index' columns to save_dir
    """
    if top_pct is not None and bottom_pct is not None:
        raise ValueError("top_pct and bottom_pct are mutually exclusive")

    if print_percentiles is None:
        print_percentiles = [0, 1, 5, 10, 20, 80, 90, 95, 99, 100]

    manifest_path = cache_dir / "manifest.json"
    with open(manifest_path) as f:
        manifest = Manifest.model_validate_json(f.read())
    total_chunks = manifest.total_chunks

    # Stream chunks and compute similarities
    completed_chunks = sorted(manifest.completed_chunks)
    similarities: dict[tuple[int, int], float] = {}
    logger.info(f"Computing {method} similarities across {len(completed_chunks)}/{total_chunks} completed chunks...")

    for chunk_idx in tqdm(completed_chunks, desc="Processing chunks"):
        chunk = load_chunk(cache_dir, chunk_idx, manifest)
        for local_idx, sample_diffs in enumerate(chunk["activation_diffs"]):
            if layer not in sample_diffs:
                raise ValueError(f"Layer {layer} not in sample_diffs")
            diff = sample_diffs[layer].to(dtype=vector.dtype, device=vector.device, non_blocking=True)
            if method == "dot":
                sim = torch.dot(diff, vector).item()
            else:  # cosine
                sim = F.cosine_similarity(diff, vector, dim=0).item()
            similarities[(chunk_idx, local_idx)] = sim

    # Print distribution stats
    n_available = len(similarities)
    sim_arr = np.array(list(similarities.values()))

    # Determine which samples are selected based on top_pct or bottom_pct
    selected_indices: set[tuple[int, int]] = set()
    cutoff: float | None = None
    if top_pct is not None:
        cutoff = np.percentile(sim_arr, 100 - top_pct)
        selected_indices = {idx for idx, sim in similarities.items() if sim >= cutoff}
        logger.info(f"Selected top {top_pct}% (cutoff={cutoff:.4f}): {len(selected_indices)} samples")
    elif bottom_pct is not None:
        cutoff = np.percentile(sim_arr, bottom_pct)
        selected_indices = {idx for idx, sim in similarities.items() if sim <= cutoff}
        logger.info(f"Selected bottom {bottom_pct}% (cutoff={cutoff:.4f}): {len(selected_indices)} samples")

    # Apply action to selected samples
    flip_indices: set[tuple[int, int]] = set()
    if action == "prune":
        kept_indices = [idx for idx in similarities.keys() if idx not in selected_indices]
        logger.info(f"Pruning {len(selected_indices)} samples, keeping {len(kept_indices)}/{n_available} ({100*len(kept_indices)/n_available:.1f}%)")
    elif action == "flip":
        kept_indices = list(similarities.keys())
        flip_indices = selected_indices
        logger.info(f"Flipping {len(flip_indices)} samples (swapping chosen/rejected)")

    # Ensure save_dir exists
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save distribution figure and percentile examples
    _save_distribution_figure(sim_arr, method, cutoff, save_dir / "distribution.png")
    _save_percentile_examples(cache_dir, manifest, similarities, print_percentiles, print_num_examples, save_dir / "examples.json")

    # Build filtered dataset from cache
    kept_set = set(kept_indices)

    chosen, rejected, cache_index, flipped = [], [], [], []
    n_flipped = 0

    logger.info(f"Building filtered dataset from {len(completed_chunks)} chunks...")
    for chunk_idx in tqdm(completed_chunks, desc="Collecting samples"):
        chunk = load_chunk(cache_dir, chunk_idx, manifest)
        start_idx = chunk["start_idx"]

        for local_idx, chosen_chat in enumerate(chunk["chosen"]):
            if (chunk_idx, local_idx) not in kept_set:
                continue
            
            rejected_chat = chunk["rejected"][local_idx]
            cache_index.append(start_idx + local_idx)
            if dataset["prompt_id"][start_idx + local_idx] != chunk["prompt_id"][local_idx]:
                raise ValueError(f"Prompt ID mismatch at {start_idx + local_idx}")

            # Apply flip if needed
            if (chunk_idx, local_idx) in flip_indices:
                chosen.append(rejected_chat)
                rejected.append(chosen_chat)
                flipped.append(True)
                n_flipped += 1
            else:
                chosen.append(chosen_chat)
                rejected.append(rejected_chat)
                flipped.append(False)

    # Save dataset as JSON
    filtered_dataset = dataset.select(cache_index)
    data_dict = filtered_dataset.to_dict()
    data_dict["chosen"] = chosen
    data_dict["rejected"] = rejected
    data_dict["flipped"] = flipped
    data_dict["cache_index"] = cache_index
    filtered_dataset = Dataset.from_dict(data_dict)
    output_path = save_dir / "dataset.jsonl"
    filtered_dataset.to_json(output_path)
    logger.success(f"Saved filtered dataset with {len(chosen)} samples ({n_flipped} flipped) to {output_path}")

def _extract_example(chunk: dict, local_idx: int, similarity: float) -> dict:
    """Extract example data from a loaded chunk."""
    chosen_chat = chunk["chosen"][local_idx]
    rejected_chat = chunk["rejected"][local_idx]
    return {
        "local_idx": local_idx,
        "similarity": similarity,
        "chosen_prompt": chosen_chat[0]["content"],
        "rejected_prompt": rejected_chat[0]["content"],
        "prompts_match": chosen_chat[:-1] == rejected_chat[:-1],
        "chosen_response": chosen_chat[-1]["content"],
        "rejected_response": rejected_chat[-1]["content"],
    }


def _save_distribution_figure(
    sim_arr: np.ndarray,
    method: Literal["dot", "cosine"],
    cutoff: float | None,
    save_path: Path,
) -> None:
    """Save histogram of similarity distribution."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(sim_arr, bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Cosine Similarity" if method == "cosine" else "Dot Product")
    ax.set_ylabel("Count")
    ax.set_title(f"{method.capitalize()} Similarity Distribution (n={len(sim_arr)})")
    if cutoff is not None:
        ax.axvline(cutoff, color="red", linestyle="--", label=f"Cutoff: {cutoff:.4f}")
        ax.legend()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved distribution figure to {save_path}")


def _save_percentile_examples(
    cache_dir: Path,
    manifest: Manifest,
    similarities: dict[tuple[int, int], float],
    percentiles: list[float],
    num_examples: int,
    save_path: Path,
) -> None:
    """Collect and save examples for each percentile bucket as nested dict."""
    sim_arr = np.array(list(similarities.values()))
    percentile_pairs = list(zip(percentiles[:-1], percentiles[1:]))

    # Build example requests grouped by percentile bucket
    requests_by_percentile: dict[str, list[tuple[tuple[int, int], float]]] = {}
    for lo_pct, hi_pct in percentile_pairs:
        lo_val = np.percentile(sim_arr, lo_pct)
        hi_val = np.percentile(sim_arr, hi_pct)
        if hi_pct == 100:
            indices = [idx for idx, sim in similarities.items() if lo_val <= sim <= hi_val]
        else:
            indices = [idx for idx, sim in similarities.items() if lo_val <= sim < hi_val]
        logger.info(f"Percentile [{lo_pct}-{hi_pct}%] (val {lo_val:.4f} to {hi_val:.4f}): {len(indices)} samples")

        if not indices:
            continue

        sorted_indices = sorted(indices, key=lambda i: similarities[i], reverse=True)
        sample_indices = sorted_indices[:num_examples]
        bucket_key = f"[{lo_pct}-{hi_pct}%]"
        requests_by_percentile[bucket_key] = [(idx, similarities[idx]) for idx in sample_indices]

    # Batch-load examples by chunk
    by_chunk: dict[int, list[tuple[str, int, float]]] = defaultdict(list)
    for bucket_key, requests in requests_by_percentile.items():
        for (chunk_idx, local_idx), sim in requests:
            by_chunk[chunk_idx].append((bucket_key, local_idx, sim))

    examples_by_percentile: dict[str, list[dict]] = {k: [] for k in requests_by_percentile}
    for chunk_idx in sorted(by_chunk.keys()):
        chunk = load_chunk(cache_dir, chunk_idx, manifest)
        for bucket_key, local_idx, sim in by_chunk[chunk_idx]:
            example = _extract_example(chunk, local_idx, sim)
            example["chunk_idx"] = chunk_idx
            examples_by_percentile[bucket_key].append(example)

    with open(save_path, "w") as f:
        json.dump(examples_by_percentile, f, indent=4)
    total = sum(len(v) for v in examples_by_percentile.values())
    logger.info(f"Saved {total} examples across {len(examples_by_percentile)} percentile buckets to {save_path}")


def get_persona_vectors(model: str, trait: str) -> Tensor:
    """returns [num_layers, hidden_size]"""
    model_slug = model.split("/")[-1]
    logger.info(f"Generating positive responses for {model} for {trait}...")
    eval_persona_main(
        model=model,
        trait=trait,
        output_path=f"eval_persona_extract/{model_slug}/{trait}_pos_instruct.csv",
        persona_instruction_type="pos",
        assistant_name=trait,
        judge_model="gpt-5-mini",
        version="extract"
    )
    
    logger.info(f"Generating negative responses for {model} for {trait}...")
    eval_persona_main(
        model=model,
        trait=trait,
        output_path=f"eval_persona_extract/{model_slug}/{trait}_neg_instruct.csv",
        persona_instruction_type="neg",
        assistant_name=trait,
        judge_model="gpt-5-mini",
        version="extract"
    )

    logger.info(f"Saving persona vectors for {model} for {trait}...")
    save_persona_vector(
        model_name=model,
        pos_path=f"eval_persona_extract/{model_slug}/{trait}_pos_instruct.csv",
        neg_path=f"eval_persona_extract/{model_slug}/{trait}_neg_instruct.csv",
        trait=trait,
        save_dir=f"persona_vectors/{model_slug}/",
        threshold=50
    )
    persona_vectors = torch.load(f"persona_vectors/{model_slug}/{trait}_response_avg_diff.pt")
    return persona_vectors



if __name__ == "__main__":
    model_name = "allenai/Olmo-3-7B-Instruct-SFT"
    LAYER = 23
    trait = "sycophantic"
    model_slug = model_name.split("/")[-1]
    persona_vector = torch.load(f"persona_vectors/{model_slug}/{trait}_response_avg_diff.pt")[LAYER + 1]  # offset by 1
    feedback_syco_vector = torch.load("sycophancy_eval/vectors/feedback_L23.pt")["vector"]
    dataset_path = Path("dpo_filter_data/16K-baseline/dataset.jsonl")
    cache_dir = Path("dpo_embedding_analysis/Olmo-3-7B-Instruct-SFT-L23")

    # Full runs
    RUNS = [
        {"vector": persona_vector, "top_pct": 50.0, "action": "prune", "method": "cosine", "save_dir": "dpo_filter_data/16K-persona-50.0pct-prune"},
        {"vector": persona_vector, "top_pct": 20.0, "action": "prune", "method": "cosine", "save_dir": "dpo_filter_data/16K-persona-20.0pct-prune"},
        {"vector": persona_vector, "top_pct": 10.0, "action": "prune", "method": "cosine", "save_dir": "dpo_filter_data/16K-persona-10.0pct-prune"},
        {"vector": persona_vector, "top_pct": 50.0, "action": "flip", "method": "cosine", "save_dir": "dpo_filter_data/16K-persona-50.0pct-flip"},
        {"vector": persona_vector, "top_pct": 20.0, "action": "flip", "method": "cosine", "save_dir": "dpo_filter_data/16K-persona-20.0pct-flip"},
        {"vector": persona_vector, "top_pct": 10.0, "action": "flip", "method": "cosine", "save_dir": "dpo_filter_data/16K-persona-10.0pct-flip"},
    ]

    for run in RUNS:
        filter_dataset(
            cache_dir=cache_dir,
            save_dir=Path(run["save_dir"]),  # type: ignore
            vector=run["vector"],  # type: ignore
            layer=LAYER,
            top_pct=run["top_pct"],  # type: ignore
            action=run["action"],  # type: ignore
            method=run["method"],  # type: ignore
        )
