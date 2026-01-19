from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from loguru import logger
from torch import Tensor
from tqdm import tqdm
from transformers import AutoConfig

from dpo_embedding_analysis import cache_embedding_diffs_multi, load_chunk, load_manifest
from persona_vectors.eval.eval_persona import main as eval_persona_main
from persona_vectors.generate_vec import save_persona_vector


def filter_dataset(
    cache_dir: Path,
    save_dir: Path,
    vector: Tensor,  # [hidden_dim]
    layer: int,
    top_pct: float | None = None,
    bottom_pct: float | None = None,
    action: Literal["prune", "flip"] = "prune",
    print_thresholds: list[float] | None = None,
    print_num_examples: int = 10,
) -> Dataset:
    """
    Compute cosine similarity between (chosen - rejected) activation diff and persona vector,
    then filter or flip samples based on similarity.

    Args:
        cache_dir: Directory containing cached embedding diffs
        save_dir: Directory to save the filtered HuggingFace Dataset
        vector: Persona vector to compare against
        layer: Model layer to use for activation diffs
        top_pct: Select top X% most similar samples (mutually exclusive with bottom_pct)
        bottom_pct: Select bottom X% least similar samples (mutually exclusive with top_pct)
        action: "prune" removes selected samples, "flip" swaps their chosen/rejected
        print_thresholds: Bucket boundaries for printing example distribution
        print_num_examples: Number of examples to print per bucket

    Returns:
        HuggingFace Dataset with 'chosen' and 'rejected' columns, saved to save_dir
    """
    if top_pct is not None and bottom_pct is not None:
        raise ValueError("top_pct and bottom_pct are mutually exclusive")

    if print_thresholds is None:
        print_thresholds = [-1.0, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.25, 1.0]

    manifest = load_manifest(cache_dir)
    assert manifest is not None, "Manifest not found"
    total_chunks = manifest["total_chunks"]

    # Check layer availability in first chunk
    first_chunk = load_chunk(cache_dir, 0)
    if layer not in first_chunk["activation_diffs"][0]:
        available_layers = list(first_chunk["activation_diffs"][0].keys())
        raise ValueError(f"Layer {layer} not in cache. Available layers: {available_layers}")
    del first_chunk

    # Stream chunks and compute cosine similarities (only completed chunks)
    completed_chunks = sorted(manifest["completed_chunks"])
    cos_sims = {}  # global_idx -> cos_sim
    logger.info(f"Computing cosine similarities across {len(completed_chunks)}/{total_chunks} completed chunks...")

    for chunk_idx in tqdm(completed_chunks, desc="Processing chunks"):
        chunk = load_chunk(cache_dir, chunk_idx)
        start_idx = chunk["start_idx"]
        for local_idx, sample_diffs in enumerate(chunk["activation_diffs"]):
            global_idx = start_idx + local_idx
            if layer not in sample_diffs:
                raise ValueError(f"Layer {layer} not in sample_diffs")
            diff = sample_diffs[layer]
            sim = F.cosine_similarity(diff.unsqueeze(0), vector.unsqueeze(0)).item()
            cos_sims[global_idx] = sim
        del chunk

    # Print distribution stats
    n_available = len(cos_sims)
    cos_sims_arr = np.array(list(cos_sims.values()))
    logger.info(f"Layer {layer} cosine similarity stats ({n_available} samples):")
    logger.info(f"  min={cos_sims_arr.min():.4f}, max={cos_sims_arr.max():.4f}")
    logger.info(f"  mean={cos_sims_arr.mean():.4f}, std={cos_sims_arr.std():.4f}")
    logger.info(f"  median={np.median(cos_sims_arr):.4f}")

    # Determine which samples are selected based on top_pct or bottom_pct
    selected_indices: set[int] = set()
    if top_pct is not None:
        cutoff = np.percentile(cos_sims_arr, 100 - top_pct)
        selected_indices = {idx for idx, sim in cos_sims.items() if sim >= cutoff}
        logger.info(f"Selected top {top_pct}% (cutoff={cutoff:.4f}): {len(selected_indices)} samples")
    elif bottom_pct is not None:
        cutoff = np.percentile(cos_sims_arr, bottom_pct)
        selected_indices = {idx for idx, sim in cos_sims.items() if sim <= cutoff}
        logger.info(f"Selected bottom {bottom_pct}% (cutoff={cutoff:.4f}): {len(selected_indices)} samples")

    # Apply action to selected samples
    flip_indices: set[int] = set()
    if action == "prune":
        kept_indices = [idx for idx in cos_sims.keys() if idx not in selected_indices]
        logger.info(f"Pruning {len(selected_indices)} samples, keeping {len(kept_indices)}/{n_available} ({100*len(kept_indices)/n_available:.1f}%)")
    elif action == "flip":
        kept_indices = list(cos_sims.keys())
        flip_indices = selected_indices
        logger.info(f"Flipping {len(flip_indices)} samples (swapping chosen/rejected)")

    # Print examples in each bucket (for analysis)
    buckets = list(zip(print_thresholds[:-1], print_thresholds[1:]))
    for lo, hi in buckets:
        indices = [idx for idx, sim in cos_sims.items() if lo <= sim < hi]
        logger.info(f"\n{'='*60}")
        logger.info(f"Bucket [{lo:.2f}, {hi:.2f}): {len(indices)} samples ({100*len(indices)/n_available:.1f}%)")
        logger.info(f"{'='*60}")

        if not indices:
            logger.info("  (no samples in this bucket)")
            continue

        # Sort by cosine sim descending and take top examples
        sorted_indices = sorted(indices, key=lambda i: cos_sims[i], reverse=True)
        sample_indices = sorted_indices[:print_num_examples]
        for idx in sample_indices:
            _print_example_from_cache(cache_dir, idx, cos_sims[idx])

    # Build filtered dataset from cache
    kept_set = set(kept_indices)
    chosen_list = []
    rejected_list = []
    n_flipped = 0

    logger.info(f"Building filtered dataset from {len(completed_chunks)} chunks...")
    for chunk_idx in tqdm(completed_chunks, desc="Collecting samples"):
        chunk = load_chunk(cache_dir, chunk_idx)
        start_idx = chunk["start_idx"]
        for local_idx, chosen_chat in enumerate(chunk["chosen_chats"]):
            global_idx = start_idx + local_idx
            if global_idx not in kept_set:
                continue

            rejected_chat = chunk["rejected_chats"][local_idx]

            # Apply flip if needed
            if global_idx in flip_indices:
                chosen_list.append(rejected_chat)
                rejected_list.append(chosen_chat)
                n_flipped += 1
            else:
                chosen_list.append(chosen_chat)
                rejected_list.append(rejected_chat)
        del chunk

    # Create and save HuggingFace Dataset
    dataset = Dataset.from_dict({"chosen": chosen_list, "rejected": rejected_list})
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(save_dir)
    logger.success(f"Saved filtered dataset with {len(dataset)} samples ({n_flipped} flipped) to {save_dir}")

    return dataset


def _print_example_from_cache(cache_dir: Path, global_idx: int, cos_sim: float):
    """Print a single example by loading from the appropriate chunk."""
    manifest = load_manifest(cache_dir)
    assert manifest is not None, "Manifest not found"
    chunk_size = manifest["chunk_size"]
    chunk_idx = global_idx // chunk_size
    local_idx = global_idx % chunk_size

    chunk = load_chunk(cache_dir, chunk_idx)
    chosen_prompt = chunk["chosen_prompts"][local_idx]
    chosen_response = chunk["chosen_responses"][local_idx]
    rejected_response = chunk["rejected_responses"][local_idx]

    # Truncate for display
    max_len = 500
    logger.info(f"\n--- Sample {global_idx} (cos_sim={cos_sim:.4f}) ---")
    logger.info(f"PROMPT: {chosen_prompt[:max_len]}...")
    logger.info(f"CHOSEN RESPONSE: {chosen_response[:max_len]}...")
    logger.info(f"REJECTED RESPONSE: {rejected_response[:max_len]}...")


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


def main(
    model_name: str,
    vector: Tensor,  # [hidden_size]
    layer: int,
    num_samples: int = 32768,
    chunk_size: int = 512,
    top_pct: float | None = None,
    bottom_pct: float | None = None,
    action: Literal["prune", "flip"] = "prune",
    save_dir: Path | str | None = None,
):
    """Run embedding analysis and optionally filter dataset."""
    model_slug = model_name.split("/")[-1]
        
    dataset = load_dataset("allenai/Dolci-Instruct-DPO", split="train").filter(
        lambda ex: (
            ex["chosen"] is not None
            and len(ex["chosen"]) >= 2
            and ex["chosen"][0]["content"] is not None
            and ex["chosen"][-1]["role"] == "assistant"
            and ex["chosen"][-1]["content"] is not None
            and ex["rejected"] is not None
            and len(ex["rejected"]) >= 2
            and ex["rejected"][0]["content"] is not None
            and ex["rejected"][-1]["role"] == "assistant"
            and ex["rejected"][-1]["content"] is not None
        ),
        num_proc=16,
    )
    dataset = dataset.select(range(min(num_samples, len(dataset))))

    # Use existing cache if available, otherwise compute
    cache_dir = Path(f"dpo_embedding_analysis/{model_slug}-{num_samples}")
    if not cache_dir.exists():
        logger.info(f"Caching embedding diffs for {model_name}...")
        cache_dir = cache_embedding_diffs_multi(
            dataset=dataset,
            model_name=model_name,
            num_samples=num_samples,
            layers=[layer],
            batch_size=8,
            chunk_size=chunk_size,
            output_dir=cache_dir,
        )
    else:
        logger.info(f"Using existing cache at {cache_dir}")

    # Determine save path for filtered dataset
    if save_dir is not None:
        save_dir = Path(save_dir)
    else:
        save_dir = Path(f"filtered_dpo/{model_slug}/layer_{layer}")

    filter_dataset(
        cache_dir=cache_dir,
        save_dir=save_dir,
        vector=vector,
        layer=layer,
        top_pct=top_pct,
        bottom_pct=bottom_pct,
        action=action,
    )


if __name__ == "__main__":
    # import fire
    # fire.Fire(main)

    main(
        model_name="allenai/Olmo-3-7B-Instruct-SFT",
        vector=torch.load("sycophancy_eval/vectors/feedback_L23.pt")["vector"],
        layer=23,
        num_samples=32768,
        chunk_size=256,
        top_pct=5.0,
        action="prune",
        save_dir="dpo_filter_data/33K-5pct.jsonl",
    )
