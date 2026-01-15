from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from datasets import Dataset
from loguru import logger
from torch import Tensor
from tqdm import tqdm
from transformers import AutoConfig

from dpo_embedding_analysis import cache_embedding_diffs, load_chunk, load_manifest
from persona_vectors.eval.eval_persona import main as eval_persona_main
from persona_vectors.generate_vec import save_persona_vector


def filter_dataset(
    cache_dir: Path,
    persona_vectors: Tensor,
    layer: int,
    prune_top_pct: float | None = None,
    thresholds: list[float] | None = None,
    num_examples_per_bucket: int = 3,
) -> tuple[list[float], list[int]]:
    """
    Compute cosine similarity between (chosen - rejected) activation diff and persona vector.

    Streams chunks from disk to handle large datasets.

    Args:
        cache_dir: Path to cache directory from cache_embedding_diffs
        persona_vectors: Tensor of shape [num_layers, hidden_dim]
        layer: Which layer to use for comparison
        prune_top_pct: If set, remove top X% most similar samples (keeps bottom 100-X%)
        thresholds: Bucket boundaries for printing examples (default: fine-grained)
        num_examples_per_bucket: How many examples to print per bucket

    Returns:
        Tuple of (cos_sims, kept_indices) where kept_indices are the sample indices to keep
    """
    if thresholds is None:
        thresholds = [-1.0, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.25, 1.0]

    manifest = load_manifest(cache_dir)
    persona_vec = persona_vectors[layer]  # (hidden_dim,)
    n_samples = manifest["num_samples"]
    total_chunks = manifest["total_chunks"]

    # Check layer availability in first chunk
    first_chunk = load_chunk(cache_dir, 0)
    if layer not in first_chunk["activation_diffs"][0]:
        available_layers = list(first_chunk["activation_diffs"][0].keys())
        raise ValueError(f"Layer {layer} not in cache. Available layers: {available_layers}")
    del first_chunk

    # Stream chunks and compute cosine similarities
    cos_sims = []
    logger.info(f"Computing cosine similarities for {n_samples} samples across {total_chunks} chunks...")

    for chunk_idx in tqdm(range(total_chunks), desc="Processing chunks"):
        chunk = load_chunk(cache_dir, chunk_idx)
        for sample_diffs in chunk["activation_diffs"]:
            diff = sample_diffs[layer]  # (hidden_dim,)
            sim = F.cosine_similarity(diff.unsqueeze(0), persona_vec.unsqueeze(0)).item()
            cos_sims.append(sim)
        del chunk

    # Print distribution stats
    cos_sims_arr = np.array(cos_sims)
    logger.info(f"Layer {layer} cosine similarity stats:")
    logger.info(f"  min={cos_sims_arr.min():.4f}, max={cos_sims_arr.max():.4f}")
    logger.info(f"  mean={cos_sims_arr.mean():.4f}, std={cos_sims_arr.std():.4f}")
    logger.info(f"  median={np.median(cos_sims_arr):.4f}")

    # Compute kept indices based on prune_top_pct
    if prune_top_pct is not None:
        cutoff_percentile = 100 - prune_top_pct
        cutoff_value = np.percentile(cos_sims_arr, cutoff_percentile)
        kept_indices = [i for i, s in enumerate(cos_sims) if s <= cutoff_value]
        logger.info(f"Pruning top {prune_top_pct}% (cutoff={cutoff_value:.4f})")
        logger.info(f"Keeping {len(kept_indices)}/{n_samples} samples ({100*len(kept_indices)/n_samples:.1f}%)")
    else:
        kept_indices = list(range(n_samples))

    # Print examples in each bucket (for analysis)
    buckets = list(zip(thresholds[:-1], thresholds[1:]))
    for lo, hi in buckets:
        indices = [i for i, s in enumerate(cos_sims) if lo <= s < hi]
        logger.info(f"\n{'='*60}")
        logger.info(f"Bucket [{lo:.2f}, {hi:.2f}): {len(indices)} samples ({100*len(indices)/n_samples:.1f}%)")
        logger.info(f"{'='*60}")

        if not indices:
            logger.info("  (no samples in this bucket)")
            continue

        # Sort by cosine sim descending and take top examples
        sorted_indices = sorted(indices, key=lambda i: cos_sims[i], reverse=True)
        sample_indices = sorted_indices[:num_examples_per_bucket]
        for idx in sample_indices:
            _print_example_from_cache(cache_dir, idx, cos_sims[idx])

    return cos_sims, kept_indices


def _print_example_from_cache(cache_dir: Path, global_idx: int, cos_sim: float):
    """Print a single example by loading from the appropriate chunk."""
    manifest = load_manifest(cache_dir)
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


def save_filtered_dataset(
    cache_dir: Path,
    kept_indices: list[int],
    cos_sims: list[float],
    output_path: Path,
) -> Dataset:
    """Save filtered dataset as HuggingFace dataset.

    Args:
        cache_dir: Path to cache directory from cache_embedding_diffs
        kept_indices: List of sample indices to include
        cos_sims: Cosine similarities for all samples (for reference)
        output_path: Path to save the filtered dataset

    Returns:
        The filtered HuggingFace Dataset
    """
    manifest = load_manifest(cache_dir)
    chunk_size = manifest["chunk_size"]

    # Build lookup of which indices are in which chunk
    indices_by_chunk: dict[int, list[int]] = {}
    for idx in kept_indices:
        chunk_idx = idx // chunk_size
        if chunk_idx not in indices_by_chunk:
            indices_by_chunk[chunk_idx] = []
        indices_by_chunk[chunk_idx].append(idx)

    # Collect samples from chunks
    chosen_messages = []
    rejected_messages = []
    original_indices = []
    similarities = []

    logger.info(f"Collecting {len(kept_indices)} samples from {len(indices_by_chunk)} chunks...")
    for chunk_idx in tqdm(sorted(indices_by_chunk.keys()), desc="Loading chunks"):
        chunk = load_chunk(cache_dir, chunk_idx)
        chunk_start = chunk["start_idx"]

        for global_idx in indices_by_chunk[chunk_idx]:
            local_idx = global_idx - chunk_start

            # Reconstruct message format (prompt + response as conversation)
            chosen_prompt = chunk["chosen_prompts"][local_idx]
            chosen_response = chunk["chosen_responses"][local_idx]
            rejected_prompt = chunk["rejected_prompts"][local_idx]
            rejected_response = chunk["rejected_responses"][local_idx]

            # Store as formatted text (prompt already has chat template applied)
            chosen_messages.append(chosen_prompt + chosen_response)
            rejected_messages.append(rejected_prompt + rejected_response)
            original_indices.append(global_idx)
            similarities.append(cos_sims[global_idx])

        del chunk

    # Create HuggingFace Dataset
    dataset = Dataset.from_dict({
        "chosen": chosen_messages,
        "rejected": rejected_messages,
        "original_idx": original_indices,
        "cos_sim": similarities,
    })

    # Save to disk
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_path))
    logger.success(f"Saved filtered dataset with {len(dataset)} samples to {output_path}")

    return dataset



def main(
    model: str,
    trait: str,
    num_samples: int = 8192,
    chunk_size: int = 128,
    num_gpus: int | None = None,
    prune_top_pct: float | None = None,
    output_dataset_path: Path | str | None = None,
):
    """Run embedding analysis and optionally filter dataset.

    Args:
        model: HuggingFace model name
        trait: Personality trait for persona vector (e.g., "sycophantic")
        num_samples: Number of samples to process
        chunk_size: Samples per chunk for checkpointing
        num_gpus: Number of GPUs to use (None = auto-detect)
        prune_top_pct: If set, prune top X% most similar samples
        output_dataset_path: If set, save filtered dataset to this path
    """
    model_slug = model.split("/")[-1]
    
    # logger.info(f"Generating positive responses for {model} for {trait}...")
    # eval_persona_main(
    #     model=model,
    #     trait=trait,
    #     output_path=f"eval_persona_extract/{model_slug}/{trait}_pos_instruct.csv",
    #     persona_instruction_type="pos",
    #     assistant_name=trait,
    #     judge_model="gpt-5-mini",
    #     version="extract"
    # )
    
    # logger.info(f"Generating negative responses for {model} for {trait}...")
    # eval_persona_main(
    #     model=model,
    #     trait=trait,
    #     output_path=f"eval_persona_extract/{model_slug}/{trait}_neg_instruct.csv",
    #     persona_instruction_type="neg",
    #     assistant_name=trait,
    #     judge_model="gpt-5-mini",
    #     version="extract"
    # )

    # logger.info(f"Saving persona vectors for {model} for {trait}...")
    # save_persona_vector(
    #     model_name=model,
    #     pos_path=f"eval_persona_extract/{model_slug}/{trait}_pos_instruct.csv",
    #     neg_path=f"eval_persona_extract/{model_slug}/{trait}_neg_instruct.csv",
    #     trait=trait,
    #     save_dir=f"persona_vectors/{model_slug}/",
    #     threshold=50
    # )
    persona_vectors = torch.load(f"persona_vectors/{model_slug}/{trait}_response_avg_diff.pt")  # [num_layers, hidden_size]

    logger.info(f"Caching embedding diffs for {model} for {trait}...")

    config = AutoConfig.from_pretrained(model)
    num_layers = config.num_hidden_layers
    layers = [int(num_layers * 0.5), int(num_layers * 0.66), int(num_layers * 0.75)]

    cache_dir = cache_embedding_diffs(
        model_name=model,
        num_samples=num_samples,
        layers=layers,
        batch_size=8,
        chunk_size=chunk_size,
        num_gpus=num_gpus,
    )

    # Analyze cosine similarities for each layer
    manifest = load_manifest(cache_dir)
    for layer in manifest["layers"]:
        logger.info(f"\n\n{'#'*60}")
        logger.info(f"# ANALYZING LAYER {layer}")
        logger.info(f"{'#'*60}")
        cos_sims, kept_indices = filter_dataset(
            cache_dir,
            persona_vectors,
            layer=layer,
            prune_top_pct=prune_top_pct,
        )

        # Save filtered dataset if requested (only for the last layer analyzed)
        if output_dataset_path is not None and layer == manifest["layers"][-1]:
            save_filtered_dataset(
                cache_dir,
                kept_indices,
                cos_sims,
                Path(output_dataset_path),
            )


if __name__ == "__main__":
    main(
        model="allenai/Olmo-3-7B-Instruct-SFT",
        trait="sycophantic",
        num_samples=32768,
        chunk_size=128,
        prune_top_pct=1,
        output_dataset_path="dpo_filter_data/30K-1pct"
    )
