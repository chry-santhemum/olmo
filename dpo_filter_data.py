import random

import torch
import torch.nn.functional as F
from loguru import logger
from torch import Tensor

from dpo_embedding_analysis import cache_embedding_diffs
from persona_vectors.eval.eval_persona import main as eval_persona_main
from persona_vectors.generate_vec import save_persona_vector


def filter_dataset(
    embeddings_cache: dict,
    persona_vectors: Tensor,
    layer: int,
    thresholds: list[float] | None = None,
    num_examples_per_bucket: int = 3,
) -> list[float]:
    """
    Compute cosine similarity between (chosen - rejected) activation diff and persona vector.

    Prints examples at different threshold buckets to help choose filtering threshold.

    Args:
        embeddings_cache: Dict from cache_embedding_diffs containing activation_diffs
        persona_vectors: Tensor of shape [num_layers, hidden_dim]
        layer: Which layer to use for comparison
        thresholds: Bucket boundaries for printing examples (default: [-1, -0.5, -0.25, 0, 0.25, 0.5, 1])
        num_examples_per_bucket: How many examples to print per bucket

    Returns:
        List of cosine similarities for each sample
    """
    if thresholds is None:
        thresholds = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0]

    activation_diffs = embeddings_cache["activation_diffs"]
    persona_vec = persona_vectors[layer]  # (hidden_dim,)
    n_samples = len(activation_diffs)

    # Check that the requested layer exists in the cache
    if layer not in activation_diffs[0]:
        available_layers = list(activation_diffs[0].keys())
        raise ValueError(f"Layer {layer} not in cache. Available layers: {available_layers}")

    # Compute cosine similarities
    cos_sims = []
    for sample_diffs in activation_diffs:
        diff = sample_diffs[layer]  # (hidden_dim,)
        sim = F.cosine_similarity(diff.unsqueeze(0), persona_vec.unsqueeze(0)).item()
        cos_sims.append(sim)

    # Print distribution stats
    import numpy as np
    cos_sims_arr = np.array(cos_sims)
    logger.info(f"Layer {layer} cosine similarity stats:")
    logger.info(f"  min={cos_sims_arr.min():.4f}, max={cos_sims_arr.max():.4f}")
    logger.info(f"  mean={cos_sims_arr.mean():.4f}, std={cos_sims_arr.std():.4f}")
    logger.info(f"  median={np.median(cos_sims_arr):.4f}")

    # Print examples in each bucket
    buckets = list(zip(thresholds[:-1], thresholds[1:]))
    for lo, hi in buckets:
        indices = [i for i, s in enumerate(cos_sims) if lo <= s < hi]
        logger.info(f"\n{'='*60}")
        logger.info(f"Bucket [{lo:.2f}, {hi:.2f}): {len(indices)} samples ({100*len(indices)/n_samples:.1f}%)")
        logger.info(f"{'='*60}")

        if not indices:
            logger.info("  (no samples in this bucket)")
            continue

        sample_indices = random.sample(indices, min(num_examples_per_bucket, len(indices)))
        for idx in sample_indices:
            _print_example(embeddings_cache, idx, cos_sims[idx])

    return cos_sims


def _print_example(cache: dict, idx: int, cos_sim: float):
    """Print a single example with its prompt and responses."""
    chosen_prompt = cache["chosen_prompts"][idx]
    chosen_response = cache["chosen_responses"][idx]
    rejected_response = cache["rejected_responses"][idx]

    # Truncate for display
    max_len = 500
    logger.info(f"\n--- Sample {idx} (cos_sim={cos_sim:.4f}) ---")
    logger.info(f"PROMPT: {chosen_prompt[:max_len]}...")
    logger.info(f"CHOSEN RESPONSE: {chosen_response[:max_len]}...")
    logger.info(f"REJECTED RESPONSE: {rejected_response[:max_len]}...")



def main(model, trait, num_samples=8192):

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
    persona_vectors = torch.load(f"persona_vectors/{model_slug}/{trait}_response_avg_diff.pt")  # [num_layers, hidden_size]

    logger.info(f"Caching embedding diffs for {model} for {trait}...")
    num_layers = model.config.num_hidden_layers
    embeddings_cache = cache_embedding_diffs(
        model_name=model,
        num_samples=num_samples,
        layers=[int(num_layers * 0.5), int(num_layers * 0.66), int(num_layers * 0.75)],
        batch_size=8,
    )

    # Analyze cosine similarities for each layer
    for layer in embeddings_cache["config"]["layers"]:
        logger.info(f"\n\n{'#'*60}")
        logger.info(f"# ANALYZING LAYER {layer}")
        logger.info(f"{'#'*60}")
        filter_dataset(embeddings_cache, persona_vectors, layer=layer)

