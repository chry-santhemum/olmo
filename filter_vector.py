import json
from pathlib import Path
from typing import Literal
from pydantic import BaseModel
from loguru import logger

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm import tqdm

from dpo_emb_cache import cache_embedding_diffs_multi, Manifest
from persona_vectors.eval.eval_persona import main as eval_persona_main
from persona_vectors.generate_vec import save_persona_vector

# We need the original dataset to have an index column.
# If up/downsampled, it is guaranteed that the first pass through the data is in-order.
class DatasetMask(BaseModel):
    original_path: str
    index: list[int]  # index in original dataset: we use this to mask & upsample 
    flip: list[bool]  # same length as index


NUM_PROC = 16

def load_chunk(cache_dir: Path, chunk_idx: int, manifest: Manifest) -> dict:
    num_digits = len(str(manifest.total_chunks - 1))
    chunk_path = cache_dir / f"chunk_{chunk_idx:0{num_digits}d}.pt"
    if not chunk_path.exists():
        raise FileNotFoundError(f"Chunk {chunk_idx} not found at {chunk_path}")
    return torch.load(chunk_path)

def filter_vector(
    cache_dir: Path,
    save_dir: Path,
    vector: Tensor,  # [hidden_dim]
    layer: int,
    top_pct: float | None = None,
    bottom_pct: float | None = None,
    action: Literal["discard", "flip"] = "discard",
    method: Literal["dot", "cosine"] = "cosine",
    target_dataset_size: int | None = None,  # will upsample or downsample if provided
    print_num: int = 10,
    random_seed: int = 42,
) -> DatasetMask:
    manifest_path = cache_dir / "manifest.json"
    with open(manifest_path) as f:
        manifest = Manifest.model_validate_json(f.read())
    chunk_size = manifest.chunk_size
    dataset_path = cache_dir.parent / "dataset.jsonl"
    assert dataset_path.exists(), f"Dataset path {dataset_path} does not exist"

    # Compute sims
    sims: list[float] = []
    logger.info(f"Computing {method} similarities...")
    for chunk_idx in tqdm(range(manifest.total_chunks), desc="Processing chunks"):
        chunk = load_chunk(cache_dir, chunk_idx, manifest)
        chunk_diffs = chunk["activation_diffs"]
        is_last_chunk = chunk_idx == manifest.total_chunks - 1
        if not is_last_chunk:
            assert len(chunk_diffs) == chunk_size
        for local_idx, sample_diffs in enumerate(chunk_diffs):
            if layer not in sample_diffs:
                raise ValueError(f"Layer {layer} not in sample_diffs")
            diff = sample_diffs[layer].to(dtype=vector.dtype, device=vector.device, non_blocking=True)
            if method == "dot":
                sim = torch.dot(diff, vector).item()
            else:  # cosine
                sim = F.cosine_similarity(diff, vector, dim=0).item()
            sims.append(sim)
            assert chunk_idx * chunk_size + local_idx == len(sims) - 1

    # Save some examples in percentiles, save distribution
    save_dir.mkdir(parents=True, exist_ok=True)
    save_distribution_figure(sims, save_dir / "distr.png")
    save_percentile_examples(cache_dir, save_dir / "examples.json", manifest, sims, print_num)

    # Determine which samples are selected
    selected_indices: set[int] = set()
    if top_pct is not None or bottom_pct is not None:
        sims_arr = np.array(sims)
    if top_pct is not None:
        cutoff = float(np.percentile(sims_arr, 100 - top_pct))
        selected_indices.update({idx for idx, sim in enumerate(sims) if sim >= cutoff})
        logger.info(f"Selected top {top_pct}% (cutoff={cutoff:.4f}): {len(selected_indices)} samples")
    if bottom_pct is not None:
        cutoff = float(np.percentile(sims_arr, bottom_pct))
        selected_indices.update({idx for idx, sim in enumerate(sims) if sim <= cutoff})
        logger.info(f"Selected bottom {bottom_pct}% (cutoff={cutoff:.4f}): {len(selected_indices)} samples")

    # Build dataset mask
    output_index = []
    output_flip = []
    output_sims = []
    for i, sim in enumerate(sims):
        if action == "discard":
            if i in selected_indices:
                continue
            output_index.append(i)
            output_flip.append(False)
            output_sims.append(sim)
        elif action == "flip":
            output_index.append(i)
            if i in selected_indices:
                output_flip.append(True)
                output_sims.append(-sim)
            else:
                output_flip.append(False)
                output_sims.append(sim)

    # Apply up/downsampling
    if target_dataset_size is not None:
        rng = np.random.RandomState(random_seed)
        n = len(output_index)
        if n >= target_dataset_size:
            output_index = output_index[:target_dataset_size]
            output_flip = output_flip[:target_dataset_size]
            output_sims = output_sims[:target_dataset_size]
        else:
            extra_needed = target_dataset_size - n
            full_repeats = extra_needed // n
            remainder = extra_needed % n
            for _ in range(full_repeats):
                sampled = rng.permutation(n)
                output_index.extend([output_index[i] for i in sampled])
                output_flip.extend([output_flip[i] for i in sampled])
                output_sims.extend([output_sims[i] for i in sampled])
            sampled = rng.permutation(n)[:remainder]
            output_index.extend([output_index[i] for i in sampled])
            output_flip.extend([output_flip[i] for i in sampled])
            output_sims.extend([output_sims[i] for i in sampled])

    # Compute metrics on final dataset (after up/downsampling)
    mean_sim = np.mean(output_sims).item()
    sum_sim = np.sum(output_sims).item()

    # Save metadata
    metadata = {
        "filter_config": {
            "top_pct": top_pct,
            "bottom_pct": bottom_pct,
            "action": action,
            "method": method,
            "layer": layer,
            "target_dataset_size": target_dataset_size,
        },
        "sum_similarity": sum_sim,
        "mean_similarity": mean_sim,
        "original_dataset_size": len(sims),
        "original_num_selected": len(selected_indices),
    }
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {save_dir / 'metadata.json'}")

    dataset_mask = DatasetMask(
        original_path=str(dataset_path),
        index=output_index,
        flip=output_flip,
    )
    with open(save_dir / "dataset_mask.json", "w") as f:
        json.dump(dataset_mask.model_dump(), f, indent=2)
    return dataset_mask 

def save_distribution_figure(sims_arr: np.ndarray, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(sims_arr, bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Similarity")
    ax.set_ylabel("Count")
    ax.set_title(f"Similarity Distribution (n={len(sims_arr)})")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved distribution figure to {save_path}")

def save_percentile_examples(
    cache_dir: Path,
    save_path: Path,
    manifest: Manifest,
    sims: list[float],
    num_examples: int,
    percentiles: list[float] = [0, 2, 5, 10, 50, 90, 95, 98, 100],
) -> None:
    sims_arr = np.array(sims)

    pct_to_indices = {}
    for pct in percentiles:
        cutoff = np.percentile(sims_arr, pct)
        higher = [idx for idx, sim in enumerate(sims) if sim >= cutoff]
        lower = [idx for idx, sim in enumerate(sims) if sim < cutoff]
        sorted_higher = sorted(higher, key=lambda i: sims[i])
        sorted_lower = sorted(lower, key=lambda i: sims[i], reverse=True)

        if len(sorted_higher) < num_examples // 2:
            sample_indices = sorted_higher + sorted_lower[:num_examples - len(sorted_higher)]
        elif len(sorted_lower) < num_examples // 2:
            sample_indices = sorted_higher[:num_examples - len(sorted_lower)] + sorted_lower
        else:
            sample_indices = sorted_higher[:num_examples // 2] + sorted_lower[:num_examples // 2]
        pct_to_indices[pct] = sample_indices

    # Load examples
    pct_to_examples = {k: [None for _ in range(len(v))] for k, v in pct_to_indices.items()}
    chunks_to_load = set()
    for indices in pct_to_indices.values():
        chunks_to_load.update(idx // manifest.chunk_size for idx in indices)

    for chunk_idx in sorted(chunks_to_load):
        chunk = load_chunk(cache_dir, chunk_idx, manifest)
        for pct, indices in pct_to_indices.items():
            for i, idx in enumerate(indices):
                if idx // manifest.chunk_size != chunk_idx:
                    continue
                local_idx = idx % manifest.chunk_size
                pct_to_examples[pct][i] = {
                    "prompt": chunk["chosen"][local_idx][0]["content"],
                    "chosen": chunk["chosen"][local_idx][1]["content"],
                    "rejected": chunk["rejected"][local_idx][1]["content"],
                    "similarity": sims[idx],
                    "similarity_percentile": (100.0 * np.mean(sims_arr < sims[idx]), 100.0 * np.mean(sims_arr <= sims[idx]))
                }

    with open(save_path, "w") as f:
        json.dump(pct_to_examples, f, indent=4)
    logger.info(f"Saved percentile examples to {save_path}")


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
