import json
from collections import defaultdict
from pathlib import Path
from typing import Literal
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset, Dataset
from loguru import logger
from torch import Tensor
from tqdm import tqdm

from dpo_emb_cache import cache_embedding_diffs_multi, Manifest
from persona_vectors.eval.eval_persona import main as eval_persona_main
from persona_vectors.generate_vec import save_persona_vector

# We need the original dataset to have an index column.
@dataclass
class DatasetMask:
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

def vector_filter(
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

    # Compute sims
    sims: list[float] = []
    logger.info(f"Computing {method} similarities...")
    for chunk_idx in tqdm(manifest.total_chunks, desc="Processing chunks"):
        chunk = load_chunk(cache_dir, chunk_idx, manifest)
        print(f"Chunk {chunk_idx} contains {len(chunk["activation_diffs"])} samples.")
        assert len(chunk["activation_diffs"]) == chunk_size
        for local_idx, sample_diffs in enumerate(chunk["activation_diffs"]):
            if layer not in sample_diffs:
                raise ValueError(f"Layer {layer} not in sample_diffs")
            diff = sample_diffs[layer].to(dtype=vector.dtype, device=vector.device, non_blocking=True)
            if method == "dot":
                sim = torch.dot(diff, vector).item()
            else:  # cosine
                sim = F.cosine_similarity(diff, vector, dim=0).item()
            sims.append(sim)
            assert chunk_idx * chunk_size + local_idx == len(sims)

    # Save some examples in percentiles, save distribution
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    _save_distribution_figure(sims, method, )

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
        output_index_index = list(range(len(output_index)))
        if len(output_index) >= target_dataset_size:
            sampled = rng.shuffle(output_index_index)[:target_dataset_size]
            output_index = [output_index[i] for i in sampled]
            output_flip = [output_flip[i] for i in sampled]
            output_sims = [output_sims[i] for i in sampled]
        elif len(output_index) < target_dataset_size:
            full_repeats = target_dataset_size // len(output_index)
            remainder = target_dataset_size % len(output_index)
            for _ in range(full_repeats):
                sampled = rng.shuffle(output_index_index)
                output_index.extend([output_index[i] for i in sampled])
                output_flip.extend([output_flip[i] for i in sampled])
                output_sims.extend([output_sims[i] for i in sampled])
            sampled = rng.shuffle(output_index_index)[:remainder]
            output_index.extend([output_index[i] for i in sampled])
            output_flip.extend([output_flip[i] for i in sampled])
            output_sims.extend([output_sims[i] for i in sampled])

    # Compute metrics on final dataset (after up/downsampling)
    mean_cosine_sim = np.mean(output_sims).item()
    sum_cosine_sim = np.sum(output_sims).item()

    # Save metadata
    metadata = {
        "filter_config": {
            "top_pct": top_pct,
            "bottom_pct": bottom_pct,
            "action": action,
            "method": method,
            "layer": layer,
            "num_samples_target": num_samples,
        },
        "sum_cosine_similarity": sum_cosine_sim,
        "mean_cosine_similarity": mean_cosine_sim,
        "num_samples": len(kept_indices),
        "num_unique_samples": num_unique_samples,
        "num_flipped": sum(flipped),
    }
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {save_dir / 'metadata.json'}")

    # Save dataset as JSON
    dataset = Dataset.from_dict({
        "chosen": chosen,
        "rejected": rejected,
        "flipped": flipped,
        "cache_index": cache_index,
    })
    output_path = save_dir / "dataset.jsonl"
    dataset.to_json(output_path)
    logger.success(f"Saved filtered dataset with {len(chosen)} samples (sum_cos_sim={sum_cosine_sim:.2f}, mean={mean_cosine_sim:.4f}) to {output_path}")



def save_distribution_figure(sims_arr: np.ndarray, cutoff: float | None, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(sims_arr, bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Similarity")
    ax.set_ylabel("Count")
    ax.set_title(f"Similarity Distribution (n={len(sims_arr)})")
    if cutoff is not None:
        ax.axvline(cutoff, color="red", linestyle="--", label=f"Cutoff: {cutoff:.4f}")
        ax.legend()
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
        chunks_to_load.update(set(indices) // manifest.chunk_size)

    for chunk_idx in sorted(chunks_to_load):
        chunk = load_chunk(cache_dir, chunk_idx, manifest)
        for pct, indices in pct_to_indices.items():
            for idx in indices:
                if idx 
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
    # trait = "sycophantic"
    model_slug = model_name.split("/")[-1]
    # persona_vector = torch.load(f"persona_vectors/{model_slug}/{trait}_response_avg_diff.pt")[LAYER + 1]  # offset by 1
    cache_dir = Path("dpo_emb_cache/Olmo-3-7B-Instruct-SFT-L23")
    vector = torch.load("sycophancy_eval/vectors/20260202_011515/contrast_L23.pt")["vector"]

    # Fixed dataset size for all experiments (enables fair comparison)
    NUM_SAMPLES = 16384

    # Sweep runs: all normalized to NUM_SAMPLES via up/downsampling
    RUNS = [
        # Baseline (no filtering, just downsample to NUM_SAMPLES)
        # {"vector": None, "top_pct": None, "action": "discard", "save_dir": "dpo_filter_data/16K-baseline"},
        {"vector": vector, "top_pct": 50.0, "action": "discard", "save_dir": "dpo_filter_data/16K-feedback-50.0pct-discard"},
        {"vector": vector, "top_pct": 20.0, "action": "discard", "save_dir": "dpo_filter_data/16K-feedback-20.0pct-discard"},
        {"vector": vector, "top_pct": 5.0, "action": "discard", "save_dir": "dpo_filter_data/16K-feedback-5.0pct-discard"},
        {"vector": vector, "top_pct": 50.0, "action": "flip", "save_dir": "dpo_filter_data/16K-feedback-50.0pct-flip"},
        {"vector": vector, "top_pct": 20.0, "action": "flip", "save_dir": "dpo_filter_data/16K-feedback-20.0pct-flip"},
        {"vector": vector, "top_pct": 5.0, "action": "flip", "save_dir": "dpo_filter_data/16K-feedback-5.0pct-flip"},
    ]

    for run in RUNS:
        filter_dataset(
            cache_dir=cache_dir,
            save_dir=Path(run["save_dir"]),  # type: ignore
            vector=run["vector"],  # type: ignore
            layer=LAYER,
            top_pct=run["top_pct"],  # type: ignore
            action=run["action"],  # type: ignore
            method="cosine",
            num_samples=NUM_SAMPLES,
        )

    RUNS = [
        # Baseline (no filtering, just downsample to NUM_SAMPLES)
        # {"vector": None, "top_pct": None, "action": "discard", "save_dir": "dpo_filter_data/16K-baseline"},
        # {"vector": None, "top_pct": None, "action": "discard", "save_dir": "dpo_filter_data/33K-baseline"},
        {"vector": vector, "top_pct": 33.0, "action": "discard", "save_dir": "dpo_filter_data/33K-feedback-33.0pct-discard"},
        {"vector": vector, "top_pct": 15.0, "action": "discard", "save_dir": "dpo_filter_data/33K-feedback-15.0pct-discard"},
        {"vector": vector, "top_pct": 50.0, "action": "flip", "save_dir": "dpo_filter_data/33K-feedback-50.0pct-flip"},
        {"vector": vector, "top_pct": 33.0, "action": "flip", "save_dir": "dpo_filter_data/33K-feedback-33.0pct-flip"},
        {"vector": vector, "top_pct": 10.0, "action": "flip", "save_dir": "dpo_filter_data/33K-feedback-10.0pct-flip"},
    ]

    for run in RUNS:
        filter_dataset(
            cache_dir=cache_dir,
            save_dir=Path(run["save_dir"]),  # type: ignore
            vector=run["vector"],  # type: ignore
            layer=LAYER,
            top_pct=run["top_pct"],  # type: ignore
            action=run["action"],  # type: ignore
            method="cosine",
        )
