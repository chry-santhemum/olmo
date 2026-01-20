import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset, Dataset
from loguru import logger
from torch import Tensor
from tqdm import tqdm

from dpo_embedding_analysis import cache_embedding_diffs_multi, load_manifest
from persona_vectors.eval.eval_persona import main as eval_persona_main
from persona_vectors.generate_vec import save_persona_vector


def load_chunk(cache_dir: Path, chunk_idx: int) -> dict:
    """Load a single chunk from the cache directory."""
    manifest = load_manifest(cache_dir / "manifest.json")
    assert manifest is not None, "Manifest not found"
    chunk_path = cache_dir / f"chunk_{chunk_idx:0{manifest['num_digits']}d}.pt"
    if not chunk_path.exists():
        raise FileNotFoundError(f"Chunk {chunk_idx} not found at {chunk_path}")
    return torch.load(chunk_path)


def filter_dataset(
    cache_dir: Path,
    save_dir: Path,
    vector: Tensor,  # [hidden_dim]
    layer: int,
    dataset: Dataset,
    top_pct: float | None = None,
    bottom_pct: float | None = None,
    action: Literal["prune", "flip"] = "prune",
    method: Literal["dot", "cosine"] = "cosine",
    print_percentiles: list[float] | None = None,
    print_num_examples: int = 10,
) -> dict:
    """
    Compute dot product between (chosen - rejected) activation diff and persona vector,
    then filter or flip samples based on similarity.

    Returns:
        HuggingFace Dataset with 'chosen' and 'rejected' columns, saved to save_dir
    """
    if top_pct is not None and bottom_pct is not None:
        raise ValueError("top_pct and bottom_pct are mutually exclusive")

    if print_percentiles is None:
        print_percentiles = [0, 1, 5, 10, 20, 80, 90, 95, 99, 100]

    manifest = load_manifest(cache_dir / "manifest.json")
    assert manifest is not None, "Manifest not found"
    total_chunks = manifest["total_chunks"]

    # Stream chunks and compute similarities (only completed chunks)
    completed_chunks = sorted(manifest["completed_chunks"])
    similarities: dict[tuple[int, int], float] = {}
    logger.info(f"Computing {method} similarities across {len(completed_chunks)}/{total_chunks} completed chunks...")

    for chunk_idx in tqdm(completed_chunks, desc="Processing chunks"):
        chunk = load_chunk(cache_dir, chunk_idx)
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
    selected_indices: set[int] = set()
    if top_pct is not None:
        cutoff = np.percentile(sim_arr, 100 - top_pct)
        selected_indices = {idx for idx, sim in similarities.items() if sim >= cutoff}
        logger.info(f"Selected top {top_pct}% (cutoff={cutoff:.4f}): {len(selected_indices)} samples")
    elif bottom_pct is not None:
        cutoff = np.percentile(sim_arr, bottom_pct)
        selected_indices = {idx for idx, sim in similarities.items() if sim <= cutoff}
        logger.info(f"Selected bottom {bottom_pct}% (cutoff={cutoff:.4f}): {len(selected_indices)} samples")

    # Apply action to selected samples
    flip_indices: set[int] = set()
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

    # Collect examples for each percentile bucket
    all_examples = []
    percentile_pairs = list(zip(print_percentiles[:-1], print_percentiles[1:]))
    for lo_pct, hi_pct in percentile_pairs:
        lo_val = np.percentile(sim_arr, lo_pct)
        hi_val = np.percentile(sim_arr, hi_pct)
        # Use <= for last bucket to include the max value
        if hi_pct == 100:
            indices = [idx for idx, sim in similarities.items() if lo_val <= sim <= hi_val]
        else:
            indices = [idx for idx, sim in similarities.items() if lo_val <= sim < hi_val]
        logger.info(f"Percentile [{lo_pct}-{hi_pct}%] (val {lo_val:.4f} to {hi_val:.4f}): {len(indices)} samples")

        if not indices:
            continue

        sorted_indices = sorted(indices, key=lambda i: similarities[i], reverse=True)
        sample_indices = sorted_indices[:print_num_examples]
        for idx in sample_indices:
            example = _get_example_from_cache(cache_dir, idx, similarities[idx])
            example["percentile"] = f"[{lo_pct}-{hi_pct}%]"
            all_examples.append(example)

    # Write examples to JSON
    examples_path = save_dir / "examples.json"
    with open(examples_path, "w") as f:
        json.dump(all_examples, f, indent=4)
    logger.info(f"Wrote {len(all_examples)} examples to {examples_path}")

    # Build filtered dataset from cache
    kept_set = set(kept_indices)

    chosen_list = []
    rejected_list = []
    original_indices = []
    flipped = []
    n_flipped = 0

    logger.info(f"Building filtered dataset from {len(completed_chunks)} chunks...")
    for chunk_idx in tqdm(completed_chunks, desc="Collecting samples"):
        chunk = load_chunk(cache_dir, chunk_idx)
        start_idx = chunk["start_idx"]

        for local_idx, chosen_chat in enumerate(chunk["chosen"]):
            if (chunk_idx, local_idx) not in kept_set:
                continue
            
            rejected_chat = chunk["rejected"][local_idx]
            original_indices.append(start_idx + local_idx)
            if dataset["prompt_id"][start_idx + local_idx] != chunk["prompt_id"][local_idx]:
                raise ValueError(f"Prompt ID mismatch at {start_idx + local_idx}")

            # Apply flip if needed
            if (chunk_idx, local_idx) in flip_indices:
                chosen_list.append(rejected_chat)
                rejected_list.append(chosen_chat)
                flipped.append(True)
                n_flipped += 1
            else:
                chosen_list.append(chosen_chat)
                rejected_list.append(rejected_chat)
                flipped.append(False)

    # Save dataset as JSON
    filtered_dataset = dataset.select(original_indices)
    data_dict = filtered_dataset.to_dict()
    data_dict["chosen"] = chosen_list
    data_dict["rejected"] = rejected_list
    data_dict["flipped"] = flipped
    data_dict["original_index"] = original_indices
    filtered_dataset = Dataset.from_dict(data_dict)
    dataset_path = save_dir / "dataset.jsonl"
    filtered_dataset.to_json(dataset_path)
    logger.success(f"Saved filtered dataset with {len(chosen_list)} samples ({n_flipped} flipped) to {dataset_path}")

    return filtered_dataset


def _get_example_from_cache(cache_dir: Path, idx: tuple[int, int], similarity: float) -> dict:
    """Load a single example from the appropriate chunk."""
    manifest = load_manifest(cache_dir / "manifest.json")
    assert manifest is not None, "Manifest not found"
    chunk_idx, local_idx = idx

    chunk = load_chunk(cache_dir, chunk_idx)
    chosen_chat = chunk["chosen"][local_idx]
    rejected_chat = chunk["rejected"][local_idx]
    return {
        "chunk_idx": chunk_idx,
        "local_idx": local_idx,
        "similarity": similarity,
        "chosen_prompt": chosen_chat[0]["content"],
        "rejected_prompt": rejected_chat[0]["content"],
        "prompts_match": chosen_chat[:-1] == rejected_chat[:-1],
        "chosen_response": chosen_chat[-1]["content"],
        "rejected_response": rejected_chat[-1]["content"],
    }


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
    dataset: Dataset,
    chunk_size: int,
    top_pct: float | None = None,
    bottom_pct: float | None = None,
    action: Literal["prune", "flip"] = "prune",
    method: Literal["dot", "cosine"] = "cosine",
    save_dir: Path | str | None = None,
):
    """Run embedding analysis and optionally filter dataset."""
    model_slug = model_name.split("/")[-1]
    num_samples = len(dataset)

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
            output_dir=Path("dpo_embedding_analysis"),
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
        dataset=dataset,
        top_pct=top_pct,
        bottom_pct=bottom_pct,
        action=action,
        method=method,
    )


if __name__ == "__main__":
    # import fire
    # fire.Fire(main)

    model_name = "allenai/Olmo-3-7B-Instruct-SFT"
    model_slug = model_name.split("/")[-1]
    trait = "sycophantic"
    LAYER = 23
    num_samples = 32768
    chunk_size = 1024
    persona_vector = torch.load(f"persona_vectors/{model_slug}/{trait}_response_avg_diff.pt")[LAYER + 1]  # offset by 1
    feedback_syco_vector = torch.load("sycophancy_eval/vectors/feedback_L23.pt")["vector"]

    dataset = load_dataset("allenai/Dolci-Instruct-DPO", split="train").filter(
        lambda ex: (
            ex["chosen"] is not None
            and len(ex["chosen"]) >= 2
            and ex["chosen"][0]["content"] is not None
            and ex["chosen"][-1]["role"] == "assistant"
            and ex["chosen"][-1]["content"] is not None
            and ex["chosen"][-1]["content"] != ""
            and ex["rejected"] is not None
            and len(ex["rejected"]) >= 2
            and ex["rejected"][0]["content"] is not None
            and ex["rejected"][-1]["role"] == "assistant"
            and ex["rejected"][-1]["content"] is not None
            and ex["rejected"][-1]["content"] != ""
            and ex["chosen"][-1]["content"] != ex["rejected"][-1]["content"]
        ),
        num_proc=16,
    )
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(min(num_samples, len(dataset))))
    dataset = dataset.add_column("flipped", [False] * len(dataset))

    dataset_path = "dpo_filter_data/33K-baseline/dataset.jsonl"
    Path(dataset_path).parent.mkdir(parents=True, exist_ok=True)
    dataset.to_json(dataset_path)
    logger.success(f"Saved filtered dataset with {len(dataset)} samples to {dataset_path}")


    # main(
    #     model_name=model_name,
    #     vector=persona_vector,
    #     layer=LAYER,
    #     num_samples=512,
    #     chunk_size=128,
    #     top_pct=5.0,
    #     action="prune",
    #     method="cosine",
    #     save_dir="dpo_filter_data/debug",
    # )

    main(
        model_name=model_name,
        vector=persona_vector,
        layer=LAYER,
        dataset=dataset,
        chunk_size=chunk_size,
        top_pct=5.0,
        action="prune",
        method="cosine",
        save_dir="dpo_filter_data/33K-persona-5.0pct-prune-cosine",
    )
    main(
        model_name=model_name,
        vector=persona_vector,
        layer=LAYER,
        dataset=dataset,
        chunk_size=chunk_size,
        top_pct=1.0,
        action="prune",
        method="cosine",
        save_dir="dpo_filter_data/33K-persona-1.0pct-prune-cosine",
    )
    main(
        model_name=model_name,
        vector=persona_vector,
        layer=LAYER,
        dataset=dataset,
        chunk_size=chunk_size,
        top_pct=0.25,
        action="prune",
        method="cosine",
        save_dir="dpo_filter_data/33K-persona-0.25pct-prune-cosine",
    )
    main(
        model_name=model_name,
        vector=persona_vector,
        layer=LAYER,
        dataset=dataset,
        chunk_size=chunk_size,
        top_pct=1.0,
        action="flip",
        method="cosine",
        save_dir="dpo_filter_data/33K-persona-1.0pct-flip-cosine",
    )
    main(
        model_name=model_name,
        vector=feedback_syco_vector,
        layer=LAYER,
        dataset=dataset,
        chunk_size=chunk_size,
        top_pct=1.0,
        action="prune",
        method="cosine",
        save_dir="dpo_filter_data/33K-feedback-1.0pct-prune-cosine",
    )
    main(
        model_name=model_name,
        vector=persona_vector,
        layer=LAYER,
        dataset=dataset,
        chunk_size=chunk_size,
        top_pct=1.0,
        action="prune",
        method="dot",
        save_dir="dpo_filter_data/33K-persona-1.0pct-prune-dot",
    )
