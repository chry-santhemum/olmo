import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch import Tensor
from tqdm import tqdm
from dpo_emb_cache import Manifest


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def normalize_training_row(row: dict, default_original_index: int, force_new: bool = False) -> dict:
    out = dict(row)
    out.pop("index", None)

    if force_new:
        original_index = -1
        mask = 1
    else:
        original_index = row.get("original_index", default_original_index)
        mask = row.get("mask", 1)
        if original_index == -1:
            mask = 1
        elif mask not in (-1, 1):
            raise ValueError(f"Expected reused row mask in {{-1, 1}}, got {mask}")

    out["original_index"] = int(original_index)
    out["mask"] = int(mask)
    out["new_index"] = -1
    return out


def flip_training_row(row: dict) -> dict:
    out = dict(row)
    out["chosen"], out["rejected"] = out["rejected"], out["chosen"]
    if out["original_index"] >= 0:
        out["mask"] = -int(out["mask"])
    else:
        out["mask"] = 1
    return out


def finalize_training_rows(rows: list[dict]) -> list[dict]:
    finalized = []
    next_new_index = 0

    for row in rows:
        out = dict(row)
        out.pop("index", None)
        if out["original_index"] == -1:
            out["mask"] = 1
            out["new_index"] = next_new_index
            next_new_index += 1
        else:
            if out["mask"] not in (-1, 1):
                raise ValueError(f"Expected reused row mask in {{-1, 1}}, got {out['mask']}")
            out["new_index"] = -1
        finalized.append(out)

    return finalized


def resample_items(items, target_size: int | None, random_seed: int):
    if target_size is None:
        return list(items)
    if target_size < 0:
        raise ValueError(f"target_size must be non-negative, got {target_size}")
    if target_size == 0:
        return []
    if len(items) == 0:
        raise ValueError("Cannot resample an empty dataset to a positive target size")
    if len(items) >= target_size:
        return list(items[:target_size])

    rng = np.random.RandomState(random_seed)
    output = list(items)
    extra_needed = target_size - len(items)

    while extra_needed > 0:
        sampled = rng.permutation(len(items))[: min(extra_needed, len(items))]
        output.extend(items[i] for i in sampled)
        extra_needed -= len(sampled)

    return output


def load_extra_rows(extra_dataset_path: Path | None) -> list[dict]:
    if extra_dataset_path is None:
        return []
    extra_rows = load_jsonl(extra_dataset_path)
    normalized_rows = [normalize_training_row(row, i, force_new=True) for i, row in enumerate(extra_rows)]
    logger.info(f"Loaded {len(normalized_rows)} extra rows from {extra_dataset_path}")
    return normalized_rows


def load_layer_vector(vector_path: Path, layer: int) -> Tensor:
    vector = torch.load(vector_path, map_location="cpu")
    if not isinstance(vector, torch.Tensor):
        raise ValueError(f"Expected {vector_path} to contain a torch.Tensor")
    if vector.ndim == 1:
        return vector

    vector_index = layer + 1
    if vector_index >= vector.shape[0]:
        raise ValueError(f"Layer {layer} maps to vector index {vector_index}, but {vector_path} only has {vector.shape[0]} rows")
    return vector[vector_index]


def build_training_rows_from_selection(
    source_rows: list[dict],
    selected_indices: set[int],
    action: Literal["discard", "flip"],
    target_dataset_size: int | None = None,
    random_seed: int = 42,
    extra_rows: list[dict] | None = None,
) -> list[dict]:
    base_rows = []
    for i, row in enumerate(source_rows):
        out = normalize_training_row(row, i)
        if action == "discard" and i in selected_indices:
            continue
        if action == "flip" and i in selected_indices:
            out = flip_training_row(out)
        base_rows.append(out)

    extra_rows = [] if extra_rows is None else [dict(row) for row in extra_rows]
    if target_dataset_size is not None:
        if len(extra_rows) > target_dataset_size:
            raise ValueError(
                f"target_dataset_size={target_dataset_size} is smaller than the number of extra rows={len(extra_rows)}"
            )
        base_rows = resample_items(base_rows, target_dataset_size - len(extra_rows), random_seed)

    return finalize_training_rows(base_rows + extra_rows)


def load_chunk(cache_dir: Path, chunk_idx: int, manifest: "Manifest") -> dict:
    num_digits = len(str(manifest.total_chunks - 1))
    chunk_path = cache_dir / f"chunk_{chunk_idx:0{num_digits}d}.pt"
    if not chunk_path.exists():
        raise FileNotFoundError(f"Chunk {chunk_idx} not found at {chunk_path}")
    return torch.load(chunk_path)


def filter_vector(
    cache_dir: Path,
    save_dir: Path,
    vector: Tensor,
    layer: int,
    top_pct: float | None = None,
    bottom_pct: float | None = None,
    action: Literal["discard", "flip"] = "discard",
    method: Literal["dot", "cosine"] = "cosine",
    target_dataset_size: int | None = None,
    print_num: int = 10,
    random_seed: int = 42,
    extra_dataset_path: Path | None = None,
) -> Path:
    manifest_path = cache_dir / "manifest.json"
    with open(manifest_path) as f:
        manifest = Manifest.model_validate_json(f.read())
    chunk_size = manifest.chunk_size
    dataset_path = cache_dir.parent / "dataset.jsonl"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path {dataset_path} does not exist")

    source_rows = load_jsonl(dataset_path)
    extra_rows = load_extra_rows(extra_dataset_path)

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
            else:
                sim = F.cosine_similarity(diff, vector, dim=0).item()
            sims.append(sim)
            assert chunk_idx * chunk_size + local_idx == len(sims) - 1

    if len(sims) != len(source_rows):
        raise ValueError(f"Similarity count {len(sims)} does not match dataset size {len(source_rows)}")

    save_dir.mkdir(parents=True, exist_ok=True)
    save_distribution_figure(np.array(sims), save_dir / "distr.png")
    save_percentile_examples(cache_dir, save_dir / "examples.json", manifest, sims, print_num)

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

    base_examples = []
    for i, row in enumerate(source_rows):
        out = normalize_training_row(row, i)
        if action == "discard" and i in selected_indices:
            continue
        if action == "flip" and i in selected_indices:
            out = flip_training_row(out)
            base_examples.append((out, -sims[i]))
        else:
            base_examples.append((out, sims[i]))

    base_target_size = target_dataset_size
    if target_dataset_size is not None:
        if len(extra_rows) > target_dataset_size:
            raise ValueError(
                f"target_dataset_size={target_dataset_size} is smaller than the number of extra rows={len(extra_rows)}"
            )
        base_target_size = target_dataset_size - len(extra_rows)

    base_examples = resample_items(base_examples, base_target_size, random_seed)
    base_rows = [row for row, _ in base_examples]
    base_sims = [sim for _, sim in base_examples]
    output_rows = finalize_training_rows(base_rows + extra_rows)

    mean_sim = float(np.mean(base_sims)) if base_sims else 0.0
    sum_sim = float(np.sum(base_sims)) if base_sims else 0.0

    metadata = {
        "filter_config": {
            "top_pct": top_pct,
            "bottom_pct": bottom_pct,
            "action": action,
            "method": method,
            "layer": layer,
            "target_dataset_size": target_dataset_size,
            "extra_dataset_path": None if extra_dataset_path is None else str(extra_dataset_path),
        },
        "sum_similarity": sum_sim,
        "mean_similarity": mean_sim,
        "original_dataset_size": len(sims),
        "original_num_selected": len(selected_indices),
        "final_dataset_size": len(output_rows),
        "final_num_new_rows": sum(row["original_index"] == -1 for row in output_rows),
        "num_extra_rows": len(extra_rows),
    }
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {save_dir / 'metadata.json'}")

    output_path = save_dir / "dataset.jsonl"
    save_jsonl(output_path, output_rows)
    logger.info(f"Saved filtered dataset to {output_path}")
    return output_path


def save_distribution_figure(sims_arr: np.ndarray, save_path: Path) -> None:
    import matplotlib.pyplot as plt

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
            sample_indices = sorted_higher + sorted_lower[: num_examples - len(sorted_higher)]
        elif len(sorted_lower) < num_examples // 2:
            sample_indices = sorted_higher[: num_examples - len(sorted_lower)] + sorted_lower
        else:
            sample_indices = sorted_higher[: num_examples // 2] + sorted_lower[: num_examples // 2]
        pct_to_indices[pct] = sample_indices

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
                    "similarity_percentile": (
                        100.0 * np.mean(sims_arr < sims[idx]),
                        100.0 * np.mean(sims_arr <= sims[idx]),
                    ),
                }

    with open(save_path, "w") as f:
        json.dump(pct_to_examples, f, indent=4)
    logger.info(f"Saved percentile examples to {save_path}")


def get_persona_vectors(model: str, trait: str):
    from persona_vectors.eval.eval_persona import main as eval_persona_main
    from persona_vectors.generate_vec import save_persona_vector

    model_slug = model.split("/")[-1]
    logger.info(f"Generating positive responses for {model} for {trait}...")
    eval_persona_main(
        model=model,
        trait=trait,
        use_hf=True,
        output_path=f"eval_persona_extract/{model_slug}/{trait}_pos_instruct.csv",
        persona_instruction_type="pos",
        assistant_name=trait,
        judge_model="gpt-5-mini",
        version="extract",
    )

    logger.info(f"Generating negative responses for {model} for {trait}...")
    eval_persona_main(
        model=model,
        trait=trait,
        use_hf=True,
        output_path=f"eval_persona_extract/{model_slug}/{trait}_neg_instruct.csv",
        persona_instruction_type="neg",
        assistant_name=trait,
        judge_model="gpt-5-mini",
        version="extract",
    )

    logger.info(f"Saving persona vectors for {model} for {trait}...")
    save_persona_vector(
        model_name=model,
        pos_path=f"eval_persona_extract/{model_slug}/{trait}_pos_instruct.csv",
        neg_path=f"eval_persona_extract/{model_slug}/{trait}_neg_instruct.csv",
        trait=trait,
        save_dir=f"vectors/{model_slug}/",
        threshold=50,
    )

