"""Print the examples with the highest and lowest cosine similarity
for both the feedback vector and sycophancy persona vector on the 32K dataset."""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from dpo_embedding_analysis import Manifest
from dpo_filter_data import load_chunk

LAYER = 23
CACHE_DIR = Path("dpo_embedding_analysis/Olmo-3-7B-Instruct-SFT-L23")
FEEDBACK_VECTOR_PATH = "sycophancy_eval/vectors/20260202_011515/contrast_L23.pt"
PERSONA_VECTOR_PATH = "persona_vectors/Olmo-3-7B-Instruct-SFT/sycophantic_response_avg_diff.pt"


def find_extremes(cache_dir: Path, vector: torch.Tensor, vector_name: str) -> dict:
    with open(cache_dir / "manifest.json") as f:
        manifest = Manifest.model_validate_json(f.read())
    completed_chunks = sorted(manifest.completed_chunks)

    best_sim, worst_sim = -float("inf"), float("inf")
    best_idx, worst_idx = None, None

    for chunk_idx in tqdm(completed_chunks, desc=f"[{vector_name}] Computing similarities"):
        chunk = load_chunk(cache_dir, chunk_idx, manifest)
        for local_idx, sample_diffs in enumerate(chunk["activation_diffs"]):
            diff = sample_diffs[LAYER].to(dtype=vector.dtype, device=vector.device)
            sim = F.cosine_similarity(diff, vector, dim=0).item()
            if sim > best_sim:
                best_sim = sim
                best_idx = (chunk_idx, local_idx)
            if sim < worst_sim:
                worst_sim = sim
                worst_idx = (chunk_idx, local_idx)

    results = {}
    for label, idx, sim in [("highest", best_idx, best_sim), ("lowest", worst_idx, worst_sim)]:
        chunk = load_chunk(cache_dir, idx[0], manifest)
        chosen_chat = chunk["chosen"][idx[1]]
        rejected_chat = chunk["rejected"][idx[1]]
        results[label] = {
            "cosine_similarity": sim,
            "chunk_idx": idx[0],
            "local_idx": idx[1],
            "prompt": chosen_chat[0]["content"],
            "chosen_response": chosen_chat[-1]["content"],
            "rejected_response": rejected_chat[-1]["content"],
        }
    return results


if __name__ == "__main__":
    feedback_vector = torch.load(FEEDBACK_VECTOR_PATH)["vector"]
    persona_vectors = torch.load(PERSONA_VECTOR_PATH)
    sycophancy_vector = persona_vectors[LAYER + 1]  # offset by 1

    output = {
        "feedback": find_extremes(CACHE_DIR, feedback_vector, "feedback"),
        "sycophancy_persona": find_extremes(CACHE_DIR, sycophancy_vector, "sycophancy persona"),
    }

    save_path = Path("dpo_filter_data/extreme_examples.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {save_path}")
