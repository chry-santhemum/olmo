# %% Configuration
from pathlib import Path
from datetime import datetime

def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")

MODEL_NAME = "allenai/Olmo-3-7B-Instruct-SFT"
LAYER = None  # None = auto 2/3 through model
POOLING = "sum"  # "mean", "sum", "last"
NUM_SAMPLES = 10_000
OUTPUT_DIR = Path(f"dpo_svd_analysis/20260111_034425")
CACHE_PATH = OUTPUT_DIR / "embeddings_cache.pt"
STARTING_BATCH_SIZE = 8

# %% Imports and setup
from tqdm import tqdm
from typing import Literal
from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from jaxtyping import Float
from loguru import logger
from torch import Tensor

from dpo_emb_cache import (
    load_model,
    load_tokenizer,
    extract_prompt_and_response,
    find_executable_batch_size,
    _oom_cleanup,
)
from utils_model import fwd_record_resid

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# %% Helper functions
def compute_layer_index(model, fraction: float = 2/3) -> int:
    """Get layer index as fraction through model."""
    num_layers = model.config.num_hidden_layers
    return int(num_layers * fraction)


def pool_embeddings(
    hidden_states: Float[Tensor, "seq hidden"],
    pooling: Literal["mean", "sum", "last"],
) -> Float[Tensor, "hidden"]:
    """Pool hidden states across sequence dimension."""
    if pooling == "mean":
        return hidden_states.mean(dim=0)
    elif pooling == "sum":
        return hidden_states.sum(dim=0)
    elif pooling == "last":
        return hidden_states[-1]
    else:
        raise ValueError(f"Unknown pooling method: {pooling}")

@dataclass
class EmbeddingBatch:
    """Embeddings for a batch of samples."""
    prompt_embeddings: list[Tensor]  # [hidden_dim] each
    chosen_response_embeddings: list[Tensor]
    rejected_response_embeddings: list[Tensor]
    sample_indices: list[int]


def _extract_embeddings_slice(
    batch_size: int,
    model,
    tokenizer,
    chosen_prompts: list[str],
    chosen_responses: list[str],
    rejected_prompts: list[str],
    rejected_responses: list[str],
    valid_indices: list[int],
    layer: int,
    pooling: str,
    start_pos: int,
) -> tuple[EmbeddingBatch, int]:
    """Extract embeddings for a slice of samples."""
    device = next(model.parameters()).device
    n = len(chosen_prompts)
    end_pos = min(start_pos + batch_size, n)

    batch_chosen_prompts = chosen_prompts[start_pos:end_pos]
    batch_chosen_responses = chosen_responses[start_pos:end_pos]
    batch_rejected_prompts = rejected_prompts[start_pos:end_pos]
    batch_rejected_responses = rejected_responses[start_pos:end_pos]
    batch_indices = valid_indices[start_pos:end_pos]
    actual_bs = len(batch_chosen_prompts)

    if actual_bs == 0:
        return EmbeddingBatch([], [], [], []), end_pos

    prompt_embeddings = []
    chosen_response_embeddings = []
    rejected_response_embeddings = []

    # Process each sample individually to handle variable lengths
    # (batching with hooks is tricky due to padding and position tracking)
    for i in range(actual_bs):
        prompt_text = batch_chosen_prompts[i]
        chosen_resp = batch_chosen_responses[i]
        rejected_resp = batch_rejected_responses[i]

        # Tokenize prompt + chosen response
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        chosen_ids = tokenizer(chosen_resp, add_special_tokens=False)["input_ids"]
        rejected_ids = tokenizer(rejected_resp, add_special_tokens=False)["input_ids"]

        prompt_len = len(prompt_ids)

        # Get prompt embedding (last token of prompt)
        full_chosen_ids = prompt_ids + chosen_ids
        input_ids = torch.tensor([full_chosen_ids], device=device)
        attention_mask = torch.ones_like(input_ids)

        # Record at all positions, then extract what we need
        hidden = fwd_record_resid(
            model,
            {"input_ids": input_ids, "attention_mask": attention_mask},
            layer,
            slice(None),  # all tokens
        )  # [1, seq_len, hidden]

        # Prompt embedding: last token before response (index prompt_len - 1)
        prompt_emb = hidden[0, prompt_len - 1, :].cpu()
        prompt_embeddings.append(prompt_emb)

        # Chosen response embedding: tokens from prompt_len onwards
        chosen_hidden = hidden[0, prompt_len:, :]
        chosen_emb = pool_embeddings(chosen_hidden, pooling).cpu()
        chosen_response_embeddings.append(chosen_emb)

        # Get rejected response embedding
        full_rejected_ids = prompt_ids + rejected_ids
        input_ids = torch.tensor([full_rejected_ids], device=device)
        attention_mask = torch.ones_like(input_ids)

        hidden = fwd_record_resid(
            model,
            {"input_ids": input_ids, "attention_mask": attention_mask},
            layer,
            slice(None),
        )

        rejected_hidden = hidden[0, prompt_len:, :]
        rejected_emb = pool_embeddings(rejected_hidden, pooling).cpu()
        rejected_response_embeddings.append(rejected_emb)

    return EmbeddingBatch(
        prompt_embeddings=prompt_embeddings,
        chosen_response_embeddings=chosen_response_embeddings,
        rejected_response_embeddings=rejected_response_embeddings,
        sample_indices=batch_indices,
    ), end_pos


def extract_all_embeddings(
    model,
    tokenizer,
    chosen_prompts: list[str],
    chosen_responses: list[str],
    rejected_prompts: list[str],
    rejected_responses: list[str],
    valid_indices: list[int],
    layer: int,
    pooling: str,
    starting_batch_size: int = STARTING_BATCH_SIZE,
) -> EmbeddingBatch:
    """Extract embeddings for all samples with OOM handling."""
    n = len(chosen_prompts)

    decorated_fn = find_executable_batch_size(starting_batch_size)(_extract_embeddings_slice)

    all_prompt_embs = []
    all_chosen_embs = []
    all_rejected_embs = []
    all_indices = []

    start_pos = 0
    pbar = tqdm(total=n, desc="Extracting embeddings")

    while start_pos < n:
        batch, end_pos = decorated_fn(
            model, tokenizer,
            chosen_prompts, chosen_responses,
            rejected_prompts, rejected_responses,
            valid_indices, layer, pooling, start_pos,
        )

        all_prompt_embs.extend(batch.prompt_embeddings)
        all_chosen_embs.extend(batch.chosen_response_embeddings)
        all_rejected_embs.extend(batch.rejected_response_embeddings)
        all_indices.extend(batch.sample_indices)

        pbar.update(end_pos - start_pos)
        start_pos = end_pos

    pbar.close()
    return EmbeddingBatch(
        prompt_embeddings=all_prompt_embs,
        chosen_response_embeddings=all_chosen_embs,
        rejected_response_embeddings=all_rejected_embs,
        sample_indices=all_indices,
    )


# %% Covariance and SVD
@dataclass
class SVDResult:
    """Results from SVD analysis."""
    U: Tensor  # [hidden_dim, k] - prompt-space singular vectors
    S: Tensor  # [k] - singular values
    Vh: Tensor  # [k, hidden_dim] - response-diff-space singular vectors
    covariance_matrix: Tensor
    prompt_mean: Tensor


def compute_covariance_matrix(
    prompt_embeddings: Tensor,  # [N, hidden_dim]
    response_diffs: Tensor,  # [N, hidden_dim]
) -> tuple[Tensor, Tensor]:
    """Compute covariance between centered prompts and response diffs.

    Returns (covariance_matrix, prompt_mean).
    """
    prompt_mean = prompt_embeddings.mean(dim=0)
    centered = prompt_embeddings - prompt_mean

    # Outer product sum: [N, d] x [N, d] -> [d, d]
    covariance = torch.einsum('ni,nj->ij', centered, response_diffs)

    return covariance, prompt_mean


def perform_svd(covariance_matrix: Tensor, k: int | None = None) -> SVDResult:
    """Perform SVD on covariance matrix."""
    U, S, Vh = torch.linalg.svd(covariance_matrix, full_matrices=False)

    if k is not None:
        U = U[:, :k]
        S = S[:k]
        Vh = Vh[:k, :]

    return SVDResult(
        U=U,
        S=S,
        Vh=Vh,
        covariance_matrix=covariance_matrix,
        prompt_mean=torch.zeros(U.shape[0]),  # placeholder, set properly in main
    )


# %% Lookup functions
def find_top_aligned_samples(
    prompt_embeddings: Tensor,  # [N, hidden_dim]
    response_diffs: Tensor,  # [N, hidden_dim]
    sample_indices: list[int],
    svd_result: SVDResult,
    component_idx: int,
    top_k: int = 10,
    space: Literal["prompt", "response_diff"] = "prompt",
) -> list[tuple[int, float]]:
    """Find samples with highest cosine similarity to a singular vector.

    Returns list of (sample_idx, similarity_score).
    """
    if space == "prompt":
        singular_vec = svd_result.U[:, component_idx]
        vectors = prompt_embeddings - svd_result.prompt_mean
    else:
        singular_vec = svd_result.Vh[component_idx, :]
        vectors = response_diffs

    # Filter out zero-norm vectors to avoid NaN
    norms = vectors.norm(dim=1)
    valid_mask = norms > 1e-8
    valid_indices_local = valid_mask.nonzero(as_tuple=True)[0]

    if len(valid_indices_local) == 0:
        return []

    vectors_valid = vectors[valid_mask]
    norms_valid = norms[valid_mask]

    # Normalize
    singular_vec = singular_vec / singular_vec.norm()
    vectors_normed = vectors_valid / norms_valid.unsqueeze(1).float()

    # Cosine similarities
    similarities = vectors_normed @ singular_vec

    # Get top-k by absolute value
    abs_sim = similarities.abs()
    k = min(top_k, len(similarities))
    top_local_indices = abs_sim.argsort(descending=True)[:k]

    # Map back to original sample indices
    return [(sample_indices[valid_indices_local[i].item()], similarities[i].item()) for i in top_local_indices]


# %% Load dataset
print("Loading dataset...")
dataset = load_dataset("allenai/Dolci-Instruct-DPO", split="train")
print(f"Dataset loaded: {len(dataset)} samples")

if NUM_SAMPLES is not None:
    subset = dataset.select(range(min(NUM_SAMPLES, len(dataset))))
else:
    subset = dataset
print(f"Will process {len(subset)} samples")

# %% Load model and prepare data
print(f"\nLoading model: {MODEL_NAME}")
model = load_model(MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda")
tokenizer = load_tokenizer(MODEL_NAME)

# Determine layer
if LAYER is None:
    layer = compute_layer_index(model)
else:
    layer = LAYER
print(f"Using layer {layer} (out of {model.config.num_hidden_layers})")

# Extract prompts and responses
print("\nExtracting prompts and responses...")
chosen_prompts = []
chosen_responses = []
rejected_prompts = []
rejected_responses = []
valid_indices = []

for i, sample in enumerate(tqdm(subset, desc="Extracting")):
    try:
        prompt_c, response_c = extract_prompt_and_response(tokenizer, sample["chosen"])
        prompt_r, response_r = extract_prompt_and_response(tokenizer, sample["rejected"])
        if not response_c or not response_r:
            continue
        chosen_prompts.append(prompt_c)
        chosen_responses.append(response_c)
        rejected_prompts.append(prompt_r)
        rejected_responses.append(response_r)
        valid_indices.append(i)
    except Exception as e:
        logger.warning(f"Error extracting sample {i}: {e}")
        continue

print(f"Extracted {len(valid_indices)} valid samples")

# %% Extract embeddings
print("\nExtracting embeddings...")
embeddings = extract_all_embeddings(
    model, tokenizer,
    chosen_prompts, chosen_responses,
    rejected_prompts, rejected_responses,
    valid_indices, layer, POOLING,
    starting_batch_size=STARTING_BATCH_SIZE,
)

# Free model memory
del model
_oom_cleanup()

# Stack embeddings into tensors
prompt_embeddings = torch.stack(embeddings.prompt_embeddings)
chosen_embeddings = torch.stack(embeddings.chosen_response_embeddings)
rejected_embeddings = torch.stack(embeddings.rejected_response_embeddings)
response_diffs = chosen_embeddings - rejected_embeddings

print(f"Prompt embeddings shape: {prompt_embeddings.shape}")
print(f"Response diffs shape: {response_diffs.shape}")

# %% Save embeddings cache
cache = {
    "prompt_embeddings": prompt_embeddings,
    "response_diffs": response_diffs,
    "sample_indices": embeddings.sample_indices,
    "config": {
        "model_name": MODEL_NAME,
        "layer": layer,
        "pooling": POOLING,
        "num_samples": len(valid_indices),
        "hidden_dim": prompt_embeddings.shape[1],
    },
}
torch.save(cache, CACHE_PATH)
print(f"Embeddings cached to {CACHE_PATH}")

# %% Compute covariance matrix

cache = torch.load("dpo_svd_analysis/20260111_034425/embeddings_cache.pt")
prompt_embeddings = cache["prompt_embeddings"]
response_diffs = cache["response_diffs"]
sample_indices = cache["sample_indices"]

prompt_embeddings /= prompt_embeddings.shape[0]

print("\nComputing covariance matrix...")
covariance, prompt_mean = compute_covariance_matrix(prompt_embeddings, response_diffs)
print(f"Covariance matrix shape: {covariance.shape}")

# %% Perform SVD
print("\nPerforming SVD...")
svd_result = perform_svd(covariance.float())
svd_result.prompt_mean = prompt_mean

# Save SVD result
svd_cache = {
    "U": svd_result.U,
    "S": svd_result.S,
    "Vh": svd_result.Vh,
    "covariance_matrix": svd_result.covariance_matrix,
    "prompt_mean": svd_result.prompt_mean,
}
torch.save(svd_cache, OUTPUT_DIR / "svd_result.pt")
print(f"SVD result saved to {OUTPUT_DIR / 'svd_result.pt'}")

# %% Analyze top components
print("\n=== Top Singular Values ===")
for i in range(min(10, len(svd_result.S))):
    print(f"  Component {i}: {svd_result.S[i].item():.4f}")

print("\n=== Top Aligned Samples per Component ===")
for comp_idx in range(min(5, len(svd_result.S))):
    print(f"\n--- Component {comp_idx} (S={svd_result.S[comp_idx].item():.4f}) ---")

    # Prompt space
    top_prompt = find_top_aligned_samples(
        prompt_embeddings, response_diffs, sample_indices,
        svd_result, comp_idx, top_k=3, space="prompt"
    )
    print("  Top prompt-aligned samples:")
    for idx, sim in top_prompt:
        prompt_preview = chosen_prompts[valid_indices.index(idx)][:100].replace('\n', ' ')
        print(f"    idx={idx}, sim={sim:.3f}:\n    prompt: {prompt_preview}...")

    # Response diff space
    top_resp = find_top_aligned_samples(
        prompt_embeddings, response_diffs, sample_indices,
        svd_result, comp_idx, top_k=3, space="response_diff"
    )
    print("  Top response-diff-aligned samples:")
    for idx, sim in top_resp:
        chosen_preview = chosen_responses[valid_indices.index(idx)][:500].replace('\n', ' ')
        rejected_preview = rejected_responses[valid_indices.index(idx)][:500].replace('\n', ' ')
        print(f"    idx={idx}, sim={sim:.3f}:\n    chosen: {chosen_preview}...\n    rejected: {rejected_preview}...")

# %% Visualization: Singular value spectrum
plt.figure(figsize=(10, 6))
plt.semilogy(svd_result.S.numpy()[:100], marker='o', markersize=3)
plt.xlabel("Component Index")
plt.ylabel("Singular Value (log scale)")
plt.title("Singular Value Spectrum of Prompt-Response Covariance")
plt.grid(True, alpha=0.3)
plt.savefig(OUTPUT_DIR / "singular_spectrum.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Spectrum plot saved to {OUTPUT_DIR / 'singular_spectrum.png'}")

# %%
