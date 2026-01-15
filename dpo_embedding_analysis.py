# %% Imports and Configuration
import gc
import inspect
import json
import multiprocessing as mp
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import torch
import torch.nn.functional as F
from datasets import load_dataset
from loguru import logger
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import hook utilities from introspection
sys.path.insert(0, str(Path(__file__).parent.parent / "introspection"))
from utils_model import get_module, get_resid_block_name

# %%


def _make_cpu_record_hook(layer_idx: int, captured: dict):
    """Create hook that records hidden states directly to CPU."""
    def hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured[layer_idx] = hidden.detach().cpu()  # Move to CPU immediately
    return hook


def _is_oom_exception(e: Exception) -> bool:
    msg = str(e)
    return (
        isinstance(e, RuntimeError)
        and any(s in msg for s in (
            "CUDA out of memory",
            "CUDNN_STATUS_NOT_SUPPORTED",
            "can't allocate memory",
            "out of memory",
        ))
    )

def _oom_cleanup(device: int | str | None = None):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()  # Sometimes helps reclaim CUDA IPC allocations
        logger.info(f"CUDA allocated memory: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
        logger.info(f"CUDA reserved memory: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")

def find_executable_batch_size(starting_batch_size: int, function=None):
    """Decorator that auto-reduces batch_size on OOM.

    The decorated function must take batch_size as its first argument.
    """
    if function is None:
        return partial(find_executable_batch_size, starting_batch_size)

    def decorator(*args, **kwargs):
        batch_size = starting_batch_size
        _oom_cleanup()
        params = list(inspect.signature(function).parameters.keys())
        if len(params) < (len(args) + 1):
            arg_str = ", ".join([f"{arg}={value}" for arg, value in zip(params[1:], args[1:])])
            raise TypeError(
                f"Batch size was passed into `{function.__name__}` as the first argument when called."
                f"Remove this as the decorator already does so: `{function.__name__}({arg_str})`"
            )
        while True:
            if batch_size == 0:
                raise RuntimeError("No executable batch size found, reached zero.")
            try:
                result = function(batch_size, *args, **kwargs)
                return result
            except Exception as e:
                oom = _is_oom_exception(e)
                try:
                    e.__traceback__ = None  # aggressively drop traceback references
                finally:
                    del e
                if oom:
                    _oom_cleanup()
                    batch_size //= 2
                    logger.warning(f"Decreasing batch size to {batch_size}.")
                else:
                    raise

    return decorator


@dataclass
class EmbeddingDiffs:
    """Activation diffs (chosen - rejected) and NLLs for a batch."""
    nlls_chosen: list[float]
    nlls_rejected: list[float]
    activation_diffs: list[dict[int, Tensor]]  # chosen - rejected per layer


def load_model(model_name: str, **kwargs):
    """Load model in eval mode with gradients disabled."""
    logger.info(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    logger.success("Model loaded (eval mode, grads disabled)")
    return model


def load_tokenizer(model_name: str):
    """Load tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def extract_prompt_response_text(
    tokenizer,
    messages: list[dict],
) -> tuple[str, str]:
    """Extract FORMATTED prompt and response from multi-turn conversation."""
    if len(messages) < 2:
        raise ValueError(f"Need at least 2 messages, got {len(messages)}")

    if messages[-1]["role"] != "assistant":
        raise ValueError(f"Last message should be assistant, got {messages[-1]['role']}")

    prompt_messages = messages[:-1]
    response_text = messages[-1]["content"]

    prompt_text = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    return prompt_text, response_text


def compute_embedding_diffs(
    model,
    tokenizer,
    chosen_prompts: list[str],
    chosen_responses: list[str],
    rejected_prompts: list[str],
    rejected_responses: list[str],
    layers: list[int],
    batch_size: int,
) -> EmbeddingDiffs:
    """Compute activation diffs (chosen - rejected) for each sample."""
    device = next(model.parameters()).device
    n = len(chosen_prompts)
    assert n == len(chosen_responses) == len(rejected_prompts) == len(rejected_responses)

    def _extract_mean_activations(
        outputs, captured_hidden: dict[int, Tensor], input_ids, attention_mask,
        prompt_lens: list[int], layers: list[int], actual_bs: int
    ) -> tuple[list[float], list[dict[int, Tensor]]]:
        """Extract NLLs and mean activations from model outputs.

        captured_hidden: dict mapping layer_idx -> tensor [batch, seq_len, hidden] on CPU
        """
        nlls = []
        activations = []
        for i in range(actual_bs):
            seq_len = attention_mask[i].sum().item()
            prompt_len = prompt_lens[i]
            response_start, response_end = prompt_len, seq_len

            if response_end - response_start <= 0:
                raise ValueError(f"Response length <= 0 for sample {i}; please filter your data.")

            # NLL
            logits = outputs.logits[i, response_start - 1 : response_end - 1]
            labels = input_ids[i, response_start:response_end]
            nll = F.cross_entropy(logits, labels, reduction="mean").item()
            nlls.append(nll)

            # Mean activations per layer (captured_hidden is already on CPU)
            item_acts = {}
            for layer_idx in layers:
                hidden = captured_hidden[layer_idx]  # [batch, seq_len, hidden] on CPU
                response_hidden = hidden[i, response_start:response_end]
                item_acts[layer_idx] = response_hidden.mean(dim=0)
            activations.append(item_acts)
        return nlls, activations

    def _compute_diffs_inner(batch_size: int, start_pos: int) -> tuple[EmbeddingDiffs, int]:
        """Process a slice of samples, computing diffs."""
        end_pos = min(start_pos + batch_size, n)
        actual_bs = end_pos - start_pos
        if actual_bs == 0:
            return EmbeddingDiffs(nlls_chosen=[], nlls_rejected=[], activation_diffs=[]), end_pos

        # Prepare batch: interleave chosen and rejected for same forward pass
        batch_chosen_prompts = chosen_prompts[start_pos:end_pos]
        batch_chosen_responses = chosen_responses[start_pos:end_pos]
        batch_rejected_prompts = rejected_prompts[start_pos:end_pos]
        batch_rejected_responses = rejected_responses[start_pos:end_pos]

        # Concatenate chosen + rejected into one batch
        all_prompts = batch_chosen_prompts + batch_rejected_prompts
        all_responses = batch_chosen_responses + batch_rejected_responses
        full_texts = [p + r for p, r in zip(all_prompts, all_responses)]

        inputs = tokenizer(full_texts, return_tensors="pt", padding=True, padding_side="right")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        prompt_lens = [len(tokenizer.encode(p, add_special_tokens=False)) for p in all_prompts]

        # Register hooks to capture hidden states (moved to CPU immediately in hook)
        captured_hidden: dict[int, Tensor] = {}
        handles = []
        for layer_idx in layers:
            module = get_module(model, get_resid_block_name(model, layer_idx))
            handle = module.register_forward_hook(_make_cpu_record_hook(layer_idx, captured_hidden))
            handles.append(handle)

        outputs = None
        try:
            with torch.inference_mode():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

            nlls, activations = _extract_mean_activations(
                outputs, captured_hidden, input_ids, attention_mask, prompt_lens, layers, actual_bs * 2
            )
        finally:
            # Always remove hooks
            for handle in handles:
                handle.remove()
            # Clean up GPU tensors
            try:
                del input_ids
            except Exception:
                pass
            try:
                del attention_mask
            except Exception:
                pass
            try:
                del outputs
            except Exception:
                pass

        # Split results: first half is chosen, second half is rejected
        nlls_chosen = nlls[:actual_bs]
        nlls_rejected = nlls[actual_bs:]
        acts_chosen = activations[:actual_bs]
        acts_rejected = activations[actual_bs:]

        # Compute diffs
        activation_diffs = []
        for c_acts, r_acts in zip(acts_chosen, acts_rejected):
            diff = {layer: c_acts[layer] - r_acts[layer] for layer in layers}
            activation_diffs.append(diff)

        return EmbeddingDiffs(
            nlls_chosen=nlls_chosen,
            nlls_rejected=nlls_rejected,
            activation_diffs=activation_diffs,
        ), end_pos

    # Note: batch_size here refers to number of *pairs*, so we process 2x that many samples
    decorated_inner = find_executable_batch_size(starting_batch_size=batch_size)(_compute_diffs_inner)

    all_nlls_chosen: list[float] = []
    all_nlls_rejected: list[float] = []
    all_diffs: list[dict[int, Tensor]] = []

    start_pos = 0
    pbar = tqdm(total=n, desc="Computing embedding diffs")

    while start_pos < n:
        diffs, end_pos = decorated_inner(start_pos)
        all_nlls_chosen.extend(diffs.nlls_chosen)
        all_nlls_rejected.extend(diffs.nlls_rejected)
        all_diffs.extend(diffs.activation_diffs)
        pbar.update(end_pos - start_pos)
        start_pos = end_pos
        del diffs
        _oom_cleanup()

    pbar.close()
    return EmbeddingDiffs(
        nlls_chosen=all_nlls_chosen,
        nlls_rejected=all_nlls_rejected,
        activation_diffs=all_diffs,
    )


# def compute_similarity_metric(
#     h_chosen: Float[Tensor, "hidden_dim"],
#     h_rejected: Float[Tensor, "hidden_dim"],
# ) -> float:
#     """Compute similarity metric: dot(h_chosen, h_rejected - h_chosen).

#     Lower absolute value indicates more similar embeddings.
#     """
#     diff = h_rejected - h_chosen
#     return torch.dot(h_chosen, diff).item()

# def compute_similarity_metric(
#     h_chosen: Float[Tensor, "hidden_dim"],
#     h_rejected: Float[Tensor, "hidden_dim"],
# ) -> float:
#     return torch.norm(h_rejected - h_chosen).item()


# %% Visualization Functions
def plot_correlation(df: pd.DataFrame, output_dir: Path):
    """Plot correlation between similarity metric and NLL diff."""
    valid_df = df.dropna(subset=["similarity_metric", "nll_diff"])

    if len(valid_df) < 10:
        logger.warning("Not enough valid samples for plotting")
        return

    x = valid_df["similarity_metric"].values
    y = valid_df["nll_diff"].values

    pearson_r, pearson_p = scipy.stats.pearsonr(x, y)
    spearman_r, spearman_p = scipy.stats.spearmanr(x, y)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(x, y, alpha=0.5, s=20, c="steelblue")

    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_sorted = np.sort(x)
    ax.plot(x_sorted, p(x_sorted), "r--", alpha=0.8, linewidth=2, label="Linear fit")
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)

    ax.set_xlabel("Similarity Metric: dot(h_chosen, h_rejected - h_chosen)", fontsize=12)
    ax.set_ylabel("NLL Diff (pre - post): positive = DPO improved", fontsize=12)
    ax.set_title(
        f"DPO Effect vs Embedding Similarity\n"
        f"Pearson r={pearson_r:.3f} (p={pearson_p:.2e}), "
        f"Spearman rho={spearman_r:.3f} (p={spearman_p:.2e})",
        fontsize=12,
    )
    ax.legend()

    plot_path = output_dir / "correlation_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.show()
    logger.info(f"Plot saved to {plot_path}")

    # Distributions
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(x, bins=50, alpha=0.7, color="steelblue")
    axes[0].set_xlabel("Similarity Metric")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of Similarity Metric")

    axes[1].hist(y, bins=50, alpha=0.7, color="coral")
    axes[1].set_xlabel("NLL Diff (pre - post)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Distribution of NLL Diff")
    axes[1].axvline(x=0, color="gray", linestyle="--", alpha=0.5)

    hist_path = output_dir / "distributions.png"
    plt.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.show()
    logger.info(f"Distributions plot saved to {hist_path}")

    logger.info("=== Summary Statistics ===")
    logger.info(f"Samples: {len(valid_df)}")
    logger.info(f"Pearson correlation: r={pearson_r:.4f}, p={pearson_p:.2e}")
    logger.info(f"Spearman correlation: rho={spearman_r:.4f}, p={spearman_p:.2e}")
    logger.info(f"Similarity metric: mean={x.mean():.2e}, std={x.std():.2e}")
    logger.info(f"NLL diff: mean={y.mean():.4f}, std={y.std():.4f}")
    logger.info(f"Samples where DPO improved (nll_diff > 0): {(y > 0).sum()} ({100*(y > 0).mean():.1f}%)")



# %%


def _load_manifest(manifest_path: Path) -> dict | None:
    """Load manifest file if it exists."""
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)
    return None


def _save_manifest(manifest_path: Path, manifest: dict) -> None:
    """Save manifest file atomically."""
    tmp_path = manifest_path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(manifest, f, indent=2)
    tmp_path.rename(manifest_path)


def _save_chunk(chunk_path: Path, chunk_data: dict) -> None:
    """Save a single chunk to disk."""
    tmp_path = chunk_path.with_suffix(".tmp")
    torch.save(chunk_data, tmp_path)
    tmp_path.rename(chunk_path)


def load_chunk(cache_dir: Path, chunk_idx: int) -> dict:
    """Load a single chunk from the cache directory."""
    chunk_path = cache_dir / f"chunk_{chunk_idx:04d}.pt"
    if not chunk_path.exists():
        raise FileNotFoundError(f"Chunk {chunk_idx} not found at {chunk_path}")
    return torch.load(chunk_path)


def load_manifest(cache_dir: Path) -> dict:
    """Load manifest from cache directory."""
    manifest_path = cache_dir / "manifest.json"
    manifest = _load_manifest(manifest_path)
    if manifest is None:
        raise FileNotFoundError(f"No manifest found at {manifest_path}")
    return manifest


def _claim_next_chunk(cache_dir: Path, lock) -> int | None:
    """Atomically claim the next uncompleted chunk.

    Returns chunk index if one is available, None if all chunks are done or in progress.
    """
    manifest_path = cache_dir / "manifest.json"
    with lock:
        manifest = _load_manifest(manifest_path)
        if manifest is None:
            return None

        completed = set(manifest["completed_chunks"])
        in_progress = set(manifest.get("in_progress_chunks", []))

        for chunk_idx in range(manifest["total_chunks"]):
            if chunk_idx not in completed and chunk_idx not in in_progress:
                manifest.setdefault("in_progress_chunks", []).append(chunk_idx)
                _save_manifest(manifest_path, manifest)
                return chunk_idx
        return None


def _mark_chunk_complete(cache_dir: Path, lock, chunk_idx: int) -> None:
    """Mark a chunk as completed and remove from in_progress."""
    manifest_path = cache_dir / "manifest.json"
    with lock:
        manifest = _load_manifest(manifest_path)
        if manifest is None:
            raise RuntimeError("Manifest not found")

        # Move from in_progress to completed
        in_progress = manifest.get("in_progress_chunks", [])
        if chunk_idx in in_progress:
            in_progress.remove(chunk_idx)
        manifest["in_progress_chunks"] = in_progress

        if chunk_idx not in manifest["completed_chunks"]:
            manifest["completed_chunks"].append(chunk_idx)

        _save_manifest(manifest_path, manifest)


def _reset_in_progress_chunks(cache_dir: Path) -> None:
    """Reset all in_progress chunks to unclaimed state (for recovery after crash)."""
    manifest_path = cache_dir / "manifest.json"
    manifest = _load_manifest(manifest_path)
    if manifest is not None and manifest.get("in_progress_chunks"):
        logger.warning(f"Resetting {len(manifest['in_progress_chunks'])} in_progress chunks")
        manifest["in_progress_chunks"] = []
        _save_manifest(manifest_path, manifest)


def _gpu_worker(
    gpu_id: int,
    model_name: str,
    layers: list[int],
    batch_size: int,
    chunk_size: int,
    cache_dir: Path,
    lock,
    dataset_path: str,
    num_samples: int,
) -> None:
    """Worker function that processes chunks on a specific GPU.

    Each worker:
    1. Loads the model on its assigned GPU
    2. Loops claiming uncompleted chunks
    3. Processes each chunk and saves it
    4. Marks chunk as complete
    """
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)  # Set default device for this process
    logger.info(f"[GPU {gpu_id}] Starting worker on {device}")

    # Load model on this GPU
    model = load_model(model_name, torch_dtype=torch.bfloat16, device_map=device)
    tokenizer = load_tokenizer(model_name)

    # Load dataset (HF caches this so it's fast)
    dataset = load_dataset(dataset_path, split="train")
    dataset = dataset.filter(
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
        num_proc=1,  # Single proc in worker to avoid nested multiprocessing issues
    )
    subset = dataset.select(range(min(num_samples, len(dataset))))

    chunks_processed = 0
    while True:
        # Claim next available chunk
        chunk_idx = _claim_next_chunk(cache_dir, lock)
        if chunk_idx is None:
            logger.info(f"[GPU {gpu_id}] No more chunks to process")
            break

        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, num_samples)
        logger.info(f"[GPU {gpu_id}] Processing chunk {chunk_idx} (samples {start_idx}-{end_idx})")

        # Extract prompts/responses for this chunk
        chunk_chosen_prompts = []
        chunk_chosen_responses = []
        chunk_rejected_prompts = []
        chunk_rejected_responses = []

        for i in range(start_idx, end_idx):
            sample = subset[i]
            prompt_c, response_c = extract_prompt_response_text(tokenizer, sample["chosen"])
            prompt_r, response_r = extract_prompt_response_text(tokenizer, sample["rejected"])
            chunk_chosen_prompts.append(prompt_c)
            chunk_chosen_responses.append(response_c)
            chunk_rejected_prompts.append(prompt_r)
            chunk_rejected_responses.append(response_r)

        # Compute embedding diffs for this chunk
        diffs = compute_embedding_diffs(
            model,
            tokenizer,
            chunk_chosen_prompts,
            chunk_chosen_responses,
            chunk_rejected_prompts,
            chunk_rejected_responses,
            layers=layers,
            batch_size=batch_size,
        )

        # Save chunk
        chunk_data = {
            "chunk_idx": chunk_idx,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "nlls_chosen": diffs.nlls_chosen,
            "nlls_rejected": diffs.nlls_rejected,
            "activation_diffs": diffs.activation_diffs,
            "chosen_prompts": chunk_chosen_prompts,
            "chosen_responses": chunk_chosen_responses,
            "rejected_prompts": chunk_rejected_prompts,
            "rejected_responses": chunk_rejected_responses,
        }
        chunk_path = cache_dir / f"chunk_{chunk_idx:04d}.pt"
        _save_chunk(chunk_path, chunk_data)

        # Mark complete
        _mark_chunk_complete(cache_dir, lock, chunk_idx)
        chunks_processed += 1
        logger.info(f"[GPU {gpu_id}] Completed chunk {chunk_idx}")

        # Cleanup
        del diffs
        _oom_cleanup(gpu_id)

    del model
    _oom_cleanup(gpu_id)
    logger.info(f"[GPU {gpu_id}] Worker finished, processed {chunks_processed} chunks")


def cache_embedding_diffs(
    model_name: str,
    num_samples: int,
    layers: list[int],
    batch_size: int,
    chunk_size: int = 1000,
    num_gpus: int | None = None,
    dataset_name: str = "allenai/Dolci-Instruct-DPO",
    output_dir: Path = Path("dpo_embedding_analysis"),
) -> Path:
    """Cache embedding diffs with checkpointing for large datasets.

    Processes data in chunks and saves incrementally to disk.
    Supports resuming from last completed chunk if interrupted.
    Supports multi-GPU processing when num_gpus > 1.

    Args:
        model_name: HuggingFace model name
        num_samples: Number of samples to process
        layers: Which layers to extract activations from
        batch_size: Batch size for model forward pass
        chunk_size: Number of samples per chunk (for checkpointing)
        num_gpus: Number of GPUs to use (None = auto-detect)
        dataset_name: HuggingFace dataset name
        output_dir: Base output directory

    Returns:
        Path to cache directory containing chunks and manifest
    """
    # Auto-detect GPUs if not specified
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()
    num_gpus = max(1, num_gpus)  # At least 1
    logger.info(f"Using {num_gpus} GPU(s)")

    model_slug = model_name.split("/")[-1]
    cache_dir = output_dir / f"{model_slug}-{num_samples}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"

    # Load or create manifest
    manifest = _load_manifest(manifest_path)
    total_chunks = (num_samples + chunk_size - 1) // chunk_size

    if manifest is not None:
        # Verify configuration matches
        if manifest["model_name"] != model_name:
            raise ValueError(f"Model mismatch: {manifest['model_name']} vs {model_name}")
        if manifest["num_samples"] != num_samples:
            raise ValueError(f"num_samples mismatch: {manifest['num_samples']} vs {num_samples}")
        if set(manifest["layers"]) != set(layers):
            raise ValueError(f"layers mismatch: {manifest['layers']} vs {layers}")

        completed_chunks = set(manifest["completed_chunks"])
        logger.info(f"Resuming from manifest: {len(completed_chunks)}/{total_chunks} chunks completed")

        # Reset any in_progress chunks from previous crashed run
        _reset_in_progress_chunks(cache_dir)
    else:
        completed_chunks = set()
        manifest = {
            "model_name": model_name,
            "layers": layers,
            "num_samples": num_samples,
            "chunk_size": chunk_size,
            "completed_chunks": [],
            "in_progress_chunks": [],
            "total_chunks": total_chunks,
        }
        _save_manifest(manifest_path, manifest)
        logger.info(f"Created new manifest for {total_chunks} chunks")

    # Check if all chunks complete
    if len(completed_chunks) == total_chunks:
        logger.info("All chunks already computed")
        return cache_dir

    # Multi-GPU path: spawn workers
    if num_gpus > 1:
        logger.info(f"Starting {num_gpus} GPU workers...")

        # Use spawn to avoid CUDA fork issues
        ctx = mp.get_context("spawn")
        manager = ctx.Manager()
        lock = manager.Lock()

        processes = []
        for gpu_id in range(num_gpus):
            p = ctx.Process(
                target=_gpu_worker,
                args=(
                    gpu_id,
                    model_name,
                    layers,
                    batch_size,
                    chunk_size,
                    cache_dir,
                    lock,
                    dataset_name,
                    num_samples,
                ),
            )
            p.start()
            processes.append(p)

        # Wait for all workers to complete
        for p in processes:
            p.join()

        logger.success(f"All {total_chunks} chunks saved to {cache_dir}")
        return cache_dir

    # Single-GPU path: process sequentially (original logic)
    logger.info("Loading dataset...")
    dataset = load_dataset(dataset_name, split="train")
    logger.info(f"Before filtering: {len(dataset)} samples")
    dataset = dataset.filter(
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
        num_proc=8,
    )
    logger.info(f"After filtering: {len(dataset)} samples")

    if len(dataset) < num_samples:
        logger.warning(f"Only {len(dataset)} samples available, adjusting num_samples")
        num_samples = len(dataset)
        total_chunks = (num_samples + chunk_size - 1) // chunk_size
        manifest["num_samples"] = num_samples
        manifest["total_chunks"] = total_chunks
        _save_manifest(manifest_path, manifest)

    subset = dataset.select(range(num_samples))

    # Load tokenizer and model
    tokenizer = load_tokenizer(model_name)
    model = load_model(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    # Process chunks
    chunks_to_process = [i for i in range(total_chunks) if i not in completed_chunks]
    logger.info(f"Processing {len(chunks_to_process)} remaining chunks...")

    for chunk_idx in tqdm(chunks_to_process, desc="Processing chunks"):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, num_samples)

        # Extract prompts/responses for this chunk
        chunk_chosen_prompts = []
        chunk_chosen_responses = []
        chunk_rejected_prompts = []
        chunk_rejected_responses = []

        for i in range(start_idx, end_idx):
            sample = subset[i]
            prompt_c, response_c = extract_prompt_response_text(tokenizer, sample["chosen"])
            prompt_r, response_r = extract_prompt_response_text(tokenizer, sample["rejected"])
            chunk_chosen_prompts.append(prompt_c)
            chunk_chosen_responses.append(response_c)
            chunk_rejected_prompts.append(prompt_r)
            chunk_rejected_responses.append(response_r)

        # Compute embedding diffs for this chunk
        diffs = compute_embedding_diffs(
            model,
            tokenizer,
            chunk_chosen_prompts,
            chunk_chosen_responses,
            chunk_rejected_prompts,
            chunk_rejected_responses,
            layers=layers,
            batch_size=batch_size,
        )

        # Save chunk
        chunk_data = {
            "chunk_idx": chunk_idx,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "nlls_chosen": diffs.nlls_chosen,
            "nlls_rejected": diffs.nlls_rejected,
            "activation_diffs": diffs.activation_diffs,
            "chosen_prompts": chunk_chosen_prompts,
            "chosen_responses": chunk_chosen_responses,
            "rejected_prompts": chunk_rejected_prompts,
            "rejected_responses": chunk_rejected_responses,
        }
        chunk_path = cache_dir / f"chunk_{chunk_idx:04d}.pt"
        _save_chunk(chunk_path, chunk_data)

        # Update manifest
        manifest["completed_chunks"].append(chunk_idx)
        _save_manifest(manifest_path, manifest)
        logger.info(f"Saved chunk {chunk_idx} ({end_idx - start_idx} samples)")

        # Cleanup
        del diffs
        _oom_cleanup()

    del model
    _oom_cleanup()

    logger.success(f"All {total_chunks} chunks saved to {cache_dir}")
    return cache_dir

    # # %% Phase 3: Compute metrics and build results
    # print("\n=== Phase 3: Computing metrics ===")
    # results = []

    # def compute_similarity_metric(
    #     h_chosen: Float[Tensor, "hidden_dim"],
    #     h_rejected: Float[Tensor, "hidden_dim"],
    # ) -> float:
    #     return torch.norm(h_rejected - h_chosen).item()

    # for i in range(len(valid_indices)):
    #     similarity_metric = compute_similarity_metric(
    #         pre_chosen_stats.hidden_sums[i],
    #         pre_rejected_stats.hidden_sums[i],
    #     )

    #     nll_diff = pre_chosen_stats.nlls[i] - post_chosen_stats.nlls[i]

    #     results.append({
    #         "idx": valid_indices[i],
    #         "prompt_chosen": chosen_prompts[i][:500],
    #         "response_chosen": chosen_responses[i][:500],
    #         "response_rejected": rejected_responses[i][:500],
    #         "pre_chosen_nll": pre_chosen_stats.nlls[i],
    #         "pre_rejected_nll": pre_rejected_stats.nlls[i],
    #         "post_chosen_nll": post_chosen_stats.nlls[i],
    #         "similarity_metric": similarity_metric,
    #         "nll_diff": nll_diff,
    #         "chosen_hidden_norm": pre_chosen_stats.hidden_sums[i].norm().item(),
    #         "rejected_hidden_norm": pre_rejected_stats.hidden_sums[i].norm().item(),
    #     })

    # df = pd.DataFrame(results)

    # csv_path = OUTPUT_DIR / "results.csv"
    # df.to_csv(csv_path, index=False)
    # print(f"Results saved to {csv_path}")

    # # %% Phase 4: Visualization
    # print("\n=== Phase 4: Generating plots ===")
    # plot_correlation(df, OUTPUT_DIR)

    # # %% Inspect results
    # df.head(10)
