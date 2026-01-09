# %% Imports and Configuration
import gc
import inspect
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import torch
import torch.nn.functional as F
from datasets import load_dataset
from jaxtyping import Float
from loguru import logger
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer

# %%
# Configuration
PRE_DPO_MODEL = "allenai/Olmo-3-7B-Instruct-SFT"
POST_DPO_MODEL = "allenai/Olmo-3-7B-Instruct-DPO"
STARTING_BATCH_SIZE = 8

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

def _oom_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Sometimes helps reclaim CUDA IPC allocations
        torch.cuda.ipc_collect()
        logger.info(f"CUDA allocated memory: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        logger.info(f"CUDA reserved memory: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

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
class ResponseStats:
    nll: float
    hidden_sum: Float[Tensor, "hidden_dim"] | None


@dataclass
class BatchResponseStats:
    """Stats for a batch of responses."""
    nlls: list[float]
    hidden_sums: list[Tensor] | None


def load_model(model_name: str, **kwargs):
    """Load model in eval mode with gradients disabled."""
    print(f"Loading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print("Model loaded (eval mode, grads disabled)")
    return model


def load_tokenizer(model_name: str):
    """Load tokenizer for the model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _compute_stats_for_slice_inner(
    batch_size: int,
    model,
    tokenizer,
    prompt_texts: list[str],
    response_texts: list[str],
    start_pos: int,
    compute_hidden: bool = True,
) -> tuple[BatchResponseStats, int]:
    """
    Compute stats for a single slice: indices [start_pos : start_pos + batch_size).
    Returns (stats_for_slice, end_pos).
    """
    device = next(model.parameters()).device
    n = len(prompt_texts)
    end_pos = min(start_pos + batch_size, n)

    batch_prompts = prompt_texts[start_pos:end_pos]
    batch_responses = response_texts[start_pos:end_pos]
    actual_bs = len(batch_prompts)
    if actual_bs == 0:
        return BatchResponseStats(nlls=[], hidden_sums=[] if compute_hidden else None), end_pos

    # Tokenize each sample separately (keeps per-sample prompt/response boundaries)
    batch_input_ids: list[list[int]] = []
    batch_prompt_lens: list[int] = []
    batch_response_lens: list[int] = []

    for prompt, response in zip(batch_prompts, batch_responses):
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
        full_ids = prompt_ids + response_ids

        batch_input_ids.append(full_ids)
        batch_prompt_lens.append(len(prompt_ids))
        batch_response_lens.append(len(response_ids))

    # Left-pad to the max length in this slice
    max_len = max(len(ids) for ids in batch_input_ids)
    padded_ids: list[list[int]] = []
    attention_masks: list[list[int]] = []

    for ids in batch_input_ids:
        pad_len = max_len - len(ids)
        padded_ids.append([tokenizer.pad_token_id] * pad_len + ids)
        attention_masks.append([0] * pad_len + [1] * len(ids))

    input_ids = torch.tensor(padded_ids, device=device)
    attention_mask = torch.tensor(attention_masks, device=device)

    slice_nlls: list[float] = []
    slice_hidden_sums: Optional[list[Tensor]] = [] if compute_hidden else None

    outputs = None
    try:
        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=compute_hidden,
            )

        for i in range(actual_bs):
            pad_len = max_len - len(batch_input_ids[i])
            prompt_len = batch_prompt_lens[i]
            response_len = batch_response_lens[i]

            if response_len == 0:
                slice_nlls.append(float("nan"))
                if compute_hidden:
                    slice_hidden_sums.append(torch.zeros(model.config.hidden_size))
                continue

            seq_start = pad_len
            prompt_end = seq_start + prompt_len
            seq_end = prompt_end + response_len

            # logits[prompt_end-1:seq_end-1] predict labels[prompt_end:seq_end]
            logits = outputs.logits[i, prompt_end - 1 : seq_end - 1]
            labels = input_ids[i, prompt_end:seq_end]
            nll = F.cross_entropy(logits, labels, reduction="mean").item()
            slice_nlls.append(nll)

            if compute_hidden:
                last_hidden = outputs.hidden_states[-1]
                response_hidden = last_hidden[i, prompt_end:seq_end]
                hidden_sum = response_hidden.sum(dim=0).cpu()
                slice_hidden_sums.append(hidden_sum)
    finally:
        if outputs is not None:
            try:
                del outputs
            except Exception:
                pass

    return BatchResponseStats(nlls=slice_nlls, hidden_sums=slice_hidden_sums), end_pos

# --- NEW: outer driver that resets batch size each slice ---

def compute_response_stats_batch(
    model,
    tokenizer,
    prompt_texts: list[str],
    response_texts: list[str],
    compute_hidden: bool = True,
    starting_batch_size: int = STARTING_BATCH_SIZE,
) -> BatchResponseStats:
    """
    Process the dataset in slices. Each slice starts at `start_pos` and attempts `starting_batch_size`.
    If OOM occurs for that slice, the decorator halves batch size until it fits, then the NEXT slice
    starts again at `starting_batch_size`.
    """
    assert len(prompt_texts) == len(response_texts), "prompt_texts and response_texts must align"
    n = len(prompt_texts)

    decorated_slice_fn = find_executable_batch_size(starting_batch_size)(_compute_stats_for_slice_inner)

    all_nlls: list[float] = [float("nan")] * n
    all_hidden_sums: Optional[list[Tensor]] = [None] * n if compute_hidden else None  # type: ignore

    start_pos = 0
    pbar = tqdm(total=n, desc="Processing slices")

    while start_pos < n:
        stats, end_pos = decorated_slice_fn(
            model,
            tokenizer,
            prompt_texts,
            response_texts,
            start_pos,
            compute_hidden,
        )

        # write back into full arrays
        all_nlls[start_pos:end_pos] = stats.nlls
        if compute_hidden and all_hidden_sums is not None:
            all_hidden_sums[start_pos:end_pos] = stats.hidden_sums  # type: ignore

        pbar.update(end_pos - start_pos)
        start_pos = end_pos

    pbar.close()
    return BatchResponseStats(nlls=all_nlls, hidden_sums=all_hidden_sums)


# def compute_similarity_metric(
#     h_chosen: Float[Tensor, "hidden_dim"],
#     h_rejected: Float[Tensor, "hidden_dim"],
# ) -> float:
#     """Compute similarity metric: dot(h_chosen, h_rejected - h_chosen).

#     Lower absolute value indicates more similar embeddings.
#     """
#     diff = h_rejected - h_chosen
#     return torch.dot(h_chosen, diff).item()

def compute_similarity_metric(
    h_chosen: Float[Tensor, "hidden_dim"],
    h_rejected: Float[Tensor, "hidden_dim"],
) -> float:
    return torch.norm(h_rejected - h_chosen).item()

def extract_prompt_and_response(
    tokenizer,
    messages: list[dict],
) -> tuple[str, str]:
    """Extract prompt and response from multi-turn conversation."""
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


# %% Visualization Functions
def plot_correlation(df: pd.DataFrame, output_dir: Path):
    """Plot correlation between similarity metric and NLL diff."""
    valid_df = df.dropna(subset=["similarity_metric", "nll_diff"])

    if len(valid_df) < 10:
        print("Not enough valid samples for plotting")
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
    print(f"Plot saved to {plot_path}")

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
    print(f"Distributions plot saved to {hist_path}")

    print("\n=== Summary Statistics ===")
    print(f"Samples: {len(valid_df)}")
    print(f"Pearson correlation: r={pearson_r:.4f}, p={pearson_p:.2e}")
    print(f"Spearman correlation: rho={spearman_r:.4f}, p={spearman_p:.2e}")
    print(f"Similarity metric: mean={x.mean():.2e}, std={x.std():.2e}")
    print(f"NLL diff: mean={y.mean():.4f}, std={y.std():.4f}")
    print(f"Samples where DPO improved (nll_diff > 0): {(y > 0).sum()} ({100*(y > 0).mean():.1f}%)")


# %% Load Dataset
print("Loading dataset...")
dataset = load_dataset("allenai/Dolci-Instruct-DPO", split="train")
print(f"Dataset loaded: {len(dataset)} samples")

# %%
subset = dataset.select(range(500))
print(f"Will process {len(subset)} samples")

# %% Prepare samples
OUTPUT_DIR = Path("dpo_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
tokenizer = load_tokenizer(PRE_DPO_MODEL)

# Pre-extract all prompts and responses
print("Extracting prompts and responses...")
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
            print(f"Skipping sample {i} because no response")
            continue
        chosen_prompts.append(prompt_c)
        chosen_responses.append(response_c)
        rejected_prompts.append(prompt_r)
        rejected_responses.append(response_r)
        valid_indices.append(i)
    except Exception as e:
        print(f"Error extracting sample {i}: {e}")
        continue

print(f"Extracted {len(valid_indices)} valid samples")

# %% Phase 1: Process with pre-DPO model
print("\n=== Phase 1: Processing with pre-DPO model ===")

pre_model = load_model(
    PRE_DPO_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)

print("Computing chosen response stats...")
pre_chosen_stats = compute_response_stats_batch(
    pre_model, tokenizer, chosen_prompts, chosen_responses, compute_hidden=True
)

print("Computing rejected response stats...")
pre_rejected_stats = compute_response_stats_batch(
    pre_model, tokenizer, rejected_prompts, rejected_responses, compute_hidden=True
)

# Free pre-model memory
del pre_model
_oom_cleanup()
print("Pre-DPO processing complete.")

# %% Phase 2: Process with post-DPO model
print("\n=== Phase 2: Processing with post-DPO model ===")
post_model = load_model(
    POST_DPO_MODEL,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)

print("Computing chosen response stats (post-DPO)...")
post_chosen_stats = compute_response_stats_batch(
    post_model, tokenizer, chosen_prompts, chosen_responses, compute_hidden=False
)

# Free post-model memory
del post_model
_oom_cleanup()
print("Post-DPO processing complete.")

# %% Phase 3: Compute metrics and build results
print("\n=== Phase 3: Computing metrics ===")
results = []

def compute_similarity_metric(
    h_chosen: Float[Tensor, "hidden_dim"],
    h_rejected: Float[Tensor, "hidden_dim"],
) -> float:
    return torch.norm(h_rejected - h_chosen).item()

for i in range(len(valid_indices)):
    similarity_metric = compute_similarity_metric(
        pre_chosen_stats.hidden_sums[i],
        pre_rejected_stats.hidden_sums[i],
    )

    nll_diff = pre_chosen_stats.nlls[i] - post_chosen_stats.nlls[i]

    results.append({
        "idx": valid_indices[i],
        "prompt_chosen": chosen_prompts[i][:500],
        "response_chosen": chosen_responses[i][:500],
        "response_rejected": rejected_responses[i][:500],
        "pre_chosen_nll": pre_chosen_stats.nlls[i],
        "pre_rejected_nll": pre_rejected_stats.nlls[i],
        "post_chosen_nll": post_chosen_stats.nlls[i],
        "similarity_metric": similarity_metric,
        "nll_diff": nll_diff,
        "chosen_hidden_norm": pre_chosen_stats.hidden_sums[i].norm().item(),
        "rejected_hidden_norm": pre_rejected_stats.hidden_sums[i].norm().item(),
    })

df = pd.DataFrame(results)

csv_path = OUTPUT_DIR / "results.csv"
df.to_csv(csv_path, index=False)
print(f"Results saved to {csv_path}")

# %% Phase 4: Visualization
print("\n=== Phase 4: Generating plots ===")
plot_correlation(df, OUTPUT_DIR)

# %% Inspect results
df.head(10)
