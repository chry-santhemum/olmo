# %% Imports and Configuration
import gc
import inspect
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

# %%

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
        outputs, input_ids, attention_mask, prompt_lens: list[int], layers: list[int], actual_bs: int
    ) -> tuple[list[float], list[dict[int, Tensor]]]:
        """Extract NLLs and mean activations from model outputs."""
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

            # Mean activations per layer
            item_acts = {}
            for layer_idx in layers:
                hidden = outputs.hidden_states[layer_idx]
                response_hidden = hidden[i, response_start:response_end]
                item_acts[layer_idx] = response_hidden.mean(dim=0).cpu()
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

        outputs = None
        try:
            with torch.inference_mode():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )

            nlls, activations = _extract_mean_activations(
                outputs, input_ids, attention_mask, prompt_lens, layers, actual_bs * 2
            )
        finally:
            if outputs is not None:
                del outputs

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

def cache_embedding_diffs(
    model_name: str,
    num_samples: int,
    layers: list[int],
    batch_size: int,
    output_dir: Path = Path("dpo_embedding_analysis"),
) -> dict:
    """Cache embedding diffs with incremental layer support.

    Output path does NOT include layer numbers. If cache exists, only computes missing layers.
    """
    model_slug = model_name.split("/")[-1]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_slug}-first_{num_samples}.pt"

    # Try to load existing cache
    cache = None
    existing_layers: set[int] = set()
    if output_path.exists():
        cache = torch.load(output_path)
        existing_layers = set(cache["config"]["layers"])
        logger.info(f"Loaded existing cache with layers {sorted(existing_layers)}")

    # Determine which layers to compute
    layers_to_compute = [l for l in layers if l not in existing_layers]
    if not layers_to_compute:
        logger.info("All requested layers already cached")
        return cache

    logger.info(f"Computing missing layers: {layers_to_compute}")

    # Load dataset and extract prompts/responses (or reuse from cache)
    if cache is not None:
        chosen_prompts = cache["chosen_prompts"]
        chosen_responses = cache["chosen_responses"]
        rejected_prompts = cache["rejected_prompts"]
        rejected_responses = cache["rejected_responses"]
    else:
        logger.info("Loading dataset...")
        dataset = load_dataset("allenai/Dolci-Instruct-DPO", split="train")
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

        subset = dataset.select(range(num_samples))
        logger.info(f"Will process {len(subset)} samples")

        tokenizer = load_tokenizer(model_name)
        logger.info("Extracting prompts and responses...")
        chosen_prompts = []
        chosen_responses = []
        rejected_prompts = []
        rejected_responses = []

        for sample in tqdm(subset, desc="Extracting"):
            prompt_c, response_c = extract_prompt_response_text(tokenizer, sample["chosen"])
            prompt_r, response_r = extract_prompt_response_text(tokenizer, sample["rejected"])
            chosen_prompts.append(prompt_c)
            chosen_responses.append(response_c)
            rejected_prompts.append(prompt_r)
            rejected_responses.append(response_r)

    # Load model and compute diffs for missing layers
    tokenizer = load_tokenizer(model_name)
    model = load_model(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )

    logger.info(f"Computing embedding diffs for layers {layers_to_compute}...")
    new_diffs = compute_embedding_diffs(
        model,
        tokenizer,
        chosen_prompts,
        chosen_responses,
        rejected_prompts,
        rejected_responses,
        layers=layers_to_compute,
        batch_size=batch_size,
    )

    del model
    _oom_cleanup()

    # Merge into existing cache or create new
    if cache is None:
        cache = {
            "nlls_chosen": new_diffs.nlls_chosen,
            "nlls_rejected": new_diffs.nlls_rejected,
            "activation_diffs": new_diffs.activation_diffs,
            "chosen_prompts": chosen_prompts,
            "chosen_responses": chosen_responses,
            "rejected_prompts": rejected_prompts,
            "rejected_responses": rejected_responses,
            "config": {
                "model_name": model_name,
                "layers": layers_to_compute,
                "num_samples": num_samples,
            },
        }
    else:
        # Merge new layers into existing activation_diffs
        for i, new_sample_diffs in enumerate(new_diffs.activation_diffs):
            for layer_idx, diff in new_sample_diffs.items():
                cache["activation_diffs"][i][layer_idx] = diff
        cache["config"]["layers"] = sorted(existing_layers | set(layers_to_compute))

    torch.save(cache, output_path)
    logger.success(f"Embedding diffs cached to {output_path} (layers: {cache['config']['layers']})")
    return cache

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
