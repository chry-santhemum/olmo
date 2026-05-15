import json
import urllib.error
import urllib.request
from numbers import Real
from pathlib import Path
from typing import Literal

import numpy as np
from loguru import logger
from tqdm import tqdm
from transformers import AutoTokenizer

from filter_vector import build_training_rows_from_selection, load_extra_rows, load_jsonl, save_jsonl


DEFAULT_DATASET_PATH = Path("filtered/33K-baseline/dataset.jsonl")
DEFAULT_SCORE_DIR = Path("cached_repr_33K_SFT/LLS")
DEFAULT_MODEL = "allenai/Olmo-3-7B-Instruct-SFT"
DEFAULT_SERVER_URL = "http://127.0.0.1:8020/v1/completions"
DEFAULT_MAX_SEQ_LENGTH = 16384
EXAMPLE_PERCENTILES = [0, 2, 5, 10, 50, 90, 95, 98, 100]
TRUNCATED_RESPONSE_SKIP_REASON = "final_assistant_response_truncated"


def clean_messages(messages: list[dict]) -> list[dict]:
    return [{"role": message["role"], "content": message["content"]} for message in messages]


def read_default_system_prompt(tokenizer) -> str:
    """Return the implicit system prompt added by the tokenizer chat template."""
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "x"}],
        tokenize=False,
        add_generation_prompt=False,
    )
    start_marker = "<|im_start|>system\n"
    end_marker = "<|im_end|>"
    if start_marker not in rendered:
        return ""
    start = rendered.index(start_marker) + len(start_marker)
    end = rendered.index(end_marker, start)
    return rendered[start:end]


def append_system_prompt(messages: list[dict], system_prompt: str, default_system_prompt: str) -> list[dict]:
    """Append the steering instruction to the active system prompt for this conversation."""
    messages = clean_messages(messages)
    if messages and messages[0]["role"] == "system":
        first = dict(messages[0])
        first["content"] = first["content"] + "\n\n" + system_prompt
        return [first] + messages[1:]

    if default_system_prompt:
        system_content = default_system_prompt + "\n\n" + system_prompt
    else:
        system_content = system_prompt
    return [{"role": "system", "content": system_content}] + messages


def chat_token_ids(tokenizer, messages: list[dict], max_seq_length: int, add_generation_prompt: bool) -> list[int]:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        truncation=True,
        max_length=max_seq_length,
        return_dict=False,
    )


def build_scoring_input(
    tokenizer,
    messages: list[dict],
    max_seq_length: int,
) -> dict:
    """Tokenize a chat and return the final-assistant token span to score."""
    if not messages:
        raise ValueError("Cannot score an empty conversation")
    if messages[-1]["role"] != "assistant":
        raise ValueError(f"Expected final message role to be assistant, got {messages[-1]['role']}")

    input_ids = chat_token_ids(tokenizer, messages, max_seq_length, add_generation_prompt=False)
    prefix_ids = chat_token_ids(tokenizer, messages[:-1], max_seq_length, add_generation_prompt=True)
    response_start = len(prefix_ids)
    response_len = len(input_ids) - response_start
    if response_len <= 0:
        raise ValueError("Final assistant response has no scoreable tokens after truncation")

    return {
        "input_ids": input_ids,
        "response_start": response_start,
        "response_len": response_len,
    }


def score_request_for_row(
    tokenizer,
    row: dict,
    system_prompt: str,
    default_system_prompt: str,
    max_seq_length: int,
) -> list[dict]:
    chosen = clean_messages(row["chosen"])
    rejected = clean_messages(row["rejected"])
    return [
        build_scoring_input(tokenizer, chosen, max_seq_length),
        build_scoring_input(tokenizer, rejected, max_seq_length),
        build_scoring_input(
            tokenizer,
            append_system_prompt(chosen, system_prompt, default_system_prompt),
            max_seq_length,
        ),
        build_scoring_input(
            tokenizer,
            append_system_prompt(rejected, system_prompt, default_system_prompt),
            max_seq_length,
        ),
    ]


def request_prompt_logprobs(
    token_id_batches: list[list[int]],
    model_name: str,
    server_url: str,
    timeout_s: int,
) -> list[dict]:
    """Call a vLLM OpenAI-compatible completions endpoint with tokenized prompts."""
    request_body = {
        "model": model_name,
        "prompt": token_id_batches,
        "max_tokens": 1,
        "temperature": 0,
        "prompt_logprobs": 0,
        "return_token_ids": True,
    }
    request = urllib.request.Request(
        server_url,
        data=json.dumps(request_body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Could not reach vLLM server at {server_url}. Start it with "
            "`./serve_model.sh --config config_instr_sft.yml`."
        ) from exc

    choices = sorted(data["choices"], key=lambda choice: choice["index"])
    if len(choices) != len(token_id_batches):
        raise ValueError(f"Expected {len(token_id_batches)} choices from vLLM, got {len(choices)}")
    return choices


def sum_response_logprob(choice: dict, response_start: int, response_len: int) -> float:
    """Sum vLLM prompt logprobs over the final assistant response token span."""
    token_ids = choice.get("prompt_token_ids")
    prompt_logprobs = choice.get("prompt_logprobs")
    if token_ids is None or prompt_logprobs is None:
        raise ValueError("vLLM response is missing prompt_token_ids or prompt_logprobs")
    if len(token_ids) != len(prompt_logprobs):
        raise ValueError(f"Token/logprob length mismatch: {len(token_ids)} vs {len(prompt_logprobs)}")

    total = 0.0
    for pos in range(response_start, response_start + response_len):
        token_id = token_ids[pos]
        logprobs_for_pos = prompt_logprobs[pos]
        if logprobs_for_pos is None:
            raise ValueError(f"Missing prompt logprob at token position {pos}")
        token_key = str(token_id)
        if token_key not in logprobs_for_pos:
            raise ValueError(f"Actual token id {token_id} missing from vLLM prompt logprobs at position {pos}")
        total += float(logprobs_for_pos[token_key]["logprob"])
    return total


def build_score_row(row_index: int, request_inputs: list[dict], choices: list[dict]) -> dict:
    base_chosen = sum_response_logprob(
        choices[0],
        request_inputs[0]["response_start"],
        request_inputs[0]["response_len"],
    )
    base_rejected = sum_response_logprob(
        choices[1],
        request_inputs[1]["response_start"],
        request_inputs[1]["response_len"],
    )
    system_chosen = sum_response_logprob(
        choices[2],
        request_inputs[2]["response_start"],
        request_inputs[2]["response_len"],
    )
    system_rejected = sum_response_logprob(
        choices[3],
        request_inputs[3]["response_start"],
        request_inputs[3]["response_len"],
    )
    chosen_len = request_inputs[0]["response_len"]
    rejected_len = request_inputs[1]["response_len"]
    base_diff = base_chosen - base_rejected
    system_diff = system_chosen - system_rejected
    raw_score = system_diff - base_diff
    score = raw_score / float(chosen_len + rejected_len)

    return {
        "row_index": row_index,
        "skip": False,
        "base_chosen_logprob": base_chosen,
        "base_rejected_logprob": base_rejected,
        "system_chosen_logprob": system_chosen,
        "system_rejected_logprob": system_rejected,
        "chosen_len": chosen_len,
        "rejected_len": rejected_len,
        "base_diff": base_diff,
        "system_diff": system_diff,
        "raw_score": raw_score,
        "score": score,
    }


def append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def save_distribution_figure(scores: list[float], save_path: Path) -> None:
    import matplotlib.pyplot as plt

    scores_arr = np.array(scores)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(scores_arr, bins=200, edgecolor="black", alpha=0.7)
    ax.set_yscale("log")
    ax.set_xlabel("Logprob Linear Score")
    ax.set_ylabel("Count (log scale)")
    ax.set_title(f"Logprob Linear Score Distribution (n={len(scores_arr)})")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved score distribution to {save_path}")


def percentile_indices(scores: list[float], percentiles: list[float], num_examples: int) -> dict[float, list[int]]:
    scores_arr = np.array(scores)
    output = {}
    for pct in percentiles:
        cutoff = np.percentile(scores_arr, pct)
        higher = [idx for idx, score in enumerate(scores) if score >= cutoff]
        lower = [idx for idx, score in enumerate(scores) if score < cutoff]
        sorted_higher = sorted(higher, key=lambda idx: scores[idx])
        sorted_lower = sorted(lower, key=lambda idx: scores[idx], reverse=True)

        if len(sorted_higher) < num_examples // 2:
            indices = sorted_higher + sorted_lower[: num_examples - len(sorted_higher)]
        elif len(sorted_lower) < num_examples // 2:
            indices = sorted_higher[: num_examples - len(sorted_lower)] + sorted_lower
        else:
            indices = sorted_higher[: num_examples // 2] + sorted_lower[: num_examples // 2]
        output[pct] = indices
    return output


def save_percentile_examples(
    dataset_rows: list[dict],
    score_rows: list[dict],
    save_path: Path,
    num_examples: int,
    percentiles: list[float] = EXAMPLE_PERCENTILES,
) -> None:
    scores = [row["score"] for row in score_rows]
    scores_arr = np.array(scores)
    examples = {}
    for pct, indices in percentile_indices(scores, percentiles, num_examples).items():
        examples[pct] = []
        for idx in indices:
            score_row = score_rows[idx]
            row = dataset_rows[score_row["row_index"]]
            examples[pct].append(
                {
                    "row_index": score_row["row_index"],
                    "prompt": clean_messages(row["chosen"][:-1]),
                    "chosen": row["chosen"][-1]["content"],
                    "rejected": row["rejected"][-1]["content"],
                    "score": score_row["score"],
                    "raw_score": score_row["raw_score"],
                    "base_diff": score_row["base_diff"],
                    "system_diff": score_row["system_diff"],
                    "score_percentile": (
                        100.0 * float(np.mean(scores_arr < score_row["score"])),
                        100.0 * float(np.mean(scores_arr <= score_row["score"])),
                    ),
                }
            )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(examples, f, indent=2)
    logger.info(f"Saved percentile examples to {save_path}")


def validate_score_cache(score_rows: list[dict], dataset_size: int) -> None:
    if len(score_rows) != dataset_size:
        raise ValueError(f"Score cache has {len(score_rows)} rows, but dataset has {dataset_size}")
    for idx, row in enumerate(score_rows):
        if row.get("row_index") != idx:
            raise ValueError(f"Expected score row {idx} to have row_index={idx}, got {row.get('row_index')}")
        if row.get("skip", False):
            if row.get("skip_reason") is None:
                raise ValueError(f"Skipped score row {idx} is missing skip_reason")
        elif not isinstance(row.get("score"), Real):
            raise ValueError(f"Score row {idx} is missing a numeric score")


def valid_score_rows(score_rows: list[dict]) -> list[dict]:
    return [row for row in score_rows if not row.get("skip", False)]


def cache_logprob_diffs(
    system_prompt: str,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    save_dir: Path = DEFAULT_SCORE_DIR,
    model_name: str = DEFAULT_MODEL,
    server_url: str = DEFAULT_SERVER_URL,
    pair_batch_size: int = 1,
    max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
    print_num: int = 10,
    limit: int | None = None,
    timeout_s: int = 600,
    resume: bool = True,
) -> Path:
    """Cache logprob-linear scores for a preference dataset.

    Each row is scored from four logprob sums: chosen/rejected under the base
    chat, and chosen/rejected after appending `system_prompt` to the active
    system prompt. The cached score is normalized by chosen_len + rejected_len.
    """
    dataset_path = Path(dataset_path)
    save_dir = Path(save_dir)
    dataset_rows = load_jsonl(dataset_path)
    if limit is not None:
        dataset_rows = dataset_rows[:limit]
    save_dir.mkdir(parents=True, exist_ok=True)
    scores_path = save_dir / "scores.jsonl"

    if scores_path.exists() and resume:
        score_rows = load_jsonl(scores_path)
        if len(score_rows) > len(dataset_rows):
            raise ValueError(f"Existing score cache has {len(score_rows)} rows for {len(dataset_rows)} input rows")
        for idx, row in enumerate(score_rows):
            if row.get("row_index") != idx:
                raise ValueError(f"Existing score cache is not sequential at row {idx}")
        start_row = len(score_rows)
        logger.info(f"Resuming from {scores_path}: {start_row}/{len(dataset_rows)} rows already scored")
    else:
        if scores_path.exists():
            scores_path.unlink()
        score_rows = []
        start_row = 0

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    default_system_prompt = read_default_system_prompt(tokenizer)

    for start in tqdm(range(start_row, len(dataset_rows), pair_batch_size), desc="Scoring logprobs"):
        end = min(start + pair_batch_size, len(dataset_rows))
        batch_requests = []
        for row_idx in range(start, end):
            try:
                request_inputs = score_request_for_row(
                    tokenizer,
                    dataset_rows[row_idx],
                    system_prompt,
                    default_system_prompt,
                    max_seq_length,
                )
            except ValueError as exc:
                if "no scoreable tokens after truncation" not in str(exc):
                    raise
                logger.warning(
                    f"Skipping row {row_idx}: final assistant response has no scoreable tokens after truncation"
                )
                batch_requests.append((row_idx, None))
            else:
                batch_requests.append((row_idx, request_inputs))

        flat_inputs = [
            item
            for _, request_inputs in batch_requests
            if request_inputs is not None
            for item in request_inputs
        ]
        choices = []
        if flat_inputs:
            choices = request_prompt_logprobs(
                [item["input_ids"] for item in flat_inputs],
                model_name,
                server_url,
                timeout_s,
            )

        new_score_rows = []
        choice_offset = 0
        for row_idx, request_inputs in batch_requests:
            if request_inputs is None:
                new_score_rows.append(
                    {
                        "row_index": row_idx,
                        "skip": True,
                        "skip_reason": TRUNCATED_RESPONSE_SKIP_REASON,
                        "score": None,
                    }
                )
                continue

            new_score_rows.append(
                build_score_row(row_idx, request_inputs, choices[choice_offset : choice_offset + 4])
            )
            choice_offset += 4
        append_jsonl(scores_path, new_score_rows)
        score_rows.extend(new_score_rows)

    validate_score_cache(score_rows, len(dataset_rows))
    valid_rows = valid_score_rows(score_rows)
    if not valid_rows:
        raise ValueError("No valid score rows were cached")
    scores = [row["score"] for row in valid_rows]
    save_distribution_figure(scores, save_dir / "distr.png")
    save_percentile_examples(dataset_rows, valid_rows, save_dir / "examples.json", print_num)

    metadata = {
        "dataset_path": str(dataset_path),
        "model_name": model_name,
        "system_prompt": system_prompt,
        "server_url": server_url,
        "max_seq_length": max_seq_length,
        "pair_batch_size": pair_batch_size,
        "num_rows": len(score_rows),
        "num_skipped": len(score_rows) - len(valid_rows),
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "min_score": float(np.min(scores)),
        "max_score": float(np.max(scores)),
    }
    with open(save_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved logprob score cache to {scores_path}")
    return scores_path


def select_indices_from_scores(
    score_rows: list[dict],
    top_pct: float | None,
    bottom_pct: float | None,
    top_threshold: float | None,
    bottom_threshold: float | None,
) -> set[int]:
    selected_indices: set[int] = set()
    scores = [row["score"] for row in score_rows]
    if not scores:
        raise ValueError("Cannot select from an empty score cache")

    scores_arr = np.array(scores)
    if top_pct is not None:
        cutoff = float(np.percentile(scores_arr, 100 - top_pct))
        selected_indices.update(
            row["row_index"] for row in score_rows if row["score"] >= cutoff
        )
        logger.info(f"Selected top {top_pct}% (cutoff={cutoff:.6f})")
    if top_threshold is not None:
        selected_indices.update(
            row["row_index"] for row in score_rows if row["score"] >= top_threshold
        )
        logger.info(f"Selected top threshold {top_threshold:.6f}")
    if bottom_pct is not None:
        cutoff = float(np.percentile(scores_arr, bottom_pct))
        selected_indices.update(
            row["row_index"] for row in score_rows if row["score"] <= cutoff
        )
        logger.info(f"Selected bottom {bottom_pct}% (cutoff={cutoff:.6f})")
    if bottom_threshold is not None:
        selected_indices.update(
            row["row_index"] for row in score_rows if row["score"] <= bottom_threshold
        )
        logger.info(f"Selected bottom threshold {bottom_threshold:.6f}")
    return selected_indices


def filter_logprob(
    score_path: Path = DEFAULT_SCORE_DIR / "scores.jsonl",
    dataset_path: Path = DEFAULT_DATASET_PATH,
    save_dir: Path = Path("filtered/16K-logprob-bottom-discard"),
    top_pct: float | None = None,
    bottom_pct: float | None = None,
    top_threshold: float | None = None,
    bottom_threshold: float | None = None,
    action: Literal["discard", "flip"] = "discard",
    target_dataset_size: int | None = 16384,
    random_seed: int = 42,
    extra_dataset_path: Path | None = None,
    extra_dataset_size: int | None = None,
) -> Path:
    """Create a filtered preference dataset from cached logprob-linear scores."""
    assert top_pct is None or top_threshold is None, "Only one of top_pct and top_threshold can be supplied"
    assert bottom_pct is None or bottom_threshold is None, "Only one of bottom_pct and bottom_threshold can be supplied"

    score_path = Path(score_path)
    dataset_path = Path(dataset_path)
    save_dir = Path(save_dir)
    extra_dataset_path = None if extra_dataset_path is None else Path(extra_dataset_path)

    source_rows = load_jsonl(dataset_path)
    score_rows = load_jsonl(score_path)
    validate_score_cache(score_rows, len(source_rows))

    valid_rows = valid_score_rows(score_rows)
    skipped_indices = {row["row_index"] for row in score_rows if row.get("skip", False)}
    scores = [row["score"] for row in valid_rows]
    selected_indices = select_indices_from_scores(
        valid_rows,
        top_pct,
        bottom_pct,
        top_threshold,
        bottom_threshold,
    )
    extra_rows = load_extra_rows(extra_dataset_path, extra_dataset_size, random_seed)
    output_rows = build_training_rows_from_selection(
        source_rows,
        selected_indices,
        action=action,
        target_dataset_size=target_dataset_size,
        random_seed=random_seed,
        extra_rows=extra_rows,
        always_discard_indices=skipped_indices,
    )

    save_dir.mkdir(parents=True, exist_ok=True)
    output_path = save_dir / "dataset.jsonl"
    metadata_path = save_dir / "metadata.json"
    save_jsonl(output_path, output_rows)
    save_distribution_figure(scores, save_dir / "score_distr.png")

    metadata = {
        "filter_config": {
            "score_path": str(score_path),
            "dataset_path": str(dataset_path),
            "top_pct": top_pct,
            "bottom_pct": bottom_pct,
            "top_threshold": top_threshold,
            "bottom_threshold": bottom_threshold,
            "action": action,
            "target_dataset_size": target_dataset_size,
            "random_seed": random_seed,
            "extra_dataset_path": None if extra_dataset_path is None else str(extra_dataset_path),
            "extra_dataset_size": extra_dataset_size,
        },
        "mean_score": float(np.mean(scores)),
        "original_dataset_size": len(source_rows),
        "original_num_skipped": len(skipped_indices),
        "original_num_selected": len(selected_indices),
        "final_dataset_size": len(output_rows),
        "final_num_new_rows": sum(row["original_index"] == -1 for row in output_rows),
        "num_extra_rows": len(extra_rows),
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Saved filtered dataset to {output_path}")
    logger.info(f"Saved metadata to {metadata_path}")
    return output_path


if __name__ == "__main__":
    cache_logprob_diffs(
        system_prompt = "Answer truthfully and with your honest assessment, not biased by what the user seems to want to hear.",
        save_dir=Path("cached_repr_33K_SFT/LLS/full"),
        pair_batch_size=4,
    )
