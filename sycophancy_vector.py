"""Compute a sycophancy contrast vector: DPO model responses vs SFT model responses.

The vector captures the direction in activation space corresponding to
"how a DPO model responds to biased prompts" vs "how an SFT model responds".

Two-phase workflow:
  Phase 1 (--sample-only or default): Serve SFT and DPO models via vLLM,
    sample responses to biased prompts from feedback_v2.jsonl.
  Phase 2 (--responses-path or default): Load SFT model weights, compute
    activation diffs between DPO and SFT response chats, average to get vector.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch
from loguru import logger
from openai import OpenAI
from tqdm import tqdm

from dpo_emb_cache import (
    compute_embedding_diffs,
    load_model,
    load_tokenizer,
)
from sycophancy_eval.run_evals import (
    _cleanup_vllm,
    start_vllm_server,
    wait_for_server,
)

VLLM_PORT = 8234


def load_biased_prompts(
    dataset_path: Path,
    max_per_bias: int | None = None,
) -> list[dict]:
    """Load biased prompts from feedback_v2.jsonl.

    Returns list of dicts with keys: prompt (str), bias_type, dataset_type.
    """
    prompts = {"positive_bias": [], "negative_bias": []}

    with open(dataset_path) as f:
        for line in f:
            record = json.loads(line)
            bias_type = record["metadata"]["bias_type"]
            if bias_type not in prompts:
                continue
            # Convert {"type": "human", "content": ...} -> plain string
            content = record["prompt"][0]["content"]
            prompts[bias_type].append({
                "prompt": content,
                "bias_type": bias_type,
                "dataset_type": record["metadata"]["dataset_type"],
            })

    if max_per_bias is not None:
        for k in prompts:
            prompts[k] = prompts[k][:max_per_bias]

    result = prompts["positive_bias"] + prompts["negative_bias"]
    logger.info(
        f"Loaded {len(result)} biased prompts "
        f"(positive={len(prompts['positive_bias'])}, negative={len(prompts['negative_bias'])})"
    )
    return result


def sample_responses_from_served_model(
    prompts: list[str],
    port: int,
    model_name: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    num_workers: int = 32,
) -> list[str]:
    """Sample chat completions from a vLLM-served model.

    Args:
        prompts: List of user message strings.
        port: vLLM server port.
        model_name: Served model name for the OpenAI client.
        max_tokens: Max tokens per response.
        temperature: Sampling temperature.
        num_workers: ThreadPoolExecutor concurrency.

    Returns:
        List of response strings, same order as prompts.
    """
    client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="unused")

    def _get_response(idx_prompt):
        idx, prompt = idx_prompt
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return idx, resp.choices[0].message.content

    responses = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = [pool.submit(_get_response, (i, p)) for i, p in enumerate(prompts)]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Sampling ({model_name})"):
            idx, text = future.result()
            responses[idx] = text

    return responses


def serve_and_sample(
    model_path: str,
    served_name: str,
    prompts: list[str],
    port: int = VLLM_PORT,
    gpu_id: int = 0,
) -> list[str]:
    """Start vLLM server, sample responses, then shut down."""
    start_vllm_server(model_path, port=port, gpu_id=gpu_id, served_name=served_name)
    if not wait_for_server(port):
        raise RuntimeError(f"vLLM server for {model_path} failed to start")

    try:
        responses = sample_responses_from_served_model(prompts, port, served_name)
    finally:
        _cleanup_vllm()

    return responses


def sample_responses(
    sft_model: str,
    dpo_model: str,
    prompt_records: list[dict],
    save_path: Path,
) -> list[dict]:
    """Phase 1: Sample responses from both SFT and DPO models.

    Saves and returns list of dicts with keys:
        prompt, bias_type, dataset_type, sft_response, dpo_response
    """
    prompts = [r["prompt"] for r in prompt_records]

    logger.info(f"Sampling {len(prompts)} responses from SFT model: {sft_model}")
    sft_responses = serve_and_sample(sft_model, "sft-model", prompts, port=VLLM_PORT)

    logger.info(f"Sampling {len(prompts)} responses from DPO model: {dpo_model}")
    dpo_responses = serve_and_sample(dpo_model, "dpo-model", prompts, port=VLLM_PORT + 1)

    records = []
    for rec, sft_resp, dpo_resp in zip(prompt_records, sft_responses, dpo_responses):
        records.append({
            **rec,
            "sft_response": sft_resp,
            "dpo_response": dpo_resp,
        })

    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    logger.success(f"Saved {len(records)} sampled responses to {save_path}")

    return records


def compute_sycophancy_vector(
    model,
    tokenizer,
    chosen_chats: list[list[dict]],
    rejected_chats: list[list[dict]],
    layer: int,
    batch_size: int = 8,
) -> torch.Tensor:
    """Compute sycophancy vector by averaging activation diffs."""
    activation_diffs = compute_embedding_diffs(
        model=model,
        tokenizer=tokenizer,
        chosen_chats=chosen_chats,
        rejected_chats=rejected_chats,
        layers=[layer],
        batch_size=batch_size,
    )

    logger.info(f"Computed {len(activation_diffs)} activation diffs")

    # Average across all samples
    vectors = [diff[layer] for diff in activation_diffs]
    sycophancy_vector = torch.stack(vectors).mean(dim=0)

    logger.info(f"Sycophancy vector shape: {sycophancy_vector.shape}")
    return sycophancy_vector


def main():
    p = argparse.ArgumentParser(description="Compute sycophancy contrast vector (DPO vs SFT)")
    p.add_argument("--sft-model", default="allenai/Olmo-3-7B-Instruct-SFT")
    p.add_argument("--dpo-model", default="allenai/Olmo-3-7B-Instruct")
    p.add_argument("--dataset", type=Path, default=Path("sycophancy_eval/datasets/feedback_v2.jsonl"))
    p.add_argument("--save-dir", type=Path, default=None)
    p.add_argument("--layer", type=int, default=23)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-samples", type=int, default=256)
    p.add_argument("--responses-path", type=Path, default=None)
    p.add_argument("--sample-only", action="store_true")
    args = p.parse_args()

    # Default save dir based on model names
    if args.save_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_dir = Path(f"sycophancy_eval/vectors/{timestamp}")
    args.save_dir.mkdir(parents=True, exist_ok=True)

    responses_path = args.responses_path or args.save_dir / "sampled_responses.jsonl"

    # Phase 1: Sample responses (unless pre-existing responses provided)
    if args.responses_path is None:
        prompt_records = load_biased_prompts(args.dataset, max_per_bias=args.max_samples)
        response_records = sample_responses(
            sft_model=args.sft_model,
            dpo_model=args.dpo_model,
            prompt_records=prompt_records,
            save_path=responses_path,
        )
    else:
        with open(responses_path) as f:
            response_records = [json.loads(line) for line in f]
        if args.max_samples is not None:
            response_records = response_records[:args.max_samples]
        logger.info(f"Loaded {len(response_records)} pre-existing responses from {responses_path}")

    if args.sample_only:
        logger.info("--sample-only: skipping vector computation")
        return

    # Phase 2: Compute contrast vector
    # chosen = (biased_prompt, DPO response), rejected = (biased_prompt, SFT response)
    chosen_chats = []
    rejected_chats = []
    for rec in response_records:
        chosen_chats.append([
            {"role": "user", "content": rec["prompt"]},
            {"role": "assistant", "content": rec["dpo_response"]},
        ])
        rejected_chats.append([
            {"role": "user", "content": rec["prompt"]},
            {"role": "assistant", "content": rec["sft_response"]},
        ])

    # Extract hidden states from SFT model
    model = load_model(args.sft_model, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = load_tokenizer(args.sft_model)

    sycophancy_vector = compute_sycophancy_vector(
        model=model,
        tokenizer=tokenizer,
        chosen_chats=chosen_chats,
        rejected_chats=rejected_chats,
        layer=args.layer,
        batch_size=args.batch_size,
    )

    # Save with metadata
    output_path = args.save_dir / f"contrast_L{args.layer}.pt"
    output_data = {
        "vector": sycophancy_vector,
        "layer": args.layer,
        "sft_model": args.sft_model,
        "dpo_model": args.dpo_model,
        "num_samples": len(chosen_chats),
        "bias_types": ["positive_bias", "negative_bias"],
        "dataset": str(args.dataset),
        "responses_path": str(responses_path),
    }
    torch.save(output_data, output_path)
    logger.success(f"Saved sycophancy contrast vector to {output_path}")


if __name__ == "__main__":
    main()
