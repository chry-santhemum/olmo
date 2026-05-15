"""
DPO - SFT response.
"""

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import torch
from loguru import logger
from openai import OpenAI
from tqdm import tqdm

from dpo_emb_cache import compute_embedding_diffs, load_model, load_tokenizer
from sycophancy_eval.run_evals import start_vllm_server, stop_vllm_server, wait_for_server


DEFAULT_DATASET_PATH = Path("sycophancy_eval/datasets/feedback_v2.jsonl")
DEFAULT_SAVE_ROOT = Path("vectors")
MODEL_ALIASES = {
    "instr-sft": "allenai/Olmo-3-7B-Instruct-SFT",
    "instr-dpo": "allenai/Olmo-3-7B-Instruct-DPO",
    "instr-final": "allenai/Olmo-3-7B-Instruct",
}


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def resolve_model(model: str) -> str:
    return MODEL_ALIASES.get(model, model)


def load_biased_prompts(
    dataset_path: Path,
    bias_types: list[str],
    max_per_bias: int | None,
) -> list[dict]:
    """Load current feedback_v2 biased prompts, ignoring baseline records."""
    prompts_by_bias = {bias_type: [] for bias_type in bias_types}

    with open(dataset_path) as f:
        for line in f:
            record = json.loads(line)
            metadata = record["metadata"]
            bias_type = metadata["bias_type"]
            if bias_type not in prompts_by_bias:
                continue

            prompts_by_bias[bias_type].append(
                {
                    "prompt": record["prompt"][0]["content"],
                    "bias_type": bias_type,
                    "dataset_type": metadata["dataset_type"],
                    "bias_phrase": metadata["bias_phrase"],
                    "tail_phrase": metadata["tail_phrase"],
                    "phrase_position": metadata["phrase_position"],
                }
            )

    for bias_type, prompts in prompts_by_bias.items():
        if not prompts:
            raise ValueError(f"No feedback_v2 prompts found for bias_type={bias_type!r}")
        if max_per_bias is not None:
            prompts_by_bias[bias_type] = prompts[:max_per_bias]

    prompt_records = [record for bias_type in bias_types for record in prompts_by_bias[bias_type]]
    logger.info(
        "Loaded "
        + ", ".join(f"{bias_type}={len(prompts_by_bias[bias_type])}" for bias_type in bias_types)
        + f" prompts from {dataset_path}"
    )
    return prompt_records


def sample_responses_from_server(
    prompts: list[str],
    port: int,
    served_name: str,
    max_tokens: int,
    temperature: float,
    num_workers: int,
) -> list[str]:
    """Sample one response per prompt from an OpenAI-compatible local server."""
    client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="unused")

    def sample_one(index_prompt: tuple[int, str]) -> tuple[int, str]:
        index, prompt = index_prompt
        response = client.chat.completions.create(
            model=served_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        content = response.choices[0].message.content
        if content is None:
            raise ValueError(f"Empty response for prompt index {index}")
        return index, content

    responses = [""] * len(prompts)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(sample_one, item) for item in enumerate(prompts)]
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Sampling {served_name}"):
            index, content = future.result()
            responses[index] = content
    return responses


def serve_and_sample(
    model_path: str,
    served_name: str,
    prompts: list[str],
    gpu: int,
    port: int,
    log_dir: Path,
    max_tokens: int,
    temperature: float,
    num_workers: int,
) -> list[str]:
    """Start vLLM for one model, sample all prompts, then stop the server."""
    previous_deep_gemm_warmup = os.environ.get("VLLM_DEEP_GEMM_WARMUP")
    os.environ["VLLM_DEEP_GEMM_WARMUP"] = "skip"
    process, log_handle, log_path = start_vllm_server(
        model_path=model_path,
        gpu=gpu,
        port=port,
        served_name=served_name,
        log_dir=log_dir,
    )
    if previous_deep_gemm_warmup is None:
        os.environ.pop("VLLM_DEEP_GEMM_WARMUP", None)
    else:
        os.environ["VLLM_DEEP_GEMM_WARMUP"] = previous_deep_gemm_warmup

    try:
        wait_for_server(process, port, log_path)
        return sample_responses_from_server(
            prompts,
            port=port,
            served_name=served_name,
            max_tokens=max_tokens,
            temperature=temperature,
            num_workers=num_workers,
        )
    finally:
        stop_vllm_server(process)
        log_handle.close()


def save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def sample_response_records(
    prompt_records: list[dict],
    sft_model: str,
    dpo_model: str,
    save_path: Path,
    gpu: int,
    sft_port: int,
    dpo_port: int,
    max_tokens: int,
    temperature: float,
    num_workers: int,
) -> list[dict]:
    """Sample SFT and DPO responses to the same feedback_v2 prompts."""
    prompts = [record["prompt"] for record in prompt_records]

    logger.info(f"Sampling {len(prompts)} SFT responses from {sft_model}")
    sft_responses = serve_and_sample(
        model_path=sft_model,
        served_name="sft-model",
        prompts=prompts,
        gpu=gpu,
        port=sft_port,
        log_dir=save_path.parent,
        max_tokens=max_tokens,
        temperature=temperature,
        num_workers=num_workers,
    )

    logger.info(f"Sampling {len(prompts)} DPO responses from {dpo_model}")
    dpo_responses = serve_and_sample(
        model_path=dpo_model,
        served_name="dpo-model",
        prompts=prompts,
        gpu=gpu,
        port=dpo_port,
        log_dir=save_path.parent,
        max_tokens=max_tokens,
        temperature=temperature,
        num_workers=num_workers,
    )

    response_records = []
    for prompt_record, sft_response, dpo_response in zip(prompt_records, sft_responses, dpo_responses):
        response_records.append(
            {
                **prompt_record,
                "sft_response": sft_response,
                "dpo_response": dpo_response,
            }
        )

    save_jsonl(save_path, response_records)
    logger.success(f"Saved {len(response_records)} sampled response records to {save_path}")
    return response_records


def build_contrast_chats(response_records: list[dict]) -> tuple[list[list[dict]], list[list[dict]]]:
    """Return DPO-response chats and SFT-response chats for activation diffs."""
    chosen_chats = []
    rejected_chats = []

    for record in response_records:
        prompt = record["prompt"]
        chosen_chats.append(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": record["dpo_response"]},
            ]
        )
        rejected_chats.append(
            [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": record["sft_response"]},
            ]
        )

    return chosen_chats, rejected_chats


def compute_feedback_v2_vector(
    sft_model: str,
    response_records: list[dict],
    layer: int,
    batch_size: int,
) -> torch.Tensor:
    """Compute the mean DPO-minus-SFT response activation difference."""
    chosen_chats, rejected_chats = build_contrast_chats(response_records)
    model = load_model(sft_model, torch_dtype=torch.bfloat16, device_map="auto")
    tokenizer = load_tokenizer(sft_model)

    activation_diffs = compute_embedding_diffs(
        model=model,
        tokenizer=tokenizer,
        chosen_chats=chosen_chats,
        rejected_chats=rejected_chats,
        layers=[layer],
        batch_size=batch_size,
    )

    vectors = [diff[layer] for diff in activation_diffs]
    vector = torch.stack(vectors).mean(dim=0)
    logger.info(f"Computed feedback_v2 contrast vector with shape {tuple(vector.shape)}")
    return vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute a feedback_v2 DPO-vs-SFT contrast vector.")
    parser.add_argument("--sft-model", default="instr-sft")
    parser.add_argument("--dpo-model", default="instr-dpo")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--save-dir", type=Path)
    parser.add_argument("--responses-path", type=Path)
    parser.add_argument("--sample-only", action="store_true")
    parser.add_argument("--bias-types", nargs="+", default=["positive", "negative"])
    parser.add_argument("--max-per-bias", type=int, default=512)
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--sft-port", type=int, default=8234)
    parser.add_argument("--dpo-port", type=int, default=8235)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-workers", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sft_model = resolve_model(args.sft_model)
    dpo_model = resolve_model(args.dpo_model)

    save_dir = args.save_dir or DEFAULT_SAVE_ROOT / f"feedback_v2_contrast_{timestamp()}"
    save_dir.mkdir(parents=True, exist_ok=True)
    responses_path = args.responses_path or save_dir / "sampled_responses.jsonl"

    if args.responses_path is None:
        prompt_records = load_biased_prompts(args.dataset, args.bias_types, args.max_per_bias)
        response_records = sample_response_records(
            prompt_records=prompt_records,
            sft_model=sft_model,
            dpo_model=dpo_model,
            save_path=responses_path,
            gpu=args.gpu,
            sft_port=args.sft_port,
            dpo_port=args.dpo_port,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            num_workers=args.num_workers,
        )
    else:
        response_records = load_jsonl(responses_path)
        logger.info(f"Loaded {len(response_records)} sampled response records from {responses_path}")

    if args.sample_only:
        return

    vector = compute_feedback_v2_vector(
        sft_model=sft_model,
        response_records=response_records,
        layer=args.layer,
        batch_size=args.batch_size,
    )

    output_path = save_dir / f"contrast_L{args.layer}.pt"
    torch.save(
        {
            "vector": vector,
            "layer": args.layer,
            "sft_model": sft_model,
            "dpo_model": dpo_model,
            "num_samples": len(response_records),
            "bias_types": args.bias_types,
            "dataset": str(args.dataset),
            "responses_path": str(responses_path),
        },
        output_path,
    )
    logger.success(f"Saved feedback_v2 contrast vector to {output_path}")


if __name__ == "__main__":
    main()
