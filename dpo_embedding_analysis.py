# %% Imports and Configuration
import contextlib
import gc
import inspect
import json
import multiprocessing as mp
from functools import partial
from pathlib import Path
from loguru import logger
from tqdm import tqdm
from datasets import load_dataset, Dataset

import numpy as np
import torch
from torch import Tensor, nn
from torch.profiler import profile, ProfilerActivity, schedule

from transformers import AutoModelForCausalLM, AutoTokenizer


def get_resid_block_name(model, layer: int) -> str:
    """Get the residual block module name for a given layer."""
    name = ""
    # print(model)
    if hasattr(model, "model"):
        name += "model."
        model = model.model
    if hasattr(model, "language_model"):
        name += "language_model."
        model = model.language_model
    if hasattr(model, "transformer"):
        name += "transformer."
        model = model.transformer
    if hasattr(model, "layers"):
        name += "layers."
        model = model.layers
    if hasattr(model, "h"):
        name += "h."
        model = model.h
    return name + f"{layer}"

def get_module(model, module_name: str) -> nn.Module:
    module = model
    for part in module_name.split("."):
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


class EarlyExitException(Exception):
    """Signal to stop forward pass after capturing target layer."""
    pass


def _make_record_mean_hook(layer_idx: int, slices: list[slice], captured: dict[int, list[Tensor]], early_exit: bool=True):
    """Record mean of hidden states between `starts` and `ends` for each sample."""
    def hook(module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        batch_size = hidden.shape[0]
        assert len(slices) == batch_size, f"Slices ({len(slices)}) and batch size ({batch_size}) must have the same length"

        means = [hidden[i, slices[i], :].mean(dim=0) for i in range(batch_size)]
        captured[layer_idx] = means

        if early_exit:
            raise EarlyExitException()
    return hook

def _is_oom_exception(e: Exception) -> bool:
    msg = str(e)
    return (
        isinstance(e, RuntimeError)
        and any(s in msg for s in (
            "CUDNN_STATUS_NOT_SUPPORTED",
            "can't allocate memory",
            "out of memory",
        ))
    )

def _oom_cleanup(device: torch.device|None = None):
    """If `device` is provided, print memory status for that device."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()  # Sometimes helps reclaim CUDA IPC allocations
        if device is not None:
            logger.info(f"[Device {device}] allocated memory: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
            logger.info(f"[Device {device}] reserved memory: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")

def find_executable_batch_size(starting_batch_size: int, function=None):
    """Decorator that auto-reduces batch_size on OOM.

    The decorated function must take batch_size as its first argument.
    """
    if function is None:
        return partial(find_executable_batch_size, starting_batch_size)

    def decorated_fn(*args, **kwargs):
        batch_size = starting_batch_size
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
                e.__traceback__ = None
                del e  # aggressively drop traceback references
                if oom:
                    _oom_cleanup()
                    batch_size //= 2
                    logger.warning(f"Decreasing batch size to {batch_size}.")
                else:
                    raise

    return decorated_fn



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


def compute_embedding_diffs(
    model,
    tokenizer,
    chosen_chats: list[list[dict]],
    rejected_chats: list[list[dict]],
    layers: list[int],
    batch_size: int,
) -> list[dict[int, Tensor]]:
    """Compute activation diffs (chosen - rejected) for each sample.
    
    Note: batch_size here refers to number of *pairs*, so the real batch size is twice that."""

    device = next(model.parameters()).device
    n = len(chosen_chats)
    assert n == len(rejected_chats), f"Chosen ({n}) and rejected ({len(rejected_chats)}) chats must have the same length"

    @find_executable_batch_size(starting_batch_size=batch_size)
    def _compute_diffs_inner(batch_size: int, start_pos: int) -> tuple[list[dict[int, Tensor]], int]:
        """Process a slice of samples, computing diffs.
        
        Returns (activation diffs, end pos)"""
        end_pos = min(start_pos + batch_size, n)
        actual_bs = end_pos - start_pos
        if actual_bs == 0:
            return [], end_pos

        batch_chosen_chats = chosen_chats[start_pos:end_pos]
        batch_rejected_chats = rejected_chats[start_pos:end_pos]
        all_chats = batch_chosen_chats + batch_rejected_chats
        all_chats_formatted = [
            tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
            for chat in all_chats
        ]
        inputs = tokenizer(
            all_chats_formatted,
            return_tensors="pt",
            padding=True,
            padding_side="right"
        )
        inputs = {k: v.to(device, non_blocking=True) for k, v in inputs.items()}

        response_starts = [
            tokenizer.apply_chat_template(
                chat[:-1],
                tokenize=True,
                return_tensors="pt",
                add_generation_prompt=True
            ).shape[-1] 
            for chat in all_chats
        ]
        response_lengths = [
            len(tokenizer(chat[-1]["content"], add_special_tokens=False)["input_ids"])
            for chat in all_chats
        ]

        for i in range(len(all_chats)):
            if response_lengths[i] <= 0:
                logger.error(f"Response length = {response_lengths[i]} <= 0.")
                logger.error(f"Chat messages: {all_chats[i]}")
                raise ValueError(f"Response length = {response_lengths[i]} <= 0.")
                
        # Register hooks to capture hidden states
        captured_hidden: dict[int, list[Tensor]] = {}
        handles = []
        for layer_idx in layers:
            module = get_module(model, get_resid_block_name(model, layer_idx))
            handle = module.register_forward_hook(_make_record_mean_hook(
                layer_idx, 
                [slice(response_starts[i], response_starts[i] + response_lengths[i]) for i in range(len(all_chats))], 
                captured_hidden, 
                early_exit=(layer_idx == max(layers)),
            ))
            handles.append(handle)

        # Forward pass
        try:
            with torch.no_grad():
                try:
                    model(**inputs)
                except EarlyExitException:
                    pass
        finally:
            for handle in handles:
                handle.remove()

        activation_diffs = []
        for i in range(actual_bs):
            diff = {}
            for layer_idx in layers:
                chosen_acts = captured_hidden[layer_idx][i]
                rejected_acts = captured_hidden[layer_idx][i + actual_bs]
                diff[layer_idx] = chosen_acts - rejected_acts
            activation_diffs.append(diff)

        return activation_diffs, end_pos

    all_diffs: list[dict[int, Tensor]] = []
    start_pos = 0
    pbar = tqdm(total=n, desc="Computing embedding diffs")

    while start_pos < n:
        diffs, end_pos = _compute_diffs_inner(start_pos)
        all_diffs.extend(diffs)
        pbar.update(end_pos - start_pos)
        start_pos = end_pos

    pbar.close()
    return all_diffs


# %% Workers

def load_manifest(path: Path) -> dict | None:
    """Load manifest file from `path` if it exists, otherwise None."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None

def save_manifest(path: Path, manifest: dict) -> None:
    """Atomic save."""
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w") as f:
        json.dump(manifest, f, indent=2)
    tmp_path.rename(path)


def _gpu_worker(
    device: torch.device,
    dataset: Dataset,
    model_name: str,
    layers: list[int],
    batch_size: int,
    chunk_size: int,
    cache_dir: Path,
    lock,
    enable_profiling: bool = False,
) -> None:
    """Worker function that processes chunks on a specific GPU."""
    num_samples = len(dataset)
    manifest = load_manifest(cache_dir / "manifest.json")
    assert manifest is not None, "Manifest not found"
    if manifest["num_samples"] != num_samples:
        raise ValueError(f"Dataset length {num_samples} does not match manifest num_samples {manifest['num_samples']}")

    torch.cuda.set_device(device)  # Set default device for this process
    logger.info(f"[Device {device}] Starting worker")
    model = load_model(model_name, torch_dtype=torch.bfloat16).to(device)
    tokenizer = load_tokenizer(model_name)

    def _claim_next_chunk(cache_dir: Path, lock) -> int | None:
        """Atomically claim the next uncompleted chunk.

        Returns chunk index if one is available, None if all chunks are done or in progress.
        """
        manifest_path = cache_dir / "manifest.json"
        with lock:
            manifest = load_manifest(manifest_path)
            if manifest is None:
                raise RuntimeError("Manifest not found")

            completed = set(manifest["completed_chunks"])
            in_progress = set(manifest.get("in_progress_chunks", []))

            for chunk_idx in range(manifest["total_chunks"]):
                if chunk_idx not in completed.union(in_progress):
                    manifest.setdefault("in_progress_chunks", []).append(chunk_idx)
                    save_manifest(manifest_path, manifest)
                    return chunk_idx
            return None

    def _mark_chunk_complete(cache_dir: Path, lock, chunk_idx: int) -> None:
        """Mark a chunk as completed and remove from in_progress."""
        manifest_path = cache_dir / "manifest.json"
        with lock:
            manifest = load_manifest(manifest_path)
            if manifest is None:
                raise RuntimeError("Manifest not found")

            in_progress = manifest.get("in_progress_chunks", [])
            if chunk_idx in in_progress:
                in_progress.remove(chunk_idx)
            else:
                logger.warning(f"Chunk {chunk_idx} was not in_progress")
            manifest["in_progress_chunks"] = in_progress

            if chunk_idx not in manifest["completed_chunks"]:
                manifest["completed_chunks"].append(chunk_idx)
            else:
                logger.warning(f"Chunk {chunk_idx} was already completed")

            save_manifest(manifest_path, manifest)

    if enable_profiling:
        profile_dir = cache_dir / "profiles"
        profile_dir.mkdir(exist_ok=True)
        profiler_ctx = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=0, warmup=1, active=2, repeat=1),  # Profile chunks 2-3
            on_trace_ready=lambda p: p.export_chrome_trace(
                str(profile_dir / f"trace_device_{device.index}.json")
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
    else:
        profiler_ctx = contextlib.nullcontext()

    with profiler_ctx as prof:
        chunks_processed = 0
        while True:
            chunk_idx = _claim_next_chunk(cache_dir, lock)
            if chunk_idx is None:
                logger.info(f"[Device {device}] No more chunks to process")
                break

            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, num_samples)
            logger.info(f"[Device {device}] Processing chunk {chunk_idx} ({start_idx} to {end_idx})")

            chosen_chats = dataset["chosen"][start_idx:end_idx]
            rejected_chats = dataset["rejected"][start_idx:end_idx]

            activation_diffs = compute_embedding_diffs(
                model,
                tokenizer,
                chosen_chats=chosen_chats,
                rejected_chats=rejected_chats,
                layers=layers,
                batch_size=batch_size,
            )

            # save chunk
            chunk_data = {
                "chunk_idx": chunk_idx,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "activation_diffs": activation_diffs,
                "chosen_chats": chosen_chats,
                "rejected_chats": rejected_chats,
            }
            chunk_path = cache_dir / f"chunk_{chunk_idx:0{manifest['num_digits']}d}.pt"
            tmp_path = chunk_path.with_suffix(".tmp")
            torch.save(chunk_data, tmp_path)
            tmp_path.rename(chunk_path)
            _mark_chunk_complete(cache_dir, lock, chunk_idx)
            chunks_processed += 1
            logger.info(f"[Device {device}] Completed chunk {chunk_idx}")
            if prof is not None:
                prof.step()

    logger.info(f"[Device {device}] Worker finished, processed {chunks_processed} chunks")


def cache_embedding_diffs_multi(
    dataset: Dataset,
    model_name: str,
    num_samples: int,
    layers: list[int],
    batch_size: int,
    chunk_size: int,
    devices: list[torch.device] | None = None,
    output_dir: Path = Path("dpo_embedding_analysis"),
    enable_profiling: bool = False,
) -> Path:
    """Cache DPO embedding diffs in chunks.

    Supports resuming from last completed chunk if interrupted.

    Args:
        dataset: Must have "chosen" and "rejected" columns with full conversations

    Returns:
        Path to cache directory containing chunks and manifest
    """
    if len(dataset) < num_samples:
        logger.warning(f"Dataset length {len(dataset)} is less than num_samples {num_samples}. Setting num_samples to {len(dataset)}.")
        num_samples = len(dataset)
    elif len(dataset) > num_samples:
        logger.warning(f"Dataset length {len(dataset)} is greater than num_samples {num_samples}. Only evaluating first {num_samples} samples.")
        dataset = dataset.select(range(num_samples))

    # Auto-detect GPUs if not specified
    if devices is None:
        devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
    logger.info(f"Using devices: {devices}")

    model_slug = model_name.split("/")[-1]
    cache_dir = output_dir / f"{model_slug}-{num_samples}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"

    # Add file logging to the cache directory
    log_file = cache_dir / "run.log"
    file_logger_id = logger.add(log_file, rotation="10 MB", retention="7 days")

    # Load or create manifest
    manifest = load_manifest(manifest_path)
    total_chunks = (num_samples + chunk_size - 1) // chunk_size
    num_digits = int(np.log10(total_chunks)) + 1

    if manifest is not None:
        # Verify configuration matches
        if set(manifest["layers"]) != set(layers):
            raise ValueError(f"layers mismatch: {manifest['layers']} vs {layers}")
        if manifest["chunk_size"] != chunk_size:
            raise ValueError(f"chunk_size mismatch: {manifest['chunk_size']} vs {chunk_size}")

        completed_chunks = set(manifest["completed_chunks"])
        logger.info(f"Resuming from manifest: {len(completed_chunks)}/{total_chunks} chunks completed")

        # Reset any in_progress chunks from previous run
        if manifest.get("in_progress_chunks"):
            logger.warning(f"Resetting {len(manifest['in_progress_chunks'])} in_progress chunks")
            manifest["in_progress_chunks"] = []
            save_manifest(manifest_path, manifest)
        
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
            "num_digits": num_digits,
        }
        save_manifest(manifest_path, manifest)
        logger.info(f"Created new manifest for {total_chunks} chunks")

    # Check if all chunks are complete
    if len(completed_chunks) == total_chunks:
        logger.info("All chunks already computed")
        logger.remove(file_logger_id)
        return cache_dir

    # Spawn workers
    logger.info(f"Starting {len(devices)} GPU workers...")
    ctx = mp.get_context("spawn")
    manager = ctx.Manager()
    lock = manager.Lock()

    processes = []
    for device in devices:
        p = ctx.Process(
            target=_gpu_worker,
            args=(
                device,
                dataset,
                model_name,
                layers,
                batch_size,
                chunk_size,
                cache_dir,
                lock,
                enable_profiling,
            ),
        )
        p.start()
        processes.append(p)

    # Wait for all workers to complete
    for p in processes:
        p.join()

    failed_devices = []
    for p, device in zip(processes, devices):
        if p.exitcode != 0:
            failed_devices.append((device, p.exitcode))

    if failed_devices:
        logger.remove(file_logger_id)
        error_msg = "; ".join(f"device {d} exited with code {c}" for d, c in failed_devices)
        raise RuntimeError(f"Worker(s) failed: {error_msg}")

    logger.success(f"All {total_chunks} chunks saved to {cache_dir}")
    logger.remove(file_logger_id)
    return cache_dir


if __name__ == "__main__":

    dataset = load_dataset("allenai/Dolci-Instruct-DPO", split="train").filter(
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
        num_proc=16,
    )
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(1024))


    cache_embedding_diffs_multi(
        dataset=dataset,
        model_name="allenai/Olmo-3-7B-Instruct-SFT",
        num_samples=64,
        layers=[23],
        batch_size=8,
        chunk_size=32,
        output_dir=Path("dpo_embedding_analysis"),
        enable_profiling=True,
    )

