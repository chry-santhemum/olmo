from dataclasses import dataclass
from contextlib import contextmanager
from typing import Literal, Optional, Union
from jaxtyping import Float, Int


import torch
from torch import nn, Tensor
from dpo_embedding_analysis import find_executable_batch_size

def get_module(model, module_name: str) -> nn.Module:
    module = model
    for part in module_name.split("."):
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


# --- Forward hook configuration ---

TokenSpec = Union[list[int], slice]

@dataclass(kw_only=True)
class FwdHook:
    module_name: str
    pos: Literal["input", "output"]
    op: Literal["record", "replace", "add", "proj_ablate"]
    tokens: TokenSpec

    tensor: Optional[Tensor] = None  # will be broadcast to x[tokens]
    grad: Optional[Tensor] = None  
    # populated when backward() is called in fwd_with_hooks(allow_grad=True).
    # this captures grads wrt the modified in/outputs.

    def __post_init__(self):
        if self.op != "record" and self.tensor is None:
            raise ValueError(f"tensor is required for '{self.op}' operation")


def _apply_hook_op(h: FwdHook, x: Tensor) -> Optional[Tensor]:
    """Apply hook operation to x.

    Assumes KV cache is used when seq_len=1.
    """
    idx = h.tokens

    # Handle KV cache decode step (seq_len=1)
    # Generated tokens should be steered if h.tokens is unbounded
    if x.shape[1] == 1:
        if isinstance(idx, slice) and idx.stop is None:
            idx = slice(None)  # steer the single generated token
        else:
            return None

    if h.op == "record":
        if h.tensor is None:
            h.tensor = x[:, idx, :].clone().detach()
        else:
            h.tensor.copy_(x[:, idx, :])
        return None

    new_tensor = x.clone()
    vec = h.tensor.to(device=x.device, dtype=x.dtype)

    if h.op == "add":
        new_tensor[:, idx, :] += vec
    elif h.op == "replace":
        new_tensor[:, idx, :] = vec
    elif h.op == "proj_ablate":
        v_hat = vec / vec.norm(dim=-1, keepdim=True)
        proj_coef = (new_tensor[:, idx, :] * v_hat).sum(dim=-1, keepdim=True)  # [batch, num_tokens, 1]
        new_tensor[:, idx, :] -= proj_coef * vec

    return new_tensor


def _get_tensor(x):
    return x[0] if isinstance(x, tuple) else x

def _set_tensor(original, new_tensor):
    if isinstance(original, tuple):
        return (new_tensor,) + original[1:]
    return new_tensor


def make_fwd_hook(h: FwdHook):
    if h.pos == "output":
        def output_hook(module, input, output):
            result = _apply_hook_op(h, _get_tensor(output))
            return _set_tensor(output, result) if result is not None else None
        return output_hook
    else:
        def input_hook(module, input):
            result = _apply_hook_op(h, _get_tensor(input))
            return _set_tensor(input, result) if result is not None else None
        return input_hook


def make_bwd_hook(h: FwdHook):
    """Capture gradients for a FwdHook."""
    def grad_hook(module, grad_input, grad_output):
        if h.pos == "output":
            grad_tensor = grad_output[0]
        else:
            grad_tensor = grad_input[0]
        
        if grad_tensor is not None:
            idx = h.tokens
            h.grad = grad_tensor[:, idx, :].clone().detach()

        return None
    return grad_hook


@contextmanager
def fwd_with_hooks(hooks: list[FwdHook], model, allow_grad: bool = False):
    """Context manager for running forward pass with hooks.
    
    allow_grad: If False (default), runs in inference_mode with no gradients.
                If True, gradients are enabled and backward hooks are registered.
    """
    handles = []

    # register hooks
    for hook in hooks:
        module = get_module(model, hook.module_name)
        if hook.pos == "output":
            handle = module.register_forward_hook(make_fwd_hook(hook))
        else:
            handle = module.register_forward_pre_hook(make_fwd_hook(hook))
        handles.append(handle)

        # Register backward hooks to capture gradients
        if allow_grad:
            bwd_handle = module.register_full_backward_hook(make_bwd_hook(hook))
            handles.append(bwd_handle)

    if allow_grad:
        with torch.enable_grad():
            yield
    else:
        with torch.inference_mode():
            yield

    for handle in handles:
        handle.remove()


def get_resid_block_name(model, layer: int) -> str:
    """Get the residual block module name for a given layer.

    Detects model architecture and returns appropriate path.
    """
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


def fwd_record_resid(
    model,
    inputs: dict[str, Tensor],
    layer: int,
    tokens: TokenSpec,
) -> Float[Tensor, "batch tok_captured hidden"]:
    """Record residual stream activations at a specific layer and position."""
    module_name = get_resid_block_name(model, layer)
    record_hook = FwdHook(
        module_name=module_name,
        pos="output",
        op="record",
        tokens=tokens,
    )

    with fwd_with_hooks([record_hook], model, allow_grad=False):
        model(**inputs)

    return record_hook.tensor







