"""Batched Pipeline Executor - GPU-accelerated LoRA application and merging.

This module provides the core primitives for batched GPU evaluation:
- OpSignature: hashable key for grouping parameters by (affecting_sets, shape, ndim)
- DeltaSpec: typed specification for LoRA delta factors
- compute_batch_size: VRAM-aware batch sizing (targets 70% of free VRAM)
- apply_lora_batch_gpu: torch.bmm-based batched LoRA application
- chunked_evaluation: OOM backoff wrapper for reliable GPU evaluation

This module is pure torch and stdlib - no ComfyUI imports.
"""

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import torch

T = TypeVar("T")


@dataclass(frozen=True)
class OpSignature:
    """Hashable key for grouping parameters with identical operations.

    Parameters with the same OpSignature can be batched together because
    they have identical shape and are affected by the same LoRA sets.

    # AC: @batched-executor ac-1
    Keys with identical shape and affecting sets are in the same group.
    """

    affecting_sets: frozenset  # frozenset of set_id strings
    shape: tuple  # tensor shape (without batch dim)
    ndim: int  # len(shape) — determines WIDEN dispatch


@dataclass
class DeltaSpec:
    """Typed spec for a single LoRA delta factor, used by batched GPU apply.

    Loaders produce these from get_delta_specs(keys) so the engine can
    partition by (kind, rank) and use torch.bmm for same-rank groups.

    # AC: @batched-executor ac-2
    torch.bmm produces correct deltas matching per-key application.

    # AC: @batched-executor ac-7
    LoKr weights use per-key torch.kron on GPU instead of bmm.
    """

    kind: str  # 'standard' | 'lokr' | 'qkv_q' | 'qkv_k' | 'qkv_v'
    key_index: int  # position in the batch [0..B)
    up: torch.Tensor | None = None  # (out, rank) — None for lokr
    down: torch.Tensor | None = None  # (rank, in) — None for lokr
    scale: float = 1.0  # strength * alpha / rank
    # For lokr:
    w1: torch.Tensor | None = None
    w2: torch.Tensor | None = None
    # For conv2d reshape:
    target_shape: tuple | None = None  # original 4D shape if applicable


def compile_batch_groups(
    keys: list[str],
    base_state: dict[str, torch.Tensor],
    set_affected: dict[str, set[str]],
) -> dict[OpSignature, list[str]]:
    """Group parameter keys by OpSignature for batched evaluation.

    Keys in the same group have identical affecting_sets and shapes,
    so they can be stacked and processed with a single batched operation.

    # AC: @batched-executor ac-1
    Keys with identical shape and affecting sets are in the same group.

    Args:
        keys: Parameter keys to group
        base_state: Base model state dict (CPU)
        set_affected: set_id -> {affected param keys}

    Returns:
        Dict mapping OpSignature -> list of param keys
    """
    all_set_ids = list(set_affected.keys())

    groups: dict[OpSignature, list[str]] = {}
    for key in keys:
        if key not in base_state:
            continue
        affecting = frozenset(
            sid for sid in all_set_ids if key in set_affected.get(sid, set())
        )
        shape = tuple(base_state[key].shape)
        sig = OpSignature(affecting, shape, len(shape))
        groups.setdefault(sig, []).append(key)
    return groups


def compute_batch_size(
    shape: tuple,
    n_models: int,
    dtype: torch.dtype,
    vram_budget_gb: float = 0.0,
) -> int:
    """Estimate safe batch size for batched operations given shape and VRAM budget.

    Conservative: accounts for base + models + merge intermediates.
    When vram_budget_gb=0 (default), queries actual free VRAM via CUDA API.

    # AC: @batched-executor ac-3
    Returns a batch size targeting 70 percent of free VRAM.

    Args:
        shape: Parameter shape (without batch dim)
        n_models: Number of models being merged
        dtype: Data type
        vram_budget_gb: Override VRAM budget in GB (0 = auto-detect)

    Returns:
        Max safe batch size (>= 1)
    """
    if vram_budget_gb <= 0 and torch.cuda.is_available():
        free_bytes, _ = torch.cuda.mem_get_info()
        # Use 70% of actually-free VRAM to leave headroom for fragmentation
        vram_budget_bytes = int(free_bytes * 0.7)
    else:
        vram_budget_bytes = int(vram_budget_gb * (1024**3))

    # Handle case where no budget could be determined
    if vram_budget_bytes <= 0:
        return 1

    element_bytes = torch.finfo(dtype).bits // 8 if dtype.is_floating_point else 4
    tensor_bytes = element_bytes
    for d in shape:
        tensor_bytes *= d

    # Multiplier accounts for: base + per-set weight copies + merge intermediates.
    # Measured: actual peak is ~3 + 3*n_models for typical workloads.
    multiplier = 3 + 3 * n_models
    max_batch = int(vram_budget_bytes / (tensor_bytes * multiplier))
    return max(1, max_batch)


def chunked(lst: list[T], n: int):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def apply_lora_batch_gpu(
    keys: list[str],
    base_batch: torch.Tensor,
    delta_specs: list[DeltaSpec],
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Apply LoRA deltas to a batch via GPU bmm/kron.

    # AC: @batched-executor ac-2
    torch.bmm produces correct deltas matching per-key application.

    # AC: @batched-executor ac-7
    LoKr weights use per-key torch.kron on GPU instead of bmm.

    Args:
        keys: B parameter names
        base_batch: [B, *shape] base weights on GPU
        delta_specs: list of DeltaSpec for each key/lora pair
        device: GPU device string
        dtype: computation dtype

    Returns:
        [B, *shape] base + accumulated deltas on GPU
    """
    if not delta_specs:
        return base_batch

    result = base_batch.clone()

    # Partition specs by (kind, rank) for bmm compatibility
    partitions: dict[tuple[str, int], list[DeltaSpec]] = defaultdict(list)
    for spec in delta_specs:
        if spec.kind == "lokr":
            partitions[("lokr", 0)].append(spec)
        elif spec.kind in ("qkv_q", "qkv_k", "qkv_v"):
            rank = spec.down.shape[0] if spec.down is not None else 0
            partitions[(spec.kind, rank)].append(spec)
        else:
            rank = spec.down.shape[0] if spec.down is not None else 0
            partitions[("standard", rank)].append(spec)

    for (kind, rank), group in partitions.items():
        if kind == "lokr":
            # AC: @batched-executor ac-7
            # Per-key GPU kron (params are small, not worth batching)
            for spec in group:
                delta = spec.scale * torch.kron(
                    spec.w1.to(device, dtype=dtype),
                    spec.w2.to(device, dtype=dtype),
                )
                if spec.target_shape is not None:
                    delta = delta.view(spec.target_shape)
                result[spec.key_index] += delta
                del delta

        elif kind in ("qkv_q", "qkv_k", "qkv_v"):
            # QKV components: apply delta to the correct slice
            qkv_map = {"qkv_q": 0, "qkv_k": 1, "qkv_v": 2}
            section_idx = qkv_map[kind]

            if len(group) == 1:
                spec = group[0]
                up_gpu = spec.up.to(device, dtype=dtype)
                down_gpu = spec.down.to(device, dtype=dtype)
                delta = spec.scale * (up_gpu @ down_gpu)
                hidden = result.shape[1] // 3
                start = section_idx * hidden
                end = start + hidden
                result[spec.key_index, start:end] += delta
                del up_gpu, down_gpu, delta
            else:
                # Stack same-rank factors → single bmm
                ups = torch.stack([s.up for s in group]).to(device, dtype=dtype)
                downs = torch.stack([s.down for s in group]).to(device, dtype=dtype)
                scales = torch.tensor(
                    [s.scale for s in group], device=device, dtype=dtype
                )
                deltas = torch.bmm(ups, downs) * scales.view(-1, 1, 1)
                hidden = result.shape[1] // 3
                start = section_idx * hidden
                end = start + hidden
                for i, spec in enumerate(group):
                    result[spec.key_index, start:end] += deltas[i]
                del ups, downs, scales, deltas

        else:
            # AC: @batched-executor ac-2
            # Standard LoRA: stack same-rank factors → single bmm
            if len(group) == 1:
                spec = group[0]
                up_gpu = spec.up.to(device, dtype=dtype)
                down_gpu = spec.down.to(device, dtype=dtype)
                delta = spec.scale * (up_gpu @ down_gpu)
                if spec.target_shape is not None:
                    delta = delta.view(spec.target_shape)
                result[spec.key_index] += delta
                del up_gpu, down_gpu, delta
            else:
                ups = torch.stack([s.up for s in group]).to(device, dtype=dtype)
                downs = torch.stack([s.down for s in group]).to(device, dtype=dtype)
                scales = torch.tensor(
                    [s.scale for s in group], device=device, dtype=dtype
                )
                deltas = torch.bmm(ups, downs) * scales.view(-1, 1, 1)
                for i, spec in enumerate(group):
                    d = deltas[i]
                    if spec.target_shape is not None:
                        d = d.view(spec.target_shape)
                    result[spec.key_index] += d
                del ups, downs, scales, deltas

    return result


def chunked_evaluation(
    keys: list[str],
    base_tensors: dict[str, torch.Tensor],
    eval_fn: Callable[[list[str], torch.Tensor], torch.Tensor],
    batch_size: int,
    device: str,
    dtype: torch.dtype,
    storage_dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    """Evaluate keys in chunks with OOM backoff, returning CPU tensors.

    # AC: @batched-executor ac-4
    OOM backoff: failed chunk retries at batch size 1 while others continue normally.

    # AC: @batched-executor ac-5
    All result tensors are on CPU ready for set patch installation.

    # AC: @batched-executor ac-6
    Output tensors match the base model storage dtype.

    Args:
        keys: List of parameter keys to evaluate
        base_tensors: Dict of key -> CPU tensor for base weights
        eval_fn: Function (keys, base_batch_gpu) -> merged_batch_gpu
        batch_size: Initial batch size for chunking
        device: GPU device string
        dtype: Computation dtype (fp32 for numerical stability)
        storage_dtype: Output dtype (matches base model)

    Returns:
        Dict of key -> CPU tensor with merged weights
    """
    results: dict[str, torch.Tensor] = {}

    for chunk_keys in chunked(keys, batch_size):
        try:
            # Stack base tensors and move to GPU
            base_stack = torch.stack([base_tensors[k] for k in chunk_keys])
            base_gpu = base_stack.to(device, dtype=dtype)
            del base_stack

            # Evaluate the chunk
            merged_gpu = eval_fn(chunk_keys, base_gpu)
            del base_gpu

            # AC: @batched-executor ac-5, ac-6
            # Transfer to CPU with storage dtype
            merged_cpu = merged_gpu.to("cpu", dtype=storage_dtype)
            del merged_gpu

            # Unpack results
            for i, key in enumerate(chunk_keys):
                results[key] = merged_cpu[i]

        except torch.cuda.OutOfMemoryError:
            # AC: @batched-executor ac-4
            # OOM backoff: clear cache and retry with batch_size=1
            torch.cuda.empty_cache()

            for key in chunk_keys:
                try:
                    base_tensor = base_tensors[key].unsqueeze(0)
                    base_gpu = base_tensor.to(device, dtype=dtype)

                    merged_gpu = eval_fn([key], base_gpu)
                    del base_gpu

                    # AC: @batched-executor ac-5, ac-6
                    merged_cpu = merged_gpu.to("cpu", dtype=storage_dtype)
                    del merged_gpu

                    results[key] = merged_cpu[0]
                except torch.cuda.OutOfMemoryError:
                    # Even single-key evaluation failed; propagate
                    torch.cuda.empty_cache()
                    raise

    return results


__all__ = [
    "OpSignature",
    "DeltaSpec",
    "compile_batch_groups",
    "compute_batch_size",
    "chunked",
    "apply_lora_batch_gpu",
    "chunked_evaluation",
]
