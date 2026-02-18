"""GPU operations for batched LoRA application and chunked evaluation.

Provides:
- DeltaSpec: typed specification for LoRA delta factors
- compute_batch_size: VRAM-aware batch sizing (targets 70% of free VRAM)
- chunked: simple list chunking utility
- apply_lora_batch_gpu: torch.bmm-based batched LoRA application
- chunked_evaluation: OOM backoff wrapper for reliable GPU evaluation

This module is pure torch and stdlib - no ComfyUI imports.
"""

from __future__ import annotations

import gc
import logging
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import torch

logger = logging.getLogger("ecaj.gpu_ops")

# Conservative fallback when /proc/meminfo is unavailable (non-Linux).
_FALLBACK_RAM_BYTES = 4 * 1024**3  # 4 GB
_meminfo_warned = False

T = TypeVar("T")


def get_available_ram_bytes() -> int:
    """Return available system RAM in bytes.

    # AC: @memory-management ac-9
    # AC: @memory-management ac-10

    Parses /proc/meminfo for MemAvailable (Linux). On non-Linux or parse
    failure, returns a conservative fallback (4 GB) so protections stay
    active, and logs a warning on the first call.
    """
    global _meminfo_warned  # noqa: PLW0603
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024  # kB -> bytes
    except (OSError, ValueError):
        pass

    if not _meminfo_warned:
        logger.warning(
            "/proc/meminfo unavailable; using conservative %d MB RAM estimate. "
            "System RAM protections may be less accurate.",
            _FALLBACK_RAM_BYTES // (1024**2),
        )
        _meminfo_warned = True
    return _FALLBACK_RAM_BYTES


def estimate_peak_ram(
    merged_state_bytes: int,
    worst_chunk_bytes: int,
    save_model: bool,
    loader_bytes: int = 0,
) -> int:
    """Estimate NEW system RAM needed beyond what's already allocated.

    # AC: @memory-management ac-10
    # AC: @accurate-ram-preflight ac-1
    # AC: @accurate-ram-preflight ac-4
    # AC: @accurate-ram-preflight ac-5

    MemAvailable already reflects the base model, ComfyUI, loaders, and other
    loaded models. This function estimates only the additional RAM that
    execute() will allocate:

    - merged_state: new CPU tensors accumulating results (~merged_state_bytes)
    - pin_memory: temporary 2x cost for one chunk at a time

    Loader memory is NOT added to the estimate because loaders are populated
    before preflight runs, so their cost is already reflected in MemAvailable.

    Args:
        merged_state_bytes: Total bytes of processed keys (not the full base
            state dict — only keys actually being merged).
        worst_chunk_bytes: Largest single-chunk allocation in bytes
            (element_size * prod(shape) * batch_size for that group)
        save_model: Whether model will be saved (adds streaming overhead)
        loader_bytes: Measured bytes already consumed by loaders (logged for
            diagnostics only, not added to the estimate).

    Returns:
        Estimated additional RAM needed in bytes
    """
    # AC: @accurate-ram-preflight ac-3
    logger.debug("Preflight loader_bytes=%d (already in MemAvailable)", loader_bytes)

    # pin_memory allocates a copy: original stack + pinned = 2x
    chunk_pinned_cost = 2 * worst_chunk_bytes

    # merged_state accumulates new CPU tensors (up to ~merged_state_bytes)
    # + temporary pin_memory cost
    peak = merged_state_bytes + chunk_pinned_cost
    if save_model:
        # AC: @accurate-ram-preflight ac-4
        # Streaming writer holds one tensor at a time, but metadata + temp
        # file overhead adds ~5% of merged state
        peak += int(merged_state_bytes * 0.05)
    return peak


def check_ram_preflight(
    merged_state_bytes: int,
    worst_chunk_bytes: int,
    save_model: bool,
    loader_bytes: int = 0,
) -> None:
    """Raise RuntimeError if available RAM is insufficient for merge.

    # AC: @memory-management ac-10
    # AC: @exit-node ac-9
    # AC: @clip-exit-node ac-11

    Called by exit nodes before the GPU loop to fail early with a clear message.

    Args:
        merged_state_bytes: Total bytes of processed keys (only keys being merged)
        worst_chunk_bytes: Largest single-chunk allocation in bytes
        save_model: Whether model will be saved
        loader_bytes: Measured bytes already consumed by loaders (for logging)

    Raises:
        RuntimeError: With shortfall in MB if RAM is insufficient
    """
    avail = get_available_ram_bytes()
    peak = estimate_peak_ram(
        merged_state_bytes, worst_chunk_bytes, save_model, loader_bytes
    )
    if avail < peak:
        raise RuntimeError(
            f"Insufficient system RAM for merge: {avail // (1024**2)} MB available, "
            f"estimated {peak // (1024**2)} MB needed. Close other applications or "
            f"reduce model count."
        )


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
    # For QKV slicing (AC: @zimage-loader ac-3):
    # offset=(start, length) for the slice into fused weight
    # e.g. q=(0, 3840), k=(3840, 3840), v=(7680, 3840)
    offset: tuple[int, int] | None = None


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


def _compute_deltas(
    group: list[DeltaSpec],
    device: str,
    dtype: torch.dtype,
) -> list[tuple[DeltaSpec, torch.Tensor]]:
    """Compute LoRA deltas for a group of same-rank standard specs.

    Returns list of (spec, delta_tensor) pairs so callers can access
    spec.key_index, spec.offset, spec.target_shape without a linear search.

    Args:
        group: List of DeltaSpec with same rank
        device: GPU device string
        dtype: Computation dtype

    Returns:
        List of (DeltaSpec, delta) tuples
    """
    if len(group) == 1:
        spec = group[0]
        up_gpu = spec.up.to(device, dtype=dtype)
        down_gpu = spec.down.to(device, dtype=dtype)
        delta = spec.scale * (up_gpu @ down_gpu)
        del up_gpu, down_gpu
        return [(spec, delta)]
    else:
        ups = torch.stack([s.up for s in group]).to(device, dtype=dtype)
        downs = torch.stack([s.down for s in group]).to(device, dtype=dtype)
        scales = torch.tensor(
            [s.scale for s in group], device=device, dtype=dtype
        )
        deltas = torch.bmm(ups, downs) * scales.view(-1, 1, 1)
        del ups, downs, scales
        return [(spec, deltas[i]) for i, spec in enumerate(group)]


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

        elif kind in ("qkv_q", "qkv_k", "qkv_v", "offset_mlp"):
            # QKV components: apply delta to the correct slice
            # Use DeltaSpec.offset as source of truth for slice placement
            # when available; fall back to hardcoded hidden // 3 otherwise
            pairs = _compute_deltas(group, device, dtype)
            for spec, delta in pairs:
                key_index = spec.key_index
                if spec.offset is not None:
                    start, length = spec.offset
                    result[key_index, start : start + length] += delta
                else:
                    # Fallback: use qkv section index
                    qkv_map = {"qkv_q": 0, "qkv_k": 1, "qkv_v": 2}
                    section_idx = qkv_map[kind]
                    hidden = result.shape[1] // 3
                    start = section_idx * hidden
                    end = start + hidden
                    result[key_index, start:end] += delta
                del delta

        else:
            # AC: @batched-executor ac-2
            # Standard LoRA: deduplicated single/batched path
            pairs = _compute_deltas(group, device, dtype)
            for spec, delta in pairs:
                if spec.target_shape is not None:
                    delta = delta.view(spec.target_shape)
                result[spec.key_index] += delta
                del delta

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

    # AC: @memory-management ac-1
    After chunk completes and results transfer to CPU, GPU tensors are freed.

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

    # Cache available RAM once before the loop to avoid reading /proc/meminfo
    # per chunk. RAM doesn't change significantly within a single evaluation.
    avail_ram = get_available_ram_bytes() if device != "cpu" else 0

    for chunk_keys in chunked(keys, batch_size):
        try:
            # Stack base tensors and move to GPU
            # AC: @memory-management ac-9 — gate pin_memory on available RAM
            base_stack = torch.stack([base_tensors[k].cpu() for k in chunk_keys])
            tensor_bytes = base_stack.nelement() * base_stack.element_size()
            if device != "cpu" and base_stack.numel() > 65536:
                # 3x headroom: original stack + pinned copy + GC margin
                if avail_ram > tensor_bytes * 3:
                    base_gpu = base_stack.pin_memory().to(device, dtype=dtype)
                else:
                    base_gpu = base_stack.to(device, dtype=dtype)
            else:
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

            # AC: @memory-management ac-1
            # GPU tensors freed via del statements above after results
            # transfer to CPU. Periodic gc.collect() runs between
            # OpSignature groups (in exit.py), not per-chunk, to avoid
            # GPU sync overhead that blocks kernel queuing.

        except torch.cuda.OutOfMemoryError:
            # AC: @batched-executor ac-4
            # OOM backoff: clear cache and retry with batch_size=1
            gc.collect()
            if torch.cuda.is_available():
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

                    # AC: @memory-management ac-1
                    # Free GPU memory after each single-key evaluation in OOM path
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except torch.cuda.OutOfMemoryError:
                    # Even single-key evaluation failed; propagate
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise
                except (MemoryError, RuntimeError) as inner_e:
                    # AC: @memory-management ac-11
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    if isinstance(inner_e, MemoryError) or (
                        isinstance(inner_e, RuntimeError)
                        and ("not enough memory" in str(inner_e).lower()
                             or "out of memory" in str(inner_e).lower())
                    ):
                        raise RuntimeError(
                            f"System memory exhausted during single-key retry for '{key}'"
                        ) from inner_e
                    raise

        except MemoryError as e:
            # AC: @memory-management ac-11
            # System RAM exhaustion during chunk evaluation
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise RuntimeError(
                "System memory exhausted during chunked evaluation"
            ) from e

        except RuntimeError as e:
            # AC: @memory-management ac-11
            # PyTorch CPU allocator OOM or similar
            if "not enough memory" in str(e).lower() or "out of memory" in str(e).lower():
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise RuntimeError(
                    "System memory exhausted during chunked evaluation"
                ) from e
            raise

    return results
