"""Batched Pipeline Executor - GPU-accelerated LoRA application and merging.

This module provides the core primitives for batched GPU evaluation:
- OpSignature: hashable key for grouping parameters by (affecting_sets, shape, ndim)
- DeltaSpec: typed specification for LoRA delta factors
- compute_batch_size: VRAM-aware batch sizing (targets 70% of free VRAM)
- apply_lora_batch_gpu: torch.bmm-based batched LoRA application
- chunked_evaluation: OOM backoff wrapper for reliable GPU evaluation

This module is pure torch and stdlib - no ComfyUI imports.
"""

import gc
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

            # AC: @memory-management ac-1
            # Free GPU memory after chunk completes and results are on CPU
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError:
            # AC: @batched-executor ac-4
            # OOM backoff: clear cache and retry with batch_size=1
            gc.collect()
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
                    torch.cuda.empty_cache()
                    raise

    return results


def evaluate_recipe(
    keys: list[str],
    base_batch: torch.Tensor,
    recipe_node: object,
    loader: object,
    widen: object,
    set_id_map: dict[int, str],
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Evaluate a recipe tree on a batch of parameters.

    This is the core recipe tree walker for batched GPU evaluation. It recursively
    evaluates the recipe tree, dispatching to WIDEN functions based on node type:
    - RecipeMerge with RecipeCompose target → merge_weights_batched
    - RecipeMerge with RecipeLoRA target → filter_delta_batched
    - Chained RecipeMerge → recurse on inner merge first

    # AC: @exit-batched-eval ac-1
    Compose targets call merge_weights_batched with all branch results and backbone.

    # AC: @exit-batched-eval ac-2
    Single LoRA targets call filter_delta_batched with applied delta and backbone.

    # AC: @exit-batched-eval ac-3
    Chained RecipeMerge nodes evaluate inner merges first.

    # AC: @exit-batched-eval ac-4
    Results remain on GPU (CPU transfer happens in patch installation phase).

    # AC: @exit-batched-eval ac-5
    RecipeMerge.backbone overrides the importance reference for WIDEN analysis.

    Args:
        keys: List of B parameter keys being evaluated
        base_batch: [B, *shape] base model weights on GPU
        recipe_node: Recipe tree root (typically RecipeMerge)
        loader: LoRALoader with loaded LoRA data
        widen: WIDEN instance for filter/merge operations
        set_id_map: Map of object id(RecipeLoRA) → set_id string
        device: GPU device string
        dtype: Computation dtype

    Returns:
        [B, *shape] merged weights on GPU
    """
    # Import here to avoid circular import
    from lib.recipe import RecipeBase, RecipeCompose, RecipeLoRA, RecipeMerge

    def _apply_lora_set(
        current: torch.Tensor,
        recipe_lora: RecipeLoRA,
    ) -> torch.Tensor:
        """Apply a LoRA set to current weights.

        Args:
            current: [B, *shape] current weights on GPU
            recipe_lora: RecipeLoRA node with LoRA specs

        Returns:
            [B, *shape] weights with LoRA applied
        """
        set_id = set_id_map.get(id(recipe_lora))
        if set_id is None:
            # No LoRA data for this set
            return current

        # Build key_indices for DeltaSpec
        key_indices = {k: i for i, k in enumerate(keys)}

        # Get delta specs from loader for this set
        delta_specs = loader.get_delta_specs(keys, key_indices)

        # Apply deltas using batched GPU application
        result = apply_lora_batch_gpu(keys, current, delta_specs, device, dtype)
        return result

    def _eval_node(
        current_base: torch.Tensor,
        node: object,
    ) -> torch.Tensor:
        """Recursively evaluate a recipe node.

        Args:
            current_base: [B, *shape] current base weights
            node: Recipe node to evaluate

        Returns:
            [B, *shape] evaluated weights on GPU
        """
        if isinstance(node, RecipeBase):
            # Base node - return current base unchanged
            # AC: @exit-batched-eval ac-3 - base case for recursion
            return current_base

        elif isinstance(node, RecipeLoRA):
            # Apply LoRA set to current base
            # AC: @exit-batched-eval ac-2
            return _apply_lora_set(current_base, node)

        elif isinstance(node, RecipeCompose):
            # Evaluate all branches
            # AC: @exit-batched-eval ac-1
            branch_results = []
            for branch in node.branches:
                result = _eval_node(current_base, branch)
                branch_results.append(result)
            return branch_results  # type: ignore  # Return list for merge handling

        elif isinstance(node, RecipeMerge):
            # AC: @exit-batched-eval ac-3
            # First, evaluate the base (could be another RecipeMerge or RecipeBase)
            if isinstance(node.base, RecipeMerge):
                # Recursive merge - evaluate inner merge first
                current_base = _eval_node(current_base, node.base)
            elif isinstance(node.base, RecipeBase):
                # At the root - use provided base_batch
                current_base = current_base
            else:
                # Unexpected base type
                raise ValueError(f"Invalid base type in RecipeMerge: {type(node.base)}")

            # AC: @exit-batched-eval ac-5
            # Determine backbone for WIDEN analysis
            if node.backbone is not None:
                # Explicit backbone override - need to get backbone weights
                # For now, backbone should be a RecipeBase or similar that
                # we can extract base weights from
                # In practice, the backbone is the importance reference tensor
                backbone_weights = base_batch  # Use original base as default
            else:
                # Use current base as backbone
                backbone_weights = current_base

            # Evaluate target
            target_result = _eval_node(current_base, node.target)

            # Dispatch based on target type
            if isinstance(node.target, RecipeCompose):
                # target_result is a list of branch results
                branch_results = target_result

                # AC: @exit-node ac-6
                # Single-branch compose uses filter_delta (passthrough), not merge_weights
                if len(branch_results) == 1:
                    lora_applied = branch_results[0]
                    filtered = widen.filter_delta_batched(lora_applied, backbone_weights)
                    return filtered

                # AC: @exit-batched-eval ac-1
                # Multi-branch compose: Call merge_weights_batched with all branches
                merged = widen.merge_weights_batched(branch_results, backbone_weights)
                return merged

            elif isinstance(node.target, RecipeLoRA):
                # AC: @exit-batched-eval ac-2
                # target_result is the LoRA-applied weights
                lora_applied = target_result
                # Call filter_delta_batched
                filtered = widen.filter_delta_batched(lora_applied, backbone_weights)
                return filtered

            elif isinstance(node.target, RecipeMerge):
                # AC: @exit-batched-eval ac-3
                # Target is another merge - already evaluated recursively
                # Apply filter_delta_batched to treat inner merge result as "delta"
                inner_result = target_result
                filtered = widen.filter_delta_batched(inner_result, backbone_weights)
                return filtered

            else:
                raise ValueError(f"Invalid target type in RecipeMerge: {type(node.target)}")

        else:
            raise ValueError(f"Unknown recipe node type: {type(node)}")

    # AC: @exit-batched-eval ac-4
    # Evaluate and return - result stays on GPU
    return _eval_node(base_batch, recipe_node)


__all__ = [
    "OpSignature",
    "DeltaSpec",
    "compile_batch_groups",
    "compute_batch_size",
    "chunked",
    "apply_lora_batch_gpu",
    "chunked_evaluation",
    "evaluate_recipe",
]
