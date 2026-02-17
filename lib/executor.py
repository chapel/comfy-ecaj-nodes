"""Batched Pipeline Executor - compatibility facade.

This module re-exports all public symbols from the focused submodules
so that existing ``from lib.executor import ...`` statements continue
to work without modification.

Actual implementations live in:
- lib.batch_groups  — OpSignature, compile_batch_groups
- lib.gpu_ops       — DeltaSpec, compute_batch_size, chunked,
                      apply_lora_batch_gpu, chunked_evaluation
- lib.per_block     — _apply_per_block_lora_strength, _get_block_t_factors,
                      _apply_widen_filter_per_block, _apply_widen_merge_per_block
- lib.recipe_eval   — compile_plan, execute_plan, evaluate_recipe, EvalPlan
"""

from __future__ import annotations

# Re-export gc so tests that patch "lib.executor.gc.collect" still work.
import gc  # noqa: F401

from .batch_groups import OpSignature, compile_batch_groups
from .gpu_ops import (
    DeltaSpec,
    apply_lora_batch_gpu,
    check_ram_preflight,
    chunked,
    chunked_evaluation,
    compute_batch_size,
    estimate_peak_ram,
    get_available_ram_bytes,
)
from .per_block import (
    _apply_per_block_lora_strength,
    _apply_widen_filter_per_block,
    _apply_widen_merge_per_block,
    _get_block_t_factors,
)
from .recipe_eval import EvalPlan, compile_plan, evaluate_recipe, execute_plan

__all__ = [
    "OpSignature",
    "DeltaSpec",
    "EvalPlan",
    "check_ram_preflight",
    "compile_batch_groups",
    "compile_plan",
    "compute_batch_size",
    "chunked",
    "apply_lora_batch_gpu",
    "chunked_evaluation",
    "estimate_peak_ram",
    "evaluate_recipe",
    "execute_plan",
    "get_available_ram_bytes",
    "_apply_per_block_lora_strength",
    "_get_block_t_factors",
    "_apply_widen_filter_per_block",
    "_apply_widen_merge_per_block",
]
