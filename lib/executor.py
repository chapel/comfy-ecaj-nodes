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
- lib.recipe_eval   — evaluate_recipe (+ EvalContext, helpers)
"""

from __future__ import annotations

# Re-export gc so tests that patch "lib.executor.gc.collect" still work.
import gc  # noqa: F401

from .batch_groups import OpSignature, compile_batch_groups
from .gpu_ops import (
    DeltaSpec,
    apply_lora_batch_gpu,
    chunked,
    chunked_evaluation,
    compute_batch_size,
)
from .per_block import (
    _apply_per_block_lora_strength,
    _apply_widen_filter_per_block,
    _apply_widen_merge_per_block,
    _get_block_t_factors,
)
from .recipe_eval import evaluate_recipe

__all__ = [
    "OpSignature",
    "DeltaSpec",
    "compile_batch_groups",
    "compute_batch_size",
    "chunked",
    "apply_lora_batch_gpu",
    "chunked_evaluation",
    "evaluate_recipe",
    "_apply_per_block_lora_strength",
    "_get_block_t_factors",
    "_apply_widen_filter_per_block",
    "_apply_widen_merge_per_block",
]
