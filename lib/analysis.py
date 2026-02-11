"""Exit Recipe Analysis — tree walk, set ID assignment, and LoRA loading.

This module provides the recipe analysis phase that runs at the start of
Exit node execution. It handles:
1. Walking the recipe tree to find RecipeBase (root)
2. Assigning synthetic set IDs to each unique RecipeLoRA
3. Loading LoRA files with architecture-appropriate loaders
4. Building the affected-key map for batched evaluation

AC: @exit-recipe-analysis ac-1 through ac-6
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

try:
    from .lora import LoRALoader, get_loader
    from .recipe import RecipeBase, RecipeCompose, RecipeLoRA, RecipeMerge, RecipeNode
except ImportError:
    from lib.lora import LoRALoader, get_loader
    from lib.recipe import RecipeBase, RecipeCompose, RecipeLoRA, RecipeMerge, RecipeNode

if TYPE_CHECKING:
    pass

__all__ = [
    "AnalysisResult",
    "analyze_recipe",
]


@dataclass
class AnalysisResult:
    """Result of recipe tree analysis.

    Contains everything needed to execute the recipe:
    - model_patcher: The base model from RecipeBase
    - arch: Architecture tag for LoRA loading
    - set_affected: Map of set_id -> set of base model keys affected
    - loader: Loaded LoRALoader instance (caller must cleanup)
    - affected_keys: Union of all keys affected by any LoRA set
    """

    model_patcher: object
    arch: str
    set_affected: dict[str, set[str]]
    loader: LoRALoader
    affected_keys: set[str]


def _walk_to_base(node: RecipeNode) -> RecipeBase:
    """Walk the recipe tree to find the RecipeBase root.

    AC: @exit-recipe-analysis ac-1
    Given a recipe tree, walk to the root and find RecipeBase.

    Args:
        node: Any recipe node (typically RecipeMerge root)

    Returns:
        The RecipeBase at the root of the tree

    Raises:
        ValueError: If tree structure is invalid (no RecipeBase found)
    """
    if isinstance(node, RecipeBase):
        return node
    elif isinstance(node, RecipeMerge):
        # Recurse through base link until we hit RecipeBase
        return _walk_to_base(node.base)
    elif isinstance(node, RecipeLoRA):
        raise ValueError(
            "RecipeLoRA cannot be the root of a recipe tree. "
            "Use Entry node to create RecipeBase first."
        )
    elif isinstance(node, RecipeCompose):
        raise ValueError(
            "RecipeCompose cannot be the root of a recipe tree. "
            "Use Merge node to connect Compose output to a base."
        )
    else:
        raise ValueError(f"Unknown recipe node type: {type(node)}")


def _collect_lora_sets(node: RecipeNode) -> dict[int, RecipeLoRA]:
    """Collect all unique RecipeLoRA nodes with synthetic set IDs.

    AC: @exit-recipe-analysis ac-2
    Each unique RecipeLoRA gets a distinct set ID. Two LoRAs chained via
    prev (accumulated into the same RecipeLoRA tuple) share the same set ID.

    Uses object identity (id()) for set assignment because frozen dataclasses
    with the same content are still distinct objects in the recipe tree.

    Args:
        node: Root recipe node to walk

    Returns:
        Dict mapping set_id (int) -> RecipeLoRA for each unique node
    """
    lora_sets: dict[int, RecipeLoRA] = {}

    def _walk(n: RecipeNode) -> None:
        if isinstance(n, RecipeBase):
            # Base has no LoRAs
            pass
        elif isinstance(n, RecipeLoRA):
            # Use object id as set ID - each RecipeLoRA instance is a set
            # Chained LoRAs (via prev) are accumulated in the same RecipeLoRA
            set_id = id(n)
            if set_id not in lora_sets:
                lora_sets[set_id] = n
        elif isinstance(n, RecipeCompose):
            # Walk all branches
            for branch in n.branches:
                _walk(branch)
        elif isinstance(n, RecipeMerge):
            # Walk base, target, and backbone
            _walk(n.base)
            _walk(n.target)
            if n.backbone is not None:
                _walk(n.backbone)
        else:
            raise ValueError(f"Unknown recipe node type: {type(n).__name__}")

    _walk(node)
    return lora_sets


def _resolve_lora_path(lora_name: str, lora_base_path: str | None = None) -> str:
    """Resolve a LoRA name to its full path.

    Args:
        lora_name: LoRA filename (from RecipeLoRA)
        lora_base_path: Base directory for LoRA files (for tests, or runtime)

    Returns:
        Full path to LoRA file

    Note:
        In production, the caller (Exit node) should pass a lora_base_path
        obtained from folder_paths.get_folder_paths("loras")[0]. This keeps
        the lib module pure (no ComfyUI imports).
    """
    if lora_base_path:
        full_path = os.path.join(lora_base_path, lora_name)
    else:
        # No base path - assume lora_name is already a full path
        full_path = lora_name

    return full_path


def analyze_recipe(
    node: RecipeNode,
    lora_base_path: str | None = None,
) -> AnalysisResult:
    """Analyze a recipe tree and load all LoRA files.

    This is the main entry point for recipe analysis. It:
    1. Walks to the root to find RecipeBase (AC-1)
    2. Collects all RecipeLoRA nodes with set IDs (AC-2)
    3. Loads LoRA files with architecture loader (AC-3)
    4. Builds the affected-key map (AC-4)

    AC: @exit-recipe-analysis ac-1 through ac-4

    Args:
        node: Root recipe node (typically RecipeMerge)
        lora_base_path: Base path for LoRA files (for testing)

    Returns:
        AnalysisResult with all analysis data

    Raises:
        FileNotFoundError: If any LoRA file does not exist (AC-6)
        ValueError: If recipe structure is invalid
    """
    # AC-1: Walk to base and extract model_patcher and arch
    base = _walk_to_base(node)
    model_patcher = base.model_patcher
    arch = base.arch

    # AC-2: Collect LoRA sets with IDs
    lora_sets = _collect_lora_sets(node)

    # AC-3: Get architecture-appropriate loader
    loader = get_loader(arch)

    # Load each LoRA set and track affected keys per set
    set_affected: dict[str, set[str]] = {}

    for set_id, recipe_lora in lora_sets.items():
        set_key = str(set_id)  # Convert int id to string key

        # Load all LoRAs in this set, tagged with set_key
        for lora_spec in recipe_lora.loras:
            lora_name = lora_spec["path"]
            strength = lora_spec["strength"]

            # Resolve path (AC-6: raises FileNotFoundError if missing)
            full_path = _resolve_lora_path(lora_name, lora_base_path)
            if not os.path.exists(full_path):
                raise FileNotFoundError(
                    f"LoRA file not found: {lora_name} "
                    f"(referenced by LoRA node with strength {strength})"
                )

            # Load the LoRA file into the specific set
            loader.load(full_path, strength, set_id=set_key)

        # AC-4: Keys added by this set (queried from the set-scoped API)
        set_affected[set_key] = loader.affected_keys_for_set(set_key)

    # All affected keys across all sets
    affected_keys = set(loader.affected_keys)

    return AnalysisResult(
        model_patcher=model_patcher,
        arch=arch,
        set_affected=set_affected,
        loader=loader,
        affected_keys=affected_keys,
    )


def get_keys_to_process(
    all_keys: set[str],
    affected_keys: set[str],
) -> set[str]:
    """Filter keys to only those affected by at least one LoRA set.

    AC: @exit-recipe-analysis ac-5
    Keys not affected by any LoRA set are skipped entirely.

    Args:
        all_keys: All parameter keys in the base model
        affected_keys: Keys affected by at least one LoRA

    Returns:
        Set of keys that need processing
    """
    return all_keys & affected_keys
