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
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .lora import LoRALoader, get_loader
from .model_loader import ModelLoader
from .recipe import (
    RecipeBase,
    RecipeCompose,
    RecipeLoRA,
    RecipeMerge,
    RecipeModel,
    RecipeNode,
)

if TYPE_CHECKING:
    pass

__all__ = [
    "AnalysisResult",
    "ModelAnalysisResult",
    "analyze_recipe",
    "analyze_recipe_models",
    "walk_to_base",
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


@dataclass
class ModelAnalysisResult:
    """Result of recipe model analysis.

    Contains model loaders and affected keys for full checkpoint merging:
    - model_loaders: Map of model_id -> ModelLoader (streaming access)
    - model_affected: Map of model_id -> set of keys affected by that model
    - all_model_keys: Union of all keys affected by any model

    AC: @full-model-execution ac-1
    """

    model_loaders: dict[str, ModelLoader]
    model_affected: dict[str, frozenset[str]]
    all_model_keys: frozenset[str]


def walk_to_base(node: RecipeNode) -> RecipeBase:
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
        return walk_to_base(node.base)
    elif isinstance(node, RecipeLoRA):
        raise ValueError(
            "RecipeLoRA cannot be the root of a recipe tree. "
            "Use Entry node to create RecipeBase first."
        )
    elif isinstance(node, RecipeModel):
        raise ValueError(
            "RecipeModel cannot be the root of a recipe tree. "
            "Use Entry node to create RecipeBase first, then Merge with the model."
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
        elif isinstance(n, RecipeModel):
            # RecipeModel has no LoRAs - skip
            pass
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


def _resolve_lora_path(
    lora_name: str,
    lora_path_resolver: Callable[[str], str | None] | None = None,
) -> str:
    """Resolve a LoRA name to its full path.

    Args:
        lora_name: LoRA filename (from RecipeLoRA), may include subdirectories
            (e.g. "z-image/Mystic.safetensors")
        lora_path_resolver: Callable that takes a LoRA name and returns the
            full path, or None if not found. In production, this wraps
            folder_paths.get_full_path("loras", name), which searches all
            registered LoRA directories. This keeps the lib module pure
            (no ComfyUI imports).

    Returns:
        Full path to LoRA file
    """
    if lora_path_resolver is not None:
        resolved = lora_path_resolver(lora_name)
        if resolved is not None:
            return resolved
        # Resolver was provided but couldn't find the file — fail immediately
        # rather than falling back to the raw name (which could accidentally
        # match a file in CWD)
        raise FileNotFoundError(
            f"LoRA file not found: {lora_name} "
            f"(resolver could not locate file in any registered directory)"
        )

    # No resolver — assume lora_name is already a full path
    return lora_name


def analyze_recipe(
    node: RecipeNode,
    lora_path_resolver: Callable[[str], str | None] | None = None,
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
        lora_path_resolver: Callable that resolves a LoRA name to its full
            filesystem path, or None if not found. In production, wraps
            folder_paths.get_full_path("loras", name). For testing, use
            lambda name: os.path.join(test_dir, name).

    Returns:
        AnalysisResult with all analysis data

    Raises:
        FileNotFoundError: If any LoRA file does not exist (AC-6)
        ValueError: If recipe structure is invalid
    """
    # AC-1: Walk to base and extract model_patcher and arch
    base = walk_to_base(node)
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
            full_path = _resolve_lora_path(lora_name, lora_path_resolver)
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


def _collect_model_refs(node: RecipeNode) -> dict[int, RecipeModel]:
    """Collect all unique RecipeModel nodes with synthetic model IDs.

    AC: @full-model-execution ac-1
    Each unique RecipeModel gets a distinct ID for loader management.

    Args:
        node: Root recipe node to walk

    Returns:
        Dict mapping model_id (int) -> RecipeModel for each unique node
    """
    model_refs: dict[int, RecipeModel] = {}

    def _walk(n: RecipeNode) -> None:
        if isinstance(n, RecipeBase):
            pass
        elif isinstance(n, RecipeLoRA):
            pass
        elif isinstance(n, RecipeModel):
            model_id = id(n)
            if model_id not in model_refs:
                model_refs[model_id] = n
        elif isinstance(n, RecipeCompose):
            for branch in n.branches:
                _walk(branch)
        elif isinstance(n, RecipeMerge):
            _walk(n.base)
            _walk(n.target)
            if n.backbone is not None:
                _walk(n.backbone)
        else:
            raise ValueError(f"Unknown recipe node type: {type(n).__name__}")

    _walk(node)
    return model_refs


def analyze_recipe_models(
    node: RecipeNode,
    base_arch: str,
    model_path_resolver: Callable[[str, str], str | None] | None = None,
) -> ModelAnalysisResult:
    """Analyze a recipe tree for full model checkpoints.

    AC: @full-model-execution ac-1, ac-6, ac-10, ac-12

    Opens ModelLoader instances for each unique RecipeModel path,
    validates architecture consistency, and builds affected-key maps.

    Args:
        node: Root recipe node (typically RecipeMerge)
        base_arch: Architecture of the base model (for validation)
        model_path_resolver: Callable that resolves (model_name, source_dir) to
            full filesystem path. In production, wraps folder_paths.get_full_path.

    Returns:
        ModelAnalysisResult with loaders and affected key sets

    Raises:
        FileNotFoundError: If any checkpoint file doesn't exist (AC-10)
        ValueError: If checkpoint architecture doesn't match base (AC-6)
    """
    model_refs = _collect_model_refs(node)

    model_loaders: dict[str, ModelLoader] = {}
    model_affected: dict[str, frozenset[str]] = {}
    all_model_keys: set[str] = set()
    opened_loaders: list[ModelLoader] = []  # For cleanup on error

    try:
        for model_id, recipe_model in model_refs.items():
            model_key = str(model_id)
            model_name = recipe_model.path
            source_dir = recipe_model.source_dir

            # Resolve path using source_dir from RecipeModel
            full_path = model_name
            if model_path_resolver is not None:
                resolved = model_path_resolver(model_name, source_dir)
                if resolved is not None:
                    full_path = resolved

            # AC-10: Check file exists before opening loader
            if not os.path.exists(full_path):
                raise FileNotFoundError(
                    f"Checkpoint file not found: {model_name}\n"
                    f"Referenced by Model Input node with strength {recipe_model.strength}"
                )

            # Open streaming loader
            loader = ModelLoader(full_path)
            opened_loaders.append(loader)

            # AC-6: Validate architecture matches base model
            if loader.arch is not None and loader.arch != base_arch:
                raise ValueError(
                    f"Architecture mismatch: checkpoint '{model_name}' has "
                    f"architecture '{loader.arch}' but base model has '{base_arch}'\n"
                    f"Both models must have the same architecture for merging."
                )

            model_loaders[model_key] = loader

            # AC-12: All diffusion model keys in the checkpoint are affected
            model_affected[model_key] = loader.affected_keys
            all_model_keys.update(loader.affected_keys)

    except Exception:
        # Cleanup any opened loaders on error
        for loader in opened_loaders:
            loader.cleanup()
        raise

    return ModelAnalysisResult(
        model_loaders=model_loaders,
        model_affected=model_affected,
        all_model_keys=frozenset(all_model_keys),
    )
