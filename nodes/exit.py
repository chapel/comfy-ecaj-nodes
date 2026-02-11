"""WIDEN Exit Node — Executes the recipe tree, returns ComfyUI MODEL."""

from __future__ import annotations

import gc
import hashlib
import os
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from ..lib.analysis import analyze_recipe, get_keys_to_process
from ..lib.executor import (
    chunked_evaluation,
    compile_batch_groups,
    compute_batch_size,
    evaluate_recipe,
)
from ..lib.recipe import RecipeBase, RecipeCompose, RecipeLoRA, RecipeMerge, RecipeNode
from ..lib.widen import WIDEN, WIDENConfig

if TYPE_CHECKING:
    pass


def _validate_recipe_tree(node: RecipeNode, path: str = "root") -> None:
    """Recursively validate the recipe tree structure.

    AC: @exit-node ac-2
    Raises ValueError naming the invalid type and its position in the tree.

    Args:
        node: Recipe node to validate
        path: Current position in tree (for error messages)

    Raises:
        ValueError: If tree structure is invalid with position info
    """
    if isinstance(node, RecipeBase):
        # Valid leaf node
        return

    elif isinstance(node, RecipeLoRA):
        # Valid branch node (must be used as target or branch, not root)
        return

    elif isinstance(node, RecipeCompose):
        # Validate each branch
        if not node.branches:
            raise ValueError(f"RecipeCompose at {path} has no branches")
        for i, branch in enumerate(node.branches):
            branch_path = f"{path}.branches[{i}]"
            if not isinstance(branch, (RecipeLoRA, RecipeCompose, RecipeMerge)):
                raise ValueError(
                    f"Invalid branch type at {branch_path}: expected RecipeLoRA, "
                    f"RecipeCompose, or RecipeMerge, got {type(branch).__name__}"
                )
            _validate_recipe_tree(branch, branch_path)

    elif isinstance(node, RecipeMerge):
        # Validate base
        base_path = f"{path}.base"
        if not isinstance(node.base, (RecipeBase, RecipeMerge)):
            raise ValueError(
                f"Invalid base type at {base_path}: expected RecipeBase or "
                f"RecipeMerge, got {type(node.base).__name__}"
            )
        _validate_recipe_tree(node.base, base_path)

        # Validate target
        target_path = f"{path}.target"
        if not isinstance(node.target, (RecipeLoRA, RecipeCompose, RecipeMerge)):
            raise ValueError(
                f"Invalid target type at {target_path}: expected RecipeLoRA, "
                f"RecipeCompose, or RecipeMerge, got {type(node.target).__name__}"
            )
        _validate_recipe_tree(node.target, target_path)

        # Validate backbone (optional)
        if node.backbone is not None:
            backbone_path = f"{path}.backbone"
            _validate_recipe_tree(node.backbone, backbone_path)

    else:
        raise ValueError(
            f"Unknown recipe node type at {path}: {type(node).__name__}"
        )


def _unpatch_loaded_clones(model_patcher: object) -> None:
    """Force-unpatch any loaded clone sharing our model's weights.

    ComfyUI keeps models patched in-place between prompts for performance.
    When a clone with "set" patches is loaded, the shared model's weights
    are overwritten. model_state_dict() returns these patched values.

    This finds any loaded clone sharing the same underlying model and fully
    unloads it, which restores the original weights from its backup.

    Args:
        model_patcher: ComfyUI ModelPatcher (from Entry node)
    """
    try:
        from comfy.model_management import current_loaded_models  # noqa: E402
    except (ImportError, AttributeError):
        return  # Testing without ComfyUI

    loaded_models = current_loaded_models
    for i in range(len(loaded_models) - 1, -1, -1):
        loaded = loaded_models[i]
        if loaded.model is not None and loaded.model.is_clone(model_patcher):
            loaded.model_unload()
            loaded_models.pop(i)


def install_merged_patches(
    model_patcher: object,
    merged_state: dict[str, torch.Tensor],
    storage_dtype: torch.dtype,
) -> object:
    """Install merged tensors as set patches on a cloned ModelPatcher.

    AC: @exit-patch-install ac-1 — clone model, add as set patches
    AC: @exit-patch-install ac-2 — keys use diffusion_model. prefix
    AC: @exit-patch-install ac-3 — tensors transferred to CPU
    AC: @exit-patch-install ac-4 — tensors match base model storage dtype

    Args:
        model_patcher: Original ComfyUI ModelPatcher
        merged_state: Dict of {key: merged_tensor} from batched evaluation
            Keys already have diffusion_model. prefix (from LoRA loaders)
        storage_dtype: Base model storage dtype for casting output tensors

    Returns:
        Cloned ModelPatcher with merged weights installed as set patches
    """
    # Clone model (AC-1)
    cloned = model_patcher.clone()  # type: ignore[attr-defined]

    # Build set patches: transfer to CPU (AC-3), cast to base dtype (AC-4)
    # Keys already have diffusion_model. prefix (AC-2)
    patches = {}
    for key, tensor in merged_state.items():
        cpu_tensor = tensor.cpu().to(storage_dtype)
        # "set" patch format: replaces the weight entirely
        # ComfyUI expects value wrapped in a tuple: ("set", (tensor,))
        patches[key] = ("set", (cpu_tensor,))

    # Install patches (AC-1)
    cloned.add_patches(patches, strength_patch=1.0)  # type: ignore[attr-defined]

    return cloned


def _collect_lora_paths(node: RecipeNode) -> list[str]:
    """Recursively collect all LoRA file paths from a recipe tree.

    Args:
        node: Any recipe node

    Returns:
        List of LoRA file paths in deterministic order
    """
    paths: list[str] = []

    if isinstance(node, RecipeBase):
        # Base node has no LoRAs
        pass
    elif isinstance(node, RecipeLoRA):
        # Extract paths from loras tuple
        for lora_spec in node.loras:
            paths.append(lora_spec["path"])
    elif isinstance(node, RecipeCompose):
        # Collect from all branches
        for branch in node.branches:
            paths.extend(_collect_lora_paths(branch))
    elif isinstance(node, RecipeMerge):
        # Collect from base, target, and backbone
        paths.extend(_collect_lora_paths(node.base))
        paths.extend(_collect_lora_paths(node.target))
        if node.backbone is not None:
            paths.extend(_collect_lora_paths(node.backbone))

    return paths


def _compute_recipe_hash(
    widen: RecipeNode,
    lora_path_resolver: Callable[[str], str | None] | None = None,
) -> str:
    """Compute a hash of the recipe based on LoRA file paths and mtimes.

    AC: @exit-patch-install ac-5 — identical hash when no LoRA changes
    AC: @exit-patch-install ac-6 — different hash when LoRA modified

    Args:
        widen: Recipe tree root
        lora_path_resolver: Callable that resolves a LoRA name to its full
            filesystem path, or None if not found. Same resolver as
            used by analyze_recipe.

    Returns:
        Hex digest of SHA-256 hash
    """
    paths = _collect_lora_paths(widen)

    # Sort for deterministic ordering
    paths = sorted(set(paths))

    # Build hash from (path, mtime, size) tuples
    hasher = hashlib.sha256()

    for path in paths:
        # Resolve full path using resolver if available
        full_path = path
        if lora_path_resolver is not None:
            resolved = lora_path_resolver(path)
            if resolved is not None:
                full_path = resolved

        try:
            stat = os.stat(full_path)
            mtime = stat.st_mtime
            size = stat.st_size
        except OSError:
            # File doesn't exist or inaccessible — use sentinel values
            mtime = 0.0
            size = 0

        # Add to hash: path|mtime|size
        hasher.update(f"{path}|{mtime}|{size}\n".encode())

    return hasher.hexdigest()


def _build_lora_resolver() -> Callable[[str], str | None]:
    """Build a LoRA path resolver using ComfyUI's folder_paths.

    Returns a callable that resolves LoRA names (including nested paths like
    "z-image/Mystic.safetensors") to their full filesystem path by searching
    all registered LoRA directories.
    """
    import folder_paths

    def resolver(lora_name: str) -> str | None:
        return folder_paths.get_full_path("loras", lora_name)

    return resolver


class WIDENExitNode:
    """The only node that computes. Runs full batched GPU pipeline."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "widen": ("WIDEN",),
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "execute"
    CATEGORY = "ecaj/merge"
    OUTPUT_NODE = False

    @classmethod
    def IS_CHANGED(cls, widen: RecipeNode) -> str:
        """Compute cache key based on LoRA file modification times.

        AC: @exit-patch-install ac-5 — identical hash on no LoRA changes
        AC: @exit-patch-install ac-6 — different hash on LoRA modifications

        Returns:
            Hash string for ComfyUI caching
        """
        return _compute_recipe_hash(widen, lora_path_resolver=_build_lora_resolver())

    def execute(self, widen: RecipeNode) -> tuple[object]:
        """Execute the recipe tree and return merged MODEL.

        AC: @exit-node ac-1 — returns ComfyUI MODEL with set patches
        AC: @exit-node ac-2 — validates tree, raises ValueError on type mismatches
        AC: @exit-node ac-3 — compose targets call merge_weights
        AC: @exit-node ac-4 — single LoRA targets call filter_delta
        AC: @exit-node ac-5 — chained merges evaluate inner first
        AC: @exit-node ac-6 — single-branch compose uses filter_delta
        AC: @exit-node ac-7 — downstream LoRA patches apply additively
        AC: @exit-node ac-8 — patch tensors match base model dtype

        Args:
            widen: Recipe tree root (should be RecipeMerge or RecipeBase)

        Returns:
            Tuple containing cloned ModelPatcher with merged weights as set patches

        Raises:
            ValueError: If recipe tree structure is invalid
        """
        # AC-2: Validate recipe tree structure
        _validate_recipe_tree(widen)

        # Quick check: must end in RecipeMerge for actual merging
        if isinstance(widen, RecipeBase):
            return (widen.model_patcher.clone(),)  # type: ignore[attr-defined]

        if not isinstance(widen, RecipeMerge):
            raise ValueError(
                f"Exit node expects RecipeMerge or RecipeBase at root, "
                f"got {type(widen).__name__}. Connect a Merge node to Exit."
            )

        # Phase 1: Analyze recipe tree (loads LoRAs, builds set map)
        # Build resolver that searches all ComfyUI LoRA directories
        lora_path_resolver = _build_lora_resolver()

        analysis = analyze_recipe(widen, lora_path_resolver=lora_path_resolver)

        try:
            model_patcher = analysis.model_patcher
            loader = analysis.loader
            set_affected = analysis.set_affected
            affected_keys = analysis.affected_keys
            arch = analysis.arch

            # Force-unpatch any loaded clone from a previous run so that
            # model_state_dict() returns clean (unpatched) base weights.
            # Without this, "set" patches from the prior clone remain applied
            # in-place on the shared model, causing LoRA double-application.
            _unpatch_loaded_clones(model_patcher)

            # Get base model state dict (prefixed keys: diffusion_model.X.weight)
            # Must match the key format produced by LoRA loaders
            base_state = model_patcher.model_state_dict()  # type: ignore[attr-defined]

            # Determine storage dtype from base model
            storage_dtype = next(iter(base_state.values())).dtype

            # Computation dtype is fp32 for numerical stability
            compute_dtype = torch.float32

            # Get device for GPU computation
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # AC-6: Filter keys to only those affected by LoRAs
            all_keys = set(base_state.keys())
            keys_to_process = get_keys_to_process(all_keys, affected_keys)

            if not keys_to_process:
                # No keys affected - return clone
                return (model_patcher.clone(),)  # type: ignore[attr-defined]

            # Build set_id_map from object ids to string keys
            # This maps id(RecipeLoRA) -> str for evaluate_recipe
            set_id_map: dict[int, str] = {}
            for set_key, affected in set_affected.items():
                # set_key is str(id(RecipeLoRA)), convert back to int
                set_id = int(set_key)
                set_id_map[set_id] = set_key

            # Create WIDEN instance with t_factor from the root merge
            # AC-6: Single-branch compose will be handled by evaluate_recipe
            # dispatching to filter_delta for len(branches)==1
            widen_config = WIDENConfig(
                t_factor=widen.t_factor,
                dtype=compute_dtype,
            )
            widen_merger = WIDEN(widen_config)

            # Group keys by OpSignature for batched evaluation
            batch_groups = compile_batch_groups(
                list(keys_to_process),
                base_state,
                arch=arch,
            )

            # Phase 2: Batched GPU evaluation per group
            merged_state: dict[str, torch.Tensor] = {}

            for sig, group_keys in batch_groups.items():
                # Estimate batch size based on shape and VRAM
                n_models = len(set_affected)  # Number of LoRA sets
                batch_size = compute_batch_size(
                    sig.shape,
                    n_models,
                    compute_dtype,
                )

                # Build evaluation function that calls evaluate_recipe
                # AC: @merge-block-config ac-1, ac-2
                # Pass arch and widen_config for per-block t_factor support
                def make_eval_fn(recipe, ldr, wdn, sid_map, dev, dtype, architecture, wcfg):
                    def eval_fn(keys: list[str], base_batch: torch.Tensor) -> torch.Tensor:
                        return evaluate_recipe(
                            keys=keys,
                            base_batch=base_batch,
                            recipe_node=recipe,
                            loader=ldr,
                            widen=wdn,
                            set_id_map=sid_map,
                            device=dev,
                            dtype=dtype,
                            arch=architecture,
                            widen_config=wcfg,
                        )
                    return eval_fn

                eval_fn = make_eval_fn(
                    widen, loader, widen_merger, set_id_map, device, compute_dtype,
                    arch, widen_config
                )

                # Run chunked evaluation with OOM backoff
                group_base = {k: base_state[k] for k in group_keys}
                group_results = chunked_evaluation(
                    keys=group_keys,
                    base_tensors=group_base,
                    eval_fn=eval_fn,
                    batch_size=batch_size,
                    device=device,
                    dtype=compute_dtype,
                    storage_dtype=storage_dtype,  # AC-8: match base model dtype
                )

                merged_state.update(group_results)

                # AC: @memory-management ac-2
                # Cleanup between groups: gc.collect() and empty_cache()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        finally:
            # AC: @memory-management ac-3
            # Cleanup loader resources (delta caches and file handles)
            loader.cleanup()

        # Phase 3: Install merged weights as set patches
        # AC-1: Returns MODEL (ModelPatcher clone) with set patches
        # AC-7: Set patches work with downstream LoRA patches additively
        # AC-8: Patch tensors match base model dtype (handled by install_merged_patches)
        result = install_merged_patches(model_patcher, merged_state, storage_dtype)

        return (result,)
