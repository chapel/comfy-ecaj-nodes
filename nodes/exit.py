"""WIDEN Exit Node — Executes the recipe tree, returns ComfyUI MODEL."""

from __future__ import annotations

import gc
import hashlib
import json
import os
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from ..lib.analysis import (
    analyze_recipe,
    analyze_recipe_models,
    get_keys_to_process,
    walk_to_base,
)
from ..lib.block_classify import compute_changed_blocks, filter_changed_keys
from ..lib.executor import (
    chunked_evaluation,
    compile_batch_groups,
    compile_plan,
    compute_batch_size,
    execute_plan,
)
from ..lib.persistence import (
    atomic_save,
    build_metadata,
    check_cache,
    collect_block_configs,
    compute_base_identity,
    compute_lora_stats,
    compute_recipe_hash,
    compute_structural_fingerprint,
    load_affected_keys,
    serialize_recipe,
    validate_model_name,
)
from ..lib.recipe import (
    RecipeBase,
    RecipeCompose,
    RecipeLoRA,
    RecipeMerge,
    RecipeModel,
    RecipeNode,
)
from ..lib.widen import WIDEN, WIDENConfig

try:
    from comfy.utils import ProgressBar
except ImportError:  # testing without ComfyUI
    ProgressBar = None  # type: ignore[assignment,misc]

if TYPE_CHECKING:
    from ..lib.recipe import BlockConfig


class _CacheEntry:
    """Single incremental recompute cache entry.

    AC: @incremental-block-recompute ac-1, ac-9
    Stores the structural fingerprint, block configs, and merged state
    from a previous execution. Tensors are cloned on insertion to avoid
    aliasing with tensors passed to install_merged_patches.
    """

    __slots__ = ("structural_fingerprint", "block_configs", "merged_state", "storage_dtype")

    def __init__(
        self,
        structural_fingerprint: str,
        block_configs: list[tuple[str, BlockConfig | None]],
        merged_state: dict[str, torch.Tensor],
        storage_dtype: torch.dtype,
    ) -> None:
        self.structural_fingerprint = structural_fingerprint
        self.block_configs = block_configs
        self.merged_state = merged_state
        self.storage_dtype = storage_dtype


# LRU-1 cache: at most one entry keyed by structural fingerprint
# AC: @incremental-block-recompute ac-9
_incremental_cache: dict[str, _CacheEntry] = {}


def clear_incremental_cache() -> None:
    """Clear the incremental recompute cache.

    AC: @incremental-block-recompute ac-12
    """
    _incremental_cache.clear()


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

    elif isinstance(node, RecipeModel):
        # Valid branch node for full model merging
        return

    elif isinstance(node, RecipeCompose):
        # Validate each branch
        if not node.branches:
            raise ValueError(f"RecipeCompose at {path} has no branches")
        for i, branch in enumerate(node.branches):
            branch_path = f"{path}.branches[{i}]"
            if not isinstance(branch, (RecipeLoRA, RecipeModel, RecipeCompose, RecipeMerge)):
                raise ValueError(
                    f"Invalid branch type at {branch_path}: expected RecipeLoRA, "
                    f"RecipeModel, RecipeCompose, or RecipeMerge, got {type(branch).__name__}"
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
        if not isinstance(node.target, (RecipeLoRA, RecipeModel, RecipeCompose, RecipeMerge)):
            raise ValueError(
                f"Invalid target type at {target_path}: expected RecipeLoRA, "
                f"RecipeModel, RecipeCompose, or RecipeMerge, got {type(node.target).__name__}"
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
    elif isinstance(node, RecipeModel):
        # Model nodes have no LoRAs - skip
        pass
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


def _collect_model_paths(node: RecipeNode) -> list[str]:
    """Recursively collect all model checkpoint paths from a recipe tree.

    AC: @full-model-execution ac-11
    Returns paths for IS_CHANGED hash computation.

    Args:
        node: Any recipe node

    Returns:
        List of model checkpoint paths in deterministic order
    """
    paths: list[str] = []

    if isinstance(node, RecipeBase):
        pass
    elif isinstance(node, RecipeLoRA):
        pass
    elif isinstance(node, RecipeModel):
        paths.append(node.path)
    elif isinstance(node, RecipeCompose):
        for branch in node.branches:
            paths.extend(_collect_model_paths(branch))
    elif isinstance(node, RecipeMerge):
        paths.extend(_collect_model_paths(node.base))
        paths.extend(_collect_model_paths(node.target))
        if node.backbone is not None:
            paths.extend(_collect_model_paths(node.backbone))

    return paths


def _compute_recipe_hash(
    widen: RecipeNode,
    lora_path_resolver: Callable[[str], str | None] | None = None,
    model_path_resolver: Callable[[str], str | None] | None = None,
) -> str:
    """Compute a hash of the recipe based on LoRA and model file paths and mtimes.

    AC: @exit-patch-install ac-5 — identical hash when no LoRA changes
    AC: @exit-patch-install ac-6 — different hash when LoRA modified
    AC: @full-model-execution ac-11 — checkpoint file stats included in hash

    Args:
        widen: Recipe tree root
        lora_path_resolver: Callable that resolves a LoRA name to its full
            filesystem path, or None if not found. Same resolver as
            used by analyze_recipe.
        model_path_resolver: Callable that resolves a model name to its full
            filesystem path.

    Returns:
        Hex digest of SHA-256 hash
    """
    lora_paths = _collect_lora_paths(widen)
    model_paths = _collect_model_paths(widen)

    # Sort for deterministic ordering
    lora_paths = sorted(set(lora_paths))
    model_paths = sorted(set(model_paths))

    # Build hash from (path, mtime, size) tuples
    hasher = hashlib.sha256()

    # Hash LoRA files
    for path in lora_paths:
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
            mtime = 0.0
            size = 0

        hasher.update(f"lora:{path}|{mtime}|{size}\n".encode())

    # AC: @full-model-execution ac-11
    # Hash model checkpoint files
    for path in model_paths:
        full_path = path
        if model_path_resolver is not None:
            resolved = model_path_resolver(path)
            if resolved is not None:
                full_path = resolved

        try:
            stat = os.stat(full_path)
            mtime = stat.st_mtime
            size = stat.st_size
        except OSError:
            mtime = 0.0
            size = 0

        hasher.update(f"model:{path}|{mtime}|{size}\n".encode())

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


def _build_model_resolver() -> Callable[[str], str | None]:
    """Build a model path resolver using ComfyUI's folder_paths.

    Returns a callable that resolves model names to their full filesystem path
    by searching all registered checkpoint directories.
    """
    import folder_paths

    def resolver(model_name: str) -> str | None:
        return folder_paths.get_full_path("checkpoints", model_name)

    return resolver


def _resolve_checkpoints_path(model_name: str) -> str:
    """Resolve a model name to a full path in the first checkpoints directory.

    Args:
        model_name: Validated model filename

    Returns:
        Full path to the model file

    Raises:
        ValueError: If no checkpoints directory is configured
    """
    import folder_paths

    dirs = folder_paths.get_folder_paths("checkpoints")
    if not dirs:
        raise ValueError("No checkpoints directory configured in ComfyUI")
    return os.path.join(dirs[0], model_name)


class WIDENExitNode:
    """The only node that computes. Runs full batched GPU pipeline."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "widen": ("WIDEN",),
            },
            "optional": {
                "save_model": ("BOOLEAN", {"default": False}),
                "model_name": ("STRING", {"default": ""}),
                "save_workflow": ("BOOLEAN", {"default": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "execute"
    CATEGORY = "ecaj/merge"
    OUTPUT_NODE = False

    @classmethod
    def IS_CHANGED(
        cls,
        widen: RecipeNode,
        save_model: bool = False,
        model_name: str = "",
        save_workflow: bool = True,
        prompt: object = None,
        extra_pnginfo: object = None,
    ) -> str:
        """Compute cache key based on LoRA and model file modification times.

        AC: @exit-patch-install ac-5 — identical hash on no LoRA changes
        AC: @exit-patch-install ac-6 — different hash on LoRA modifications
        AC: @full-model-execution ac-11 — checkpoint file stats included

        Returns:
            Hash string for ComfyUI caching
        """
        base_hash = _compute_recipe_hash(
            widen,
            lora_path_resolver=_build_lora_resolver(),
            model_path_resolver=_build_model_resolver(),
        )

        if not save_model:
            return base_hash

        # Include save parameters and cached file state
        hasher = hashlib.sha256(base_hash.encode())
        hasher.update(f"|save={save_model}|name={model_name}|wf={save_workflow}".encode())
        try:
            validated = validate_model_name(model_name)
            path = _resolve_checkpoints_path(validated)
            stat = os.stat(path)
            hasher.update(f"|mtime={stat.st_mtime}|size={stat.st_size}".encode())
        except (ValueError, OSError):
            hasher.update(b"|no_cache")
        return hasher.hexdigest()

    def execute(
        self,
        widen: RecipeNode,
        save_model: bool = False,
        model_name: str = "",
        save_workflow: bool = True,
        prompt: object = None,
        extra_pnginfo: object = None,
    ) -> tuple[object]:
        """Execute the recipe tree and return merged MODEL.

        AC: @exit-node ac-1 — returns ComfyUI MODEL with set patches
        AC: @exit-node ac-2 — validates tree, raises ValueError on type mismatches
        AC: @exit-node ac-3 — compose targets call merge_weights
        AC: @exit-node ac-4 — single LoRA targets call filter_delta
        AC: @exit-node ac-5 — chained merges evaluate inner first
        AC: @exit-node ac-6 — single-branch compose uses filter_delta
        AC: @exit-node ac-7 — downstream LoRA patches apply additively
        AC: @exit-node ac-8 — patch tensors match base model dtype
        AC: @exit-model-persistence ac-1 through ac-14

        Args:
            widen: Recipe tree root (should be RecipeMerge or RecipeBase)
            save_model: Whether to save/cache the merged model
            model_name: Filename for the saved model
            save_workflow: Whether to embed workflow metadata
            prompt: ComfyUI prompt (hidden input)
            extra_pnginfo: ComfyUI workflow info (hidden input)

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

        # Build resolvers that search all ComfyUI directories
        lora_path_resolver = _build_lora_resolver()
        model_path_resolver = _build_model_resolver()

        # --- Shared setup: compute base_state ONCE ---
        model_patcher = walk_to_base(widen).model_patcher
        _unpatch_loaded_clones(model_patcher)
        base_state = model_patcher.model_state_dict()  # type: ignore[attr-defined]
        storage_dtype = next(iter(base_state.values())).dtype

        # --- Compute base_identity and lora_stats for both persistence and incremental cache ---
        base_identity = compute_base_identity(base_state)
        lora_stats = compute_lora_stats(widen, lora_path_resolver, model_path_resolver)

        # --- Persistence: pre-GPU cache check ---
        save_path = serialized = recipe_hash = None
        if save_model:
            validated_name = validate_model_name(model_name)
            save_path = _resolve_checkpoints_path(validated_name)

            serialized = serialize_recipe(widen, base_identity, lora_stats)
            recipe_hash = compute_recipe_hash(serialized)

            cached_metadata = check_cache(save_path, recipe_hash)
            if cached_metadata is not None:
                # CACHE HIT — skip GPU entirely, no LoRA/model loading
                affected = json.loads(cached_metadata["__ecaj_affected_keys__"])
                merged_state = load_affected_keys(save_path, affected)
                if ProgressBar is not None:
                    pbar = ProgressBar(1)
                    pbar.update(1)
                return (install_merged_patches(model_patcher, merged_state, storage_dtype),)

        # --- Normal GPU pipeline ---
        analysis = analyze_recipe(widen, lora_path_resolver=lora_path_resolver)

        # AC: @full-model-execution ac-1
        # Analyze recipe for full model checkpoints
        base = walk_to_base(widen)
        model_analysis = analyze_recipe_models(
            widen, base.arch, model_path_resolver=model_path_resolver
        )

        try:
            loader = analysis.loader
            set_affected = analysis.set_affected
            lora_affected_keys = analysis.affected_keys
            arch = analysis.arch

            # AC: @full-model-execution ac-12
            # Model affected keys (all diffusion model keys in both base and checkpoint)
            model_affected = model_analysis.model_affected
            model_loaders = model_analysis.model_loaders
            all_model_keys = model_analysis.all_model_keys

            # Computation dtype is fp32 for numerical stability
            compute_dtype = torch.float32

            # Get device for GPU computation
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # AC: @full-model-execution ac-12
            # For models-only recipes, process all diffusion keys in both base and model
            # For mixed recipes, union of LoRA-affected and model-affected keys
            all_keys = set(base_state.keys())
            lora_keys = get_keys_to_process(all_keys, lora_affected_keys)
            model_keys = all_keys & all_model_keys  # Keys in both base and model
            keys_to_process = lora_keys | model_keys

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

            # Build model_id_map from object ids to string keys
            # AC: @full-model-execution ac-2
            model_id_map: dict[int, str] = {}
            for model_key in model_affected.keys():
                model_id = int(model_key)
                model_id_map[model_id] = model_key

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

            # Pre-compile recipe tree into flat evaluation plan (once)
            # AC: @full-model-execution ac-2
            plan = compile_plan(widen, set_id_map, arch, model_id_map)

            # --- Incremental cache: detect which blocks changed ---
            # AC: @incremental-block-recompute ac-1 through ac-16
            structural_fp = compute_structural_fingerprint(
                widen, base_identity, lora_stats
            )
            current_block_configs = collect_block_configs(widen)
            cached_entry = _incremental_cache.get(structural_fp)
            incremental_hit = False

            if (
                cached_entry is not None
                and cached_entry.storage_dtype == storage_dtype
            ):
                diff = compute_changed_blocks(
                    cached_entry.block_configs, current_block_configs, arch
                )
                if diff is not None:
                    changed_blocks, changed_layer_types = diff

                    if not changed_blocks and not changed_layer_types:
                        # AC-2: Full cache hit — all keys identical
                        merged_state = {
                            k: v for k, v in cached_entry.merged_state.items()
                        }
                        incremental_hit = True
                        batch_groups = {}  # Skip GPU loop entirely

                        if ProgressBar is not None:
                            pbar = ProgressBar(1)
                            pbar.update(1)
                    else:
                        # AC-3, AC-5, AC-6, AC-15: Partial hit
                        recompute_keys = filter_changed_keys(
                            keys_to_process, changed_blocks,
                            changed_layer_types, arch,
                        )

                        if not recompute_keys:
                            # Edge case: changed blocks don't affect any keys
                            merged_state = {
                                k: v for k, v in cached_entry.merged_state.items()
                            }
                            incremental_hit = True
                            batch_groups = {}  # Skip GPU loop
                        else:
                            # Start from cached state, recompute subset
                            merged_state = {
                                k: v for k, v in cached_entry.merged_state.items()
                            }

                            # Rebuild batch_groups for only changed keys
                            batch_groups = compile_batch_groups(
                                list(recompute_keys), base_state, arch=arch,
                            )
                            incremental_hit = True

            if not incremental_hit:
                merged_state = {}

            # Phase 2: Batched GPU evaluation per group
            # (skipped entirely on full cache hit)
            if batch_groups:
                pbar_count = len(batch_groups)
                pbar = ProgressBar(pbar_count) if ProgressBar is not None else None

                for sig, group_keys in batch_groups.items():
                    # Estimate batch size based on shape and VRAM
                    # AC: @full-model-execution ac-13
                    # Count both LoRA sets and model loaders for memory estimation
                    n_models = len(set_affected) + len(model_loaders)
                    batch_size = compute_batch_size(
                        sig.shape,
                        n_models,
                        compute_dtype,
                    )

                    # Build evaluation function using pre-compiled plan
                    # AC: @merge-block-config ac-1, ac-2
                    # AC: @full-model-execution ac-3, ac-5
                    # Pass arch, widen_config, and model_loaders
                    def make_eval_fn(p, ldr, wdn, dev, dtype, architecture, wcfg, mdl_ldrs):
                        def eval_fn(keys: list[str], base_batch: torch.Tensor) -> torch.Tensor:
                            return execute_plan(
                                plan=p,
                                keys=keys,
                                base_batch=base_batch,
                                loader=ldr,
                                widen=wdn,
                                device=dev,
                                dtype=dtype,
                                arch=architecture,
                                widen_config=wcfg,
                                model_loaders=mdl_ldrs,
                            )
                        return eval_fn

                    eval_fn = make_eval_fn(
                        plan, loader, widen_merger, device, compute_dtype,
                        arch, widen_config, model_loaders
                    )

                    # Run chunked evaluation with OOM backoff
                    # AC: @full-model-execution ac-8
                    # OOM backoff retries at batch_size=1 (streaming loader re-reads)
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

                    # AC-9: Update progress after each batch group
                    if pbar is not None:
                        pbar.update(1)

            # AC: @memory-management ac-2
            # Cleanup after all groups complete (OOM backoff handles per-group pressure)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # AC: @incremental-block-recompute ac-1, ac-16
            # Store result in incremental cache (atomic swap)
            # Build new entry fully, then swap. On exception above,
            # old entry is preserved (we never reach this point).
            new_entry = _CacheEntry(
                structural_fingerprint=structural_fp,
                block_configs=current_block_configs,
                merged_state={k: v.clone() for k, v in merged_state.items()},
                storage_dtype=storage_dtype,
            )
            _incremental_cache.clear()
            _incremental_cache[structural_fp] = new_entry

        finally:
            # AC: @memory-management ac-3
            # Cleanup loader resources (delta caches and file handles)
            loader.cleanup()

            # AC: @full-model-execution ac-7
            # Cleanup model loaders (close file handles)
            for model_loader in model_analysis.model_loaders.values():
                model_loader.cleanup()

        # Phase 3: Install merged weights as set patches
        # AC-1: Returns MODEL (ModelPatcher clone) with set patches
        # AC-7: Set patches work with downstream LoRA patches additively
        # AC-8: Patch tensors match base model dtype (handled by install_merged_patches)
        result = install_merged_patches(model_patcher, merged_state, storage_dtype)

        # --- Persistence: save after GPU ---
        # AC: @incremental-block-recompute ac-10
        if save_model and save_path is not None:
            # Overlay merged keys into base_state in-place (base_state is
            # already a dict copy from model_state_dict, not used after this)
            for key, tensor in merged_state.items():
                base_state[key] = tensor.cpu().to(storage_dtype)
            workflow_json = (
                json.dumps(extra_pnginfo) if save_workflow and extra_pnginfo else None
            )
            metadata = build_metadata(
                serialized, recipe_hash, sorted(merged_state.keys()), workflow_json
            )
            atomic_save(base_state, save_path, metadata)

        return (result,)
