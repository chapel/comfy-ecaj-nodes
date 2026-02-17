"""WIDEN CLIP Exit Node — Executes the CLIP recipe tree, returns ComfyUI CLIP."""

from __future__ import annotations

import gc
from collections.abc import Callable

import torch

from ..lib.analysis import (
    analyze_recipe,
    analyze_recipe_models,
    compute_recipe_file_hash,
    get_keys_to_process,
    walk_to_base,
)
from ..lib.executor import (
    check_ram_preflight,
    chunked_evaluation,
    compile_batch_groups,
    compile_plan,
    compute_batch_size,
    execute_plan,
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

__all__ = [
    "WIDENCLIPExitNode",
    "install_merged_clip_patches",
]


def _validate_clip_recipe_tree(node: RecipeNode, path: str = "root") -> None:
    """Recursively validate the CLIP recipe tree structure.

    AC: @clip-exit-node ac-6
    Raises ValueError naming the invalid type and its position in the tree.

    Args:
        node: Recipe node to validate
        path: Current position in tree (for error messages)

    Raises:
        ValueError: If tree structure is invalid with position info
    """
    if isinstance(node, RecipeBase):
        # Valid leaf node — verify it has domain="clip"
        if getattr(node, "domain", "diffusion") != "clip":
            raise ValueError(
                f"RecipeBase at {path} has domain='{getattr(node, 'domain', 'diffusion')}', "
                f"expected domain='clip'. Use CLIP Entry node to create CLIP recipes."
            )
        return

    elif isinstance(node, RecipeLoRA):
        # Valid branch node
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
            _validate_clip_recipe_tree(branch, branch_path)

    elif isinstance(node, RecipeMerge):
        # Validate base
        base_path = f"{path}.base"
        if not isinstance(node.base, (RecipeBase, RecipeMerge)):
            raise ValueError(
                f"Invalid base type at {base_path}: expected RecipeBase or "
                f"RecipeMerge, got {type(node.base).__name__}"
            )
        _validate_clip_recipe_tree(node.base, base_path)

        # Validate target
        target_path = f"{path}.target"
        if not isinstance(node.target, (RecipeLoRA, RecipeModel, RecipeCompose, RecipeMerge)):
            raise ValueError(
                f"Invalid target type at {target_path}: expected RecipeLoRA, "
                f"RecipeModel, RecipeCompose, or RecipeMerge, got {type(node.target).__name__}"
            )
        _validate_clip_recipe_tree(node.target, target_path)

        # Validate backbone (optional)
        if node.backbone is not None:
            backbone_path = f"{path}.backbone"
            _validate_clip_recipe_tree(node.backbone, backbone_path)

    else:
        raise ValueError(
            f"Unknown recipe node type at {path}: {type(node).__name__}"
        )


def _unpatch_loaded_clip_clones(clip: object) -> None:
    """Force-unpatch any loaded clone sharing our CLIP's weights.

    AC: @clip-exit-node ac-4
    Same logic as diffusion model unpatch but operates on clip.patcher.

    Args:
        clip: ComfyUI CLIP object (has .patcher attribute)
    """
    try:
        from comfy.model_management import current_loaded_models  # noqa: E402
    except (ImportError, AttributeError):
        return  # Testing without ComfyUI

    # Access the patcher from CLIP
    clip_patcher = getattr(clip, "patcher", None)
    if clip_patcher is None:
        return

    loaded_models = current_loaded_models
    for i in range(len(loaded_models) - 1, -1, -1):
        loaded = loaded_models[i]
        if loaded.model is not None and loaded.model.is_clone(clip_patcher):
            loaded.model_unload()
            loaded_models.pop(i)


def install_merged_clip_patches(
    clip: object,
    merged_state: dict[str, torch.Tensor],
    storage_dtype: torch.dtype,
) -> object:
    """Install merged tensors as set patches on a cloned CLIP.

    AC: @clip-exit-node ac-4 — clone CLIP, add as set patches
    AC: @clip-exit-node ac-5 — returns usable CLIP object

    CLIP objects expose clone() and add_patches() directly, delegating
    to their internal patcher. Keys use clip_l.* and clip_g.* prefixes.

    Args:
        clip: Original ComfyUI CLIP object
        merged_state: Dict of {key: merged_tensor} from batched evaluation
            Keys have clip_l.* or clip_g.* prefixes
        storage_dtype: Base model storage dtype for casting output tensors

    Returns:
        Cloned CLIP with merged weights installed as set patches
    """
    # Clone CLIP (AC-4)
    cloned = clip.clone()  # type: ignore[attr-defined]

    # Build set patches: transfer to CPU, cast to base dtype
    patches = {}
    for key, tensor in merged_state.items():
        cpu_tensor = tensor.cpu().to(storage_dtype)
        # "set" patch format: replaces the weight entirely
        # ComfyUI expects value wrapped in a tuple: ("set", (tensor,))
        patches[key] = ("set", (cpu_tensor,))

    # Install patches (AC-4)
    # CLIP.add_patches delegates to patcher.add_patches
    cloned.add_patches(patches, strength_patch=1.0)  # type: ignore[attr-defined]

    return cloned


def _build_lora_resolver() -> Callable[[str], str | None]:
    """Build a LoRA path resolver using ComfyUI's folder_paths."""
    import folder_paths

    def resolver(lora_name: str) -> str | None:
        return folder_paths.get_full_path("loras", lora_name)

    return resolver


def _build_clip_model_resolver() -> Callable[[str, str], str | None]:
    """Build a model path resolver for CLIP checkpoints.

    CLIP checkpoints are always in the checkpoints folder since
    CLIP weights come from full model files.
    """
    import folder_paths

    def resolver(model_name: str, source_dir: str) -> str | None:
        # CLIP weights come from checkpoints (full models)
        return folder_paths.get_full_path("checkpoints", model_name)

    return resolver


class WIDENCLIPExitNode:
    """The only CLIP node that computes. Runs batched GPU pipeline for CLIP merging."""

    @classmethod
    def INPUT_TYPES(cls):
        # AC: @clip-exit-node ac-7 — accepts WIDEN_CLIP type
        return {
            "required": {
                "widen_clip": ("WIDEN_CLIP",),
            },
        }

    # AC: @clip-exit-node ac-1 — returns CLIP type
    RETURN_TYPES = ("CLIP",)
    RETURN_NAMES = ("clip",)
    FUNCTION = "execute"
    CATEGORY = "ecaj/merge/clip"
    OUTPUT_NODE = False

    @classmethod
    def IS_CHANGED(
        cls,
        widen_clip: RecipeNode,
    ) -> str:
        """Compute cache key based on LoRA and model file modification times.

        AC: @clip-exit-node ac-8 — different value when files change

        Returns:
            Hash string for ComfyUI caching
        """
        return compute_recipe_file_hash(
            widen_clip,
            lora_path_resolver=_build_lora_resolver(),
            model_path_resolver=_build_clip_model_resolver(),
        )

    def execute(
        self,
        widen_clip: RecipeNode,
    ) -> tuple[object]:
        """Execute the CLIP recipe tree and return merged CLIP.

        AC: @clip-exit-node ac-1 — returns ComfyUI CLIP with set patches
        AC: @clip-exit-node ac-2 — selects CLIP LoRA loader via domain="clip"
        AC: @clip-exit-node ac-3 — selects CLIP model loader via domain="clip"
        AC: @clip-exit-node ac-4 — clones CLIP and applies via patch mechanism
        AC: @clip-exit-node ac-5 — result is usable CLIP object
        AC: @clip-exit-node ac-6 — validates tree, raises ValueError on type mismatches
        AC: @clip-exit-node ac-9 — reports progress via ProgressBar

        Args:
            widen_clip: CLIP recipe tree root (should be RecipeMerge or RecipeBase)

        Returns:
            Tuple containing cloned CLIP with merged weights as set patches

        Raises:
            ValueError: If recipe tree structure is invalid
        """
        # AC-6: Validate recipe tree structure
        _validate_clip_recipe_tree(widen_clip)

        # Quick check: must end in RecipeMerge for actual merging
        if isinstance(widen_clip, RecipeBase):
            return (widen_clip.model_patcher.clone(),)  # type: ignore[attr-defined]

        if not isinstance(widen_clip, RecipeMerge):
            raise ValueError(
                f"CLIP Exit node expects RecipeMerge or RecipeBase at root, "
                f"got {type(widen_clip).__name__}. Connect a CLIP Merge node to CLIP Exit."
            )

        # Build resolvers
        lora_path_resolver = _build_lora_resolver()
        model_path_resolver = _build_clip_model_resolver()

        # --- Shared setup ---
        base = walk_to_base(widen_clip)
        clip = base.model_patcher

        # Unpatch any loaded clones to get clean weights
        _unpatch_loaded_clip_clones(clip)

        # Get CLIP state dict via patcher
        base_state = clip.patcher.model_state_dict()  # type: ignore[attr-defined]
        storage_dtype = next(iter(base_state.values())).dtype

        # --- AC-2: Analyze recipe with domain="clip" to get CLIP LoRA loader ---
        analysis = analyze_recipe(widen_clip, lora_path_resolver=lora_path_resolver)
        model_analysis = None

        try:
            # AC-3: Analyze recipe for CLIP model checkpoints via domain dispatch
            model_analysis = analyze_recipe_models(
                widen_clip, base.arch, model_path_resolver=model_path_resolver,
                domain="clip",
            )

            loader = analysis.loader
            set_affected = analysis.set_affected
            lora_affected_keys = analysis.affected_keys
            arch = analysis.arch

            # Model affected keys from CLIP model loaders
            model_affected = model_analysis.model_affected
            clip_model_loaders = model_analysis.model_loaders
            all_model_keys = model_analysis.all_model_keys

            # Computation dtype is fp32 for numerical stability
            compute_dtype = torch.float32

            # Get device for GPU computation
            device = "cuda" if torch.cuda.is_available() else "cpu"

            # Filter keys to process
            all_keys = set(base_state.keys())
            lora_keys = get_keys_to_process(all_keys, lora_affected_keys)
            model_keys = all_keys & all_model_keys
            keys_to_process = lora_keys | model_keys

            if not keys_to_process:
                # No keys affected - return clone
                return (clip.clone(),)  # type: ignore[attr-defined]

            # Build set_id_map from object ids to string keys
            set_id_map: dict[int, str] = {}
            for set_key in set_affected.keys():
                set_id = int(set_key)
                set_id_map[set_id] = set_key

            # Build model_id_map
            model_id_map: dict[int, str] = {}
            for model_key in model_affected.keys():
                model_id = int(model_key)
                model_id_map[model_id] = model_key

            # Create WIDEN instance
            widen_config = WIDENConfig(
                t_factor=widen_clip.t_factor,
                dtype=compute_dtype,
            )
            widen_merger = WIDEN(widen_config)

            # Group keys by OpSignature for batched evaluation
            batch_groups = compile_batch_groups(
                list(keys_to_process),
                base_state,
                arch=arch,
            )

            # Pre-compile recipe tree into flat evaluation plan
            plan = compile_plan(widen_clip, set_id_map, arch, model_id_map)

            merged_state: dict[str, torch.Tensor] = {}

            # AC: @clip-exit-node ac-11
            # Pre-flight RAM check before GPU loop
            if batch_groups:
                # Only count keys being processed — not the full base_state.
                processed_keys = {k for keys in batch_groups.values() for k in keys}
                merged_state_bytes = sum(
                    base_state[k].nelement() * base_state[k].element_size()
                    for k in processed_keys
                )
                n_models = len(set_affected) + len(clip_model_loaders)
                element_size = torch.finfo(compute_dtype).bits // 8
                # Compute worst-case chunk bytes: pair each group's batch_size
                # with its own shape (not max_batch * max_shape).
                worst_chunk_bytes = max(
                    element_size
                    * torch.Size(sig.shape).numel()
                    * compute_batch_size(sig.shape, n_models, compute_dtype)
                    for sig in batch_groups
                )
                check_ram_preflight(
                    base_state_bytes=merged_state_bytes,
                    n_models=n_models,
                    worst_chunk_bytes=worst_chunk_bytes,
                    save_model=False,  # CLIP exit doesn't save
                )

            # Phase 2: Batched GPU evaluation per group
            # AC-9: Progress reporting
            pbar_count = len(batch_groups)
            pbar = ProgressBar(pbar_count) if ProgressBar is not None else None

            for sig, group_keys in batch_groups.items():
                # Estimate batch size
                n_models = len(set_affected) + len(clip_model_loaders)
                batch_size = compute_batch_size(
                    sig.shape,
                    n_models,
                    compute_dtype,
                )

                # Build evaluation function using pre-compiled plan
                # AC: @clip-exit-node ac-10
                # Pass domain="clip" so per-block helpers use CLIP classifiers
                def make_eval_fn(p, ldr, wdn, dev, dtype, architecture, wcfg, mdl_ldrs, dom):
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
                            domain=dom,
                        )
                    return eval_fn

                eval_fn = make_eval_fn(
                    plan, loader, widen_merger, device, compute_dtype,
                    arch, widen_config, clip_model_loaders, "clip",
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
                    storage_dtype=storage_dtype,
                )

                merged_state.update(group_results)

                # AC-9: Update progress after each batch group
                if pbar is not None:
                    pbar.update(1)

            # Cleanup after all groups complete
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        finally:
            # Cleanup LoRA loader
            analysis.loader.cleanup()

            # Cleanup CLIP model loaders (if they were opened)
            if model_analysis is not None:
                for model_loader in model_analysis.model_loaders.values():
                    model_loader.cleanup()

        # Phase 3: Install merged weights as set patches
        # AC-4, AC-5: Returns CLIP with set patches
        result = install_merged_clip_patches(clip, merged_state, storage_dtype)

        return (result,)
