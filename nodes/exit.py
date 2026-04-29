"""WIDEN Exit Node — Executes the recipe tree, returns ComfyUI MODEL."""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

from ..lib.analysis import (
    analyze_recipe,
    analyze_recipe_models,
    compute_recipe_file_hash,
    get_keys_to_process,
    walk_to_base,
)
from ..lib.block_classify import compute_changed_blocks, filter_changed_keys
from ..lib.executor import (
    check_ram_preflight,
    chunked_evaluation,
    compile_batch_groups,
    compile_plan,
    compute_batch_size,
    execute_plan,
    get_available_ram_bytes,
    streaming_evaluation_to_sink,
)
from ..lib.persistence import (
    build_metadata,
    check_full_model_cache,
    collect_block_configs,
    compute_base_identity,
    compute_lora_stats,
    compute_recipe_hash,
    compute_structural_fingerprint,
    serialize_recipe,
    validate_model_name,
)
from ..lib.recipe import (
    CheckpointComponents,
    RecipeBase,
    RecipeCompose,
    RecipeLoRA,
    RecipeMerge,
    RecipeModel,
    RecipeNode,
)
from ..lib.streaming_save import MaterializationSink
from ..lib.widen import WIDEN, WIDENConfig

try:
    from comfy.utils import ProgressBar
except ImportError:  # testing without ComfyUI
    ProgressBar = None  # type: ignore[assignment,misc]

if TYPE_CHECKING:
    from ..lib.recipe import BlockConfig

logger = logging.getLogger("ecaj.exit")


def _log_memory(label: str) -> None:
    """Log current RAM and VRAM usage.

    Uses /proc/self/status for RSS (Linux, no dependency) and
    torch.cuda for VRAM. Silently no-ops on non-Linux or errors.
    """
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss_kb = int(line.split()[1])
                    break
            else:
                rss_kb = 0
    except (OSError, ValueError):
        rss_kb = 0

    parts = [f"[mem] {label}: RSS={rss_kb // 1024}MB"]
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() // (1024 * 1024)
        reserved = torch.cuda.memory_reserved() // (1024 * 1024)
        parts.append(f"VRAM={alloc}MB(reserved={reserved}MB)")
    logger.info(" ".join(parts))


class _CacheEntry:
    """Single incremental recompute cache entry.

    AC: @incremental-block-recompute ac-1, ac-9, ac-17
    AC: @memory-management ac-6
    Stores the structural fingerprint, block configs, and merged state
    from a previous execution. Tensors are stored by reference (no clone)
    — downstream consumers (install_merged_patches, persistence) are
    read-only and must not mutate cached tensors.
    """

    __slots__ = (
        "structural_fingerprint", "block_configs", "merged_state",
        "storage_dtype", "loader_bytes",
    )

    def __init__(
        self,
        structural_fingerprint: str,
        block_configs: list[tuple[str, BlockConfig | None]],
        merged_state: dict[str, torch.Tensor],
        storage_dtype: torch.dtype,
        loader_bytes: int = 0,
    ) -> None:
        self.structural_fingerprint = structural_fingerprint
        self.block_configs = block_configs
        self.merged_state = merged_state
        self.storage_dtype = storage_dtype
        self.loader_bytes = loader_bytes


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


def validate_checkpoint_components(base: RecipeBase, save_model: bool) -> None:
    """Validate that checkpoint companion components are present for save_model.

    AC: @saved-model-artifact-safety ac-missing-components-fail-before-work
    AC: @saved-model-artifact-safety ac-missing-component-diagnostic-names-requirement
    AC: @saved-model-artifact-safety ac-missing-component-diagnostic-gives-guidance

    Must be called before analyze_recipe, model_state_dict, GPU merge work,
    cache publication, temp file creation, or artifact writing.

    Args:
        base: The RecipeBase at the root of the recipe tree.
        save_model: Whether save_model mode is enabled.

    Raises:
        ValueError: If save_model=True and CLIP or VAE components are missing,
            with diagnostic naming the missing requirement and connection guidance.
    """
    if not save_model:
        return

    cc = base.checkpoint_components
    missing: list[str] = []

    if cc is None or not isinstance(cc, CheckpointComponents):
        missing = ["CLIP", "VAE"]
    else:
        if cc.clip is None:
            missing.append("CLIP")
        if cc.vae is None:
            missing.append("VAE")

    if missing:
        missing_str = " and ".join(missing)
        raise ValueError(
            f"Checkpoint save_model requires {missing_str} components. "
            f"Connect CheckpointLoaderSimple MODEL, CLIP, and VAE outputs "
            f"to WIDEN Entry."
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

    # Build set patches (AC-2, AC-3, AC-4)
    # Keys already have diffusion_model. prefix.
    # AC: @memory-management ac-6 — .to() is a no-op (returns self) when
    # tensor is already CPU + storage_dtype; no memory duplication.
    patches = {}
    for key, tensor in merged_state.items():
        patches[key] = ("set", (tensor.to(device="cpu", dtype=storage_dtype),))

    # Install patches (AC-1)
    cloned.add_patches(patches, strength_patch=1.0)  # type: ignore[attr-defined]

    return cloned


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


def _build_model_resolver() -> Callable[[str, str], str | None]:
    """Build a model path resolver using ComfyUI's folder_paths.

    Returns a callable that resolves (model_name, source_dir) to full filesystem
    path by searching the appropriate ComfyUI directory.
    """
    import folder_paths

    def resolver(model_name: str, source_dir: str) -> str | None:
        # Map source_dir to ComfyUI folder name
        # "diffusion_models" may need "unet" fallback for older ComfyUI
        if source_dir == "diffusion_models":
            result = folder_paths.get_full_path("diffusion_models", model_name)
            if result is None:
                result = folder_paths.get_full_path("unet", model_name)
            return result
        return folder_paths.get_full_path(source_dir, model_name)

    return resolver


def _load_model_from_artifact(
    save_path: str,
    model_patcher: object,
    storage_dtype: torch.dtype,
) -> object:
    """Load a full saved model from the artifact and return a ModelPatcher.

    AC: @full-saved-model-output ac-return-loaded-model
    AC: @full-saved-model-output ac-cache-reuse-is-artifact-backed
    AC: @comfy-memory-manager-compatibility ac-comfy-owns-returned-model-memory

    Loads the artifact weights into the model's own state dict (not as set
    patches) so that ComfyUI's memory manager can offload/reload the model
    normally.  The returned clone owns an independent copy of the model
    whose weights come from the artifact — they are not held as set patches
    and do not remain resident solely because Exit returned.

    Args:
        save_path: Path to the full saved model artifact.
        model_patcher: Original ModelPatcher to clone.
        storage_dtype: Base model storage dtype.

    Returns:
        Cloned ModelPatcher whose model weights come from the artifact,
        loaded through the model's own state dict (Comfy-owned memory).
    """
    from copy import deepcopy

    from safetensors import safe_open

    # Clone the model patcher so patches are independent.
    cloned = model_patcher.clone()  # type: ignore[attr-defined]

    # Clear any inherited patches from the source ModelPatcher (e.g., LoRA
    # or control patches).  Full mode returns the saved artifact model —
    # inherited patch-resident tensors must not remain attached.
    if hasattr(cloned, "patches"):
        cloned.patches = {}  # type: ignore[attr-defined]

    # Deep-copy the underlying model so the clone owns its own weight
    # storage — the original model_patcher is not affected.
    cloned.model = deepcopy(cloned.model)  # type: ignore[attr-defined]

    # Give the clone its own _state_dict if present (clone() shares it).
    # Without this, updating _state_dict would mutate the original patcher.
    if hasattr(cloned, "_state_dict"):
        cloned._state_dict = dict(cloned._state_dict)  # type: ignore[attr-defined]

    # Load artifact weights directly into the clone's model state dict.
    # This makes the weights model-owned (Comfy-managed) rather than
    # patch-owned (always resident).
    # Preserve each tensor's artifact dtype — do NOT coerce to a single
    # global storage_dtype, which would destroy mixed-dtype artifacts.
    artifact_state: dict[str, torch.Tensor] = {}
    with safe_open(save_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            artifact_state[key] = f.get_tensor(key)

    # Update the clone's underlying model weights with artifact data.
    # Try load_state_dict (real nn.Module) first, then fall back to
    # updating the diffusion_model's internal state dict.
    _DIFFUSION_PREFIX = "diffusion_model."
    dm = getattr(cloned.model, "diffusion_model", None)  # type: ignore[attr-defined]
    if dm is not None and hasattr(dm, "load_state_dict"):
        # Real nn.Module — strip prefix and load via PyTorch API.
        unprefixed = {
            k.removeprefix(_DIFFUSION_PREFIX): v
            for k, v in artifact_state.items()
        }
        try:
            dm.load_state_dict(unprefixed, strict=False)
        except (TypeError, RuntimeError):
            # Fallback for models without full load_state_dict support.
            sd = dm.state_dict()
            for k, v in unprefixed.items():
                if k in sd:
                    sd[k].copy_(v)

    # Update the patcher's _state_dict if present (MockModelPatcher and
    # some real patchers use this as the backing store for model_state_dict).
    if hasattr(cloned, "_state_dict"):
        for k, v in artifact_state.items():
            if k in cloned._state_dict:  # type: ignore[attr-defined]
                cloned._state_dict[k] = v  # type: ignore[attr-defined]

    return cloned


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
                "enable_cache": ("BOOLEAN", {"default": True}),
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
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(
        cls,
        widen: RecipeNode,
        save_model: bool = False,
        model_name: str = "",
        save_workflow: bool = True,
        enable_cache: bool = True,
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
        base_hash = compute_recipe_file_hash(
            widen,
            lora_path_resolver=_build_lora_resolver(),
            model_path_resolver=_build_model_resolver(),
        )

        if not save_model and enable_cache:
            return base_hash

        # Include save parameters, cache toggle, and cached file state
        hasher = hashlib.sha256(base_hash.encode())
        hasher.update(
            f"|save={save_model}|name={model_name}"
            f"|wf={save_workflow}|cache={enable_cache}".encode()
        )
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
        enable_cache: bool = True,
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
        AC: @streaming-full-model-materialization ac-direct-artifact-handoff
        AC: @full-saved-model-output ac-cache-reuse-is-artifact-backed

        Args:
            widen: Recipe tree root (should be RecipeMerge or RecipeBase)
            save_model: Whether to produce full saved model output
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

        # AC: @saved-model-artifact-safety ac-missing-components-fail-before-work
        # Validate checkpoint components before any expensive work
        if save_model:
            base = widen if isinstance(widen, RecipeBase) else walk_to_base(widen)
            validate_checkpoint_components(base, save_model=True)

        # Quick check: must end in RecipeMerge for actual merging
        if isinstance(widen, RecipeBase):
            if save_model:
                # AC: @full-saved-model-output ac-no-op-produces-full-artifact
                # No-op recipe in full mode: produce a full artifact with all base weights
                return self._execute_full_model_noop(
                    widen, model_name, save_workflow, enable_cache,
                    extra_pnginfo,
                )
            return (widen.model_patcher.clone(),)  # type: ignore[attr-defined]

        if not isinstance(widen, RecipeMerge):
            raise ValueError(
                f"Exit node expects RecipeMerge or RecipeBase at root, "
                f"got {type(widen).__name__}. Connect a Merge node to Exit."
            )

        if save_model:
            return self._execute_full_saved_model(
                widen, model_name, save_workflow, enable_cache,
                extra_pnginfo,
            )
        else:
            return self._execute_patch_mode(
                widen, enable_cache, save_model=False,
            )

    def _execute_full_model_noop(
        self,
        widen: RecipeBase,
        model_name: str,
        save_workflow: bool,
        enable_cache: bool,
        extra_pnginfo: object,
    ) -> tuple[object]:
        """Handle no-op recipe (RecipeBase) in full saved model mode.

        AC: @full-saved-model-output ac-no-op-produces-full-artifact
        AC: @full-saved-model-output ac-complete-artifact
        """
        model_patcher = widen.model_patcher
        _unpatch_loaded_clones(model_patcher)
        base_state = model_patcher.model_state_dict()  # type: ignore[attr-defined]
        storage_dtype = next(iter(base_state.values())).dtype

        lora_path_resolver = _build_lora_resolver()
        model_path_resolver = _build_model_resolver()
        base_identity = compute_base_identity(base_state)
        lora_stats = compute_lora_stats(widen, lora_path_resolver, model_path_resolver)

        validated_name = validate_model_name(model_name)
        save_path = _resolve_checkpoints_path(validated_name)
        serialized = serialize_recipe(widen, base_identity, lora_stats)
        recipe_hash = compute_recipe_hash(serialized)

        # Build manifest from base_state — all keys, no affected keys
        manifest = {k: (v.dtype, tuple(v.shape)) for k, v in base_state.items()}

        # AC: @full-saved-model-output ac-cache-reuses-artifact
        if enable_cache and check_full_model_cache(
            save_path, recipe_hash, expected_manifest=manifest,
        ):
            if ProgressBar is not None:
                pbar = ProgressBar(1)
                pbar.update(1)
            return (_load_model_from_artifact(save_path, model_patcher, storage_dtype),)

        # Evict in-memory cache when cache is disabled.
        # Full mode does not write to _incremental_cache, but pre-populated
        # entries from earlier patch-mode runs must be cleared.
        if not enable_cache:
            _incremental_cache.clear()
        workflow_json = (
            json.dumps(extra_pnginfo) if save_workflow and extra_pnginfo else None
        )
        metadata = build_metadata(
            serialized, recipe_hash, [], workflow_json, output_mode="full",
        )

        sink = MaterializationSink()
        try:
            sink.open(manifest, save_path, metadata)
            for name in sorted(base_state.keys()):
                sink.write_tensor(name, base_state[name])
            sink.finalize(save_path)
        except BaseException:
            sink.abort()
            raise

        if ProgressBar is not None:
            pbar = ProgressBar(1)
            pbar.update(1)

        return (_load_model_from_artifact(save_path, model_patcher, storage_dtype),)

    def _execute_full_saved_model(
        self,
        widen: RecipeMerge,
        model_name: str,
        save_workflow: bool,
        enable_cache: bool,
        extra_pnginfo: object,
    ) -> tuple[object]:
        """Execute in full saved model mode — stream to artifact, return loaded model.

        AC: @streaming-full-model-materialization ac-direct-artifact-handoff
        AC: @streaming-full-model-materialization ac-affected-results-released
        AC: @streaming-full-model-materialization ac-base-weight-bounded-copying
        AC: @streaming-full-model-materialization ac-incomplete-write-not-reused
        AC: @streaming-full-model-materialization ac-full-cache-avoids-resident-payload
        AC: @streaming-full-model-materialization
            ac-failed-materialization-releases-resident-payload
        AC: @full-saved-model-output ac-complete-artifact
        AC: @full-saved-model-output ac-return-loaded-model
        AC: @full-saved-model-output ac-cache-reuses-artifact
        AC: @full-saved-model-output ac-cache-reuse-is-artifact-backed
        AC: @comfy-memory-manager-compatibility ac-no-dynamic-vram-opt-out
        AC: @comfy-memory-manager-compatibility ac-memory-mode-preserved
        """
        lora_path_resolver = _build_lora_resolver()
        model_path_resolver = _build_model_resolver()

        model_patcher = walk_to_base(widen).model_patcher
        _unpatch_loaded_clones(model_patcher)
        base_state = model_patcher.model_state_dict()  # type: ignore[attr-defined]
        storage_dtype = next(iter(base_state.values())).dtype

        key_shapes = {k: tuple(v.shape) for k, v in base_state.items()}

        base_identity = compute_base_identity(base_state)
        lora_stats = compute_lora_stats(widen, lora_path_resolver, model_path_resolver)

        validated_name = validate_model_name(model_name)
        save_path = _resolve_checkpoints_path(validated_name)
        serialized = serialize_recipe(widen, base_identity, lora_stats)
        recipe_hash = compute_recipe_hash(serialized)

        # Build manifest early for cache validation (shapes + dtypes, not just key names).
        base_manifest = {
            k: (v.dtype, tuple(v.shape)) for k, v in base_state.items()
        }

        # AC: @full-saved-model-output ac-cache-reuses-artifact
        # AC: @full-saved-model-output ac-cache-reuse-is-artifact-backed
        # Full-mode cache: validate artifact metadata, load from artifact, no tensor payload
        if enable_cache and check_full_model_cache(
            save_path, recipe_hash, expected_manifest=base_manifest,
        ):
            del base_state
            if ProgressBar is not None:
                pbar = ProgressBar(1)
                pbar.update(1)
            return (_load_model_from_artifact(save_path, model_patcher, storage_dtype),)

        # --- GPU pipeline for full saved model mode ---
        analysis = analyze_recipe(widen, lora_path_resolver=lora_path_resolver)

        base = walk_to_base(widen)
        domain = getattr(base, "domain", "diffusion")
        model_analysis = analyze_recipe_models(
            widen, base.arch, model_path_resolver=model_path_resolver, domain=domain
        )

        sink = MaterializationSink()
        try:
            loader = analysis.loader
            set_affected = analysis.set_affected
            lora_affected_keys = analysis.affected_keys
            arch = analysis.arch

            model_affected = model_analysis.model_affected
            model_loaders = model_analysis.model_loaders
            all_model_keys = model_analysis.all_model_keys

            compute_dtype = torch.float32
            device = "cuda" if torch.cuda.is_available() else "cpu"

            all_keys = set(base_state.keys())
            lora_keys = get_keys_to_process(all_keys, lora_affected_keys)
            model_keys = all_keys & all_model_keys
            keys_to_process = lora_keys | model_keys

            # AC: @full-saved-model-output ac-no-op-produces-full-artifact
            # Even with no affected keys, produce full artifact
            affected_key_set = keys_to_process

            set_id_map: dict[int, str] = {}
            for set_key, affected in set_affected.items():
                set_id = int(set_key)
                set_id_map[set_id] = set_key

            model_id_map: dict[int, str] = {}
            for model_key in model_affected.keys():
                model_id = int(model_key)
                model_id_map[model_id] = model_key

            widen_config = WIDENConfig(
                t_factor=widen.t_factor,
                dtype=compute_dtype,
            )
            widen_merger = WIDEN(widen_config)

            plan = compile_plan(widen, set_id_map, arch, model_id_map)

            # Build manifest for entire model (base + affected keys with final dtypes/shapes)
            manifest: dict[str, tuple[torch.dtype, tuple[int, ...]]] = {}
            for k, v in base_state.items():
                manifest[k] = (v.dtype, tuple(v.shape))

            workflow_json = (
                json.dumps(extra_pnginfo) if save_workflow and extra_pnginfo else None
            )
            metadata = build_metadata(
                serialized, recipe_hash, sorted(affected_key_set), workflow_json,
                output_mode="full",
            )

            # AC: @streaming-full-model-materialization ac-base-weight-bounded-copying
            # Open sink and write header
            sink.open(manifest, save_path, metadata)

            # Pre-flight RAM check (reduced estimate — no merged_state accumulation)
            if keys_to_process:
                batch_groups = compile_batch_groups(
                    list(keys_to_process), arch=arch, key_shapes=key_shapes,
                )
            else:
                batch_groups = {}

            if batch_groups:
                n_models = len(set_affected) + len(model_loaders)
                storage_element_size = torch.finfo(storage_dtype).bits // 8
                worst_chunk_bytes = max(
                    storage_element_size
                    * torch.Size(sig.shape).numel()
                    * compute_batch_size(sig.shape, n_models, compute_dtype)
                    for sig in batch_groups
                )
                # Full mode doesn't accumulate merged_state in RAM — tensors go to sink
                check_ram_preflight(
                    merged_state_bytes=worst_chunk_bytes,
                    worst_chunk_bytes=worst_chunk_bytes,
                    save_model=True,
                    loader_bytes=loader.loaded_bytes + sum(
                        ml.loaded_bytes for ml in model_loaders.values()
                    ),
                )

            # AC: @streaming-full-model-materialization ac-direct-artifact-handoff
            # AC: @streaming-full-model-materialization ac-affected-results-released
            # AC: @streaming-full-model-materialization ac-base-weight-bounded-copying
            # Write base weights first, then evaluate and write groups one at a
            # time.  The sink supports random-access writes (seek to pre-computed
            # offsets), so groups can be written in evaluation order regardless of
            # how their keys interleave in sorted name order.  Only one group's
            # results are ever resident — each is freed before the next is
            # evaluated.
            _log_memory("before-gpu-eval-full")

            # Build a set of all affected keys for O(1) lookup
            affected_key_set_lookup: set[str] = set()
            for group_keys in batch_groups.values():
                affected_key_set_lookup.update(group_keys)

            # AC: @streaming-full-model-materialization ac-base-weight-bounded-copying
            # Write unaffected base weights to sink (one at a time, no full copy)
            for key in manifest:
                if key not in affected_key_set_lookup:
                    sink.write_tensor(key, base_state[key])

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
                arch, widen_config, model_loaders, domain,
            )

            pbar_count = len(batch_groups) if batch_groups else 0
            pbar = ProgressBar(pbar_count) if ProgressBar is not None and pbar_count else None

            # AC: @streaming-full-model-materialization ac-direct-artifact-handoff
            # AC: @streaming-full-model-materialization ac-affected-results-released
            # Evaluate and stream one group at a time.  Within each group,
            # streaming_evaluation_to_sink hands every completed chunk's
            # tensors to sink.write_tensor immediately — no dict accumulates
            # the full group's results.  Each tensor can be freed as soon as
            # write_tensor returns.
            for sig, group_keys in batch_groups.items():
                n_models = len(set_affected) + len(model_loaders)
                batch_size = compute_batch_size(
                    sig.shape, n_models, compute_dtype,
                )
                group_base = {k: base_state[k] for k in group_keys}
                streaming_evaluation_to_sink(
                    keys=group_keys,
                    base_tensors=group_base,
                    eval_fn=eval_fn,
                    batch_size=batch_size,
                    device=device,
                    dtype=compute_dtype,
                    storage_dtype=None,
                    write_fn=sink.write_tensor,
                )

                # Free this group's base tensors before evaluating next group.
                del group_base
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if pbar is not None:
                    pbar.update(1)

            _log_memory("after-streaming-full")

            # Free base_state before finalize
            del base_state

            sink.finalize(save_path)
            _log_memory("after-finalize-full")

            # Offload GPU models
            try:
                from comfy.model_management import (
                    free_memory,
                    get_torch_device,
                    soft_empty_cache,
                )
                free_memory(1e30, get_torch_device())
                soft_empty_cache()
            except (ImportError, AttributeError):
                pass

            # AC: @streaming-full-model-materialization ac-full-cache-avoids-resident-payload
            # Full mode does NOT store affected tensor payload in _incremental_cache.
            # Cache reuse is artifact-backed (check_full_model_cache).
            if not enable_cache:
                _incremental_cache.clear()

        except BaseException:
            # AC: @streaming-full-model-materialization ac-incomplete-write-not-reused
            # AC: @streaming-full-model-materialization
            #     ac-failed-materialization-releases-resident-payload
            sink.abort()
            raise
        finally:
            loader.cleanup()
            for model_loader in model_analysis.model_loaders.values():
                model_loader.cleanup()

        # AC: @full-saved-model-output ac-return-loaded-model
        # Return MODEL loaded from the saved artifact
        return (_load_model_from_artifact(save_path, model_patcher, storage_dtype),)

    def _execute_patch_mode(
        self,
        widen: RecipeMerge,
        enable_cache: bool,
        save_model: bool = False,
    ) -> tuple[object]:
        """Execute in patch mode — dict-returning evaluation, set-patch installation.

        This preserves the original patch-mode behavior unchanged.
        """
        lora_path_resolver = _build_lora_resolver()
        model_path_resolver = _build_model_resolver()

        model_patcher = walk_to_base(widen).model_patcher
        _unpatch_loaded_clones(model_patcher)
        base_state = model_patcher.model_state_dict()  # type: ignore[attr-defined]
        storage_dtype = next(iter(base_state.values())).dtype

        key_shapes = {k: tuple(v.shape) for k, v in base_state.items()}
        key_byte_sizes = {k: v.nelement() * v.element_size() for k, v in base_state.items()}

        base_identity = compute_base_identity(base_state)
        lora_stats = compute_lora_stats(widen, lora_path_resolver, model_path_resolver)

        # --- Normal GPU pipeline ---
        analysis = analyze_recipe(widen, lora_path_resolver=lora_path_resolver)

        base = walk_to_base(widen)
        domain = getattr(base, "domain", "diffusion")
        model_analysis = analyze_recipe_models(
            widen, base.arch, model_path_resolver=model_path_resolver, domain=domain
        )

        try:
            loader = analysis.loader
            set_affected = analysis.set_affected
            lora_affected_keys = analysis.affected_keys
            arch = analysis.arch

            model_affected = model_analysis.model_affected
            model_loaders = model_analysis.model_loaders
            all_model_keys = model_analysis.all_model_keys

            compute_dtype = torch.float32
            device = "cuda" if torch.cuda.is_available() else "cpu"

            all_keys = set(base_state.keys())
            lora_keys = get_keys_to_process(all_keys, lora_affected_keys)
            model_keys = all_keys & all_model_keys
            keys_to_process = lora_keys | model_keys

            if not keys_to_process:
                return (model_patcher.clone(),)  # type: ignore[attr-defined]

            set_id_map: dict[int, str] = {}
            for set_key, affected in set_affected.items():
                set_id = int(set_key)
                set_id_map[set_id] = set_key

            model_id_map: dict[int, str] = {}
            for model_key in model_affected.keys():
                model_id = int(model_key)
                model_id_map[model_id] = model_key

            widen_config = WIDENConfig(
                t_factor=widen.t_factor,
                dtype=compute_dtype,
            )
            widen_merger = WIDEN(widen_config)

            batch_groups = compile_batch_groups(
                list(keys_to_process), arch=arch, key_shapes=key_shapes,
            )

            plan = compile_plan(widen, set_id_map, arch, model_id_map)

            # --- Incremental cache: detect which blocks changed ---
            structural_fp = compute_structural_fingerprint(
                widen, base_identity, lora_stats
            )
            current_block_configs = collect_block_configs(widen)
            current_loader_bytes = loader.loaded_bytes + sum(
                ml.loaded_bytes for ml in model_loaders.values()
            )
            cached_entry = _incremental_cache.get(structural_fp) if enable_cache else None
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
                        merged_state = {
                            k: v for k, v in cached_entry.merged_state.items()
                        }
                        incremental_hit = True
                        batch_groups = {}

                        if ProgressBar is not None:
                            pbar = ProgressBar(1)
                            pbar.update(1)
                    else:
                        recompute_keys = filter_changed_keys(
                            keys_to_process, changed_blocks,
                            changed_layer_types, arch,
                        )

                        if not recompute_keys:
                            merged_state = {
                                k: v for k, v in cached_entry.merged_state.items()
                            }
                            incremental_hit = True
                            batch_groups = {}
                        else:
                            merged_state = {
                                k: v for k, v in cached_entry.merged_state.items()
                            }
                            batch_groups = compile_batch_groups(
                                list(recompute_keys), arch=arch,
                                key_shapes=key_shapes,
                            )
                            incremental_hit = True

            if incremental_hit and cached_entry is not None:
                cached_lb = cached_entry.loader_bytes
                logger.info(
                    "Incremental cache hit: cached loader_bytes=%d bytes (%.1f MB)",
                    cached_lb,
                    cached_lb / (1024 * 1024),
                )

            if not incremental_hit:
                merged_state = {}

            if batch_groups:
                processed_keys = {k for keys in batch_groups.values() for k in keys}
                merged_state_bytes = sum(
                    key_byte_sizes[k] for k in processed_keys
                )
                n_models = len(set_affected) + len(model_loaders)
                storage_element_size = torch.finfo(storage_dtype).bits // 8
                worst_chunk_bytes = max(
                    storage_element_size
                    * torch.Size(sig.shape).numel()
                    * compute_batch_size(sig.shape, n_models, compute_dtype)
                    for sig in batch_groups
                )
                check_ram_preflight(
                    merged_state_bytes=merged_state_bytes,
                    worst_chunk_bytes=worst_chunk_bytes,
                    save_model=save_model,
                    loader_bytes=current_loader_bytes,
                )

            _log_memory("before-gpu-eval")
            if batch_groups:
                pbar_count = len(batch_groups)
                pbar = ProgressBar(pbar_count) if ProgressBar is not None else None

                for sig, group_keys in batch_groups.items():
                    n_models = len(set_affected) + len(model_loaders)
                    batch_size = compute_batch_size(
                        sig.shape, n_models, compute_dtype,
                    )

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
                        arch, widen_config, model_loaders, domain,
                    )

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

                    if pbar is not None:
                        pbar.update(1)

            _log_memory("after-gpu-eval")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            del base_state

            # AC: @incremental-block-recompute ac-1, ac-16, ac-17, ac-18
            # AC: @memory-management ac-6
            # Patch-mode incremental cache storage
            if not enable_cache:
                _incremental_cache.clear()
            elif batch_groups or not incremental_hit:
                avail = get_available_ram_bytes()
                _CACHE_EVICTION_MARGIN = 512 * 1024 * 1024  # 512 MB
                if avail < _CACHE_EVICTION_MARGIN:
                    logger.info(
                        "Cache eviction: avail=%d MB < safety margin=%d MB; "
                        "evicting instead of storing",
                        avail // (1024 * 1024),
                        _CACHE_EVICTION_MARGIN // (1024 * 1024),
                    )
                    _incremental_cache.clear()
                else:
                    new_entry = _CacheEntry(
                        structural_fingerprint=structural_fp,
                        block_configs=current_block_configs,
                        merged_state=dict(merged_state),
                        storage_dtype=storage_dtype,
                        loader_bytes=current_loader_bytes,
                    )
                    cache_bytes = sum(
                        t.nelement() * t.element_size()
                        for t in merged_state.values()
                    )
                    logger.info(
                        "Cache stored: %d MB merged_state, avail=%d MB",
                        cache_bytes // (1024 * 1024),
                        avail // (1024 * 1024),
                    )
                    _incremental_cache.clear()
                    _incremental_cache[structural_fp] = new_entry
            _log_memory("before-cache-write")

        finally:
            loader.cleanup()
            for model_loader in model_analysis.model_loaders.values():
                model_loader.cleanup()

        result = install_merged_patches(model_patcher, merged_state, storage_dtype)
        return (result,)
