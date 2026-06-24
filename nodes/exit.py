"""WIDEN Exit Node — Executes the recipe tree, returns ComfyUI MODEL."""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import secrets
import time
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
    check_checkpoint_cache,
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
from ..lib.save_progress import SavedModelProgress
from ..lib.streaming_save import MaterializationSink
from ..lib.widen import WIDEN, WIDENConfig

try:
    from comfy.utils import ProgressBar
except ImportError:  # testing without ComfyUI
    ProgressBar = None  # type: ignore[assignment,misc]

if TYPE_CHECKING:
    from ..lib.recipe import BlockConfig

logger = logging.getLogger("ecaj.exit")


class _PhaseTimer:
    """Small logging helper for long-running WIDEN Exit phases."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.last = time.monotonic()
        logger.info("WIDEN Exit phase start: %s", label)

    def mark(self, phase: str) -> None:
        now = time.monotonic()
        logger.info(
            "WIDEN Exit phase: %s: %s after %.3fs",
            self.label,
            phase,
            now - self.last,
        )
        self.last = now

# Internal merged-key prefix produced by ComfyUI BaseModel.state_dict():
# `diffusion_model.X`.  ComfyUI's standalone diffusion-model loader
# (comfy.sd.load_diffusion_model_state_dict) only recognizes one of the
# `model.diffusion_model.`, `model.model.`, or `net.` prefixes, so saved
# diffusion-model artifacts are written under `model.diffusion_model.X`
# (matching what comfy_extras.nodes_model_merging.ModelSave produces via
# BaseModel.process_unet_state_dict_for_saving).
_INTERNAL_DIFFUSION_PREFIX = "diffusion_model."
_EXTERNAL_DIFFUSION_PREFIX = "model.diffusion_model."

# Source-model-kind metadata values
_SOURCE_KIND_DIFFUSION_MODEL = "diffusion_model"
_SOURCE_KIND_CHECKPOINT = "checkpoint"


def _to_external_diffusion_key(key: str) -> str:
    """Map an internal merged key to the external diffusion-model key layout.

    `diffusion_model.X` -> `model.diffusion_model.X`.  Already-external keys
    and unrelated keys (e.g. companion components) are returned unchanged.
    """
    if key.startswith(_EXTERNAL_DIFFUSION_PREFIX):
        return key
    if key.startswith(_INTERNAL_DIFFUSION_PREFIX):
        return _EXTERNAL_DIFFUSION_PREFIX + key[len(_INTERNAL_DIFFUSION_PREFIX) :]
    return key


def _comfy_load_diffusion_model(save_path: str, model_options: dict | None = None):
    """Indirection over comfy.sd.load_diffusion_model so tests can patch it."""
    import comfy.sd

    return comfy.sd.load_diffusion_model(
        save_path,
        model_options=model_options or {},
    )


def _build_save_progress(
    *,
    manifest_size: int,
    artifact_name: str,
) -> SavedModelProgress:
    """Construct a SavedModelProgress for a diffusion-only save.

    AC: @streaming-materialization-progress ac-progress-during-streaming-writes
    AC: @streaming-materialization-progress ac-no-op-save-progress
    AC: @streaming-materialization-progress ac-affected-write-progress
    AC: @streaming-materialization-progress ac-finalization-status-visible

    Total units = manifest_size (one per tensor write) + 3 phase units for
    prepare, finalize, and reload.  ProgressBar is looked up off the module
    namespace so tests can patch ``nodes.exit.ProgressBar`` to a recording
    fake or to ``None`` to verify the no-ComfyUI path.
    """
    factory = ProgressBar if ProgressBar is not None else None
    return SavedModelProgress(
        total_units=manifest_size + 3,
        progress_bar_factory=factory,
        artifact_name=artifact_name,
    )


def _build_cache_reuse_progress(*, artifact_name: str) -> SavedModelProgress:
    """Construct a 1-unit SavedModelProgress for cache-reuse status reporting.

    AC: @streaming-materialization-progress ac-cache-reuse-status-visible

    Cache hits never write tensors, so the progress only carries the
    ``cache_reuse`` phase plus a single advance to fully complete the bar.
    """
    factory = ProgressBar if ProgressBar is not None else None
    return SavedModelProgress(
        total_units=1,
        progress_bar_factory=factory,
        artifact_name=artifact_name,
    )


def _load_diffusion_model_artifact(save_path: str) -> object:
    """Load a saved diffusion-model artifact through Comfy's supported loader.

    AC: @full-saved-model-output ac-diffusion-model-source-kind-round-trip
    AC: @full-saved-model-output ac-return-loaded-model
    AC: @full-saved-model-output ac-cache-reuse-is-artifact-backed
    AC: @comfy-memory-manager-compatibility ac-comfy-owns-returned-model-memory

    Calls comfy.sd.load_diffusion_model so the returned MODEL is Comfy-owned
    and behaves like a normally loaded standalone diffusion model — no
    deepcopy of WIDEN merge internals, no transient process-resident payload.
    """
    try:
        result = _comfy_load_diffusion_model(save_path, model_options={})
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load diffusion-model artifact: {save_path} — {exc}"
        ) from exc
    if result is None:
        raise RuntimeError(f"Comfy diffusion-model loader returned None for: {save_path}")
    return result


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
        "structural_fingerprint",
        "block_configs",
        "merged_state",
        "storage_dtype",
        "loader_bytes",
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
        raise ValueError(f"Unknown recipe node type at {path}: {type(node).__name__}")


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
    #
    # Checkpoint artifacts from comfy.sd.save_checkpoint use Comfy-style
    # key prefixes: model.diffusion_model.*, conditioner.*, first_stage_model.*.
    # Internal-format artifacts use bare diffusion_model.* keys.
    # We need to extract diffusion weights from either format.
    _DIFFUSION_PREFIX = "diffusion_model."
    _COMFY_MODEL_PREFIX = "model.diffusion_model."

    # Extract diffusion weights: try Comfy checkpoint prefix first, then
    # internal diffusion_model.* prefix.  Strip to bare weight names for
    # load_state_dict.
    diffusion_weights: dict[str, torch.Tensor] = {}
    for k, v in artifact_state.items():
        if k.startswith(_COMFY_MODEL_PREFIX):
            diffusion_weights[k.removeprefix(_COMFY_MODEL_PREFIX)] = v
        elif k.startswith(_DIFFUSION_PREFIX):
            diffusion_weights[k.removeprefix(_DIFFUSION_PREFIX)] = v

    dm = getattr(cloned.model, "diffusion_model", None)  # type: ignore[attr-defined]
    if dm is not None and hasattr(dm, "load_state_dict") and diffusion_weights:
        # Real nn.Module — load via PyTorch API.
        try:
            dm.load_state_dict(diffusion_weights, strict=False)
        except (TypeError, RuntimeError):
            # Fallback for models without full load_state_dict support.
            sd = dm.state_dict()
            for k, v in diffusion_weights.items():
                if k in sd:
                    sd[k].copy_(v)

    # Update the patcher's _state_dict if present (MockModelPatcher and
    # some real patchers use this as the backing store for model_state_dict).
    # Map checkpoint-style keys back to the patcher's diffusion_model.* keys.
    if hasattr(cloned, "_state_dict"):
        for k, v in artifact_state.items():
            if k in cloned._state_dict:  # type: ignore[attr-defined]
                cloned._state_dict[k] = v  # type: ignore[attr-defined]
            elif k.startswith(_COMFY_MODEL_PREFIX):
                # Map model.diffusion_model.X → diffusion_model.X
                patcher_key = _DIFFUSION_PREFIX + k.removeprefix(_COMFY_MODEL_PREFIX)
                if patcher_key in cloned._state_dict:  # type: ignore[attr-defined]
                    cloned._state_dict[patcher_key] = v  # type: ignore[attr-defined]

    return cloned


def _trim_native_heap() -> None:
    """Ask the platform allocator to return freed native CPU arenas to the OS.

    Large torch CPU tensors are native allocations.  After Python refs are
    dropped and GC runs, glibc can still keep those arenas committed inside the
    process; under memory pressure Linux may move them to swap, which looks like
    repeated save_model=true CPU growth even when RSS falls.  malloc_trim is a
    best-effort Linux/glibc release hook and is intentionally a no-op elsewhere.
    """
    try:
        import ctypes

        libc = ctypes.CDLL(None)
        malloc_trim = getattr(libc, "malloc_trim", None)
        if malloc_trim is None:
            return
        malloc_trim.argtypes = [ctypes.c_size_t]
        malloc_trim.restype = ctypes.c_int
        malloc_trim(0)
    except Exception as exc:  # pragma: no cover - platform best effort
        logger.debug("native heap trim skipped: %s", exc)


def _clear_temporary_model_patch_payloads(temporary_model: object) -> None:
    """Sever patch-payload references held by a temporary ModelPatcher clone."""
    for attr in ("patches", "object_patches", "weight_wrapper_patches"):
        payload = getattr(temporary_model, attr, None)
        if hasattr(payload, "clear"):
            payload.clear()


def _release_temporary_checkpoint_model(temporary_model: object) -> None:
    """Release temporary checkpoint-save clones before returning loaded artifacts.

    AC: @comfy-memory-manager-compatibility ac-comfy-owns-returned-model-memory

    Checkpoint saves use a cloned ModelPatcher with dense set-patch tensors as a
    serialization vehicle.  Once the checkpoint is written, that clone must not
    remain in Comfy's loaded-model registry or WIDEN's return value; otherwise
    Comfy's output cache can keep the transient merge payload resident.
    """
    _unpatch_loaded_clones(temporary_model)
    _clear_temporary_model_patch_payloads(temporary_model)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _trim_native_heap()


def _load_checkpoint_artifact(save_path: str) -> object:
    """Load a checkpoint artifact through Comfy's supported checkpoint load path.

    AC: @checkpoint-loadable-saved-model-output ac-generated-workflow-round-trip
    AC: @checkpoint-loadable-saved-model-output ac-downstream-return-remains-usable
    AC: @comfy-memory-manager-compatibility ac-comfy-owns-returned-model-memory

    Uses comfy.sd.load_checkpoint_guess_config to load the saved checkpoint
    artifact, returning only the MODEL component.  The returned MODEL is
    Comfy-owned and compatible with Comfy's model memory lifecycle — no
    deep-copied model internals, no transient WIDEN merge payload required.

    Args:
        save_path: Path to the saved checkpoint artifact.

    Returns:
        Comfy ModelPatcher loaded from the checkpoint artifact.

    Raises:
        RuntimeError: If the Comfy loader fails, with a message naming the
            checkpoint path for diagnostics.
    """
    import comfy.sd

    try:
        result = comfy.sd.load_checkpoint_guess_config(
            save_path,
            output_vae=True,
            output_clip=True,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to load checkpoint artifact: {save_path} — {exc}") from exc

    # load_checkpoint_guess_config returns [model, clip, vae, ...].
    # Return only the MODEL for WIDEN Exit's MODEL output.
    model = result[0]
    if model is None:
        raise RuntimeError(f"Comfy checkpoint loader returned None MODEL for: {save_path}")
    return model


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


def _resolve_diffusion_models_path(model_name: str) -> str:
    """Resolve a model name to a full path in the first diffusion-models directory.

    AC: @full-saved-model-output ac-diffusion-model-source-kind-round-trip
    AC: @full-saved-model-output ac-diffusion-model-companion-separation

    Standalone diffusion-model saves must publish under Comfy's
    ``diffusion_models`` folder so Comfy's UNETLoader (which calls
    ``folder_paths.get_full_path_or_raise('diffusion_models', ...)``) can
    discover them.  Older ComfyUI installs use the legacy ``unet`` folder
    name for the same content; we fall back to it when ``diffusion_models``
    is not configured.

    Args:
        model_name: Validated model filename.

    Returns:
        Full path to the saved diffusion-model file.

    Raises:
        ValueError: If neither ``diffusion_models`` nor ``unet`` directories
            are configured.
    """
    import folder_paths

    dirs = folder_paths.get_folder_paths("diffusion_models")
    if not dirs:
        dirs = folder_paths.get_folder_paths("unet")
    if not dirs:
        raise ValueError("No diffusion_models (or unet) directory configured in ComfyUI")
    return os.path.join(dirs[0], model_name)


def _resolve_save_path(model_name: str, *, is_checkpoint: bool) -> str:
    """Resolve the save path for a saved-model artifact based on its kind.

    AC: @full-saved-model-output ac-diffusion-model-source-kind-round-trip
    AC: @checkpoint-loadable-saved-model-output ac-artifact-matches-source-model-kind

    Checkpoint-style artifacts publish under Comfy's ``checkpoints`` folder
    (CheckpointLoaderSimple discovery).  Standalone diffusion-model
    artifacts publish under Comfy's ``diffusion_models`` folder
    (UNETLoader discovery).  Routing the path by artifact kind ensures
    the saved file is reachable through the same Comfy load contract as
    its source model kind, which is required for round-trip loadability.
    """
    if is_checkpoint:
        return _resolve_checkpoints_path(model_name)
    return _resolve_diffusion_models_path(model_name)


def _recipe_has_checkpoint_components(node: RecipeNode) -> bool:
    """Check if recipe tree originates from a checkpoint-style source.

    A recipe is checkpoint-style when:
    - The RecipeBase at the root carries checkpoint_components (CLIP/VAE from
      CheckpointLoaderSimple), OR
    - The tree contains RecipeModel nodes with source_dir="checkpoints".

    RecipeModel nodes with source_dir="diffusion_models" are diffusion-only
    model inputs and do not make a recipe checkpoint-style.
    """
    if isinstance(node, RecipeBase):
        cc = node.checkpoint_components
        return (
            cc is not None
            and isinstance(cc, CheckpointComponents)
            and cc.clip is not None
            and cc.vae is not None
        )
    if isinstance(node, RecipeModel):
        return node.source_dir == "checkpoints"
    if isinstance(node, RecipeCompose):
        return any(_recipe_has_checkpoint_components(b) for b in node.branches)
    if isinstance(node, RecipeMerge):
        if _recipe_has_checkpoint_components(node.target):
            return True
        if node.backbone is not None and _recipe_has_checkpoint_components(node.backbone):
            return True
        return _recipe_has_checkpoint_components(node.base)
    return False


def _guard_non_ecaj_overwrite(save_path: str) -> None:
    """Refuse to overwrite a non-ecaj file at save_path.

    AC: @saved-model-artifact-safety ac-existing-valid-artifact-preserved

    Checks the safetensors header metadata. If the file exists and lacks
    ``__ecaj_version__`` metadata, raises ``ValueError`` to prevent
    overwriting user files that were not produced by ecaj.

    This is the same guard that ``check_checkpoint_cache`` and
    ``check_full_model_cache`` provide, extracted so it runs
    unconditionally — even when ``enable_cache=False``.
    """
    if not os.path.exists(save_path):
        return

    from safetensors import safe_open

    try:
        with safe_open(save_path, framework="pt") as f:
            file_metadata = f.metadata()
    except Exception as exc:
        raise ValueError(
            f"File exists but is not a valid safetensors file: {save_path}\n"
            f"Refusing to overwrite. Choose a different model_name.\n"
            f"Underlying error: {exc}"
        ) from exc

    if file_metadata is None or "__ecaj_version__" not in file_metadata:
        raise ValueError(
            f"File exists but is not an ecaj-saved model: {save_path}\n"
            f"Refusing to overwrite a file without ecaj metadata. "
            f"Choose a different model_name."
        )


def _classify_temp_artifact(tmp_path: str, expected_kind: str) -> None:
    """Verify the temp artifact has correct ecaj metadata before publication.

    AC: @saved-model-artifact-safety ac-no-partial-publication

    Reads the safetensors header and confirms:
    - ``__ecaj_version__`` is present
    - ``__ecaj_artifact_kind__`` matches ``expected_kind``
    - For checkpoint artifacts: all required component prefixes are present
    - For diffusion artifacts: no internal-format ``diffusion_model.*`` keys
      (every diffusion key must use the external ``model.diffusion_model.*``
      layout that Comfy's diffusion-model loader recognizes).

    Raises ``RuntimeError`` if classification fails, so the caller can
    clean up the temp file without publishing it.
    """
    from safetensors import safe_open

    from ..lib.persistence import _has_checkpoint_component_prefixes

    with safe_open(tmp_path, framework="pt") as f:
        file_metadata = f.metadata()
        tensor_keys = set(f.keys())

    if file_metadata is None or "__ecaj_version__" not in file_metadata:
        raise RuntimeError(
            f"Temp artifact missing ecaj metadata — refusing to publish: {tmp_path}"
        )

    stored_kind = file_metadata.get("__ecaj_artifact_kind__", "")
    if stored_kind != expected_kind:
        raise RuntimeError(
            f"Temp artifact kind {stored_kind!r} != expected {expected_kind!r} "
            f"— refusing to publish: {tmp_path}"
        )

    # AC: @saved-model-artifact-safety ac-no-partial-publication
    # Checkpoint artifacts must contain all three component groups.
    if expected_kind == "checkpoint" and not _has_checkpoint_component_prefixes(tensor_keys):
        raise RuntimeError(
            "Temp checkpoint artifact missing required component prefixes "
            "(need model/diffusion, conditioning, and VAE keys) "
            f"— refusing to publish: {tmp_path}"
        )

    # AC: @full-saved-model-output ac-diffusion-model-source-kind-round-trip
    # AC: @saved-model-artifact-safety ac-no-partial-publication
    # Diffusion artifacts must use the external Comfy-loadable layout; an
    # internal-format leak (bare diffusion_model.* without model. prefix)
    # would not load through comfy.sd.load_diffusion_model.
    if expected_kind == "diffusion":
        for key in tensor_keys:
            if key.startswith(_INTERNAL_DIFFUSION_PREFIX) and not key.startswith(
                _EXTERNAL_DIFFUSION_PREFIX
            ):
                raise RuntimeError(
                    f"Temp diffusion artifact contains internal-format key "
                    f"{key!r} — refusing to publish: {tmp_path}"
                )


def save_comfy_checkpoint(
    save_path: str,
    model_patcher: object,
    clip: object,
    vae: object,
    metadata: dict[str, str],
) -> None:
    """Save a checkpoint artifact using Comfy checkpoint save semantics.

    AC: @checkpoint-loadable-saved-model-output ac-artifact-matches-source-model-kind
    AC: @exit-model-persistence ac-2, ac-8, ac-10
    AC: @saved-model-artifact-safety ac-no-partial-publication
    AC: @saved-model-artifact-safety ac-existing-valid-artifact-preserved

    Calls comfy.sd.save_checkpoint to produce a Comfy-compatible checkpoint
    artifact with model.*, conditioner.*, and first_stage_model.* keys.
    Writes to a same-directory temp target and atomically replaces the final
    path only after the artifact is completely written and classified.

    Before writing, refuses to overwrite an existing non-ecaj file.
    After writing, verifies the temp artifact contains correct ecaj metadata
    and artifact kind before publishing via atomic replace.

    Args:
        save_path: Target file path for the checkpoint artifact.
        model_patcher: Comfy ModelPatcher with merged diffusion weights installed.
        clip: Comfy CLIP model from checkpoint_components.
        vae: Comfy VAE model from checkpoint_components.
        metadata: Ecaj metadata dict to embed in the safetensors header.

    Raises:
        ValueError: If save_path exists and is not an ecaj artifact.
        RuntimeError: If temp artifact fails classification.
        Exception: If Comfy save fails. Temp file is cleaned up, existing
            valid artifact at save_path is preserved.
    """
    # AC: @saved-model-artifact-safety ac-existing-valid-artifact-preserved
    # Guard runs unconditionally — even when enable_cache=False.
    _guard_non_ecaj_overwrite(save_path)

    import comfy.sd

    directory = os.path.dirname(save_path) or "."
    suffix = secrets.token_hex(4)
    tmp_path = os.path.join(directory, f".ecaj_tmp_{suffix}_{os.path.basename(save_path)}")

    try:
        # comfy.sd.save_checkpoint writes a safetensors file with model.*,
        # conditioner.*, and first_stage_model.* keys via
        # model.state_dict_for_saving(clip_sd, vae_sd).  Pass ecaj metadata
        # so it is embedded directly in the safetensors header.
        comfy.sd.save_checkpoint(
            tmp_path,
            model_patcher,
            clip=clip,
            vae=vae,
            metadata=metadata,
        )

        # fsync for crash safety
        fd = os.open(tmp_path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

        # AC: @saved-model-artifact-safety ac-no-partial-publication
        # Classify the temp artifact before publishing — verify ecaj metadata
        # and artifact kind are present and correct.
        _classify_temp_artifact(tmp_path, expected_kind="checkpoint")

        # Atomic replace — existing valid artifact is only replaced after
        # the temp file is completely written, synced, and classified.
        os.replace(tmp_path, save_path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


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
            # AC: @full-saved-model-output ac-diffusion-model-source-kind-round-trip
            # IS_CHANGED hashes file state at the same path that the save will
            # publish to.  Standalone diffusion-model recipes save under the
            # diffusion_models folder, so the cache-invalidation hash must
            # stat that folder — not the checkpoints folder.
            try:
                is_checkpoint = _recipe_has_checkpoint_components(widen)
            except Exception:
                is_checkpoint = True
            path = _resolve_save_path(validated, is_checkpoint=is_checkpoint)
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
        # Validate checkpoint components before any expensive work when the
        # recipe signals checkpoint intent.  Checkpoint intent is present when:
        # - The base carries a checkpoint_components object (even with None
        #   fields — the object's presence signals the user wired checkpoint
        #   inputs), OR
        # - The tree contains RecipeModel nodes from the "checkpoints" source
        #   directory (detected by _recipe_has_checkpoint_components).
        # Diffusion-only recipes skip this check entirely.
        if save_model:
            base = widen if isinstance(widen, RecipeBase) else walk_to_base(widen)
            has_checkpoint_intent = (
                base.checkpoint_components is not None or _recipe_has_checkpoint_components(widen)
            )
            if has_checkpoint_intent:
                validate_checkpoint_components(base, save_model=True)

        # Quick check: must end in RecipeMerge for actual merging
        if isinstance(widen, RecipeBase):
            if save_model:
                # AC: @full-saved-model-output ac-no-op-produces-full-artifact
                # No-op recipe in full mode: produce a full artifact with all base weights
                return self._execute_full_model_noop(
                    widen,
                    model_name,
                    save_workflow,
                    enable_cache,
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
                widen,
                model_name,
                save_workflow,
                enable_cache,
                extra_pnginfo,
            )
        else:
            return self._execute_patch_mode(
                widen,
                enable_cache,
                save_model=False,
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
        AC: @full-saved-model-output ac-diffusion-model-source-kind-round-trip
        AC: @checkpoint-loadable-saved-model-output ac-artifact-matches-source-model-kind
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
        # AC: @full-saved-model-output ac-diffusion-model-source-kind-round-trip
        # Detect artifact kind BEFORE resolving the save path: a no-op
        # diffusion-only save must still publish under diffusion_models so
        # Comfy's UNETLoader discovers it, not under checkpoints.
        is_checkpoint = _recipe_has_checkpoint_components(widen)
        save_path = _resolve_save_path(validated_name, is_checkpoint=is_checkpoint)
        serialized = serialize_recipe(widen, base_identity, lora_stats)
        recipe_hash = compute_recipe_hash(serialized)
        dependency_fingerprints_json = json.dumps(
            lora_stats,
            sort_keys=True,
            separators=(",", ":"),
        )

        # AC: @full-saved-model-output ac-cache-reuses-artifact
        # AC: @exit-model-persistence ac-4, ac-6
        if enable_cache:
            if is_checkpoint:
                cache_hit = check_checkpoint_cache(
                    save_path,
                    recipe_hash,
                    base_identity,
                    dependency_fingerprints_json,
                )
            else:
                # Diffusion-only cache validation uses the EXTERNAL key
                # layout (model.diffusion_model.*) — same shape we publish.
                external_manifest = {
                    _to_external_diffusion_key(k): (v.dtype, tuple(v.shape))
                    for k, v in base_state.items()
                }
                cache_hit = check_full_model_cache(
                    save_path,
                    recipe_hash,
                    expected_manifest=external_manifest,
                    expected_artifact_kind="diffusion",
                    expected_source_model_kind=_SOURCE_KIND_DIFFUSION_MODEL,
                    expected_base_identity=base_identity,
                    expected_dependency_fingerprints=dependency_fingerprints_json,
                )
            if cache_hit:
                # AC: @streaming-materialization-progress ac-cache-reuse-status-visible
                _build_cache_reuse_progress(
                    artifact_name=validated_name,
                ).cache_reuse()
                # AC: @full-saved-model-output ac-cache-reuse-is-artifact-backed
                # AC: @comfy-memory-manager-compatibility ac-comfy-owns-returned-model-memory
                # Checkpoint cache hit: load through Comfy's checkpoint loader.
                # Diffusion-only cache hit: load through Comfy's diffusion-model
                # loader so the returned MODEL is Comfy-owned.
                if is_checkpoint:
                    return (_load_checkpoint_artifact(save_path),)
                return (_load_diffusion_model_artifact(save_path),)

        # Evict in-memory cache when cache is disabled.
        # Full mode does not write to _incremental_cache, but pre-populated
        # entries from earlier patch-mode runs must be cleared.
        if not enable_cache:
            _incremental_cache.clear()
        workflow_json = json.dumps(extra_pnginfo) if save_workflow and extra_pnginfo else None

        if is_checkpoint:
            # Checkpoint-style no-op: save via Comfy checkpoint semantics.
            # Install base weights as set patches (no merge needed — this is
            # a no-op save of the original checkpoint).
            merged_model = install_merged_patches(
                model_patcher,
                {},
                storage_dtype,
            )

            metadata = build_metadata(
                serialized,
                recipe_hash,
                [],
                workflow_json,
                output_mode="full",
                artifact_kind="checkpoint",
                source_model_kind=_SOURCE_KIND_CHECKPOINT,
                base_identity=base_identity,
                dependency_fingerprints=dependency_fingerprints_json,
                checkpoint_components=True,
            )

            checkpoint_components = widen.checkpoint_components
            save_comfy_checkpoint(
                save_path,
                merged_model,
                clip=checkpoint_components.clip,
                vae=checkpoint_components.vae,
                metadata=metadata,
            )

            if ProgressBar is not None:
                pbar = ProgressBar(1)
                pbar.update(1)

            return (merged_model,)
        else:
            # AC: @full-saved-model-output ac-diffusion-model-source-kind-round-trip
            # AC: @full-saved-model-output ac-diffusion-model-companion-separation
            # Diffusion-only no-op: write external Comfy-loadable layout via
            # MaterializationSink.
            metadata = build_metadata(
                serialized,
                recipe_hash,
                [],
                workflow_json,
                output_mode="full",
                artifact_kind="diffusion",
                source_model_kind=_SOURCE_KIND_DIFFUSION_MODEL,
                base_identity=base_identity,
                dependency_fingerprints=dependency_fingerprints_json,
            )

            external_manifest = {
                _to_external_diffusion_key(k): (v.dtype, tuple(v.shape))
                for k, v in base_state.items()
            }
            # AC: @streaming-materialization-progress ac-no-op-save-progress
            # AC: @streaming-materialization-progress ac-finalization-status-visible
            # AC: @streaming-materialization-progress ac-failure-status-not-success
            progress = _build_save_progress(
                manifest_size=len(external_manifest),
                artifact_name=validated_name,
            )
            sink = MaterializationSink()
            try:
                progress.prepare()
                sink.open(external_manifest, save_path, metadata)
                for name in sorted(base_state.keys()):
                    external_key = _to_external_diffusion_key(name)
                    sink.write_tensor(external_key, base_state[name])
                    # AC: @streaming-materialization-progress ac-no-op-save-progress
                    progress.tensor_written(external_key)
                # Enter finalize phase BEFORE sink.finalize so the visible
                # status reports the active validation/fsync/atomic-replace
                # phase while it is in flight, not only after it completes.
                # AC: @streaming-materialization-progress ac-finalization-status-visible
                progress.finalize()
                sink.finalize(
                    save_path,
                    pre_publish_check=lambda p: _classify_temp_artifact(
                        p,
                        expected_kind="diffusion",
                    ),
                )
                # AC: @streaming-materialization-progress ac-failure-status-not-success
                progress.mark_published()
            except BaseException as exc:
                # AC: @streaming-materialization-progress ac-failure-status-not-success
                progress.failure(type(exc).__name__)
                sink.abort()
                raise

            # AC: @full-saved-model-output ac-return-loaded-model
            # AC: @streaming-materialization-progress ac-finalization-status-visible
            # Returned MODEL comes from the saved diffusion-model artifact
            # via Comfy's supported diffusion-model loader.
            progress.reload()
            return (_load_diffusion_model_artifact(save_path),)

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
        AC: @checkpoint-loadable-saved-model-output ac-artifact-matches-source-model-kind
        AC: @comfy-memory-manager-compatibility ac-no-dynamic-vram-opt-out
        AC: @comfy-memory-manager-compatibility ac-memory-mode-preserved
        """
        timer = _PhaseTimer("full-saved-model")
        lora_path_resolver = _build_lora_resolver()
        model_path_resolver = _build_model_resolver()
        timer.mark("resolvers-built")

        model_patcher = walk_to_base(widen).model_patcher
        timer.mark("base-walked")
        _unpatch_loaded_clones(model_patcher)
        timer.mark("loaded-clones-unpatched")
        base_state = model_patcher.model_state_dict()  # type: ignore[attr-defined]
        timer.mark(f"base-state-read keys={len(base_state)}")
        storage_dtype = next(iter(base_state.values())).dtype

        key_shapes = {k: tuple(v.shape) for k, v in base_state.items()}
        timer.mark("key-shapes-built")

        base_identity = compute_base_identity(base_state)
        timer.mark("base-identity-computed")
        lora_stats = compute_lora_stats(widen, lora_path_resolver, model_path_resolver)
        timer.mark(f"dependency-stats-computed count={len(lora_stats)}")

        validated_name = validate_model_name(model_name)
        # AC: @full-saved-model-output ac-diffusion-model-source-kind-round-trip
        # Detect artifact kind BEFORE resolving the save path so diffusion-only
        # saves publish under the diffusion_models folder (Comfy's UNETLoader
        # discovery) and checkpoint saves publish under the checkpoints folder
        # (CheckpointLoaderSimple discovery).
        is_checkpoint = _recipe_has_checkpoint_components(widen)
        save_path = _resolve_save_path(validated_name, is_checkpoint=is_checkpoint)
        timer.mark(f"save-path-resolved checkpoint={is_checkpoint}")
        serialized = serialize_recipe(widen, base_identity, lora_stats)
        recipe_hash = compute_recipe_hash(serialized)
        timer.mark("recipe-hash-computed")

        # Build manifest early for cache validation (shapes + dtypes, not just key names).
        base_manifest = {k: (v.dtype, tuple(v.shape)) for k, v in base_state.items()}
        timer.mark("base-manifest-built")

        dependency_fingerprints_json = json.dumps(
            lora_stats,
            sort_keys=True,
            separators=(",", ":"),
        )
        timer.mark("dependency-fingerprint-json-built")

        # AC: @full-saved-model-output ac-cache-reuses-artifact
        # AC: @full-saved-model-output ac-cache-reuse-is-artifact-backed
        # AC: @full-saved-model-output ac-diffusion-model-source-kind-round-trip
        # AC: @saved-model-artifact-safety ac-missing-metadata-not-reused
        # AC: @saved-model-artifact-safety ac-wrong-artifact-kind-not-reused
        # AC: @saved-model-artifact-safety ac-internal-format-not-checkpoint-cache
        # Checkpoint-style: validate artifact_kind, base identity, dependency
        # fingerprints, and checkpoint component classification.  Manifest
        # validation is skipped because base_manifest reflects only the base
        # model's state dict and may not include companion component keys
        # (CLIP, VAE) that a checkpoint artifact contains.
        # Non-checkpoint: validate full-model metadata and manifest using the
        # external diffusion-model key layout (model.diffusion_model.*).
        if enable_cache:
            if is_checkpoint:
                cache_hit = check_checkpoint_cache(
                    save_path,
                    recipe_hash,
                    base_identity,
                    dependency_fingerprints_json,
                )
            else:
                external_manifest = {
                    _to_external_diffusion_key(k): v for k, v in base_manifest.items()
                }
                cache_hit = check_full_model_cache(
                    save_path,
                    recipe_hash,
                    expected_manifest=external_manifest,
                    expected_artifact_kind="diffusion",
                    expected_source_model_kind=_SOURCE_KIND_DIFFUSION_MODEL,
                    expected_base_identity=base_identity,
                    expected_dependency_fingerprints=dependency_fingerprints_json,
                )
            if cache_hit:
                del base_state
                # AC: @streaming-materialization-progress ac-cache-reuse-status-visible
                _build_cache_reuse_progress(
                    artifact_name=validated_name,
                ).cache_reuse()
                # AC: @checkpoint-loadable-saved-model-output ac-downstream-return-remains-usable
                # AC: @full-saved-model-output ac-cache-reuse-is-artifact-backed
                # AC: @comfy-memory-manager-compatibility ac-comfy-owns-returned-model-memory
                # Checkpoint cache hit: load through Comfy's checkpoint loader.
                # Diffusion-only cache hit: load through Comfy's diffusion-model
                # loader so the returned MODEL is Comfy-owned.
                if is_checkpoint:
                    return (_load_checkpoint_artifact(save_path),)
                return (_load_diffusion_model_artifact(save_path),)

        timer.mark("cache-path-complete")
        # Branch: checkpoint-style saves use Comfy checkpoint save semantics;
        # non-checkpoint (diffusion-only) saves continue using MaterializationSink.
        if is_checkpoint:
            timer.mark("dispatch-checkpoint-save")
            return self._execute_checkpoint_save(
                widen,
                model_patcher,
                base_state,
                storage_dtype,
                key_shapes,
                save_path,
                serialized,
                recipe_hash,
                base_identity,
                dependency_fingerprints_json,
                save_workflow,
                enable_cache,
                extra_pnginfo,
                lora_path_resolver,
                model_path_resolver,
            )
        else:
            timer.mark("dispatch-diffusion-save")
            return self._execute_diffusion_save(
                widen,
                model_patcher,
                base_state,
                storage_dtype,
                key_shapes,
                save_path,
                serialized,
                recipe_hash,
                base_identity,
                dependency_fingerprints_json,
                save_workflow,
                enable_cache,
                extra_pnginfo,
                lora_path_resolver,
                model_path_resolver,
            )

    def _execute_checkpoint_save(
        self,
        widen: RecipeMerge,
        model_patcher: object,
        base_state: dict[str, torch.Tensor],
        storage_dtype: torch.dtype,
        key_shapes: dict[str, tuple[int, ...]],
        save_path: str,
        serialized: str,
        recipe_hash: str,
        base_identity: str,
        dependency_fingerprints_json: str,
        save_workflow: bool,
        enable_cache: bool,
        extra_pnginfo: object,
        lora_path_resolver: Callable,
        model_path_resolver: Callable,
    ) -> tuple[object]:
        """Execute checkpoint-style save using Comfy checkpoint save semantics.

        AC: @checkpoint-loadable-saved-model-output ac-artifact-matches-source-model-kind
        AC: @exit-model-persistence ac-2, ac-8, ac-10
        AC: @saved-model-artifact-safety ac-no-partial-publication
        AC: @saved-model-artifact-safety ac-existing-valid-artifact-preserved

        Instead of writing an internal-format diffusion-only artifact via
        MaterializationSink, this path:
        1. Computes merged diffusion weights via the WIDEN merge pipeline
        2. Installs them as set patches on a cloned ModelPatcher
        3. Calls save_comfy_checkpoint with the merged MODEL + CLIP + VAE
        4. The artifact contains Comfy checkpoint-style keys (model.*,
           conditioner.*, first_stage_model.*) and is loadable by
           CheckpointLoaderSimple.
        """
        # --- GPU pipeline: compute merged diffusion weights ---
        analysis = analyze_recipe(widen, lora_path_resolver=lora_path_resolver)

        base = walk_to_base(widen)
        domain = getattr(base, "domain", "diffusion")
        model_analysis = analyze_recipe_models(
            widen, base.arch, model_path_resolver=model_path_resolver, domain=domain
        )

        # AC: @comfy-memory-manager-compatibility ac-checkpoint-save-failure-releases-temp-payload
        # AC: @comfy-memory-manager-compatibility ac-checkpoint-cache-miss-releases-save-payload
        # AC: @saved-model-artifact-safety ac-failed-return-not-successful
        # Track the temporary save-time merge payload separately from the
        # artifact-loaded return model so release runs in finally on success
        # and on save/publication/finalization failures, exactly once.
        temporary_model: object | None = None
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
            validate_loader_keys = getattr(loader, "validate_compatible_keys", None)
            if validate_loader_keys is not None:
                validate_loader_keys(all_keys, key_shapes)
            lora_keys = get_keys_to_process(all_keys, lora_affected_keys)
            model_keys = all_keys & all_model_keys
            keys_to_process = lora_keys | model_keys

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

            if keys_to_process:
                batch_groups = compile_batch_groups(
                    list(keys_to_process),
                    arch=arch,
                    key_shapes=key_shapes,
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
                # Checkpoint save accumulates merged_state (needs model in memory
                # for Comfy save), so budget the full merged state size.
                merged_state_bytes = sum(
                    base_state[k].nelement() * base_state[k].element_size()
                    for keys in batch_groups.values()
                    for k in keys
                )
                check_ram_preflight(
                    merged_state_bytes=merged_state_bytes,
                    worst_chunk_bytes=worst_chunk_bytes,
                    save_model=True,
                    loader_bytes=loader.loaded_bytes
                    + sum(ml.loaded_bytes for ml in model_loaders.values()),
                )

            _log_memory("before-gpu-eval-checkpoint")

            # Compute merged diffusion weights via dict-returning evaluation.
            # Checkpoint save needs all merged weights in memory to install
            # into the ModelPatcher before calling comfy.sd.save_checkpoint.
            merged_state: dict[str, torch.Tensor] = {}

            if batch_groups:

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
                    plan,
                    loader,
                    widen_merger,
                    device,
                    compute_dtype,
                    arch,
                    widen_config,
                    model_loaders,
                    domain,
                )

                pbar_count = len(batch_groups)
                pbar = ProgressBar(pbar_count) if ProgressBar is not None else None

                for sig, group_keys in batch_groups.items():
                    n_models = len(set_affected) + len(model_loaders)
                    batch_size = compute_batch_size(
                        sig.shape,
                        n_models,
                        compute_dtype,
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

                    del group_base
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    if pbar is not None:
                        pbar.update(1)

            _log_memory("after-gpu-eval-checkpoint")

            # Free base_state — no longer needed after merge
            del base_state

            # Install merged diffusion weights into a cloned ModelPatcher.
            # AC: @checkpoint-loadable-saved-model-output ac-artifact-matches-source-model-kind
            # Assign to the function-scoped temporary_model so finally can
            # release it on any subsequent failure (save, publication, cache
            # bookkeeping) before control returns to ComfyUI.
            temporary_model = install_merged_patches(
                model_patcher,
                merged_state,
                storage_dtype,
            )

            # Free merged_state — weights are now held as set patches on temporary_model
            del merged_state
            gc.collect()

            # Build ecaj metadata for the checkpoint artifact.
            workflow_json = json.dumps(extra_pnginfo) if save_workflow and extra_pnginfo else None
            # AC: @exit-model-persistence ac-6
            # AC: @saved-model-artifact-safety ac-missing-metadata-not-reused
            # AC: @saved-model-artifact-safety ac-wrong-artifact-kind-not-reused
            metadata = build_metadata(
                serialized,
                recipe_hash,
                sorted(affected_key_set),
                workflow_json,
                output_mode="full",
                artifact_kind="checkpoint",
                source_model_kind=_SOURCE_KIND_CHECKPOINT,
                base_identity=base_identity,
                dependency_fingerprints=dependency_fingerprints_json,
                checkpoint_components=True,
            )

            # AC: @checkpoint-loadable-saved-model-output ac-artifact-matches-source-model-kind
            # AC: @exit-model-persistence ac-2, ac-8, ac-10
            # AC: @saved-model-artifact-safety ac-no-partial-publication
            # AC: @saved-model-artifact-safety ac-existing-valid-artifact-preserved
            # Save using Comfy checkpoint semantics: produces artifact with
            # model.*, conditioner.*, and first_stage_model.* keys.
            checkpoint_components = base.checkpoint_components
            save_comfy_checkpoint(
                save_path,
                temporary_model,
                clip=checkpoint_components.clip,
                vae=checkpoint_components.vae,
                metadata=metadata,
            )

            _log_memory("after-checkpoint-save")

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
            if not enable_cache:
                _incremental_cache.clear()

        finally:
            # See AC block above the try: release the temporary save-time
            # merge payload before returning so Comfy's output cache retains
            # only an artifact-loaded MODEL, matching checkpoint cache-hit
            # behavior, and so failures during save/publication/finalization
            # still drop the dense payload before control returns to ComfyUI.
            # Guarded by `is not None` so we never attempt to release before
            # the temporary model exists (e.g. failure during merge eval).
            if temporary_model is not None:
                _release_temporary_checkpoint_model(temporary_model)
                temporary_model = None
                _log_memory("after-checkpoint-temp-model-release")
            loader.cleanup()
            for model_loader in model_analysis.model_loaders.values():
                model_loader.cleanup()

        # AC: @checkpoint-loadable-saved-model-output ac-downstream-return-remains-usable
        # Cache-miss return: reload the saved checkpoint through Comfy's loader
        # so downstream consumers get a Comfy-owned MODEL rather than the
        # transient dense WIDEN serialization clone.
        loaded_model = _load_checkpoint_artifact(save_path)
        return (loaded_model,)

    def _execute_diffusion_save(
        self,
        widen: RecipeMerge,
        model_patcher: object,
        base_state: dict[str, torch.Tensor],
        storage_dtype: torch.dtype,
        key_shapes: dict[str, tuple[int, ...]],
        save_path: str,
        serialized: str,
        recipe_hash: str,
        base_identity: str,
        dependency_fingerprints_json: str,
        save_workflow: bool,
        enable_cache: bool,
        extra_pnginfo: object,
        lora_path_resolver: Callable,
        model_path_resolver: Callable,
    ) -> tuple[object]:
        """Execute diffusion-only save — stream to artifact via MaterializationSink.

        AC: @streaming-full-model-materialization ac-direct-artifact-handoff
        AC: @streaming-full-model-materialization ac-affected-results-released
        AC: @streaming-full-model-materialization ac-base-weight-bounded-copying
        AC: @streaming-full-model-materialization ac-incomplete-write-not-reused
        AC: @streaming-full-model-materialization ac-full-cache-avoids-resident-payload
        AC: @streaming-full-model-materialization
            ac-failed-materialization-releases-resident-payload
        """
        timer = _PhaseTimer("diffusion-save")
        analysis = analyze_recipe(widen, lora_path_resolver=lora_path_resolver)
        timer.mark(f"recipe-analyzed affected={len(analysis.affected_keys)}")

        base = walk_to_base(widen)
        domain = getattr(base, "domain", "diffusion")
        model_analysis = analyze_recipe_models(
            widen, base.arch, model_path_resolver=model_path_resolver, domain=domain
        )
        timer.mark(f"recipe-models-analyzed model_loaders={len(model_analysis.model_loaders)}")

        sink = MaterializationSink()
        # AC: @streaming-materialization-progress ac-progress-during-streaming-writes
        # AC: @streaming-materialization-progress ac-affected-write-progress
        # AC: @streaming-materialization-progress ac-no-op-save-progress
        # AC: @streaming-materialization-progress ac-finalization-status-visible
        # AC: @streaming-materialization-progress ac-failure-status-not-success
        # Progress sized from the manifest (one tick per tensor write) plus
        # explicit phase units for prepare, finalize, and reload.  Created
        # before any sink work so a failure during `open()` still carries a
        # progress object the except block can mark as failed.
        progress = _build_save_progress(
            manifest_size=len(base_state),
            artifact_name=os.path.basename(save_path),
        )
        timer.mark(f"progress-built manifest_size={len(base_state)}")
        try:
            loader = analysis.loader
            set_affected = analysis.set_affected
            lora_affected_keys = analysis.affected_keys
            arch = analysis.arch

            model_affected = model_analysis.model_affected
            model_loaders = model_analysis.model_loaders
            all_model_keys = model_analysis.all_model_keys
            timer.mark(
                f"analysis-unpacked sets={len(set_affected)} model_keys={len(all_model_keys)}"
            )

            compute_dtype = torch.float32
            device = "cuda" if torch.cuda.is_available() else "cpu"

            all_keys = set(base_state.keys())
            validate_loader_keys = getattr(loader, "validate_compatible_keys", None)
            if validate_loader_keys is not None:
                validate_loader_keys(all_keys, key_shapes)
            timer.mark(f"loader-compatible keys={len(all_keys)}")
            lora_keys = get_keys_to_process(all_keys, lora_affected_keys)
            model_keys = all_keys & all_model_keys
            keys_to_process = lora_keys | model_keys
            timer.mark(f"keys-selected lora={len(lora_keys)} model={len(model_keys)}")

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
            timer.mark("widen-config-built")

            plan = compile_plan(widen, set_id_map, arch, model_id_map)
            timer.mark(f"plan-compiled ops={len(plan.ops)}")

            # AC: @full-saved-model-output ac-diffusion-model-source-kind-round-trip
            # AC: @full-saved-model-output ac-diffusion-model-companion-separation
            # Build manifest for entire model in the EXTERNAL Comfy-loadable
            # layout (model.diffusion_model.*) — the merged internal keys
            # (diffusion_model.*) are remapped before the manifest and writes
            # so the published artifact is loadable as a standalone diffusion
            # model of the same kind as the source.
            manifest: dict[str, tuple[torch.dtype, tuple[int, ...]]] = {}
            for k, v in base_state.items():
                manifest[_to_external_diffusion_key(k)] = (v.dtype, tuple(v.shape))
            timer.mark(f"manifest-built entries={len(manifest)}")

            workflow_json = json.dumps(extra_pnginfo) if save_workflow and extra_pnginfo else None
            metadata = build_metadata(
                serialized,
                recipe_hash,
                sorted(affected_key_set),
                workflow_json,
                output_mode="full",
                artifact_kind="diffusion",
                source_model_kind=_SOURCE_KIND_DIFFUSION_MODEL,
                base_identity=base_identity,
                dependency_fingerprints=dependency_fingerprints_json,
                checkpoint_components=False,
            )

            progress.prepare()
            timer.mark("progress-prepare-done")
            sink.open(manifest, save_path, metadata)
            timer.mark("sink-opened")

            if keys_to_process:
                batch_groups = compile_batch_groups(
                    list(keys_to_process),
                    arch=arch,
                    key_shapes=key_shapes,
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
                check_ram_preflight(
                    merged_state_bytes=worst_chunk_bytes,
                    worst_chunk_bytes=worst_chunk_bytes,
                    save_model=True,
                    loader_bytes=loader.loaded_bytes
                    + sum(ml.loaded_bytes for ml in model_loaders.values()),
                )

            _log_memory("before-gpu-eval-full")

            affected_key_set_lookup: set[str] = set()
            for group_keys in batch_groups.values():
                affected_key_set_lookup.update(group_keys)

            # Unaffected base writes use the EXTERNAL key layout
            # AC: @streaming-materialization-progress ac-no-op-save-progress
            # AC: @streaming-materialization-progress ac-progress-during-streaming-writes
            for key in base_state:
                if key not in affected_key_set_lookup:
                    external_key = _to_external_diffusion_key(key)
                    sink.write_tensor(external_key, base_state[key])
                    progress.tensor_written(external_key)

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
                plan,
                loader,
                widen_merger,
                device,
                compute_dtype,
                arch,
                widen_config,
                model_loaders,
                domain,
            )

            # streaming_evaluation_to_sink calls write_fn(name, tensor) using
            # internal keys; remap to external keys at the boundary and
            # advance progress for each affected tensor handoff.
            # AC: @streaming-materialization-progress ac-affected-write-progress
            # AC: @streaming-materialization-progress ac-progress-during-streaming-writes
            def _external_write(name: str, tensor: torch.Tensor) -> None:
                external_key = _to_external_diffusion_key(name)
                sink.write_tensor(external_key, tensor)
                progress.tensor_written(external_key)

            for sig, group_keys in batch_groups.items():
                n_models = len(set_affected) + len(model_loaders)
                batch_size = compute_batch_size(
                    sig.shape,
                    n_models,
                    compute_dtype,
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
                    write_fn=_external_write,
                )

                del group_base
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            _log_memory("after-streaming-full")

            del base_state

            # Enter finalize phase BEFORE sink.finalize so the visible
            # status reports the active validation/fsync/atomic-replace
            # phase while it is in flight, not only after it completes.
            # AC: @streaming-materialization-progress ac-finalization-status-visible
            progress.finalize()
            sink.finalize(
                save_path,
                pre_publish_check=lambda p: _classify_temp_artifact(
                    p,
                    expected_kind="diffusion",
                ),
            )
            # AC: @streaming-materialization-progress ac-failure-status-not-success
            progress.mark_published()
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

            if not enable_cache:
                _incremental_cache.clear()

        except BaseException as exc:
            # AC: @streaming-materialization-progress ac-failure-status-not-success
            progress.failure(type(exc).__name__)
            sink.abort()
            raise
        finally:
            loader.cleanup()
            for model_loader in model_analysis.model_loaders.values():
                model_loader.cleanup()

        # AC: @full-saved-model-output ac-return-loaded-model
        # AC: @full-saved-model-output ac-diffusion-model-source-kind-round-trip
        # AC: @streaming-materialization-progress ac-finalization-status-visible
        # Returned MODEL comes from the saved diffusion-model artifact via
        # Comfy's supported diffusion-model loader (Comfy-owned memory).
        progress.reload()
        return (_load_diffusion_model_artifact(save_path),)

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
            validate_loader_keys = getattr(loader, "validate_compatible_keys", None)
            if validate_loader_keys is not None:
                validate_loader_keys(all_keys, key_shapes)
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
                list(keys_to_process),
                arch=arch,
                key_shapes=key_shapes,
            )

            plan = compile_plan(widen, set_id_map, arch, model_id_map)

            # --- Incremental cache: detect which blocks changed ---
            structural_fp = compute_structural_fingerprint(widen, base_identity, lora_stats)
            current_block_configs = collect_block_configs(widen)
            current_loader_bytes = loader.loaded_bytes + sum(
                ml.loaded_bytes for ml in model_loaders.values()
            )
            cached_entry = _incremental_cache.get(structural_fp) if enable_cache else None
            incremental_hit = False

            if cached_entry is not None and cached_entry.storage_dtype == storage_dtype:
                diff = compute_changed_blocks(
                    cached_entry.block_configs, current_block_configs, arch
                )
                if diff is not None:
                    changed_blocks, changed_layer_types = diff

                    if not changed_blocks and not changed_layer_types:
                        merged_state = {k: v for k, v in cached_entry.merged_state.items()}
                        incremental_hit = True
                        batch_groups = {}

                        if ProgressBar is not None:
                            pbar = ProgressBar(1)
                            pbar.update(1)
                    else:
                        recompute_keys = filter_changed_keys(
                            keys_to_process,
                            changed_blocks,
                            changed_layer_types,
                            arch,
                        )

                        if not recompute_keys:
                            merged_state = {k: v for k, v in cached_entry.merged_state.items()}
                            incremental_hit = True
                            batch_groups = {}
                        else:
                            merged_state = {k: v for k, v in cached_entry.merged_state.items()}
                            batch_groups = compile_batch_groups(
                                list(recompute_keys),
                                arch=arch,
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
                merged_state_bytes = sum(key_byte_sizes[k] for k in processed_keys)
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
                        sig.shape,
                        n_models,
                        compute_dtype,
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
                        plan,
                        loader,
                        widen_merger,
                        device,
                        compute_dtype,
                        arch,
                        widen_config,
                        model_loaders,
                        domain,
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
                        t.nelement() * t.element_size() for t in merged_state.values()
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
