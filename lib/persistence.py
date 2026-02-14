"""Model persistence — save/load merged models as safetensors.

Pure library module with zero ComfyUI imports. All external dependencies
(folder_paths, model_patcher, etc.) are passed as arguments.

AC: @exit-model-persistence ac-2 through ac-14
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from typing import TYPE_CHECKING

from safetensors.torch import save_file

if TYPE_CHECKING:
    import torch

__all__ = [
    "atomic_save",
    "build_metadata",
    "check_cache",
    "compute_base_identity",
    "compute_lora_stats",
    "compute_recipe_hash",
    "load_affected_keys",
    "serialize_recipe",
    "validate_model_name",
]

# Current metadata schema version
_ECAJ_VERSION = "1"


def validate_model_name(name: str) -> str:
    """Validate and normalize a model filename.

    AC: @exit-model-persistence ac-5, ac-11, ac-12

    Args:
        name: User-provided model name

    Returns:
        Validated name with .safetensors extension

    Raises:
        ValueError: If name is empty, contains path traversal, or has separators
    """
    stripped = name.strip()
    if not stripped:
        raise ValueError("Model name cannot be empty")

    # Reject path traversal and separators
    if ".." in stripped:
        raise ValueError(f"Model name contains path traversal: {stripped!r}")
    if "/" in stripped or "\\" in stripped:
        raise ValueError(f"Model name contains path separators: {stripped!r}")

    # Auto-append .safetensors if no extension
    if not stripped.endswith(".safetensors"):
        stripped += ".safetensors"

    return stripped


def serialize_recipe(
    node: object,
    base_identity: str,
    lora_stats: dict[str, tuple[float, int]],
) -> str:
    """Serialize a recipe tree to deterministic JSON.

    AC: @exit-model-persistence ac-6, ac-7

    Replaces model_patcher references with base_identity string.
    Includes LoRA file stats (mtime/size) for cache invalidation.

    Args:
        node: Recipe tree root (RecipeNode)
        base_identity: SHA-256 identity of the base model
        lora_stats: Map of resolved LoRA path -> (mtime, size)

    Returns:
        Deterministic JSON string
    """
    from .recipe import BlockConfig, RecipeBase, RecipeCompose, RecipeLoRA, RecipeMerge

    def _serialize_node(n: object) -> dict:
        if isinstance(n, RecipeBase):
            return {
                "type": "RecipeBase",
                "arch": n.arch,
                "base_identity": base_identity,
            }
        elif isinstance(n, RecipeLoRA):
            loras = []
            for spec in n.loras:
                path = spec["path"]
                entry: dict = {
                    "path": path,
                    "strength": spec["strength"],
                }
                # Include file stats if available
                if path in lora_stats:
                    mtime, size = lora_stats[path]
                    entry["mtime"] = mtime
                    entry["size"] = size
                loras.append(entry)
            result: dict = {"type": "RecipeLoRA", "loras": loras}
            if n.block_config is not None:
                result["block_config"] = _serialize_block_config(n.block_config)
            return result
        elif isinstance(n, RecipeCompose):
            return {
                "type": "RecipeCompose",
                "branches": [_serialize_node(b) for b in n.branches],
            }
        elif isinstance(n, RecipeMerge):
            result = {
                "type": "RecipeMerge",
                "base": _serialize_node(n.base),
                "target": _serialize_node(n.target),
                "t_factor": n.t_factor,
            }
            if n.backbone is not None:
                result["backbone"] = _serialize_node(n.backbone)
            if n.block_config is not None:
                result["block_config"] = _serialize_block_config(n.block_config)
            return result
        else:
            raise ValueError(f"Unknown recipe node type: {type(n).__name__}")

    def _serialize_block_config(bc: object) -> dict:
        if not isinstance(bc, BlockConfig):
            raise ValueError(f"Expected BlockConfig, got {type(bc).__name__}")
        result: dict = {"arch": bc.arch}
        if bc.block_overrides:
            result["block_overrides"] = [list(pair) for pair in bc.block_overrides]
        if bc.layer_type_overrides:
            result["layer_type_overrides"] = [list(pair) for pair in bc.layer_type_overrides]
        return result

    tree = _serialize_node(node)
    return json.dumps(tree, sort_keys=True, separators=(",", ":"))


def compute_base_identity(base_state: dict[str, torch.Tensor]) -> str:
    """Compute a stable identity hash for a base model.

    AC: @exit-model-persistence ac-6

    Uses sorted key signatures (key|shape|dtype) plus a small tensor data
    sample from the first key to distinguish models with identical architecture.

    Args:
        base_state: Base model state dict

    Returns:
        SHA-256 hex digest
    """
    hasher = hashlib.sha256()

    sorted_keys = sorted(base_state.keys())
    for key in sorted_keys:
        tensor = base_state[key]
        hasher.update(f"{key}|{tuple(tensor.shape)}|{tensor.dtype}\n".encode())

    # Sample tensor data from first key to catch weight differences
    if sorted_keys:
        sample_tensor = base_state[sorted_keys[0]]
        flat = sample_tensor.detach().float().reshape(-1)[:64].contiguous().cpu()
        hasher.update(bytes(flat.untyped_storage())[:flat.nelement() * flat.element_size()])

    return hasher.hexdigest()


def compute_lora_stats(
    node: object,
    resolver: Callable[[str], str | None],
) -> dict[str, tuple[float, int]]:
    """Walk recipe tree and collect LoRA file stats.

    AC: @exit-model-persistence ac-7

    Args:
        node: Recipe tree root
        resolver: Resolves LoRA name to full filesystem path

    Returns:
        Dict mapping LoRA path (as in recipe) -> (mtime, size)
    """
    from .recipe import RecipeBase, RecipeCompose, RecipeLoRA, RecipeMerge

    stats: dict[str, tuple[float, int]] = {}

    def _walk(n: object) -> None:
        if isinstance(n, RecipeBase):
            return
        elif isinstance(n, RecipeLoRA):
            for spec in n.loras:
                path = spec["path"]
                if path not in stats:
                    resolved = resolver(path)
                    full_path = resolved if resolved is not None else path
                    try:
                        st = os.stat(full_path)
                        stats[path] = (st.st_mtime, st.st_size)
                    except OSError:
                        stats[path] = (0.0, 0)
        elif isinstance(n, RecipeCompose):
            for branch in n.branches:
                _walk(branch)
        elif isinstance(n, RecipeMerge):
            _walk(n.base)
            _walk(n.target)
            if n.backbone is not None:
                _walk(n.backbone)

    _walk(node)
    return stats


def compute_recipe_hash(serialized: str) -> str:
    """Compute SHA-256 hash of a serialized recipe.

    AC: @exit-model-persistence ac-6

    Args:
        serialized: Deterministic JSON from serialize_recipe

    Returns:
        Hex digest
    """
    return hashlib.sha256(serialized.encode()).hexdigest()


def check_cache(save_path: str, expected_hash: str) -> dict | None:
    """Check if a cached model matches the expected recipe hash.

    AC: @exit-model-persistence ac-3, ac-4, ac-9

    Reads safetensors header only (cheap). Returns metadata on hash match,
    None on mismatch or missing file. Raises on non-ecaj files.

    Args:
        save_path: Path to the safetensors file
        expected_hash: Expected recipe hash

    Returns:
        Metadata dict on cache hit, None on miss/mismatch

    Raises:
        ValueError: If file exists but has no ecaj metadata (AC-9)
    """
    if not os.path.exists(save_path):
        return None

    from safetensors import safe_open

    # Read header only (metadata is in the header, no tensor data loaded)
    with safe_open(save_path, framework="pt") as f:
        metadata = f.metadata()

    if metadata is None or "__ecaj_version__" not in metadata:
        raise ValueError(
            f"File exists but is not an ecaj-saved model: {save_path}\n"
            f"Refusing to overwrite a file without ecaj metadata. "
            f"Choose a different model_name."
        )

    stored_hash = metadata.get("__ecaj_recipe_hash__", "")
    if stored_hash != expected_hash:
        return None

    return metadata


def load_affected_keys(
    save_path: str,
    keys: list[str],
) -> dict[str, torch.Tensor]:
    """Selectively load only the affected keys from a cached model.

    AC: @exit-model-persistence ac-3

    Uses safe_open for selective loading (not the full file).

    Args:
        save_path: Path to safetensors file
        keys: List of keys to load

    Returns:
        Dict of key -> tensor for the requested keys
    """
    from safetensors import safe_open

    result = {}
    with safe_open(save_path, framework="pt", device="cpu") as f:
        for key in keys:
            result[key] = f.get_tensor(key)
    return result


def build_metadata(
    serialized: str,
    recipe_hash: str,
    affected_keys: list[str],
    workflow_json: str | None = None,
) -> dict[str, str]:
    """Assemble safetensors metadata dict.

    AC: @exit-model-persistence ac-6, ac-13, ac-14

    Args:
        serialized: Deterministic JSON recipe
        recipe_hash: SHA-256 of serialized
        affected_keys: Sorted list of keys that were merged (not base-only)
        workflow_json: Optional workflow JSON string

    Returns:
        Metadata dict with string values (safetensors requirement)
    """
    metadata: dict[str, str] = {
        "__ecaj_version__": _ECAJ_VERSION,
        "__ecaj_recipe__": serialized,
        "__ecaj_recipe_hash__": recipe_hash,
        "__ecaj_affected_keys__": json.dumps(affected_keys),
    }
    if workflow_json is not None:
        metadata["__ecaj_workflow__"] = workflow_json
    return metadata


def atomic_save(
    tensors: dict[str, torch.Tensor],
    save_path: str,
    metadata: dict[str, str],
) -> None:
    """Atomically save tensors to a safetensors file.

    AC: @exit-model-persistence ac-8, ac-10

    Writes to a temp file in the same directory, fsyncs, then atomically
    replaces the target. Cleans up temp on failure.

    Args:
        tensors: Full state dict (base + merged overlays)
        save_path: Target file path
        metadata: Safetensors metadata dict
    """
    directory = os.path.dirname(save_path) or "."
    tmp_path = os.path.join(directory, ".ecaj_tmp_" + os.path.basename(save_path))

    try:
        save_file(tensors, tmp_path, metadata=metadata)

        # fsync for crash safety
        fd = os.open(tmp_path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

        os.replace(tmp_path, save_path)
    except BaseException:
        # Clean up temp file on any failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
