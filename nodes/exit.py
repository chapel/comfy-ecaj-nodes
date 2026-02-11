"""WIDEN Exit Node — Executes the recipe tree, returns ComfyUI MODEL."""

from __future__ import annotations

import hashlib
import os
from typing import TYPE_CHECKING

import torch

from lib.recipe import RecipeBase, RecipeCompose, RecipeLoRA, RecipeMerge, RecipeNode

if TYPE_CHECKING:
    pass

# Namespace prefix for diffusion model keys in ComfyUI ModelPatcher
_DIFFUSION_PREFIX = "diffusion_model."


def install_merged_patches(
    model_patcher: object,
    merged_state: dict[str, torch.Tensor],
) -> object:
    """Install merged tensors as set patches on a cloned ModelPatcher.

    AC: @exit-patch-install ac-1 — clone model, add as set patches
    AC: @exit-patch-install ac-2 — prefix keys with diffusion_model.
    AC: @exit-patch-install ac-3 — tensors transferred to CPU
    AC: @exit-patch-install ac-4 — tensors match base model storage dtype

    Args:
        model_patcher: Original ComfyUI ModelPatcher
        merged_state: Dict of {key: merged_tensor} from batched evaluation
            Keys should NOT have diffusion_model. prefix

    Returns:
        Cloned ModelPatcher with merged weights installed as set patches
    """
    # Get base model dtype from first value in state dict
    base_state = model_patcher.model_state_dict()  # type: ignore[attr-defined]
    base_dtype = next(iter(base_state.values())).dtype

    # Clone model (AC-1)
    cloned = model_patcher.clone()  # type: ignore[attr-defined]

    # Build set patches: transfer to CPU (AC-3), cast to base dtype (AC-4)
    # Prefix with diffusion_model. (AC-2)
    patches = {}
    for key, tensor in merged_state.items():
        cpu_tensor = tensor.cpu().to(base_dtype)
        prefixed_key = f"{_DIFFUSION_PREFIX}{key}"
        # "set" patch format: replaces the weight entirely
        patches[prefixed_key] = ("set", cpu_tensor)

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


def _compute_recipe_hash(widen: RecipeNode, lora_base_path: str | None = None) -> str:
    """Compute a hash of the recipe based on LoRA file paths and mtimes.

    AC: @exit-patch-install ac-5 — identical hash when no LoRA changes
    AC: @exit-patch-install ac-6 — different hash when LoRA modified

    Args:
        widen: Recipe tree root
        lora_base_path: Base path for LoRA files (for tests, optional)

    Returns:
        Hex digest of SHA-256 hash
    """
    paths = _collect_lora_paths(widen)

    # Sort for deterministic ordering
    paths = sorted(set(paths))

    # Build hash from (path, mtime, size) tuples
    hasher = hashlib.sha256()

    for path in paths:
        # Resolve full path if base path provided
        if lora_base_path:
            full_path = os.path.join(lora_base_path, path)
        else:
            full_path = path

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
        return _compute_recipe_hash(widen)

    def execute(self, widen):
        raise NotImplementedError("Exit node not yet implemented")
