"""CLIP Model Loader -- streaming loader for text encoder weights from checkpoints.

Uses safetensors.safe_open() for memory-efficient per-batch access to
text encoder weights. Handles key normalization between checkpoint file
format and CLIP base model state dict format.

Inverse of ModelLoader: includes only text encoder keys (conditioner.embedders.*)
and excludes diffusion model and VAE keys.

Only supports safetensors format. Non-safetensors files raise a clear error.
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open

from .model_loader import UnsupportedFormatError

__all__ = ["CLIPModelLoader", "CLIPKeyMappingError"]


class CLIPKeyMappingError(ValueError):
    """Raised when CLIP key mapping fails due to unexpected checkpoint structure."""

    pass


# Embedder index to CLIP model prefix mapping.
# SDXL uses:
#   embedders.0 = CLIP-L (ViT-L/14)
#   embedders.1 = CLIP-G (ViT-bigG)
_EMBEDDER_MAPPING = {
    0: "clip_l",
    1: "clip_g",
}

# Prefixes that identify text encoder keys to include.
_INCLUDED_PREFIXES = (
    "conditioner.embedders.",
    "model.conditioner.embedders.",
    "cond_stage_model.",
    "model.cond_stage_model.",
)


def _normalize_clip_key(file_key: str) -> str | None:
    """Normalize a checkpoint file key to CLIP base model format.

    Args:
        file_key: Key from checkpoint safetensors file

    Returns:
        Normalized key in CLIP base model format (clip_l.* or clip_g.*),
        or None if the key should be excluded.

    # AC: @clip-model-loader ac-1
    Exposes text encoder keys (conditioner.embedders.*) and excludes others.

    # AC: @clip-model-loader ac-6
    Maps embedders.0 keys to clip_l.* and embedders.1 keys to clip_g.*.
    """
    # Only process text encoder keys
    matched_prefix = None
    for prefix in _INCLUDED_PREFIXES:
        if file_key.startswith(prefix):
            matched_prefix = prefix
            break

    if matched_prefix is None:
        return None

    # Extract the part after the prefix
    # e.g., "conditioner.embedders.0.transformer.text_model.encoder.layers.0...."
    remaining = file_key[len(matched_prefix) :]

    # Parse embedder index
    # Format: "0.transformer.text_model...." or "1.transformer.text_model...."
    if not remaining:
        return None

    # Find the embedder index (first segment before the dot)
    if "." not in remaining:
        return None

    embedder_idx_str, rest = remaining.split(".", 1)
    try:
        embedder_idx = int(embedder_idx_str)
    except ValueError:
        return None

    # Map to CLIP model prefix
    model_prefix = _EMBEDDER_MAPPING.get(embedder_idx)
    if model_prefix is None:
        # Unknown embedder index - skip it
        return None

    # Build the normalized key
    # Input:  conditioner.embedders.0.transformer.text_model.encoder...
    # Output: clip_l.transformer.text_model.encoder...
    return f"{model_prefix}.{rest}"


def _validate_embedder_structure(
    file_keys: list[str],
) -> tuple[bool, bool]:
    """Validate the embedder structure and determine which encoders are present.

    Args:
        file_keys: All keys from the checkpoint file

    Returns:
        Tuple of (has_clip_l, has_clip_g) indicating which encoders were found

    # AC: @clip-model-loader ac-7
    Raises clear error when embedder structure is unexpected.
    """
    has_embedder_0 = False
    has_embedder_1 = False
    unexpected_embedders: set[int] = set()

    for key in file_keys:
        for prefix in _INCLUDED_PREFIXES:
            if key.startswith(prefix):
                remaining = key[len(prefix) :]
                if "." in remaining:
                    idx_str = remaining.split(".", 1)[0]
                    try:
                        idx = int(idx_str)
                        if idx == 0:
                            has_embedder_0 = True
                        elif idx == 1:
                            has_embedder_1 = True
                        else:
                            unexpected_embedders.add(idx)
                    except ValueError:
                        pass
                break

    # AC: @clip-model-loader ac-7
    # Raise error for unexpected embedders
    if unexpected_embedders:
        raise CLIPKeyMappingError(
            f"Unexpected embedder indices in checkpoint: {sorted(unexpected_embedders)}. "
            f"Expected only embedders.0 (CLIP-L) and embedders.1 (CLIP-G) for SDXL."
        )

    return has_embedder_0, has_embedder_1


class CLIPModelLoader:
    """Streaming model loader for CLIP text encoder weights from checkpoints.

    Uses safe_open() for memory-mapped access to checkpoint weights.
    Tensors are loaded on-demand via get_weights() without loading
    the full file into memory.

    # AC: @clip-model-loader ac-1
    Exposes text encoder keys (conditioner.embedders.*) and excludes
    diffusion model and VAE keys.

    # AC: @clip-model-loader ac-3
    affected_keys returns the frozenset of CLIP base model keys.

    # AC: @clip-model-loader ac-4
    cleanup() closes the file handle.

    # AC: @clip-model-loader ac-5
    Non-safetensors files raise UnsupportedFormatError.
    """

    def __init__(self, path: str) -> None:
        """Open a checkpoint file for streaming access to CLIP weights.

        Args:
            path: Path to safetensors checkpoint file

        Raises:
            UnsupportedFormatError: If file is not safetensors format
            FileNotFoundError: If file doesn't exist
            CLIPKeyMappingError: If checkpoint has unexpected embedder structure

        # AC: @clip-model-loader ac-5
        """
        # AC: @clip-model-loader ac-5
        # Only support safetensors format
        path_obj = Path(path)
        if path_obj.suffix.lower() not in (".safetensors",):
            raise UnsupportedFormatError(
                f"Only safetensors format is supported for CLIP model loading. "
                f"Got: {path_obj.suffix}. "
                f"Please convert your checkpoint to safetensors format."
            )

        # Open with safe_open for memory-mapped access
        self._handle = safe_open(path, framework="pt", device="cpu")
        self._path = path

        # Get all file keys for validation
        file_keys = list(self._handle.keys())

        # AC: @clip-model-loader ac-7
        # Validate embedder structure before building mappings
        has_clip_l, has_clip_g = _validate_embedder_structure(file_keys)

        # AC: @clip-model-loader ac-6
        # Build key mappings at open time (no tensor loading)
        # Forward: file_key -> normalized_key
        # Reverse: normalized_key -> file_key (for lookups)
        self._file_to_normalized: dict[str, str] = {}
        self._normalized_to_file: dict[str, str] = {}

        for file_key in file_keys:
            normalized = _normalize_clip_key(file_key)
            if normalized is not None:
                self._file_to_normalized[file_key] = normalized
                self._normalized_to_file[normalized] = file_key

        # AC: @clip-model-loader ac-3
        # Store affected keys as frozenset
        self._affected_keys = frozenset(self._normalized_to_file.keys())

        # Store encoder presence info
        self._has_clip_l = has_clip_l
        self._has_clip_g = has_clip_g

    @property
    def affected_keys(self) -> frozenset[str]:
        """Return set of CLIP base model keys available in this checkpoint.

        Keys are in CLIP base model format (e.g., clip_l.transformer.text_model.*).
        Excludes diffusion model and VAE keys.

        # AC: @clip-model-loader ac-3
        """
        return self._affected_keys

    @property
    def loaded_bytes(self) -> int:
        """Return 0 because CLIPModelLoader uses memory-mapped streaming access."""
        return 0

    @property
    def has_clip_l(self) -> bool:
        """Return True if CLIP-L encoder keys are present."""
        return self._has_clip_l

    @property
    def has_clip_g(self) -> bool:
        """Return True if CLIP-G encoder keys are present."""
        return self._has_clip_g

    def get_weights(self, keys: list[str]) -> list[torch.Tensor]:
        """Get weight tensors for the given CLIP base model keys.

        Args:
            keys: List of CLIP base model parameter keys

        Returns:
            List of tensors in the same order as keys

        Raises:
            KeyError: If any key doesn't exist in the checkpoint

        # AC: @clip-model-loader ac-2
        Returns text encoder weight tensors mapped to CLIP base model key format.
        """
        tensors: list[torch.Tensor] = []
        missing_keys: list[str] = []

        for key in keys:
            file_key = self._normalized_to_file.get(key)
            if file_key is None:
                missing_keys.append(key)
            else:
                tensors.append(self._handle.get_tensor(file_key))

        if missing_keys:
            raise KeyError(
                f"Checkpoint '{self._path}' is missing {len(missing_keys)} CLIP key(s):\n"
                + "\n".join(f"  - {k}" for k in missing_keys[:10])
                + (
                    f"\n  ... and {len(missing_keys) - 10} more"
                    if len(missing_keys) > 10
                    else ""
                )
            )

        return tensors

    def cleanup(self) -> None:
        """Close the file handle and release resources.

        # AC: @clip-model-loader ac-4
        """
        if hasattr(self, "_handle") and self._handle is not None:
            # safe_open returns a SafetensorHandle which doesn't have an explicit close,
            # but we can delete the reference to allow garbage collection
            del self._handle
            self._handle = None  # type: ignore[assignment]

    def __enter__(self) -> CLIPModelLoader:
        """Support context manager usage."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Cleanup on context exit."""
        self.cleanup()
