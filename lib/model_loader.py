"""Full Model Loader -- streaming loader for checkpoint merging.

Uses safetensors.safe_open() for memory-efficient per-batch access to
full checkpoint weights. Handles key normalization between checkpoint file
format and base model state dict format.

Only supports safetensors format. Non-safetensors files raise a clear error.
"""

from __future__ import annotations

from pathlib import Path

import torch
from safetensors import safe_open

__all__ = ["ModelLoader", "UnsupportedFormatError", "KeyMismatchError"]


class UnsupportedFormatError(ValueError):
    """Raised when attempting to load a non-safetensors checkpoint."""

    pass


class KeyMismatchError(ValueError):
    """Raised when checkpoint keys don't match expected base model keys."""

    pass


# Prefixes to strip from checkpoint file keys to get base model format.
# Ordered by specificity (longer prefixes first).
_FILE_KEY_PREFIXES = (
    "model.diffusion_model.",  # SDXL checkpoint format
    "model.transformer.",      # Qwen checkpoint format (model.transformer.X)
    "diffusion_model.",        # Some Z-Image/Diffusers formats
    "transformer.",            # Alternate Z-Image/Qwen format
)

# Prefixes that identify non-diffusion keys (VAE, text encoder) to exclude.
_EXCLUDED_PREFIXES = (
    "first_stage_model.",      # VAE
    "model.first_stage_model.",
    "conditioner.",            # Text encoder (SDXL)
    "model.conditioner.",
    "cond_stage_model.",       # Text encoder (SD 1.x/2.x)
    "model.cond_stage_model.",
    "encoder.",                # VAE encoder
    "decoder.",                # VAE decoder
    "quant_conv.",             # VAE quantization
    "post_quant_conv.",        # VAE post-quantization
)

# Architecture detection patterns (applied to NORMALIZED keys).
# These match the patterns in nodes/entry.py but for file-derived keys.
_ARCH_PATTERNS = (
    # Z-Image: layers.N with noise_refiner
    (
        "zimage",
        lambda keys: any("diffusion_model.layers." in k for k in keys)
        and any("noise_refiner" in k for k in keys),
    ),
    # SDXL: input_blocks, middle_block, output_blocks structure
    (
        "sdxl",
        lambda keys: any("diffusion_model.input_blocks." in k for k in keys)
        and any("diffusion_model.middle_block." in k for k in keys)
        and any("diffusion_model.output_blocks." in k for k in keys),
    ),
    # Qwen: transformer_blocks at depth 60+ (matches nodes/entry.py pattern)
    # AC: @qwen-model-loader ac-7
    (
        "qwen",
        lambda keys: sum(1 for k in keys if "transformer_blocks" in k) >= 60,
    ),
    # Flux Klein: double_blocks structure (4B: 5 double + 20 single, 9B: 8 double + 24 single)
    # Must not match Qwen which uses transformer_blocks instead of double_blocks.
    # AC: @flux-model-loader ac-8
    (
        "flux",
        lambda keys: any("double_blocks" in k for k in keys),
    ),
)


def _normalize_key(file_key: str) -> str | None:
    """Normalize a checkpoint file key to base model format.

    Args:
        file_key: Key from checkpoint safetensors file

    Returns:
        Normalized key in base model format (with diffusion_model. prefix),
        or None if the key should be excluded (VAE, text encoder).

    # AC: @full-model-loader ac-3
    Strips model.diffusion_model prefix for SDXL.

    # AC: @full-model-loader ac-4
    Handles diffusion_model or transformer prefix variants for Z-Image.
    """
    # Exclude non-diffusion keys
    for prefix in _EXCLUDED_PREFIXES:
        if file_key.startswith(prefix):
            return None

    # Strip known file prefixes
    normalized = file_key
    for prefix in _FILE_KEY_PREFIXES:
        if file_key.startswith(prefix):
            # Strip the prefix but keep diffusion_model. for base model format
            suffix = file_key[len(prefix) :]
            normalized = f"diffusion_model.{suffix}"
            break
    else:
        # If no prefix matched but key starts with diffusion_model, keep as-is
        if not file_key.startswith("diffusion_model."):
            # Not a diffusion model key we recognize
            return None

    return normalized


def _detect_architecture_from_keys(normalized_keys: frozenset[str]) -> str | None:
    """Detect architecture from normalized checkpoint keys.

    Args:
        normalized_keys: Set of normalized keys (with diffusion_model. prefix)

    Returns:
        Architecture string ("sdxl", "zimage") or None if unknown.

    # AC: @full-model-loader ac-8
    Determines architecture without loading tensor data.
    """
    for arch, pattern_fn in _ARCH_PATTERNS:
        if pattern_fn(normalized_keys):
            return arch
    return None


class ModelLoader:
    """Streaming model loader for full checkpoint merging.

    Uses safe_open() for memory-mapped access to checkpoint weights.
    Tensors are loaded on-demand via get_weights() without loading
    the full file into memory.

    # AC: @full-model-loader ac-1
    Uses safe_open() for memory-mapped access.

    # AC: @full-model-loader ac-5
    affected_keys returns diffusion model keys, excluding VAE/text encoder.

    # AC: @full-model-loader ac-6
    cleanup() closes the file handle.

    # AC: @full-model-loader ac-9
    Non-safetensors files raise UnsupportedFormatError.
    """

    def __init__(self, path: str) -> None:
        """Open a checkpoint file for streaming access.

        Args:
            path: Path to safetensors checkpoint file

        Raises:
            UnsupportedFormatError: If file is not safetensors format
            FileNotFoundError: If file doesn't exist
        """
        # AC: @full-model-loader ac-9
        # Only support safetensors format
        path_obj = Path(path)
        if path_obj.suffix.lower() not in (".safetensors",):
            raise UnsupportedFormatError(
                f"Only safetensors format is supported for model merging. "
                f"Got: {path_obj.suffix}. "
                f"Please convert your checkpoint to safetensors format."
            )

        # AC: @full-model-loader ac-1
        # Open with safe_open for memory-mapped access
        self._handle = safe_open(path, framework="pt", device="cpu")
        self._path = path

        # Build key mappings at open time (no tensor loading)
        # Forward: file_key -> normalized_key
        # Reverse: normalized_key -> file_key (for lookups)
        self._file_to_normalized: dict[str, str] = {}
        self._normalized_to_file: dict[str, str] = {}

        for file_key in self._handle.keys():
            normalized = _normalize_key(file_key)
            if normalized is not None:
                self._file_to_normalized[file_key] = normalized
                self._normalized_to_file[normalized] = file_key

        # AC: @full-model-loader ac-5
        # Store affected keys as frozenset
        self._affected_keys = frozenset(self._normalized_to_file.keys())

        # AC: @full-model-loader ac-8
        # Detect architecture from normalized keys
        self._arch = _detect_architecture_from_keys(self._affected_keys)

    @property
    def affected_keys(self) -> frozenset[str]:
        """Return set of base model keys available in this checkpoint.

        Keys are in base model format (e.g., diffusion_model.input_blocks.0.0.weight).
        Excludes VAE and text encoder keys.

        # AC: @full-model-loader ac-5
        """
        return self._affected_keys

    @property
    def arch(self) -> str | None:
        """Return detected architecture or None if unknown.

        # AC: @full-model-loader ac-8
        """
        return self._arch

    def get_weights(self, keys: list[str]) -> list[torch.Tensor]:
        """Get weight tensors for the given base model keys.

        Args:
            keys: List of base model parameter keys

        Returns:
            List of tensors in the same order as keys

        Raises:
            KeyMismatchError: If any key doesn't exist in the checkpoint

        # AC: @full-model-loader ac-2
        Returns correctly mapped weight tensors from file.

        # AC: @full-model-loader ac-7
        Raises clear error for unmatched keys.
        """
        tensors: list[torch.Tensor] = []
        missing_keys: list[str] = []

        for key in keys:
            file_key = self._normalized_to_file.get(key)
            if file_key is None:
                missing_keys.append(key)
            else:
                tensors.append(self._handle.get_tensor(file_key))

        # AC: @full-model-loader ac-7
        if missing_keys:
            raise KeyMismatchError(
                f"Checkpoint '{self._path}' is missing {len(missing_keys)} key(s) "
                f"requested by base model:\n"
                + "\n".join(f"  - {k}" for k in missing_keys[:10])
                + (f"\n  ... and {len(missing_keys) - 10} more" if len(missing_keys) > 10 else "")
            )

        return tensors

    def cleanup(self) -> None:
        """Close the file handle and release resources.

        # AC: @full-model-loader ac-6
        """
        if hasattr(self, "_handle") and self._handle is not None:
            # safe_open returns a SafetensorHandle which doesn't have an explicit close,
            # but we can delete the reference to allow garbage collection
            del self._handle
            self._handle = None  # type: ignore[assignment]

    def __enter__(self) -> ModelLoader:
        """Support context manager usage."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Cleanup on context exit."""
        self.cleanup()
