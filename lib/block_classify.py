"""Block Classification for Per-Block Weight Control.

Maps parameter keys to block groups for architecture-specific weight control.
Each architecture has its own classification function that returns the block group
name matching the BlockConfig block_overrides patterns.

This module is pure Python with no external dependencies.

# AC: @merge-block-config ac-1
# AC: @lora-block-config ac-1
"""

import functools
import re
from collections.abc import Callable

__all__ = [
    "classify_key",
    "get_block_classifier",
    "classify_key_sdxl",
    "classify_key_zimage",
]


@functools.lru_cache(maxsize=4096)
def classify_key_sdxl(key: str) -> str | None:
    """Classify an SDXL parameter key into an individual block.

    SDXL block structure matches WIDENBlockConfigSDXLNode sliders:
    - input_blocks.0-8 → IN00-IN08 (9 individual blocks)
    - middle_block → MID (single block)
    - output_blocks.0-8 → OUT00-OUT08 (9 individual blocks)

    Args:
        key: Parameter key (with or without diffusion_model. prefix)

    Returns:
        Individual block name (e.g., "IN00", "MID", "OUT05") or None if no match
    """
    # Strip common prefixes
    if key.startswith("diffusion_model."):
        key = key[len("diffusion_model.") :]

    # Match input_blocks.N
    match = re.match(r"input_blocks\.(\d+)\.", key)
    if match:
        block_num = int(match.group(1))
        if 0 <= block_num <= 8:
            return f"IN{block_num:02d}"
        # Block numbers 9-11 exist in some SDXL variants
        return None

    # Match middle_block
    if key.startswith("middle_block."):
        return "MID"

    # Match output_blocks.N
    match = re.match(r"output_blocks\.(\d+)\.", key)
    if match:
        block_num = int(match.group(1))
        if 0 <= block_num <= 8:
            return f"OUT{block_num:02d}"
        return None

    # No block match (e.g., time_embed, label_emb at top level)
    return None


@functools.lru_cache(maxsize=4096)
def classify_key_zimage(key: str) -> str | None:
    """Classify a Z-Image/S3-DiT parameter key into an individual block.

    Z-Image block structure matches WIDENBlockConfigZImageNode sliders:
    - layers.0-29 → L00-L29 (30 individual blocks)
    - noise_refiner.0-1 → NOISE_REF0, NOISE_REF1 (2 blocks)
    - context_refiner.0-1 → CTX_REF0, CTX_REF1 (2 blocks)

    Args:
        key: Parameter key (with or without transformer./diffusion_model. prefix)

    Returns:
        Individual block name (e.g., "L00", "NOISE_REF0") or None if no match
    """
    # Strip common prefixes
    for prefix in ("diffusion_model.", "transformer."):
        if key.startswith(prefix):
            key = key[len(prefix) :]

    # Match layers.N or blocks.N (S3-DiT may use either)
    match = re.match(r"(?:layers|blocks)\.(\d+)\.", key)
    if match:
        layer_num = int(match.group(1))
        if 0 <= layer_num <= 29:
            return f"L{layer_num:02d}"
        return None

    # Match noise_refiner.N (nn.ModuleList sub-modules)
    match = re.match(r"noise_refiner\.(\d+)\.", key)
    if match:
        refiner_num = int(match.group(1))
        return f"NOISE_REF{refiner_num}"

    # Match context_refiner.N (nn.ModuleList sub-modules)
    match = re.match(r"context_refiner\.(\d+)\.", key)
    if match:
        refiner_num = int(match.group(1))
        return f"CTX_REF{refiner_num}"

    # No block match
    return None


# Registry of architecture classifiers
_CLASSIFIERS: dict[str, Callable[[str], str | None]] = {
    "sdxl": classify_key_sdxl,
    "zimage": classify_key_zimage,
}


def get_block_classifier(arch: str) -> Callable[[str], str | None] | None:
    """Get the block classifier function for an architecture.

    Args:
        arch: Architecture name (e.g., "sdxl", "zimage")

    Returns:
        Classifier function or None if architecture not supported
    """
    return _CLASSIFIERS.get(arch)


@functools.lru_cache(maxsize=4096)
def classify_key(key: str, arch: str) -> str | None:
    """Classify a parameter key into a block group for the given architecture.

    Convenience function that looks up and applies the appropriate classifier.
    Cached to avoid repeated dict lookups when called in per-key loops.

    Args:
        key: Parameter key
        arch: Architecture name

    Returns:
        Block group name or None if no match or unsupported architecture
    """
    classifier = get_block_classifier(arch)
    if classifier is None:
        return None
    return classifier(key)
