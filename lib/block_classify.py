"""Block Classification for Per-Block Weight Control.

Maps parameter keys to block groups for architecture-specific weight control.
Each architecture has its own classification function that returns the block group
name matching the BlockConfig block_overrides patterns.

This module is pure Python with no external dependencies.

# AC: @merge-block-config ac-1
# AC: @lora-block-config ac-1
"""

import re
from collections.abc import Callable

__all__ = [
    "classify_key",
    "get_block_classifier",
    "classify_key_sdxl",
    "classify_key_zimage",
]


def classify_key_sdxl(key: str) -> str | None:
    """Classify an SDXL parameter key into a block group.

    SDXL block structure matches WIDENBlockConfigSDXLNode sliders:
    - input_blocks.0-2 → IN00-02
    - input_blocks.3-5 → IN03-05
    - input_blocks.6-8 → IN06-08
    - middle_block → MID
    - output_blocks.0-2 → OUT00-02
    - output_blocks.3-5 → OUT03-05
    - output_blocks.6-8 → OUT06-08

    Args:
        key: Parameter key (with or without diffusion_model. prefix)

    Returns:
        Block group name (e.g., "IN00-02", "MID") or None if no match
    """
    # Strip common prefixes
    if key.startswith("diffusion_model."):
        key = key[len("diffusion_model.") :]

    # Match input_blocks.N
    match = re.match(r"input_blocks\.(\d+)\.", key)
    if match:
        block_num = int(match.group(1))
        if 0 <= block_num <= 2:
            return "IN00-02"
        elif 3 <= block_num <= 5:
            return "IN03-05"
        elif 6 <= block_num <= 8:
            return "IN06-08"
        # Block numbers 9-11 exist in some SDXL variants
        return None

    # Match middle_block
    if key.startswith("middle_block."):
        return "MID"

    # Match output_blocks.N
    match = re.match(r"output_blocks\.(\d+)\.", key)
    if match:
        block_num = int(match.group(1))
        if 0 <= block_num <= 2:
            return "OUT00-02"
        elif 3 <= block_num <= 5:
            return "OUT03-05"
        elif 6 <= block_num <= 8:
            return "OUT06-08"
        return None

    # No block match (e.g., time_embed, label_emb at top level)
    return None


def classify_key_zimage(key: str) -> str | None:
    """Classify a Z-Image/S3-DiT parameter key into a block group.

    Z-Image block structure matches WIDENBlockConfigZImageNode sliders:
    - layers.0-4 → L00-04
    - layers.5-9 → L05-09
    - layers.10-14 → L10-14
    - layers.15-19 → L15-19
    - layers.20-24 → L20-24
    - layers.25-29 → L25-29
    - noise_refiner → noise_refiner
    - context_refiner → context_refiner

    Args:
        key: Parameter key (with or without transformer./diffusion_model. prefix)

    Returns:
        Block group name (e.g., "L00-04", "noise_refiner") or None if no match
    """
    # Strip common prefixes
    for prefix in ("diffusion_model.", "transformer."):
        if key.startswith(prefix):
            key = key[len(prefix) :]

    # Match layers.N or blocks.N (S3-DiT may use either)
    match = re.match(r"(?:layers|blocks)\.(\d+)\.", key)
    if match:
        layer_num = int(match.group(1))
        if 0 <= layer_num <= 4:
            return "L00-04"
        elif 5 <= layer_num <= 9:
            return "L05-09"
        elif 10 <= layer_num <= 14:
            return "L10-14"
        elif 15 <= layer_num <= 19:
            return "L15-19"
        elif 20 <= layer_num <= 24:
            return "L20-24"
        elif 25 <= layer_num <= 29:
            return "L25-29"
        return None

    # Match refiners
    if "noise_refiner" in key:
        return "noise_refiner"
    if "context_refiner" in key:
        return "context_refiner"

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


def classify_key(key: str, arch: str) -> str | None:
    """Classify a parameter key into a block group for the given architecture.

    Convenience function that looks up and applies the appropriate classifier.

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
