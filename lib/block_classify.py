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
    "classify_layer_type",
    "get_block_classifier",
    "classify_key_sdxl",
    "classify_key_zimage",
    "classify_key_qwen",
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


@functools.lru_cache(maxsize=4096)
def classify_key_qwen(key: str) -> str | None:
    """Classify a Qwen parameter key into an individual block.

    Qwen block structure uses dynamic index discovery (not hardcoded to 60):
    - transformer_blocks.N → TB00, TB01, ... (dynamic range based on model)

    Args:
        key: Parameter key (with or without diffusion_model./transformer. prefix)

    Returns:
        Individual block name (e.g., "TB00", "TB59") or None if no match
    """
    # Strip common prefixes
    for prefix in ("diffusion_model.", "transformer."):
        if key.startswith(prefix):
            key = key[len(prefix) :]

    # Match transformer_blocks.N
    match = re.match(r"transformer_blocks\.(\d+)\.", key)
    if match:
        block_num = int(match.group(1))
        # Dynamic range - no upper bound check, format with width for sorting
        return f"TB{block_num:02d}"

    # No block match
    return None


# Registry of architecture classifiers
_CLASSIFIERS: dict[str, Callable[[str], str | None]] = {
    "sdxl": classify_key_sdxl,
    "zimage": classify_key_zimage,
    "qwen": classify_key_qwen,
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


# Layer type patterns for SDXL (order matters - first match wins)
# Precedence: attention > feed_forward > norm (per ac-7)
_SDXL_LAYER_PATTERNS: tuple[tuple[str, str], ...] = (
    # Attention patterns (most specific first)
    ("attn1", "attention"),
    ("attn2", "attention"),
    ("to_q", "attention"),
    ("to_k", "attention"),
    ("to_v", "attention"),
    ("to_out", "attention"),
    ("proj_in", "attention"),
    ("proj_out", "attention"),
    # Feed-forward patterns
    (".ff.", "feed_forward"),
    ("ff.net", "feed_forward"),
    # Norm patterns (most general last - excludes q_norm/k_norm via precedence)
    (".norm", "norm"),
    ("_norm", "norm"),
    ("ln_", "norm"),
)

# Layer type patterns for Z-Image/S3-DiT
_ZIMAGE_LAYER_PATTERNS: tuple[tuple[str, str], ...] = (
    # Attention patterns (including q_norm/k_norm per ac-7)
    ("attn.qkv", "attention"),
    ("attn.out", "attention"),
    ("q_norm", "attention"),
    ("k_norm", "attention"),
    # Feed-forward patterns
    ("feed_forward", "feed_forward"),
    (".mlp.", "feed_forward"),
    (".w1.", "feed_forward"),
    (".w2.", "feed_forward"),
    (".w3.", "feed_forward"),
    (".fc1", "feed_forward"),
    (".fc2", "feed_forward"),
    # Norm patterns
    (".norm", "norm"),
    ("_norm", "norm"),
    (".ln", "norm"),
    (".rms", "norm"),
)

# Layer type patterns for Qwen
_QWEN_LAYER_PATTERNS: tuple[tuple[str, str], ...] = (
    # Attention patterns
    (".attn.", "attention"),
    ("to_q", "attention"),
    ("to_k", "attention"),
    ("to_v", "attention"),
    ("to_out", "attention"),
    (".qkv", "attention"),
    (".proj", "attention"),
    # Feed-forward patterns
    (".mlp.", "feed_forward"),
    (".ff.", "feed_forward"),
    (".gate_proj", "feed_forward"),
    (".up_proj", "feed_forward"),
    (".down_proj", "feed_forward"),
    # Norm patterns
    (".norm", "norm"),
    ("_norm", "norm"),
    (".ln", "norm"),
)

# Registry of layer type patterns by architecture
_LAYER_TYPE_PATTERNS: dict[str, tuple[tuple[str, str], ...]] = {
    "sdxl": _SDXL_LAYER_PATTERNS,
    "zimage": _ZIMAGE_LAYER_PATTERNS,
    "qwen": _QWEN_LAYER_PATTERNS,
}


@functools.lru_cache(maxsize=4096)
def classify_layer_type(key: str, arch: str | None) -> str | None:
    """Classify a parameter key into a layer type for the given architecture.

    # AC: @layer-type-filter ac-1
    Returns one of: attention, feed_forward, norm, or None.

    # AC: @layer-type-filter ac-6
    Keys not matching any pattern (time_embed, label_emb, adaLN_modulation,
    embedders) return None.

    # AC: @layer-type-filter ac-7
    First-match-wins with precedence: attention > feed_forward > norm.

    # AC: @layer-type-filter ac-8
    Returns None for arch=None or unsupported architectures.

    Args:
        key: Parameter key
        arch: Architecture name (e.g., "sdxl", "zimage") or None

    Returns:
        Layer type ("attention", "feed_forward", "norm") or None
    """
    if arch is None:
        return None

    patterns = _LAYER_TYPE_PATTERNS.get(arch)
    if patterns is None:
        return None

    # Strip common prefixes for cleaner matching
    for prefix in ("diffusion_model.", "transformer."):
        if key.startswith(prefix):
            key = key[len(prefix) :]

    # Exclude known non-layer-type keys early (per ac-6)
    # These are conditioning/embedding projections, not layer components
    for excluded in ("time_embed", "label_emb", "adaLN_modulation", "embedders"):
        if excluded in key:
            return None

    # First match wins (patterns are ordered by precedence)
    for pattern, layer_type in patterns:
        if pattern in key:
            return layer_type

    return None
