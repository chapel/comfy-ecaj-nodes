"""Qwen Architecture LoRA Loader.

Handles Qwen transformer_blocks key mapping from LoRA format to model format.
Qwen LoRAs can use multiple naming conventions:

1. Diffusers format:
   transformer.transformer_blocks.0.attn.to_q.lora_A.weight
   transformer.transformer_blocks.0.attn.to_q.lora_B.weight

2. A1111/kohya format:
   lora_unet_transformer_blocks_0_attn_to_q.lora_up.weight
   lora_unet_transformer_blocks_0_attn_to_q.lora_down.weight

3. LyCORIS format:
   lycoris_transformer_blocks_0_attn_to_q.lora_down.weight
   lycoris_transformer_blocks_0_attn_to_q.lora_up.weight

Unlike Z-Image, Qwen uses separate to_q/to_k/to_v weights in the base model
(no QKV fusion required).

# AC: @qwen-lora-loader ac-4
Qwen loader handles Qwen-specific key mapping from all 3 formats.

# AC: @qwen-lora-loader ac-5
Compound name preservation maintains correct key structure.

# AC: @qwen-lora-loader ac-6
Standard up/down DeltaSpec production.
"""

from collections import defaultdict
from collections.abc import Sequence

import torch

from ..executor import DeltaSpec
from .base import LoRALoader
from .validation import read_pairs

__all__ = ["QwenLoader"]

# Compound names that should preserve underscores during key normalization.
# Sorted by length descending so longer variants match first.
_COMPOUND_NAMES = sorted(
    [
        # LoKr components
        "lokr_w1_a",
        "lokr_w1_b",
        "lokr_w2_a",
        "lokr_w2_b",
        "lokr_w1",
        "lokr_w2",
        "add_q_proj",
        "add_k_proj",
        "add_v_proj",
        "to_add_out",
        # LoRA components
        "lora_down",
        "lora_up",
        "lora_A",
        "lora_B",
        # Attention components
        "to_out",
        "to_q",
        "to_k",
        "to_v",
        # Qwen specific
        "transformer_blocks",
        "adaLN_modulation",
        "feed_forward",
        "gate_proj",
        "up_proj",
        "down_proj",
        # Modulation components
        "img_mlp",
        "txt_mlp",
        "img_mod",
        "txt_mod",
    ],
    key=len,
    reverse=True,
)


def _normalize_lycoris_key(key: str) -> str:
    """Normalize LyCORIS/LoKr key format to standard dot-separated format.

    # AC: @qwen-lora-loader ac-5
    Compound name preservation during key normalization.

    LyCORIS keys use underscore-separated paths with a lycoris_ prefix.
    This normalizes them to dot-separated paths matching model format.

    Example: lycoris_transformer_blocks_0_attn_to_q -> transformer_blocks.0.attn.to_q
    """
    if not key.startswith("lycoris_"):
        return key

    # Strip lycoris_ prefix
    key = key[len("lycoris_") :]

    # Replace compound names with placeholders
    placeholders = {}
    for i, compound in enumerate(_COMPOUND_NAMES):
        if compound in key:
            placeholder = f"{{{{COMPOUND{i}}}}}"
            key = key.replace(compound, placeholder)
            placeholders[placeholder] = compound

    # Convert remaining underscores to dots
    key = key.replace("_", ".")

    # Restore compound names
    for placeholder, compound in placeholders.items():
        key = key.replace(placeholder, compound)

    return key


def _normalize_kohya_key(key: str) -> str:
    """Normalize A1111/kohya key format to standard dot-separated format.

    # AC: @qwen-lora-loader ac-5
    Compound name preservation during key normalization.

    Kohya keys use lora_unet_ prefix with underscore-separated paths.
    Example: lora_unet_transformer_blocks_0_attn_to_q -> transformer_blocks.0.attn.to_q
    """
    if not key.startswith("lora_unet_"):
        return key

    # Strip lora_unet_ prefix
    key = key[len("lora_unet_") :]

    # Replace compound names with placeholders
    placeholders = {}
    for i, compound in enumerate(_COMPOUND_NAMES):
        if compound in key:
            placeholder = f"{{{{COMPOUND{i}}}}}"
            key = key.replace(compound, placeholder)
            placeholders[placeholder] = compound

    # Convert remaining underscores to dots
    key = key.replace("_", ".")

    # Restore compound names
    for placeholder, compound in placeholders.items():
        key = key.replace(placeholder, compound)

    return key


def _parse_qwen_lora_key(lora_key: str) -> tuple[str | None, str]:
    """Parse a Qwen LoRA key into (model_key, direction).

    Handles three LoRA formats:
    1. Diffusers: transformer.transformer_blocks.0.attn.to_q.lora_A.weight
    2. A1111/kohya: lora_unet_transformer_blocks_0_attn_to_q.lora_up.weight
    3. LyCORIS: lycoris_transformer_blocks_0_attn_to_q.lora_down.weight

    Args:
        lora_key: Key from LoRA safetensors file

    Returns:
        (model_key, direction) where:
        - model_key: Corresponding base model key (None if unsupported)
        - direction: 'up' (lora_B) or 'down' (lora_A)

    # AC: @qwen-lora-loader ac-4
    """
    key = lora_key

    # Determine direction from LoRA suffix
    # lora_A/lora_down = down, lora_B/lora_up = up
    if ".lora_A." in key or ".lora_down" in key:
        direction = "down"
    elif ".lora_B." in key or ".lora_up" in key:
        direction = "up"
    else:
        return None, ""

    # Extract base path by removing LoRA suffix
    if ".lora_A." in key:
        base_path = key.split(".lora_A.")[0]
    elif ".lora_B." in key:
        base_path = key.split(".lora_B.")[0]
    elif ".lora_down" in key:
        base_path = key.split(".lora_down")[0]
    elif ".lora_up" in key:
        base_path = key.split(".lora_up")[0]
    else:
        return None, ""

    # Handle diffusion_model prefix
    if base_path.startswith("diffusion_model."):
        base_path = base_path[len("diffusion_model.") :]

    # Handle transformer prefix
    if base_path.startswith("transformer."):
        base_path = base_path[len("transformer.") :]

    # Handle LyCORIS format
    if base_path.startswith("lycoris_"):
        base_path = _normalize_lycoris_key(base_path)
    # Handle A1111/kohya format
    elif base_path.startswith("lora_unet_"):
        base_path = _normalize_kohya_key(base_path)

    # Add diffusion_model prefix and .weight suffix for base model format
    if not base_path.startswith("diffusion_model."):
        model_key = f"diffusion_model.{base_path}"
    else:
        model_key = base_path

    if not model_key.endswith(".weight"):
        model_key = f"{model_key}.weight"

    return model_key, direction


class QwenLoader(LoRALoader):
    """Qwen-specific LoRA loader.

    Loads LoRA files in diffusers, kohya/A1111, or LyCORIS format and maps
    keys to Qwen transformer_blocks format. Accumulates multiple LoRAs and
    produces DeltaSpec objects for batched GPU evaluation. Data is segmented
    by set_id for correct scoping.

    Unlike Z-Image, Qwen uses separate to_q/to_k/to_v weights in the base
    model, so no QKV fusion is needed.

    # AC: @qwen-lora-loader ac-4
    Architecture-specific loader for Qwen key mapping.

    # AC: @qwen-lora-loader ac-6
    Produces DeltaSpec objects compatible with batched executor.
    """

    def __init__(self) -> None:
        """Initialize empty loader state."""
        # Accumulated LoRA data segmented by set:
        # set_id -> model_key -> list of (up, down, scale)
        self._lora_data_by_set: dict[
            str, dict[str, list[tuple[torch.Tensor, torch.Tensor, float]]]
        ] = defaultdict(lambda: defaultdict(list))
        # Per-set affected keys
        self._affected_by_set: dict[str, set[str]] = defaultdict(set)
        # Global affected keys (union of all sets)
        self._affected: set[str] = set()

    def load(self, path: str, strength: float = 1.0, set_id: str | None = None) -> None:
        """Validate the whole package before publishing factors to this set."""
        pending = read_pairs(path, strength, _parse_qwen_lora_key, conv=True)
        effective_set_id = set_id if set_id is not None else "__default__"
        for key, up, down, scale, extra in pending:
            self._lora_data_by_set[effective_set_id][key].append((up, down, scale))
            self._affected_by_set[effective_set_id].add(key)
            self._affected.add(key)

    @property
    def affected_keys(self) -> frozenset[str]:
        """Return keys that loaded LoRAs modify (all sets).

        Returns a frozenset to prevent external mutation of internal state.

        # AC: @lora-loaders ac-4
        """
        return frozenset(self._affected)

    def affected_keys_for_set(self, set_id: str) -> set[str]:
        """Return keys modified by a specific LoRA set.

        # AC: @lora-loaders ac-4
        """
        return self._affected_by_set.get(set_id, set())

    @property
    def loaded_bytes(self) -> int:
        """Return total bytes of tensors held in memory."""
        total = 0
        for key_data in self._lora_data_by_set.values():
            for entries in key_data.values():
                for up, down, _scale in entries:
                    total += up.nbytes + down.nbytes
        return total

    def get_delta_specs(
        self,
        keys: Sequence[str],
        key_indices: dict[str, int],
        set_id: str | None = None,
    ) -> list[DeltaSpec]:
        """Produce DeltaSpec objects for batched GPU evaluation.

        # AC: @qwen-lora-loader ac-6
        Produces standard DeltaSpec objects (no QKV fusing needed for Qwen).
        When set_id is provided, only returns deltas from that set.
        """
        specs: list[DeltaSpec] = []

        # Determine which data sources to iterate
        if set_id is not None:
            data_sources = [self._lora_data_by_set.get(set_id, {})]
        else:
            data_sources = list(self._lora_data_by_set.values())

        for lora_data in data_sources:
            for key in keys:
                if key not in lora_data:
                    continue

                key_idx = key_indices[key]

                for up, down, scale in lora_data[key]:
                    # Determine kind based on weight shape
                    if up.dim() == 2 and down.dim() == 2:
                        # Standard linear LoRA
                        spec = DeltaSpec(
                            kind="standard",
                            key_index=key_idx,
                            up=up,
                            down=down,
                            scale=scale,
                        )
                    elif up.dim() == 4 and down.dim() == 4:
                        # Conv2d LoRA - flatten for bmm, store target shape
                        target_shape = (up.shape[0], down.shape[1], *down.shape[2:])
                        up_flat = up.view(up.shape[0], -1)
                        down_flat = down.view(down.shape[0], -1)
                        spec = DeltaSpec(
                            kind="standard",
                            key_index=key_idx,
                            up=up_flat,
                            down=down_flat,
                            scale=scale,
                            target_shape=target_shape,
                        )
                    else:
                        # Skip unsupported shapes
                        continue

                    specs.append(spec)

        return specs

    def cleanup(self) -> None:
        """Release loaded tensors.

        # AC: @lora-loaders ac-4
        """
        self._lora_data_by_set.clear()
        self._affected_by_set.clear()
        self._affected.clear()
