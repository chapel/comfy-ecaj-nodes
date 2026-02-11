"""SDXL Architecture LoRA Loader.

Handles SDXL UNet key mapping from LoRA format to model format.
SDXL LoRAs typically use kohya/A1111 naming conventions:

LoRA key format:
  lora_unet_{block}.{layer}.{component}.lora_{up|down}.weight
  lora_te_{encoder}_{layer}.lora_{up|down}.weight (text encoders)

Model key format:
  diffusion_model.{block}.{layer}.{component}.weight

# AC: @lora-loaders ac-1
SDXL loader handles SDXL-specific key mapping.
"""

from collections import defaultdict
from collections.abc import Sequence

import torch
from safetensors import safe_open

from lib.executor import DeltaSpec
from lib.lora.base import LoRALoader

__all__ = ["SDXLLoader"]


# Prefix mapping: LoRA naming → model state dict prefix
_LORA_TO_MODEL_PREFIX = {
    "lora_unet_": "diffusion_model.",
}


def _parse_lora_key(lora_key: str) -> tuple[str | None, str, str]:
    """Parse a LoRA key into (model_key, component, direction).

    Args:
        lora_key: Key from LoRA safetensors (e.g. 'lora_unet_input_blocks_*.lora_up.weight')

    Returns:
        (model_key, component, direction) tuple, where:
        - model_key: Corresponding base model key (None if not a unet LoRA)
        - component: 'up' or 'down'
        - direction: Full component name for matching

    # AC: @lora-loaders ac-1
    """
    # Skip non-unet keys (text encoders handled separately if needed)
    if not lora_key.startswith("lora_unet_"):
        return None, "", ""

    # Extract the component direction (lora_up or lora_down)
    if ".lora_up." in lora_key:
        direction = "up"
    elif ".lora_down." in lora_key:
        direction = "down"
    else:
        return None, "", ""

    # Remove prefix and suffix to get the layer path
    # lora_unet_input_blocks_0_0_proj_in.lora_up.weight
    # → input_blocks_0_0_proj_in
    layer_path = lora_key[len("lora_unet_") :]
    # Remove .lora_{up|down}.weight suffix
    layer_path = layer_path.rsplit(".lora_", 1)[0]

    # Convert underscores to dots for nested keys
    # input_blocks_0_0_proj_in → input_blocks.0.0.proj_in
    # This handles the kohya naming convention
    parts = layer_path.split("_")
    converted_parts = []
    i = 0
    while i < len(parts):
        part = parts[i]
        # Check if this is a numeric index that should stay attached
        if part.isdigit():
            converted_parts.append(part)
        else:
            converted_parts.append(part)
        i += 1

    # Reconstruct with proper dot separation
    # Need to handle block indices specially
    model_key = "diffusion_model."
    segment = []
    for part in converted_parts:
        if part.isdigit() and segment:
            # This is an index, append to previous segment
            segment.append(part)
        else:
            if segment:
                model_key += ".".join(segment) + "."
            segment = [part]
    if segment:
        model_key += ".".join(segment)

    # Ensure we have .weight suffix
    model_key += ".weight"

    return model_key, direction, lora_key


class SDXLLoader(LoRALoader):
    """SDXL-specific LoRA loader.

    Loads LoRA files in kohya/A1111 format and maps keys to SDXL UNet format.
    Accumulates multiple LoRAs and produces DeltaSpec objects for batched
    GPU evaluation.

    # AC: @lora-loaders ac-1
    Architecture-specific loader for SDXL key mapping.

    # AC: @lora-loaders ac-2
    Produces DeltaSpec objects compatible with batched executor.
    """

    def __init__(self) -> None:
        """Initialize empty loader state."""
        # Accumulated LoRA data: model_key → list of (up, down, scale)
        self._lora_data: dict[str, list[tuple[torch.Tensor, torch.Tensor, float]]] = (
            defaultdict(list)
        )
        self._affected: set[str] = set()

    def load(self, path: str, strength: float = 1.0) -> None:
        """Load a LoRA safetensors file.

        # AC: @lora-loaders ac-1
        Handles SDXL key mapping from kohya format.
        """
        # Collect up/down pairs keyed by layer path
        layer_tensors: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)

        with safe_open(path, framework="pt", device="cpu") as f:
            for lora_key in f.keys():
                model_key, direction, _ = _parse_lora_key(lora_key)
                if model_key is None:
                    continue

                tensor = f.get_tensor(lora_key)
                layer_tensors[model_key][direction] = tensor

        # Build delta specs for complete up/down pairs
        for model_key, tensors in layer_tensors.items():
            if "up" not in tensors or "down" not in tensors:
                continue

            up = tensors["up"]
            down = tensors["down"]

            # Compute scale: strength * alpha / rank
            # Alpha defaults to rank if not specified in the file
            rank = down.shape[0]
            alpha = rank  # Default; could read from file metadata if present
            scale = strength * alpha / rank

            self._lora_data[model_key].append((up, down, scale))
            self._affected.add(model_key)

    @property
    def affected_keys(self) -> set[str]:
        """Return keys that loaded LoRAs modify.

        # AC: @lora-loaders ac-4
        """
        return self._affected

    def get_delta_specs(
        self,
        keys: Sequence[str],
        key_indices: dict[str, int],
    ) -> list[DeltaSpec]:
        """Produce DeltaSpec objects for batched GPU evaluation.

        # AC: @lora-loaders ac-2
        Produces DeltaSpec objects compatible with batched executor.
        """
        specs: list[DeltaSpec] = []

        for key in keys:
            if key not in self._lora_data:
                continue

            key_idx = key_indices[key]

            for up, down, scale in self._lora_data[key]:
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
                    target_shape = (up.shape[0], down.shape[1], *up.shape[2:])
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
        self._lora_data.clear()
        self._affected.clear()
