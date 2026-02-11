"""Z-Image Architecture LoRA Loader.

Handles Z-Image S3-DiT key mapping with QKV fusing logic.
Z-Image uses fused attention.qkv.weight (11520x3840 = 3x3840) in the
base model, but LoRAs provide separate to_q/to_k/to_v keys.

# AC: @lora-loaders ac-1
Z-Image loader handles QKV fusing and architecture-specific key mapping.
"""

from collections import defaultdict
from collections.abc import Sequence

import torch
from safetensors import safe_open

from lib.executor import DeltaSpec
from lib.lora.base import LoRALoader

__all__ = ["ZImageLoader"]


def _parse_zimage_lora_key(lora_key: str) -> tuple[str | None, str, str | None]:
    """Parse a Z-Image LoRA key into (model_key, direction, qkv_component).

    Z-Image LoRA keys follow patterns like:
    - transformer.layers.0.attention.to_q.lora_A.weight
    - transformer.layers.0.attention.to_k.lora_B.weight
    - transformer.layers.0.ff.linear_1.lora_A.weight

    Model keys are:
    - layers.0.attention.qkv.weight (fused Q/K/V)
    - layers.0.feed_forward.linear_1.weight

    Args:
        lora_key: Key from LoRA safetensors file

    Returns:
        (model_key, direction, qkv_component) where:
        - model_key: Corresponding base model key (None if unsupported)
        - direction: 'up' (lora_B) or 'down' (lora_A)
        - qkv_component: 'q', 'k', 'v' for QKV weights, None otherwise

    # AC: @lora-loaders ac-1
    """
    # Z-Image uses lora_A (down) and lora_B (up) naming
    if ".lora_A." in lora_key:
        direction = "down"
    elif ".lora_B." in lora_key:
        direction = "up"
    else:
        return None, "", None

    # Extract layer path
    base_path = lora_key.rsplit(".lora_", 1)[0]

    # Handle transformer prefix if present
    if base_path.startswith("transformer."):
        base_path = base_path[len("transformer.") :]

    # Check for QKV components
    qkv_component = None
    if ".to_q" in base_path:
        qkv_component = "q"
        # Map to fused QKV key
        model_key = base_path.replace(".to_q", ".qkv")
    elif ".to_k" in base_path:
        qkv_component = "k"
        model_key = base_path.replace(".to_k", ".qkv")
    elif ".to_v" in base_path:
        qkv_component = "v"
        model_key = base_path.replace(".to_v", ".qkv")
    else:
        model_key = base_path

    # Add diffusion_model prefix and .weight suffix
    model_key = f"diffusion_model.{model_key}.weight"

    return model_key, direction, qkv_component


class ZImageLoader(LoRALoader):
    """Z-Image S3-DiT LoRA loader with QKV fusing support.

    Z-Image's base model uses fused attention.qkv weights, but LoRAs
    provide separate to_q/to_k/to_v. This loader maps them correctly
    and produces DeltaSpec objects with appropriate qkv_* kinds.

    # AC: @lora-loaders ac-1
    Architecture-specific loader for Z-Image key mapping with QKV fusing.

    # AC: @lora-loaders ac-2
    Produces DeltaSpec objects compatible with batched executor.
    """

    def __init__(self) -> None:
        """Initialize empty loader state."""
        # Standard LoRA data: model_key → list of (up, down, scale)
        self._lora_data: dict[str, list[tuple[torch.Tensor, torch.Tensor, float]]] = (
            defaultdict(list)
        )
        # QKV LoRA data: model_key → list of (up, down, scale, qkv_component)
        self._qkv_data: dict[
            str, list[tuple[torch.Tensor, torch.Tensor, float, str]]
        ] = defaultdict(list)
        self._affected: set[str] = set()

    def load(self, path: str, strength: float = 1.0) -> None:
        """Load a LoRA safetensors file.

        # AC: @lora-loaders ac-1
        Handles Z-Image key mapping with QKV fusing.
        """
        # Collect tensors by layer path and direction
        layer_tensors: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
        # Track which keys have QKV components
        qkv_info: dict[str, str | None] = {}

        with safe_open(path, framework="pt", device="cpu") as f:
            for lora_key in f.keys():
                model_key, direction, qkv_comp = _parse_zimage_lora_key(lora_key)
                if model_key is None:
                    continue

                tensor = f.get_tensor(lora_key)

                # For QKV, we need to track each component separately
                if qkv_comp is not None:
                    qkv_layer_key = f"{model_key}:{qkv_comp}"
                    layer_tensors[qkv_layer_key][direction] = tensor
                    qkv_info[qkv_layer_key] = qkv_comp
                else:
                    layer_tensors[model_key][direction] = tensor
                    qkv_info[model_key] = None

        # Build delta data for complete up/down pairs
        for layer_key, tensors in layer_tensors.items():
            if "up" not in tensors or "down" not in tensors:
                continue

            up = tensors["up"]
            down = tensors["down"]

            # Compute scale: strength * alpha / rank
            rank = down.shape[0]
            alpha = rank  # Default
            scale = strength * alpha / rank

            qkv_comp = qkv_info.get(layer_key)

            if qkv_comp is not None:
                # QKV component - extract actual model key
                model_key = layer_key.rsplit(":", 1)[0]
                self._qkv_data[model_key].append((up, down, scale, qkv_comp))
                self._affected.add(model_key)
            else:
                # Standard LoRA
                self._lora_data[layer_key].append((up, down, scale))
                self._affected.add(layer_key)

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
        Produces DeltaSpec objects compatible with batched executor,
        including qkv_q/qkv_k/qkv_v kinds for fused attention weights.
        """
        specs: list[DeltaSpec] = []

        for key in keys:
            key_idx = key_indices.get(key)
            if key_idx is None:
                continue

            # Handle standard LoRA data
            if key in self._lora_data:
                for up, down, scale in self._lora_data[key]:
                    if up.dim() == 2 and down.dim() == 2:
                        spec = DeltaSpec(
                            kind="standard",
                            key_index=key_idx,
                            up=up,
                            down=down,
                            scale=scale,
                        )
                        specs.append(spec)

            # Handle QKV LoRA data
            if key in self._qkv_data:
                for up, down, scale, qkv_comp in self._qkv_data[key]:
                    if up.dim() == 2 and down.dim() == 2:
                        kind = f"qkv_{qkv_comp}"
                        spec = DeltaSpec(
                            kind=kind,
                            key_index=key_idx,
                            up=up,
                            down=down,
                            scale=scale,
                        )
                        specs.append(spec)

        return specs

    def cleanup(self) -> None:
        """Release loaded tensors.

        # AC: @lora-loaders ac-4
        """
        self._lora_data.clear()
        self._qkv_data.clear()
        self._affected.clear()
