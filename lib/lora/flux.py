"""Flux Klein Architecture LoRA Loader.

Handles Flux Klein (4B/9B) double_blocks and single_blocks key mapping from
LoRA format to model format. Flux Klein uses fused QKV weights that require
offset-based delta application.

## Flux Klein Structure

**Double blocks** (8 for 9B, 5 for 4B):
- Two attention streams: img_attn and txt_attn
- Each stream has fused qkv weight: double_blocks.N.{img,txt}_attn.qkv
- LoRAs provide separate to_q/to_k/to_v for each stream
- Use qkv_q/qkv_k/qkv_v DeltaSpec kinds with offset

**Single blocks** (24 for 9B, 20 for 4B):
- Fused linear1 weight containing Q/K/V + MLP projection (4-way split)
- LoRAs provide separate to_q/to_k/to_v/proj_mlp
- Q/K/V use qkv_q/qkv_k/qkv_v kinds with offset
- MLP projection uses offset_mlp kind with offset

## LoRA Formats Supported

1. BFL/kohya format:
   lora_unet_double_blocks_0_img_attn_to_q.lora_up.weight
   lora_unet_single_blocks_0_linear1_to_q.lora_up.weight

2. Diffusers format:
   transformer.double_blocks.0.img_attn.to_q.lora_A.weight
   transformer.single_blocks.0.linear1.to_q.lora_A.weight

# AC: @flux-lora-loader ac-4
Flux loader handles Flux Klein-specific key mapping with QKV fusing.

# AC: @flux-lora-loader ac-5
Double block img_attn/txt_attn QKV fusing with offset-based DeltaSpec.

# AC: @flux-lora-loader ac-6
Single block linear1 4-way fusing (to_q/to_k/to_v/proj_mlp) with offsets.

# AC: @flux-lora-loader ac-7
Produces DeltaSpec objects compatible with batched executor.
"""

import re
from collections import defaultdict
from collections.abc import Sequence

import torch
from safetensors import safe_open

from ..executor import DeltaSpec
from .base import LoRALoader

__all__ = ["FluxLoader"]

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
        # Flux specific
        "double_blocks",
        "single_blocks",
        "img_attn",
        "txt_attn",
        "img_mlp",
        "txt_mlp",
        "img_mod",
        "txt_mod",
        "proj_mlp",
        "linear1",
        "linear2",
        "query_norm",
        "key_norm",
    ],
    key=len,
    reverse=True,
)


def _normalize_kohya_key(key: str) -> str:
    """Normalize BFL/kohya key format to standard dot-separated format.

    # AC: @flux-lora-loader ac-4
    Key normalization for BFL/kohya format.

    Kohya keys use lora_unet_ prefix with underscore-separated paths.
    Example: lora_unet_double_blocks_0_img_attn_to_q -> double_blocks.0.img_attn.to_q
    """
    if not key.startswith("lora_unet_"):
        return key

    # Strip lora_unet_ prefix
    key = key[len("lora_unet_"):]

    # Convert numeric indices
    key = re.sub(r"_(\d+)_", r".\1.", key)
    key = re.sub(r"_(\d+)$", r".\1", key)

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


def _normalize_lycoris_key(key: str) -> str:
    """Normalize LyCORIS key format to standard dot-separated format.

    # AC: @flux-lora-loader ac-4
    Key normalization for LyCORIS format.

    LyCORIS keys use lycoris_ prefix with underscore-separated paths.
    Example: lycoris_double_blocks_0_img_attn_to_q -> double_blocks.0.img_attn.to_q
    """
    if not key.startswith("lycoris_"):
        return key

    # Strip lycoris_ prefix
    key = key[len("lycoris_"):]

    # Convert numeric indices
    key = re.sub(r"_(\d+)_", r".\1.", key)
    key = re.sub(r"_(\d+)$", r".\1", key)

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


def _parse_flux_lora_key(
    lora_key: str,
) -> tuple[str | None, str, str | None, str | None]:
    """Parse a Flux Klein LoRA key into (model_key, direction, qkv_component, attn_stream).

    Handles two LoRA formats:
    1. BFL/kohya: lora_unet_double_blocks_0_img_attn_to_q.lora_up.weight
    2. Diffusers: transformer.double_blocks.0.img_attn.to_q.lora_A.weight

    Args:
        lora_key: Key from LoRA safetensors file

    Returns:
        (model_key, direction, qkv_component, attn_stream) where:
        - model_key: Corresponding base model key (None if unsupported)
        - direction: 'up' (lora_B) or 'down' (lora_A)
        - qkv_component: 'q', 'k', 'v' for QKV weights, 'mlp' for proj_mlp, None otherwise
        - attn_stream: 'img' or 'txt' for double blocks, None otherwise

    # AC: @flux-lora-loader ac-4
    # AC: @flux-lora-loader ac-5
    # AC: @flux-lora-loader ac-6
    """
    key = lora_key

    # Determine direction from LoRA suffix
    # lora_A/lora_down = down, lora_B/lora_up = up
    if ".lora_A." in key or ".lora_down" in key:
        direction = "down"
    elif ".lora_B." in key or ".lora_up" in key:
        direction = "up"
    else:
        return None, "", None, None

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
        return None, "", None, None

    # Handle diffusion_model prefix
    if base_path.startswith("diffusion_model."):
        base_path = base_path[len("diffusion_model."):]

    # Handle transformer prefix
    if base_path.startswith("transformer."):
        base_path = base_path[len("transformer."):]

    # Handle BFL/kohya format
    if base_path.startswith("lora_unet_"):
        base_path = _normalize_kohya_key(base_path)
    # Handle LyCORIS format
    elif base_path.startswith("lycoris_"):
        base_path = _normalize_lycoris_key(base_path)

    # Detect QKV/MLP components and attention stream
    qkv_component = None
    attn_stream = None

    # Check for double_blocks with img_attn or txt_attn
    # AC: @flux-lora-loader ac-5
    if "double_blocks" in base_path:
        if ".img_attn." in base_path:
            attn_stream = "img"
        elif ".txt_attn." in base_path:
            attn_stream = "txt"

        if ".to_q" in base_path:
            qkv_component = "q"
            # Map to fused qkv weight
            base_path = base_path.replace(".to_q", ".qkv")
        elif ".to_k" in base_path:
            qkv_component = "k"
            base_path = base_path.replace(".to_k", ".qkv")
        elif ".to_v" in base_path:
            qkv_component = "v"
            base_path = base_path.replace(".to_v", ".qkv")

    # Check for single_blocks with linear1 fused weight
    # AC: @flux-lora-loader ac-6
    elif "single_blocks" in base_path and ".linear1" in base_path:
        if ".to_q" in base_path:
            qkv_component = "q"
            # Map to_q to linear1 (fused weight)
            base_path = re.sub(r"\.linear1\.to_q$", ".linear1", base_path)
            base_path = re.sub(r"\.to_q$", ".linear1", base_path)
        elif ".to_k" in base_path:
            qkv_component = "k"
            base_path = re.sub(r"\.linear1\.to_k$", ".linear1", base_path)
            base_path = re.sub(r"\.to_k$", ".linear1", base_path)
        elif ".to_v" in base_path:
            qkv_component = "v"
            base_path = re.sub(r"\.linear1\.to_v$", ".linear1", base_path)
            base_path = re.sub(r"\.to_v$", ".linear1", base_path)
        elif ".proj_mlp" in base_path:
            qkv_component = "mlp"
            base_path = re.sub(r"\.linear1\.proj_mlp$", ".linear1", base_path)
            base_path = re.sub(r"\.proj_mlp$", ".linear1", base_path)
    elif "single_blocks" in base_path:
        # Keys like single_blocks.N.to_q (without .linear1 prefix)
        if ".to_q" in base_path:
            qkv_component = "q"
            # Insert linear1 before to_q replacement
            base_path = base_path.replace(".to_q", ".linear1")
        elif ".to_k" in base_path:
            qkv_component = "k"
            base_path = base_path.replace(".to_k", ".linear1")
        elif ".to_v" in base_path:
            qkv_component = "v"
            base_path = base_path.replace(".to_v", ".linear1")
        elif ".proj_mlp" in base_path:
            qkv_component = "mlp"
            base_path = base_path.replace(".proj_mlp", ".linear1")

    # Add diffusion_model prefix and .weight suffix for base model format
    if not base_path.startswith("diffusion_model."):
        model_key = f"diffusion_model.{base_path}"
    else:
        model_key = base_path

    if not model_key.endswith(".weight"):
        model_key = f"{model_key}.weight"

    return model_key, direction, qkv_component, attn_stream


class FluxLoader(LoRALoader):
    """Flux Klein LoRA loader with QKV fusing support.

    Loads LoRA files in BFL/kohya or diffusers format and maps keys to
    Flux Klein double_blocks/single_blocks format. Handles fused QKV weights
    in double_blocks (img_attn.qkv, txt_attn.qkv) and 4-way fused linear1
    in single_blocks (to_q/to_k/to_v/proj_mlp).

    Data is segmented by set_id for correct scoping.

    # AC: @flux-lora-loader ac-4
    Architecture-specific loader for Flux Klein key mapping.

    # AC: @flux-lora-loader ac-5
    Double block QKV fusing with offset-based DeltaSpec.

    # AC: @flux-lora-loader ac-6
    Single block linear1 4-way fusing with offsets.

    # AC: @flux-lora-loader ac-7
    Produces DeltaSpec objects compatible with batched executor.
    """

    def __init__(self) -> None:
        """Initialize empty loader state."""
        # Standard LoRA data segmented by set:
        # set_id -> model_key -> list of (up, down, scale)
        self._lora_data_by_set: dict[
            str, dict[str, list[tuple[torch.Tensor, torch.Tensor, float]]]
        ] = defaultdict(lambda: defaultdict(list))
        # QKV LoRA data segmented by set:
        # set_id -> model_key -> list of (up, down, scale, qkv_component, attn_stream)
        self._qkv_data_by_set: dict[
            str, dict[str, list[tuple[torch.Tensor, torch.Tensor, float, str, str | None]]]
        ] = defaultdict(lambda: defaultdict(list))
        # Per-set affected keys
        self._affected_by_set: dict[str, set[str]] = defaultdict(set)
        # Global affected keys (union of all sets)
        self._affected: set[str] = set()
        # Track hidden dimensions per key for offset calculation
        self._hidden_dims: dict[str, int] = {}

    def load(self, path: str, strength: float = 1.0, set_id: str | None = None) -> None:
        """Load a LoRA safetensors file into the given set.

        # AC: @flux-lora-loader ac-4
        Handles Flux Klein key mapping from BFL/kohya and diffusers formats.

        # AC: @flux-lora-loader ac-5
        Double block QKV keys map to fused qkv weights.

        # AC: @flux-lora-loader ac-6
        Single block linear1 keys map with 4-way offset split.
        """
        # Use a default set_id if none provided (backward compat)
        effective_set_id = set_id if set_id is not None else "__default__"

        # Collect tensors by layer path and direction
        layer_tensors: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
        # Track QKV component and attn stream for each layer key
        layer_info: dict[str, tuple[str | None, str | None]] = {}
        # Collect alpha values
        alpha_values: dict[str, float] = {}
        # Map from our layer_key to the LoRA base path (for alpha lookup)
        lora_base_paths: dict[str, str] = {}

        with safe_open(path, framework="pt", device="cpu") as f:
            for lora_key in f.keys():
                # Check for alpha keys
                if lora_key.endswith(".alpha"):
                    alpha_tensor = f.get_tensor(lora_key)
                    if alpha_tensor.numel() == 1:
                        alpha_values[lora_key[:-len(".alpha")]] = alpha_tensor.item()
                    continue

                model_key, direction, qkv_comp, attn_stream = _parse_flux_lora_key(lora_key)
                if model_key is None:
                    continue

                tensor = f.get_tensor(lora_key)

                # Extract LoRA base path for alpha lookup
                lora_base = lora_key
                for suffix in (".lora_A.weight", ".lora_B.weight",
                               ".lora_down.weight", ".lora_up.weight"):
                    if lora_base.endswith(suffix):
                        lora_base = lora_base[:-len(suffix)]
                        break

                # For QKV/MLP, track each component separately
                if qkv_comp is not None:
                    layer_key = f"{model_key}:{qkv_comp}"
                    if attn_stream is not None:
                        layer_key = f"{model_key}:{attn_stream}:{qkv_comp}"
                    layer_tensors[layer_key][direction] = tensor
                    layer_info[layer_key] = (qkv_comp, attn_stream)
                    lora_base_paths[layer_key] = lora_base
                else:
                    layer_tensors[model_key][direction] = tensor
                    layer_info[model_key] = (None, None)
                    lora_base_paths[model_key] = lora_base

        # Build delta data for complete up/down pairs
        for layer_key, tensors in layer_tensors.items():
            if "up" not in tensors or "down" not in tensors:
                continue

            up = tensors["up"]
            down = tensors["down"]

            # Compute scale: strength * alpha / rank
            rank = down.shape[0]
            alpha = float(rank)
            lora_base = lora_base_paths.get(layer_key)
            if lora_base is not None and lora_base in alpha_values:
                alpha = alpha_values[lora_base]
            scale = strength * alpha / rank

            qkv_comp, attn_stream = layer_info.get(layer_key, (None, None))

            if qkv_comp is not None:
                # QKV/MLP component - extract actual model key
                # layer_key format: "model_key:attn_stream:qkv_comp" or "model_key:qkv_comp"
                parts = layer_key.rsplit(":", 2)
                if len(parts) == 3:
                    model_key = parts[0]
                else:
                    model_key = parts[0]
                self._qkv_data_by_set[effective_set_id][model_key].append(
                    (up, down, scale, qkv_comp, attn_stream)
                )
                self._affected_by_set[effective_set_id].add(model_key)
                self._affected.add(model_key)

                # Track hidden dimension from up tensor for offset calculation
                # up shape is (out, rank), out dimension is hidden
                if model_key not in self._hidden_dims:
                    self._hidden_dims[model_key] = up.shape[0]
            else:
                # Standard LoRA
                self._lora_data_by_set[effective_set_id][layer_key].append(
                    (up, down, scale)
                )
                self._affected_by_set[effective_set_id].add(layer_key)
                self._affected.add(layer_key)

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

    def get_delta_specs(
        self,
        keys: Sequence[str],
        key_indices: dict[str, int],
        set_id: str | None = None,
    ) -> list[DeltaSpec]:
        """Produce DeltaSpec objects for batched GPU evaluation.

        # AC: @flux-lora-loader ac-5
        Double block QKV uses qkv_q/qkv_k/qkv_v kinds with offset.

        # AC: @flux-lora-loader ac-6
        Single block linear1 uses qkv_* for Q/K/V and offset_mlp for proj_mlp.

        # AC: @flux-lora-loader ac-7
        Produces DeltaSpec objects compatible with batched executor.
        """
        specs: list[DeltaSpec] = []

        # Determine which data sources to iterate
        if set_id is not None:
            lora_sources = [self._lora_data_by_set.get(set_id, {})]
            qkv_sources = [self._qkv_data_by_set.get(set_id, {})]
        else:
            lora_sources = list(self._lora_data_by_set.values())
            qkv_sources = list(self._qkv_data_by_set.values())

        for key in keys:
            key_idx = key_indices.get(key)
            if key_idx is None:
                continue

            # Handle standard LoRA data
            for lora_data in lora_sources:
                if key in lora_data:
                    for up, down, scale in lora_data[key]:
                        if up.dim() == 2 and down.dim() == 2:
                            spec = DeltaSpec(
                                kind="standard",
                                key_index=key_idx,
                                up=up,
                                down=down,
                                scale=scale,
                            )
                            specs.append(spec)

            # Handle QKV/MLP LoRA data
            for qkv_data in qkv_sources:
                if key in qkv_data:
                    for up, down, scale, qkv_comp, attn_stream in qkv_data[key]:
                        if up.dim() != 2 or down.dim() != 2:
                            continue

                        # Calculate offset based on component
                        # Use hidden dimension from the up tensor
                        hidden_dim = up.shape[0]

                        # Determine kind and offset based on component
                        if qkv_comp in ("q", "k", "v"):
                            kind = f"qkv_{qkv_comp}"
                            # For double_blocks: simple 3-way split per stream
                            # For single_blocks linear1: 4-way split (Q/K/V/MLP)
                            if "single_blocks" in key and "linear1" in key:
                                # 4-way split: Q, K, V, MLP
                                # Each takes hidden_dim slice
                                qkv_map = {"q": 0, "k": 1, "v": 2}
                                section_idx = qkv_map[qkv_comp]
                                offset = (section_idx * hidden_dim, hidden_dim)
                            else:
                                # 3-way split for double_blocks: Q, K, V
                                qkv_map = {"q": 0, "k": 1, "v": 2}
                                section_idx = qkv_map[qkv_comp]
                                offset = (section_idx * hidden_dim, hidden_dim)
                        elif qkv_comp == "mlp":
                            kind = "offset_mlp"
                            # MLP is the 4th component in single_blocks linear1
                            offset = (3 * hidden_dim, hidden_dim)
                        else:
                            continue

                        spec = DeltaSpec(
                            kind=kind,
                            key_index=key_idx,
                            up=up,
                            down=down,
                            scale=scale,
                            offset=offset,
                        )
                        specs.append(spec)

        return specs

    def cleanup(self) -> None:
        """Release loaded tensors.

        # AC: @lora-loaders ac-4
        """
        self._lora_data_by_set.clear()
        self._qkv_data_by_set.clear()
        self._affected_by_set.clear()
        self._affected.clear()
        self._hidden_dims.clear()
