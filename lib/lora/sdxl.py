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


# Prefix mapping: LoRA naming -> model state dict prefix
_LORA_TO_MODEL_PREFIX = {
    "lora_unet_": "diffusion_model.",
}

# Compound token patterns in SDXL UNet LoRA keys.
# These are ordered longest-first for greedy matching.
# Pattern: underscore-separated -> dot-separated compound name
_COMPOUND_TOKENS = [
    # Block structure
    ("input_blocks", "input_blocks"),
    ("output_blocks", "output_blocks"),
    ("middle_block", "middle_block"),
    ("transformer_blocks", "transformer_blocks"),
    # Attention components (AC-3: @sdxl-loader)
    ("proj_in", "proj_in"),
    ("proj_out", "proj_out"),
    ("to_out", "to_out"),
    ("to_q", "to_q"),
    ("to_k", "to_k"),
    ("to_v", "to_v"),
    # Attention blocks
    ("attn1", "attn1"),
    ("attn2", "attn2"),
    # Feed-forward
    ("ff_net", "ff_net"),
    ("time_embed", "time_embed"),
    ("label_emb", "label_emb"),
    ("out_layers", "out_layers"),
    ("in_layers", "in_layers"),
    ("skip_connection", "skip_connection"),
    ("emb_layers", "emb_layers"),
]


def _tokenize_lora_path(path: str) -> list[str]:
    """Tokenize a LoRA path, preserving compound identifiers.

    Splits on underscores but keeps known compound tokens together.

    Args:
        path: Layer path like 'input_blocks_0_0_proj_in'

    Returns:
        List of tokens like ['input_blocks', '0', '0', 'proj_in']
    """
    tokens: list[str] = []
    remaining = path

    while remaining:
        # Try to match a compound token at the current position
        matched = False
        for pattern, _ in _COMPOUND_TOKENS:
            if remaining.startswith(pattern):
                # Check it's followed by underscore, end of string, or digit boundary
                rest = remaining[len(pattern) :]
                if rest == "" or rest.startswith("_"):
                    tokens.append(pattern)
                    remaining = rest[1:] if rest.startswith("_") else ""
                    matched = True
                    break

        if not matched:
            # Take characters up to the next underscore as a single token
            if "_" in remaining:
                idx = remaining.index("_")
                tokens.append(remaining[:idx])
                remaining = remaining[idx + 1 :]
            else:
                tokens.append(remaining)
                remaining = ""

    return tokens


def _parse_lora_key(lora_key: str) -> tuple[str | None, str, str]:
    """Parse a LoRA key into (model_key, component, direction).

    Args:
        lora_key: Key from LoRA safetensors (e.g. 'lora_unet_input_blocks_*.lora_up.weight')

    Returns:
        (model_key, component, direction) tuple, where:
        - model_key: Corresponding base model key (None if not a unet LoRA)
        - component: 'up' or 'down'
        - direction: Full component name for matching

    # AC: @sdxl-loader ac-1
    Maps LoRA keys to diffusion_model input_blocks, middle_block, output_blocks.

    # AC: @sdxl-loader ac-3
    Handles attention keys (proj_in, proj_out, to_q/to_k/to_v).
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
    # -> input_blocks_0_0_proj_in
    layer_path = lora_key[len("lora_unet_") :]
    # Remove .lora_{up|down}.weight suffix
    layer_path = layer_path.rsplit(".lora_", 1)[0]

    # Tokenize preserving compound identifiers
    tokens = _tokenize_lora_path(layer_path)

    # Build model key with proper dot separation
    # Numeric tokens get attached to preceding segment: input_blocks.0.0
    model_key = "diffusion_model."
    parts: list[str] = []

    for token in tokens:
        if token.isdigit():
            # Numeric index - append with dot
            parts.append(token)
        else:
            # Named segment
            parts.append(token)

    model_key += ".".join(parts) + ".weight"

    return model_key, direction, lora_key


class SDXLLoader(LoRALoader):
    """SDXL-specific LoRA loader.

    Loads LoRA files in kohya/A1111 format and maps keys to SDXL UNet format.
    Accumulates multiple LoRAs and produces DeltaSpec objects for batched
    GPU evaluation. Data is segmented by set_id for correct scoping.

    # AC: @lora-loaders ac-1
    Architecture-specific loader for SDXL key mapping.

    # AC: @lora-loaders ac-2
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
        """Load a LoRA safetensors file into the given set.

        # AC: @lora-loaders ac-1
        Handles SDXL key mapping from kohya format.
        """
        # Use a default set_id if none provided (backward compat)
        effective_set_id = set_id if set_id is not None else "__default__"

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

            self._lora_data_by_set[effective_set_id][model_key].append((up, down, scale))
            self._affected_by_set[effective_set_id].add(model_key)
            self._affected.add(model_key)

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

        # AC: @lora-loaders ac-2
        Produces DeltaSpec objects compatible with batched executor.
        When set_id is provided, only returns deltas from that set.
        """
        specs: list[DeltaSpec] = []

        # Determine which data sources to iterate
        if set_id is not None:
            # Only look at the specific set
            data_sources = [self._lora_data_by_set.get(set_id, {})]
        else:
            # Legacy: iterate all sets
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
        self._lora_data_by_set.clear()
        self._affected_by_set.clear()
        self._affected.clear()
