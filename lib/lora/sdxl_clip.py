"""SDXL CLIP Architecture LoRA Loader.

Handles SDXL text encoder key mapping from LoRA format to model format.
SDXL CLIPs use kohya/A1111 naming conventions:

LoRA key format:
  lora_te1_text_model_encoder_layers_{N}_{component}.lora_{up|down}.weight (CLIP-L)
  lora_te2_text_model_encoder_layers_{N}_{component}.lora_{up|down}.weight (CLIP-G)

Model key format:
  clip_l.transformer.text_model.encoder.layers.{N}.{component}.weight (CLIP-L)
  clip_g.transformer.text_model.encoder.layers.{N}.{component}.weight (CLIP-G)

# AC: @sdxl-clip-lora-loader ac-1
CLIP-L (te1) maps lora_te1_* keys to clip_l.transformer.* keys.

# AC: @sdxl-clip-lora-loader ac-2
CLIP-G (te2) maps lora_te2_* keys to clip_g.transformer.* keys.
"""

from collections import defaultdict
from collections.abc import Sequence

import torch
from safetensors import safe_open

from ..executor import DeltaSpec
from .base import LoRALoader

__all__ = ["SDXLCLIPLoader"]


# Compound token patterns in SDXL CLIP LoRA keys.
# These are ordered longest-first for greedy matching.
# Pattern: underscore-separated -> dot-separated compound name
_CLIP_COMPOUND_TOKENS = [
    # Transformer structure
    ("text_model", "text_model"),
    ("encoder", "encoder"),
    ("layers", "layers"),
    # Attention components
    ("self_attn", "self_attn"),
    ("k_proj", "k_proj"),
    ("v_proj", "v_proj"),
    ("q_proj", "q_proj"),
    ("out_proj", "out_proj"),
    # MLP components
    ("mlp", "mlp"),
    ("fc1", "fc1"),
    ("fc2", "fc2"),
    # Layer norms
    ("layer_norm1", "layer_norm1"),
    ("layer_norm2", "layer_norm2"),
    ("final_layer_norm", "final_layer_norm"),
    # Embeddings
    ("embeddings", "embeddings"),
    ("token_embedding", "token_embedding"),
    ("position_embedding", "position_embedding"),
    # CLIP-G specific
    ("text_projection", "text_projection"),
]


def _tokenize_clip_lora_path(path: str) -> list[str]:
    """Tokenize a CLIP LoRA path, preserving compound identifiers.

    Splits on underscores but keeps known compound tokens together.

    Args:
        path: Layer path like 'text_model_encoder_layers_0_self_attn_k_proj'

    Returns:
        List of tokens like ['text_model', 'encoder', 'layers', '0', 'self_attn', 'k_proj']
    """
    tokens: list[str] = []
    remaining = path

    while remaining:
        # Try to match a compound token at the current position
        matched = False
        for pattern, _ in _CLIP_COMPOUND_TOKENS:
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


def _parse_clip_lora_key(lora_key: str) -> tuple[str | None, str]:
    """Parse a CLIP LoRA key into (model_key, direction).

    Args:
        lora_key: Key from LoRA safetensors (e.g. 'lora_te1_text_model_*.lora_up.weight')

    Returns:
        (model_key, direction) tuple, where:
        - model_key: Corresponding base model key (None if not a CLIP LoRA)
        - direction: 'up' or 'down'

    # AC: @sdxl-clip-lora-loader ac-1
    Maps lora_te1_* keys to clip_l.transformer.* keys.

    # AC: @sdxl-clip-lora-loader ac-2
    Maps lora_te2_* keys to clip_g.transformer.* keys.

    # AC: @sdxl-clip-lora-loader ac-6
    Ignores lora_unet_* keys (UNet keys handled by SDXLLoader).
    """
    # Skip non-CLIP keys (UNet keys handled by SDXLLoader)
    # AC: @sdxl-clip-lora-loader ac-6
    if lora_key.startswith("lora_unet_"):
        return None, ""

    # Determine which CLIP encoder (te1 = CLIP-L, te2 = CLIP-G)
    if lora_key.startswith("lora_te1_"):
        prefix = "lora_te1_"
        model_prefix = "clip_l.transformer."
    elif lora_key.startswith("lora_te2_"):
        prefix = "lora_te2_"
        model_prefix = "clip_g.transformer."
    else:
        return None, ""

    # Extract the component direction (lora_up or lora_down)
    if ".lora_up." in lora_key:
        direction = "up"
    elif ".lora_down." in lora_key:
        direction = "down"
    else:
        return None, ""

    # Remove prefix and suffix to get the layer path
    # lora_te1_text_model_encoder_layers_0_self_attn_k_proj.lora_up.weight
    # -> text_model_encoder_layers_0_self_attn_k_proj
    layer_path = lora_key[len(prefix) :]
    # Remove .lora_{up|down}.weight suffix
    layer_path = layer_path.rsplit(".lora_", 1)[0]

    # Tokenize preserving compound identifiers
    tokens = _tokenize_clip_lora_path(layer_path)

    # Build model key with proper dot separation
    model_key = model_prefix + ".".join(tokens) + ".weight"

    return model_key, direction


class SDXLCLIPLoader(LoRALoader):
    """SDXL CLIP-specific LoRA loader.

    Loads LoRA files in kohya/A1111 format and maps keys to SDXL CLIP format.
    Handles lora_te1_* (CLIP-L) and lora_te2_* (CLIP-G) keys.
    Accumulates multiple LoRAs and produces DeltaSpec objects for batched
    GPU evaluation. Data is segmented by set_id for correct scoping.

    # AC: @sdxl-clip-lora-loader ac-5
    Implements the LoRALoader interface with all required methods.
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

        # AC: @sdxl-clip-lora-loader ac-1
        Maps lora_te1_* keys to CLIP-L model keys.

        # AC: @sdxl-clip-lora-loader ac-2
        Maps lora_te2_* keys to CLIP-G model keys.

        # AC: @sdxl-clip-lora-loader ac-3
        LoRAs with only UNet keys result in empty affected_keys.

        # AC: @sdxl-clip-lora-loader ac-5
        Implements load(path, strength, set_id).
        """
        # Use a default set_id if none provided (backward compat)
        effective_set_id = set_id if set_id is not None else "__default__"

        # Collect up/down pairs keyed by model key
        layer_tensors: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
        # Collect alpha values keyed by LoRA base path
        alpha_values: dict[str, float] = {}
        # Map from model_key to LoRA base path (for alpha lookup)
        lora_base_paths: dict[str, str] = {}

        with safe_open(path, framework="pt", device="cpu") as f:
            for lora_key in f.keys():
                # Check for alpha keys (e.g. "lora_te1_text_model_encoder_layers_0.alpha")
                if lora_key.endswith(".alpha"):
                    alpha_tensor = f.get_tensor(lora_key)
                    if alpha_tensor.numel() == 1:
                        alpha_values[lora_key[: -len(".alpha")]] = alpha_tensor.item()
                    continue

                model_key, direction = _parse_clip_lora_key(lora_key)
                if model_key is None:
                    continue

                tensor = f.get_tensor(lora_key)
                layer_tensors[model_key][direction] = tensor

                # Extract LoRA base path for alpha lookup
                # e.g. "lora_te1_text_model_encoder_layers_0_self_attn_k_proj.lora_up.weight"
                #    → "lora_te1_text_model_encoder_layers_0_self_attn_k_proj"
                lora_base = lora_key.rsplit(".lora_", 1)[0]
                lora_base_paths[model_key] = lora_base

        # Build delta specs for complete up/down pairs
        for model_key, tensors in layer_tensors.items():
            if "up" not in tensors or "down" not in tensors:
                continue

            up = tensors["up"]
            down = tensors["down"]

            # Compute scale: strength * alpha / rank
            # Alpha is read from the file if available, otherwise defaults to rank
            rank = down.shape[0]
            alpha = float(rank)
            lora_base = lora_base_paths.get(model_key)
            if lora_base is not None and lora_base in alpha_values:
                alpha = alpha_values[lora_base]
            scale = strength * alpha / rank

            self._lora_data_by_set[effective_set_id][model_key].append((up, down, scale))
            self._affected_by_set[effective_set_id].add(model_key)
            self._affected.add(model_key)

    @property
    def affected_keys(self) -> frozenset[str]:
        """Return keys that loaded LoRAs modify (all sets).

        Returns a frozenset to prevent external mutation of internal state.

        # AC: @sdxl-clip-lora-loader ac-3
        Returns empty frozenset for UNet-only LoRAs.

        # AC: @sdxl-clip-lora-loader ac-5
        Implements affected_keys property.
        """
        return frozenset(self._affected)

    def affected_keys_for_set(self, set_id: str) -> set[str]:
        """Return keys modified by a specific LoRA set.

        # AC: @sdxl-clip-lora-loader ac-5
        Implements affected_keys_for_set(set_id).
        """
        return self._affected_by_set.get(set_id, set())

    def get_delta_specs(
        self,
        keys: Sequence[str],
        key_indices: dict[str, int],
        set_id: str | None = None,
    ) -> list[DeltaSpec]:
        """Produce DeltaSpec objects for batched GPU evaluation.

        # AC: @sdxl-clip-lora-loader ac-4
        Returns DeltaSpec objects with correct up/down weight tensors
        and dimensions matching the base model parameters.

        # AC: @sdxl-clip-lora-loader ac-5
        Implements get_delta_specs(keys, key_indices, set_id).
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
                    # CLIP LoRAs are standard linear LoRAs (no QKV fusing needed)
                    if up.dim() == 2 and down.dim() == 2:
                        spec = DeltaSpec(
                            kind="standard",
                            key_index=key_idx,
                            up=up,
                            down=down,
                            scale=scale,
                        )
                        specs.append(spec)
                    # Skip unsupported shapes (CLIP shouldn't have Conv2d)

        return specs

    def cleanup(self) -> None:
        """Release loaded tensors.

        # AC: @sdxl-clip-lora-loader ac-5
        Implements cleanup().
        """
        self._lora_data_by_set.clear()
        self._affected_by_set.clear()
        self._affected.clear()
