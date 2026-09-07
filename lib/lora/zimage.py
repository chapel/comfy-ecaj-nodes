"""Z-Image Architecture LoRA Loader.

Handles Z-Image S3-DiT key mapping with QKV fusing logic.
Z-Image uses fused attention.qkv.weight (11520x3840 = 3x3840) in the
base model, but LoRAs provide separate to_q/to_k/to_v keys.

# AC: @lora-loaders ac-1
Z-Image loader handles QKV fusing and architecture-specific key mapping.

# AC: @zimage-loader ac-1
QKV keys are fused into the base model attention.qkv.weight layout.

# AC: @zimage-loader ac-2
Diffusers-style key names are correctly mapped to S3-DiT parameter names.

# AC: @zimage-loader ac-3
QKV-fused specs have correct offset indexing for the fused weight
where q occupies 0 to 3840, k occupies 3840 to 7680, and v occupies
7680 to 11520.
"""

from collections import defaultdict
from collections.abc import Sequence

import torch

from ..executor import DeltaSpec
from .base import LoRALoader
from .validation import read_pairs

__all__ = ["ZImageLoader"]

# Z-Image hidden dimension for QKV (3840 per head component)
# Full QKV shape: (11520, 3840) where 11520 = 3 x 3840
_ZIMAGE_HIDDEN_DIM = 3840

# QKV offset mapping: component -> (start_row, length)
# # AC: @zimage-loader ac-3
_QKV_OFFSETS = {
    "q": (0, _ZIMAGE_HIDDEN_DIM),  # rows 0:3840
    "k": (_ZIMAGE_HIDDEN_DIM, _ZIMAGE_HIDDEN_DIM),  # rows 3840:7680
    "v": (2 * _ZIMAGE_HIDDEN_DIM, _ZIMAGE_HIDDEN_DIM),  # rows 7680:11520
}

# Compound names that should not have underscores converted to dots
# during LyCORIS key normalization.
# Sorted by length descending so longer variants (e.g. lokr_w1_a) are
# matched before shorter prefixes (e.g. lokr_w1) to avoid partial replacement.
_COMPOUND_NAMES = sorted(
    [
        # LoKr components (must preserve underscore!)
        "lokr_w1",
        "lokr_w2",
        "lokr_w1_a",
        "lokr_w1_b",
        "lokr_w2_a",
        "lokr_w2_b",
        "linear_1",
        "linear_2",
        "linear_3",
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
        # Z-Image specific
        "adaLN_modulation",
        "feed_forward",
        "noise_refiner",
        "context_refiner",
        "t_embedder",
        "cap_embedder",
        "x_pad_token",
        "cap_pad_token",
    ],
    key=len,
    reverse=True,
)


def _normalize_lycoris_key(key: str) -> str:
    """Normalize LyCORIS/LoKr key format to standard ComfyUI format.

    # AC: @zimage-loader ac-2

    LyCORIS keys use underscore-separated paths with a lycoris_ prefix.
    This normalizes them to dot-separated paths matching ComfyUI format.

    Example: lycoris_layers_0_adaLN_modulation_0 -> layers.0.adaLN_modulation.0
    """
    if not key.startswith("lycoris_"):
        return key

    # Strip lycoris_ prefix
    key = key[len("lycoris_") :]

    # Replace compound names with placeholders (using markers without underscores)
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


def _parse_zimage_lora_key(lora_key: str) -> tuple[str | None, str, str | None]:
    """Parse a Z-Image LoRA key into (model_key, direction, qkv_component).

    Z-Image LoRA keys follow Diffusers naming patterns:
    - transformer.layers.0.attention.to_q.lora_A.weight
    - diffusion_model.layers.0.attention.to_k.lora_B.weight
    - lycoris_layers_0_attention_to_v.lora_down.weight (LyCORIS format)
    - transformer.layers.0.ff.linear_1.lora_A.weight

    Model keys are mapped to S3-DiT format:
    - diffusion_model.layers.0.attention.qkv.weight (fused Q/K/V)
    - diffusion_model.layers.0.attention.out.weight (from to_out.0)
    - diffusion_model.layers.0.feed_forward.linear_1.weight

    Args:
        lora_key: Key from LoRA safetensors file

    Returns:
        (model_key, direction, qkv_component) where:
        - model_key: Corresponding base model key (None if unsupported)
        - direction: 'up' (lora_B) or 'down' (lora_A)
        - qkv_component: 'q', 'k', 'v' for QKV weights, None otherwise

    # AC: @lora-loaders ac-1
    # AC: @zimage-loader ac-1
    # AC: @zimage-loader ac-2
    """
    key = lora_key

    # Determine direction from LoRA suffix
    # Z-Image uses lora_A (down) and lora_B (up) naming
    # Also handle lora_down/lora_up variants
    if ".lora_A." in key or ".lora_down" in key:
        direction = "down"
    elif ".lora_B." in key or ".lora_up" in key:
        direction = "up"
    else:
        return None, "", None

    # Extract base path by removing LoRA suffix
    # Handle both .lora_A.weight and .lora_down.weight patterns
    if ".lora_A." in key:
        base_path = key.split(".lora_A.")[0]
    elif ".lora_B." in key:
        base_path = key.split(".lora_B.")[0]
    elif ".lora_down" in key:
        base_path = key.split(".lora_down")[0]
    elif ".lora_up" in key:
        base_path = key.split(".lora_up")[0]
    else:
        return None, "", None

    # Handle diffusion_model prefix
    if base_path.startswith("diffusion_model."):
        base_path = base_path[len("diffusion_model.") :]

    # Handle transformer prefix
    if base_path.startswith("transformer."):
        base_path = base_path[len("transformer.") :]

    # Handle LyCORIS format
    base_path = _normalize_lycoris_key(base_path)

    # Check for QKV components and map to fused qkv
    # # AC: @zimage-loader ac-1
    qkv_component = None
    if ".attention.to_q" in base_path or base_path.endswith(".to_q"):
        qkv_component = "q"
        model_key = base_path.replace(".to_q", ".qkv")
    elif ".attention.to_k" in base_path or base_path.endswith(".to_k"):
        qkv_component = "k"
        model_key = base_path.replace(".to_k", ".qkv")
    elif ".attention.to_v" in base_path or base_path.endswith(".to_v"):
        qkv_component = "v"
        model_key = base_path.replace(".to_v", ".qkv")
    elif ".attention.to_out.0" in base_path:
        # Map to_out.0 -> out
        model_key = base_path.replace(".attention.to_out.0", ".attention.out")
        qkv_component = None
    else:
        model_key = base_path

    # Add diffusion_model prefix and .weight suffix for base model format
    if not model_key.startswith("diffusion_model."):
        model_key = f"diffusion_model.{model_key}"
    if not model_key.endswith(".weight"):
        model_key = f"{model_key}.weight"

    return model_key, direction, qkv_component


class ZImageLoader(LoRALoader):
    """Z-Image S3-DiT LoRA loader with QKV fusing support.

    Z-Image's base model uses fused attention.qkv weights, but LoRAs
    provide separate to_q/to_k/to_v. This loader maps them correctly
    and produces DeltaSpec objects with appropriate qkv_* kinds.
    Data is segmented by set_id for correct scoping.

    # AC: @lora-loaders ac-1
    Architecture-specific loader for Z-Image key mapping with QKV fusing.

    # AC: @lora-loaders ac-2
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
        # set_id -> model_key -> list of (up, down, scale, qkv_component)
        self._qkv_data_by_set: dict[
            str, dict[str, list[tuple[torch.Tensor, torch.Tensor, float, str]]]
        ] = defaultdict(lambda: defaultdict(list))
        # Per-set affected keys
        self._affected_by_set: dict[str, set[str]] = defaultdict(set)
        # Global affected keys (union of all sets)
        self._affected: set[str] = set()

    def load(self, path: str, strength: float = 1.0, set_id: str | None = None) -> None:
        """Validate the whole package before publishing factors to this set."""
        pending = read_pairs(path, strength, _parse_zimage_lora_key)
        effective_set_id = set_id if set_id is not None else "__default__"
        for key, up, down, scale, extra in pending:
            if extra[0] is not None:
                self._qkv_data_by_set[effective_set_id][key].append((up, down, scale, *extra))
            else:
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
        for key_data in self._qkv_data_by_set.values():
            for entries in key_data.values():
                for up, down, _scale, _comp in entries:
                    total += up.nbytes + down.nbytes
        return total

    def get_delta_specs(
        self,
        keys: Sequence[str],
        key_indices: dict[str, int],
        set_id: str | None = None,
    ) -> list[DeltaSpec]:
        """Produce DeltaSpec objects for batched GPU evaluation.

        # AC: @lora-loaders ac-2
        Produces DeltaSpec objects compatible with batched executor,
        including qkv_q/qkv_k/qkv_v kinds for fused attention weights.
        When set_id is provided, only returns deltas from that set.

        # AC: @zimage-loader ac-3
        QKV-fused specs include offset indexing: q=(0,3840), k=(3840,3840), v=(7680,3840)
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

            # Handle QKV LoRA data
            # # AC: @zimage-loader ac-3
            for qkv_data in qkv_sources:
                if key in qkv_data:
                    for up, down, scale, qkv_comp in qkv_data[key]:
                        if up.dim() == 2 and down.dim() == 2:
                            kind = f"qkv_{qkv_comp}"
                            # Get offset for this QKV component
                            offset = _QKV_OFFSETS[qkv_comp]
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
