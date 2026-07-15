"""Krea 2 architecture LoRA loader.

Supports the public Krea 2 LoRA package families inspected for
@task-krea2-lora-package-compatibility:

- Diffusers-style exports: ``transformer.*.lora_A/B.weight``.
- Native Comfy-style exports: ``diffusion_model.*.lora_down/up.weight`` plus
  direct ``.diff_b`` bias deltas.
- Native LoKR exports: ``diffusion_model.*.lokr_w1/w2``.

Krea base weights are not QKV-fused. Public package names use a mixture of
Diffusers module names and Comfy checkpoint names, so this loader keeps a small
explicit compatibility table instead of relying on broad underscore rewriting.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import torch
from safetensors import safe_open

from ..executor import DeltaSpec
from .base import LoRALoader

__all__ = [
    "KREA2_COMPATIBILITY_FAMILIES",
    "Krea2CompatibilityError",
    "Krea2Loader",
    "_parse_krea2_lora_key",
]


KREA2_COMPATIBILITY_FAMILIES = {
    "diffusers_transformer_lora": (
        "transformer.*.lora_A.weight",
        "transformer.*.lora_B.weight",
    ),
    "native_diffusion_lora": (
        "diffusion_model.*.lora_down.weight",
        "diffusion_model.*.lora_up.weight",
        "diffusion_model.*.diff_b",
    ),
    "native_diffusion_lokr": (
        "diffusion_model.*.lokr_w1",
        "diffusion_model.*.lokr_w2",
    ),
}

_LORA_SUFFIXES = (
    (".lora_A.weight", "down"),
    (".lora_B.weight", "up"),
    (".lora_down.weight", "down"),
    (".lora_up.weight", "up"),
    (".lokr_w1", "lokr_w1"),
    (".lokr_w2", "lokr_w2"),
)

_ATTN_REPLACEMENTS = (
    (".attn.to_out.0", ".attn.wo"),
    (".attn.to_gate", ".attn.gate"),
    (".attn.to_q", ".attn.wq"),
    (".attn.to_k", ".attn.wk"),
    (".attn.to_v", ".attn.wv"),
)

_PATH_REPLACEMENTS = (
    ("transformer_blocks.", "blocks."),
    ("text_fusion.", "txtfusion."),
    (".ff.", ".mlp."),
)

_EXACT_REPLACEMENTS = {
    "img_in": "first",
    "final_layer.linear": "last.linear",
    # Public diffusers-style Krea packages name the timestep projections after
    # their Diffusers modules; Krea checkpoints expose the same weights as tmlp/tproj.
    "time_embed.linear_1": "tmlp.0",
    "time_embed.linear_2": "tmlp.2",
    "time_mod_proj": "tproj.1",
    "txt_in.linear_1": "txtmlp.1",
    "txt_in.linear_2": "txtmlp.3",
}


class Krea2CompatibilityError(ValueError):
    """Raised when a Krea 2 LoRA package cannot be applied completely."""


@dataclass(frozen=True)
class ParsedKrea2Key:
    """Normalized Krea 2 LoRA tensor key."""

    model_key: str
    direction: str
    group_key: str
    direct_delta: bool = False


def _strip_lora_suffix(key: str) -> tuple[str | None, str]:
    for suffix, direction in _LORA_SUFFIXES:
        if key.endswith(suffix):
            return key[: -len(suffix)], direction
    return None, ""


def _normalize_base_path(path: str) -> str | None:
    """Map a supported public Krea package path to a base checkpoint path.

    This deliberately preserves compound Krea names such as
    ``txtfusion.layerwise_blocks`` and ``txtfusion.refiner_blocks``. The
    public diffusers family already uses dotted paths, so broad underscore
    replacement would be both unnecessary and dangerous.
    """
    if path.startswith("diffusion_model."):
        path = path[len("diffusion_model.") :]
    elif path.startswith("transformer."):
        path = path[len("transformer.") :]
    else:
        return None

    if path in _EXACT_REPLACEMENTS:
        return _EXACT_REPLACEMENTS[path]

    for old, new in _PATH_REPLACEMENTS:
        path = path.replace(old, new)
    for old, new in _ATTN_REPLACEMENTS:
        path = path.replace(old, new)

    supported_prefixes = (
        "blocks.",
        "txtfusion.layerwise_blocks.",
        "txtfusion.refiner_blocks.",
        "txtfusion.projector",
        "first",
        "last.",
        "tmlp.",
        "tproj.",
        "txtmlp.",
    )
    if not path.startswith(supported_prefixes):
        return None
    return path


def _parse_krea2_lora_key(lora_key: str) -> ParsedKrea2Key | None:
    """Parse one public Krea 2 LoRA tensor key.

    Returns ``None`` for unsupported package families so ``load`` can report all
    incompatible groups in one actionable error.
    """
    if lora_key.endswith(".alpha"):
        return None

    if lora_key.endswith(".diff_b"):
        base_path = lora_key[: -len(".diff_b")]
        normalized = _normalize_base_path(base_path)
        if normalized is None:
            return None
        model_key = f"diffusion_model.{normalized}.bias"
        return ParsedKrea2Key(
            model_key=model_key,
            direction="direct",
            group_key=model_key,
            direct_delta=True,
        )

    base_path, direction = _strip_lora_suffix(lora_key)
    if base_path is None:
        return None

    normalized = _normalize_base_path(base_path)
    if normalized is None:
        return None

    model_key = f"diffusion_model.{normalized}.weight"
    return ParsedKrea2Key(
        model_key=model_key,
        direction=direction,
        group_key=model_key,
        direct_delta=False,
    )


def _format_sample(values: Iterable[str], limit: int = 6) -> str:
    items = sorted(values)
    if len(items) <= limit:
        return ", ".join(items)
    return ", ".join(items[:limit]) + f", ... (+{len(items) - limit} more)"


def _expected_lora_shape(up: torch.Tensor, down: torch.Tensor) -> tuple[int, int]:
    return (up.shape[0], down.shape[1])


def _expected_direct_shape(tensor: torch.Tensor) -> tuple[int, ...]:
    return tuple(tensor.shape)


def _expected_lokr_shape(w1: torch.Tensor, w2: torch.Tensor) -> tuple[int, ...]:
    return (w1.shape[0] * w2.shape[0], w1.shape[1] * w2.shape[1])


def _compatibility_error(
    path: str,
    *,
    unsupported: Iterable[str] = (),
    incomplete: Iterable[str] = (),
    shape_errors: Iterable[str] = (),
) -> Krea2CompatibilityError:
    parts = [f"Krea 2 LoRA package is not fully compatible: {path}"]
    unsupported = tuple(unsupported)
    incomplete = tuple(incomplete)
    shape_errors = tuple(shape_errors)
    if unsupported:
        parts.append(f"unsupported tensor groups: {_format_sample(unsupported)}")
    if incomplete:
        parts.append(f"incomplete up/down groups: {_format_sample(incomplete)}")
    if shape_errors:
        parts.append(f"shape-incompatible groups: {_format_sample(shape_errors)}")
    parts.append(
        "Supported Krea 2 package families: " + ", ".join(sorted(KREA2_COMPATIBILITY_FAMILIES))
    )
    return Krea2CompatibilityError("; ".join(parts))


class Krea2Loader(LoRALoader):
    """Krea 2 LoRA loader with complete-package validation."""

    def __init__(self) -> None:
        self._lora_data_by_set: dict[
            str, dict[str, list[tuple[torch.Tensor, torch.Tensor, float]]]
        ] = defaultdict(lambda: defaultdict(list))
        self._direct_data_by_set: dict[str, dict[str, list[tuple[torch.Tensor, float]]]] = (
            defaultdict(lambda: defaultdict(list))
        )
        self._lokr_data_by_set: dict[
            str, dict[str, list[tuple[torch.Tensor, torch.Tensor, float]]]
        ] = defaultdict(lambda: defaultdict(list))
        self._affected_by_set: dict[str, set[str]] = defaultdict(set)
        self._affected: set[str] = set()

    def load(self, path: str, strength: float = 1.0, set_id: str | None = None) -> None:
        """Load a Krea 2 LoRA package into a set.

        Unsupported keys, incomplete up/down pairs, and rank-incompatible pairs
        are rejected before any state is published on the loader. This prevents
        a later Exit execution from reporting success after applying only a
        subset of the package.
        """
        effective_set_id = set_id if set_id is not None else "__default__"

        layer_tensors: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
        lokr_tensors: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)
        direct_tensors: dict[str, torch.Tensor] = {}
        alpha_values: dict[str, float] = {}
        unsupported: list[str] = []
        shape_errors: list[str] = []

        with safe_open(path, framework="pt", device="cpu") as f:
            for lora_key in f.keys():
                if lora_key.endswith(".alpha"):
                    alpha_tensor = f.get_tensor(lora_key)
                    alpha_base_key = lora_key[: -len(".alpha")]
                    normalized_alpha = _normalize_base_path(alpha_base_key)
                    if normalized_alpha is None:
                        unsupported.append(lora_key)
                    elif alpha_tensor.numel() != 1:
                        shape_errors.append(f"{lora_key} alpha must be scalar")
                    else:
                        model_key = f"diffusion_model.{normalized_alpha}.weight"
                        alpha_values[model_key] = float(alpha_tensor.item())
                    continue

                parsed = _parse_krea2_lora_key(lora_key)
                if parsed is None:
                    unsupported.append(lora_key)
                    continue

                tensor = f.get_tensor(lora_key)
                if parsed.direct_delta:
                    direct_tensors[parsed.model_key] = tensor
                elif parsed.direction in ("lokr_w1", "lokr_w2"):
                    lokr_tensors[parsed.group_key][parsed.direction] = tensor
                else:
                    layer_tensors[parsed.group_key][parsed.direction] = tensor

        incomplete = [
            key
            for key, tensors in layer_tensors.items()
            if "up" not in tensors or "down" not in tensors
        ]
        incomplete.extend(
            key
            for key, tensors in lokr_tensors.items()
            if "lokr_w1" not in tensors or "lokr_w2" not in tensors
        )
        pending_lora: dict[str, list[tuple[torch.Tensor, torch.Tensor, float]]] = defaultdict(list)
        pending_direct: dict[str, list[tuple[torch.Tensor, float]]] = defaultdict(list)
        pending_lokr: dict[str, list[tuple[torch.Tensor, torch.Tensor, float]]] = defaultdict(list)

        for model_key, tensors in layer_tensors.items():
            if model_key in incomplete:
                continue
            up = tensors["up"]
            down = tensors["down"]
            if up.dim() != 2 or down.dim() != 2:
                shape_errors.append(f"{model_key} expected 2D LoRA factors")
                continue
            if up.shape[1] != down.shape[0]:
                shape_errors.append(
                    f"{model_key} rank mismatch: up {tuple(up.shape)} vs down {tuple(down.shape)}"
                )
                continue

            rank = down.shape[0]
            alpha = alpha_values.get(model_key, float(rank))
            scale = strength * alpha / rank
            pending_lora[model_key].append((up, down, scale))

        for model_key, tensors in lokr_tensors.items():
            if model_key in incomplete:
                continue
            w1 = tensors["lokr_w1"]
            w2 = tensors["lokr_w2"]
            if w1.dim() != 2 or w2.dim() != 2:
                shape_errors.append(f"{model_key} expected 2D LoKR factors")
                continue
            # Full w1/w2 LoKR packages match LyCORIS/ai-toolkit semantics:
            # alpha metadata is not applied unless a decomposed factor pair is
            # present. Krea 2 support intentionally accepts only full factors.
            pending_lokr[model_key].append((w1, w2, strength))

        for model_key, tensor in direct_tensors.items():
            if tensor.dim() != 1:
                shape_errors.append(
                    f"{model_key} direct bias delta must be 1D, got {tuple(tensor.shape)}"
                )
                continue
            pending_direct[model_key].append((tensor, strength))

        if unsupported or incomplete or shape_errors:
            raise _compatibility_error(
                path,
                unsupported=unsupported,
                incomplete=incomplete,
                shape_errors=shape_errors,
            )

        if not pending_lora and not pending_direct and not pending_lokr:
            raise _compatibility_error(path, unsupported=["no supported Krea 2 tensors"])

        for model_key, entries in pending_lora.items():
            self._lora_data_by_set[effective_set_id][model_key].extend(entries)
            self._affected_by_set[effective_set_id].add(model_key)
            self._affected.add(model_key)
        for model_key, entries in pending_direct.items():
            self._direct_data_by_set[effective_set_id][model_key].extend(entries)
            self._affected_by_set[effective_set_id].add(model_key)
            self._affected.add(model_key)
        for model_key, entries in pending_lokr.items():
            self._lokr_data_by_set[effective_set_id][model_key].extend(entries)
            self._affected_by_set[effective_set_id].add(model_key)
            self._affected.add(model_key)

    @property
    def affected_keys(self) -> frozenset[str]:
        return frozenset(self._affected)

    def affected_keys_for_set(self, set_id: str) -> set[str]:
        return self._affected_by_set.get(set_id, set())

    def validate_compatible_keys(
        self,
        all_keys: set[str] | frozenset[str],
        key_shapes: Mapping[str, tuple[int, ...]] | None = None,
    ) -> None:
        """Reject loaded tensors whose mapped base keys or shapes are incompatible.

        Exit nodes call this after base key discovery and before batching. It
        catches package/model mismatches that a plain key intersection would
        otherwise silently ignore or defer to a generic tensor-size error.
        """
        missing = self._affected - set(all_keys)
        shape_errors: list[str] = []
        if key_shapes is not None:
            present_keys = self._affected - missing
            for model_key in sorted(present_keys):
                if model_key not in key_shapes:
                    shape_errors.append(f"{model_key} has no current recipe shape metadata")
                    continue
                current_shape = tuple(key_shapes[model_key])

                for set_id, key_data in sorted(self._lora_data_by_set.items()):
                    for up, down, _scale in key_data.get(model_key, ()):
                        expected_shape = _expected_lora_shape(up, down)
                        if expected_shape != current_shape:
                            shape_errors.append(
                                f"{model_key} set {set_id} package delta shape {expected_shape} "
                                f"does not match current recipe shape {current_shape}"
                            )

                for set_id, key_data in sorted(self._direct_data_by_set.items()):
                    for tensor, _scale in key_data.get(model_key, ()):
                        expected_shape = _expected_direct_shape(tensor)
                        if expected_shape != current_shape:
                            shape_errors.append(
                                f"{model_key} set {set_id} package delta shape {expected_shape} "
                                f"does not match current recipe shape {current_shape}"
                            )

                for set_id, key_data in sorted(self._lokr_data_by_set.items()):
                    for w1, w2, _scale in key_data.get(model_key, ()):
                        expected_shape = _expected_lokr_shape(w1, w2)
                        if expected_shape != current_shape:
                            shape_errors.append(
                                f"{model_key} set {set_id} package delta shape {expected_shape} "
                                f"does not match current recipe shape {current_shape}"
                            )

        if missing:
            shape_errors.insert(
                0,
                "mapped tensor groups not present in the current Krea 2 recipe: "
                f"{_format_sample(missing)}",
            )
        if shape_errors:
            raise Krea2CompatibilityError(
                "Krea 2 LoRA package is not compatible with the current Krea 2 recipe; "
                f"shape-incompatible groups: {_format_sample(shape_errors)}"
            )

    @property
    def loaded_bytes(self) -> int:
        total = 0
        for key_data in self._lora_data_by_set.values():
            for entries in key_data.values():
                for up, down, _scale in entries:
                    total += up.nbytes + down.nbytes
        for key_data in self._direct_data_by_set.values():
            for entries in key_data.values():
                for tensor, _scale in entries:
                    total += tensor.nbytes
        for key_data in self._lokr_data_by_set.values():
            for entries in key_data.values():
                for w1, w2, _scale in entries:
                    total += w1.nbytes + w2.nbytes
        return total

    def get_delta_specs(
        self,
        keys: Sequence[str],
        key_indices: dict[str, int],
        set_id: str | None = None,
    ) -> list[DeltaSpec]:
        specs: list[DeltaSpec] = []

        if set_id is not None:
            lora_sources = [self._lora_data_by_set.get(set_id, {})]
            direct_sources = [self._direct_data_by_set.get(set_id, {})]
            lokr_sources = [self._lokr_data_by_set.get(set_id, {})]
        else:
            lora_sources = list(self._lora_data_by_set.values())
            direct_sources = list(self._direct_data_by_set.values())
            lokr_sources = list(self._lokr_data_by_set.values())

        for key in keys:
            key_idx = key_indices.get(key)
            if key_idx is None:
                continue

            for lora_data in lora_sources:
                for up, down, scale in lora_data.get(key, ()):
                    specs.append(
                        DeltaSpec(
                            kind="standard",
                            key_index=key_idx,
                            up=up,
                            down=down,
                            scale=scale,
                        )
                    )

            for direct_data in direct_sources:
                for tensor, scale in direct_data.get(key, ()):
                    specs.append(
                        DeltaSpec(
                            kind="direct",
                            key_index=key_idx,
                            up=tensor,
                            scale=scale,
                        )
                    )

            for lokr_data in lokr_sources:
                for w1, w2, scale in lokr_data.get(key, ()):
                    specs.append(
                        DeltaSpec(
                            kind="lokr",
                            key_index=key_idx,
                            w1=w1,
                            w2=w2,
                            scale=scale,
                        )
                    )

        return specs

    def cleanup(self) -> None:
        self._lora_data_by_set.clear()
        self._direct_data_by_set.clear()
        self._lokr_data_by_set.clear()
        self._affected_by_set.clear()
        self._affected.clear()
