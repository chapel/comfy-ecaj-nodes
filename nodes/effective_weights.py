"""Read effective Comfy weights without installing or retaining dense patches.

Metadata uses resident state only. Evaluation/serialization obtains patched keys
on demand via Comfy's return_weight contract; a shape group is NOT a cache of
materialized weights. Comfy still owns the source model and returned clones.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping

import torch


def _active(value):
    if isinstance(value, dict):
        return any(_active(v) for v in value.values())
    return bool(value)


def validate_static_patcher(patcher: object, *, for_serialization: bool = False) -> None:
    """Reject behavior which cannot be represented by one static weight mapping.

    ModelPatcherDynamic inherits the supported per-key API: its class is not a
    reason to opt out of Comfy memory management or reject an ordinary input.
    Scheduled hooks, injections and arbitrary functions, in contrast, need an
    execution context we cannot reproduce while writing a static artifact.
    """
    for field in (
        "hook_patches",
        "current_hooks",
        "forced_hooks",
        "hook_backup",
        "weight_wrapper_patches",
        "injections",
    ):
        if _active(getattr(patcher, field, None)):
            raise ValueError(
                f"WIDEN cannot materialize effective weights with {field}; "
                "scheduled/dynamic/opaque patch behavior is unsupported"
            )
    object_patches = getattr(patcher, "object_patches", {})
    # ModelSampling/latent-format configuration is not weight arithmetic and
    # remains on returned patch-mode clones. Static artifacts cannot encode
    # those runtime objects. Replacing arbitrary modules may change weights.
    if object_patches and (
        for_serialization or set(object_patches) - {"model_sampling", "latent_format"}
    ):
        raise ValueError("Unsupported object_patches for this effective-weight output path")
    if getattr(patcher, "backup", None):
        raise ValueError(
            "WIDEN requires unloaded input weights; resident patch backups "
            "would apply upstream patches twice"
        )
    patches = getattr(patcher, "patches", {})
    if patches and not callable(getattr(patcher, "patch_weight_to_device", None)):
        raise ValueError("Patched input does not expose Comfy's return_weight contract")
    for key, records in patches.items():
        for record in records:
            if not isinstance(record, (tuple, list)) or len(record) != 5 or record[4] is not None:
                raise ValueError(f"Unsupported opaque patch function/record for {key}")
            payload = record[1]
            # Native Comfy adapters own their arithmetic. Do not reimplement
            # LoRA/DoRA here or accept arbitrary custom callable payloads.
            if type(payload).__module__.startswith("comfy.weight_adapter."):
                continue
            if not isinstance(payload, tuple) or not payload:
                raise ValueError(f"Unsupported opaque patch payload for {key}")
            if len(payload) == 1 and isinstance(payload[0], torch.Tensor):
                continue  # Comfy's untagged additive diff
            if len(payload) != 2 or payload[0] not in ("diff", "set"):
                raise ValueError(f"Unsupported patch type for {key}: {payload[0]!r}")


class EffectiveWeights(Mapping[str, torch.Tensor]):
    """Lazy per-key weights; ``raw`` is metadata only, never effective identity."""

    def __init__(self, patcher: object, *, for_serialization: bool = False):
        validate_static_patcher(patcher, for_serialization=for_serialization)
        self.patcher = patcher
        self.raw = patcher.model_state_dict()
        for key, tensor in self.raw.items():
            if type(tensor) not in (torch.Tensor, torch.nn.Parameter):
                raise ValueError(
                    f"Unsupported lazy/quantized weight for {key}: {type(tensor).__name__}"
                )
            if tensor.device.type == "meta":
                raise ValueError(f"Cannot materialize meta weight {key}")

    def __iter__(self) -> Iterator[str]:
        return iter(self.raw)

    def __len__(self) -> int:
        return len(self.raw)

    def __getitem__(self, key: str) -> torch.Tensor:
        raw = self.raw[key]
        if key not in getattr(self.patcher, "patches", {}):
            return raw
        with torch.no_grad():
            value = self.patcher.patch_weight_to_device(
                key,
                device_to=torch.device("cpu"),
                return_weight=True,
            )
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != raw.shape
            or value.dtype != raw.dtype
        ):
            raise ValueError(
                f"Effective patch changed shape/dtype for {key}; unsupported static layout"
            )
        return value


class SelectedWeights(Mapping[str, torch.Tensor]):
    """Restrict keys without materializing a potentially model-sized group."""

    def __init__(self, source: Mapping[str, torch.Tensor], keys):
        self.source = source
        self.keys_selected = dict.fromkeys(keys)

    def __iter__(self):
        return iter(self.keys_selected)

    def __len__(self):
        return len(self.keys_selected)

    def __getitem__(self, key):
        if key not in self.keys_selected:
            raise KeyError(key)
        return self.source[key]


def weight_metadata(state):
    """Also accept plain mappings used by internal callers and tests."""
    return state.raw if isinstance(state, EffectiveWeights) else state
