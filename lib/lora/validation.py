"""Complete-package validation shared by ordinary LoRA loaders."""

import math

import torch
from safetensors import safe_open

_SUFFIXES = {
    ".lora_up.weight": "up",
    ".lora_B.weight": "up",
    ".lora_down.weight": "down",
    ".lora_A.weight": "down",
}
_TEXT_PREFIXES = ("lora_te", "text_encoder", "clip_l.", "clip_g.")


def validate_tensor(tensor, name):
    if tensor.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise ValueError(
            f"Unsupported LoRA dtype for {name}: {tensor.dtype}; use floating factors"
        )
    for chunk in tensor.reshape(-1).split(262144):
        if not torch.isfinite(chunk).all():
            raise ValueError(f"Non-finite LoRA tensor: {name}")


def read_pairs(path, strength, parser, *, clip=False, conv=False):
    """Read/validate every in-domain group before the caller publishes any data.

    Parser returns (canonical key, direction, *slice metadata). Source stems
    remain distinct until alias validation; two names may not overwrite factors.
    """
    if not math.isfinite(strength):
        raise ValueError("LoRA strength must be finite")
    groups = {}
    with safe_open(path, framework="pt", device="cpu") as handle:
        for name in handle.keys():
            is_text = name.startswith(_TEXT_PREFIXES)
            if is_text != clip:
                continue
            if name.endswith(".alpha"):
                stem, direction = name[:-6], "alpha"
            else:
                suffix = next((s for s in _SUFFIXES if name.endswith(s)), None)
                if suffix is None:
                    raise ValueError(
                        f"Unsupported in-domain LoRA tensor {name}; "
                        "use complete ordinary LoRA up/down pairs (no DoRA/LoKR/LoHA)"
                    )
                stem, direction = name[: -len(suffix)], _SUFFIXES[suffix]
            tensor = handle.get_tensor(name)
            validate_tensor(tensor, name)
            group = groups.setdefault(stem, {})
            if direction in group:
                raise ValueError(f"Duplicate LoRA factor aliases for {stem}: {direction}")
            group[direction] = tensor
    pending = []
    aliases = set()
    for stem, factors in groups.items():
        if "up" not in factors or "down" not in factors:
            raise ValueError(f"Incomplete LoRA group {stem}: requires up and down factors")
        parsed = parser(stem + ".lora_up.weight")
        key, _, *extra = parsed
        if key is None:
            raise ValueError(f"Unmapped in-domain LoRA group {stem}")
        identity = (key, extra[0] if extra else None)
        if identity in aliases:
            raise ValueError(f"Duplicate LoRA aliases for {key}: {stem}")
        aliases.add(identity)
        up, down = factors["up"], factors["down"]
        if up.ndim != down.ndim or up.ndim not in ((2, 4) if conv else (2,)):
            raise ValueError(
                f"Unsupported LoRA factor dimensions for {stem}: {up.shape}, {down.shape}"
            )
        if down.shape[0] == 0 or up.shape[1] != down.shape[0] or min(*up.shape, *down.shape) == 0:
            raise ValueError(f"Invalid LoRA rank/shape for {stem}: {up.shape}, {down.shape}")
        if up.ndim == 4 and tuple(up.shape[2:]) != (1, 1):
            raise ValueError(
                f"Unsupported spatial-up/LoCon adapter {stem}; requires 1x1 up factor"
            )
        alpha = factors.get("alpha")
        if alpha is not None and alpha.numel() != 1:
            raise ValueError(f"LoRA alpha must be scalar for {stem}")
        scale = strength * (alpha.item() if alpha is not None else down.shape[0]) / down.shape[0]
        if not math.isfinite(scale):
            raise ValueError(f"Non-finite LoRA scale for {stem}")
        pending.append((key, up, down, scale, extra))
    return pending
