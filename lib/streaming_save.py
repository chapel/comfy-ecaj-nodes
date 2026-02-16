"""Streaming safetensors writer — saves one tensor at a time.

Unlike safetensors.torch.save_file which materializes all tensor data as
Python bytes simultaneously (~model-sized RAM spike via _flatten()), this
writes one tensor at a time. Peak memory is +1 tensor above baseline.

Output is load-compatible with safe_open() but NOT byte-identical to
save_file() (JSON key ordering may differ). This is by design.

AC: @memory-management ac-7
"""

from __future__ import annotations

import json
import struct

import torch

__all__ = ["stream_save_file"]

# Maps torch dtype → safetensors dtype string.
# Must mirror safetensors' dtype table exactly.
_DTYPE_MAP: dict[torch.dtype, str] = {
    torch.float32: "F32",
    torch.float64: "F64",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.int8: "I8",
    torch.int16: "I16",
    torch.int32: "I32",
    torch.int64: "I64",
    torch.uint8: "U8",
    torch.bool: "BOOL",
}

# Forward compat: float8 types (available in torch >= 2.1)
for _attr, _name in (("float8_e4m3fn", "F8_E4M3"), ("float8_e5m2", "F8_E5M2")):
    _dt = getattr(torch, _attr, None)
    if _dt is not None:
        _DTYPE_MAP[_dt] = _name

# Maps torch dtype → element size in bytes.
_SIZE_MAP: dict[torch.dtype, int] = {
    torch.float32: 4,
    torch.float64: 8,
    torch.float16: 2,
    torch.bfloat16: 2,
    torch.int8: 1,
    torch.int16: 2,
    torch.int32: 4,
    torch.int64: 8,
    torch.uint8: 1,
    torch.bool: 1,
}

for _attr2 in ("float8_e4m3fn", "float8_e5m2"):
    _dt2 = getattr(torch, _attr2, None)
    if _dt2 is not None:
        _SIZE_MAP[_dt2] = 1


def _tensor_bytes(t: torch.Tensor) -> bytes:
    """Extract exact tensor bytes, safe for views with storage_offset.

    Uses .numpy().tobytes() where possible (respects view bounds).
    For dtypes that numpy doesn't support (bfloat16, float8), falls back
    to slicing untyped_storage() with correct offset.
    """
    if not t.is_contiguous():
        t = t.contiguous()
    if t.device.type != "cpu":
        t = t.cpu()
    try:
        return t.numpy().tobytes()
    except TypeError:
        # bfloat16, float8 etc. — numpy doesn't support these dtypes.
        # After contiguous(), storage_offset is 0, so untyped_storage is safe.
        nbytes = t.nelement() * t.element_size()
        return bytes(t.untyped_storage()[:nbytes])


def stream_save_file(
    tensors: dict[str, torch.Tensor],
    filename: str,
    metadata: dict[str, str] | None = None,
) -> None:
    """Save tensors to safetensors format with streaming writes.

    AC: @memory-management ac-7

    Args:
        tensors: Dict of name → tensor to save.
        filename: Output file path.
        metadata: Optional string→string metadata dict.

    Raises:
        ValueError: If tensors use unsupported layout or share storage.
    """
    sorted_names = sorted(tensors.keys())

    # Validate: reject shared-storage tensors (same as save_file)
    seen_data_ptrs: set[int] = set()
    for name in sorted_names:
        t = tensors[name]
        if t.layout != torch.strided:
            raise ValueError(f"Sparse tensor not supported: {name}")
        dtype_str = _DTYPE_MAP.get(t.dtype)
        if dtype_str is None:
            raise ValueError(f"Unsupported dtype {t.dtype} for tensor '{name}'")
        ptr = t.data_ptr()
        if ptr in seen_data_ptrs and t.nelement() > 0:
            raise ValueError(
                f"Shared storage detected for tensor '{name}'. "
                "Cannot save tensors that share underlying storage."
            )
        seen_data_ptrs.add(ptr)

    # Phase 1: Compute header (no tensor data touched)
    header_info: dict[str, dict] = {}
    current_offset = 0
    for name in sorted_names:
        t = tensors[name]
        dtype_str = _DTYPE_MAP[t.dtype]
        nbytes = t.nelement() * _SIZE_MAP[t.dtype]
        header_info[name] = {
            "dtype": dtype_str,
            "shape": list(t.shape),
            "data_offsets": [current_offset, current_offset + nbytes],
        }
        current_offset += nbytes

    # Phase 2: Build padded header (8-byte aligned)
    combined: dict = {}
    if metadata:
        combined["__metadata__"] = metadata
    combined.update(header_info)
    header_json = json.dumps(combined, separators=(",", ":")).encode()
    pad = (8 - ((8 + len(header_json)) % 8)) % 8
    padded = header_json + b" " * pad

    # Phase 3: Stream write — one tensor at a time
    with open(filename, "wb") as f:
        f.write(struct.pack("<Q", len(padded)))
        f.write(padded)
        for name in sorted_names:
            f.write(_tensor_bytes(tensors[name]))
