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
    torch.complex64: "C64",
}

# Optional dtypes — available depending on torch version
for _attr, _name in (
    ("float8_e4m3fn", "F8_E4M3"),
    ("float8_e5m2", "F8_E5M2"),
    ("uint16", "U16"),
    ("uint32", "U32"),
    ("uint64", "U64"),
):
    _dt = getattr(torch, _attr, None)
    if _dt is not None:
        _DTYPE_MAP[_dt] = _name


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
        # bfloat16, float8, complex64 etc. — numpy doesn't support these.
        # Use explicit offset for safety even though contiguous() should
        # produce offset=0.
        nbytes = t.nelement() * t.element_size()
        offset = t.storage_offset() * t.element_size()
        return bytes(t.untyped_storage()[offset:offset + nbytes])


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
        nbytes = t.nelement() * t.element_size()
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
