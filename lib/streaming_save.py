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
import logging
import os
import secrets
import struct
from collections.abc import Callable

import torch

logger = logging.getLogger("ecaj.streaming_save")

__all__ = ["stream_save_file", "MaterializationSink"]

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
    """Extract exact tensor bytes for safetensors serialization.

    Normalizes to CPU + contiguous, then exports raw bytes. The BF16/F8
    fallback uses a uint8 bitcast view — it does NOT change the saved dtype.
    The safetensors header dtype is set from t.dtype by the caller, so the
    file still records the original dtype with bit-identical data.
    """
    if t.device.type != "cpu":
        t = t.cpu()
    if not t.is_contiguous():
        t = t.contiguous()
    try:
        return t.numpy().tobytes()
    except TypeError:
        # NumPy doesn't support this dtype (BF16, F8 variants, complex, etc.).
        # Reinterpret contiguous memory as uint8 — a bitcast view, not a
        # numeric conversion. The element-wise Python loop implied by
        # bytes(untyped_storage()[start:end]) stalls on large BF16 weights;
        # this path produces the same raw bytes via a single buffer copy.
        return t.reshape(-1).view(torch.uint8).numpy().tobytes()


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


class MaterializationSink:
    """Incremental safetensors writer for full saved model materialization.

    AC: @streaming-full-model-materialization ac-direct-artifact-handoff
    AC: @streaming-full-model-materialization ac-affected-results-released
    AC: @streaming-full-model-materialization ac-base-weight-bounded-copying
    AC: @streaming-full-model-materialization ac-incomplete-write-not-reused

    The sink writes a safetensors file incrementally using random-access writes:
    1. open() — compute and write the header, pre-allocate the data region
    2. write_tensor(name, tensor) — seek to the tensor's offset and write (any order)
    3. finalize(save_path) — fsync and atomically replace save_path
    4. abort() — delete the temp file on failure

    Tensors can be written in any order (not just sorted-name order) because
    the header pre-computes byte offsets for every tensor. This allows
    group-at-a-time evaluation: evaluate a group, write all its keys, free
    the group, then evaluate the next group — even when groups' keys are
    interleaved in sorted name order.

    Tensors can be released after write_tensor returns — they are not retained.
    """

    def __init__(self) -> None:
        self._file = None
        self._tmp_path: str | None = None
        self._tensor_offsets: dict[str, int] | None = None
        self._tensor_specs: dict[str, tuple[torch.dtype, tuple[int, ...], int]] | None = None
        self._written: set[str] | None = None
        self._total_count: int = 0
        self._data_start: int = 0
        self._finalized: bool = False
        self._aborted: bool = False

    def open(
        self,
        manifest: dict[str, tuple[torch.dtype, tuple[int, ...]]],
        save_path: str,
        metadata: dict[str, str] | None = None,
    ) -> None:
        """Write the safetensors header from a manifest of key→(dtype, shape).

        Args:
            manifest: Dict of tensor_name → (dtype, shape) for every tensor
                      that will be written (base + affected).
            save_path: Target file path (used to determine temp file location).
            metadata: Optional safetensors metadata dict.
        """
        directory = os.path.dirname(save_path) or "."
        suffix = secrets.token_hex(4)
        self._tmp_path = os.path.join(
            directory, f".ecaj_tmp_{suffix}_{os.path.basename(save_path)}"
        )

        sorted_names = sorted(manifest.keys())

        # Compute header and record per-tensor byte offsets and expected specs
        header_info: dict[str, dict] = {}
        tensor_offsets: dict[str, int] = {}
        tensor_specs: dict[str, tuple[torch.dtype, tuple[int, ...], int]] = {}
        current_offset = 0
        for name in sorted_names:
            dt, shape = manifest[name]
            dtype_str = _DTYPE_MAP[dt]
            elem_size = torch.tensor([], dtype=dt).element_size()
            numel = 1
            for d in shape:
                numel *= d
            nbytes = numel * elem_size
            header_info[name] = {
                "dtype": dtype_str,
                "shape": list(shape),
                "data_offsets": [current_offset, current_offset + nbytes],
            }
            tensor_offsets[name] = current_offset
            tensor_specs[name] = (dt, shape, nbytes)
            current_offset += nbytes

        total_data_bytes = current_offset

        # Build padded header
        combined: dict = {}
        if metadata:
            combined["__metadata__"] = metadata
        combined.update(header_info)
        header_json = json.dumps(combined, separators=(",", ":")).encode()
        pad = (8 - ((8 + len(header_json)) % 8)) % 8
        padded = header_json + b" " * pad

        # Opening can fail before the caller has entered its cleanup scope.
        try:
            self._file = open(self._tmp_path, "wb")
            self._file.write(struct.pack("<Q", len(padded)))
            self._file.write(padded)
            self._data_start = 8 + len(padded)
            if total_data_bytes > 0:
                self._file.seek(self._data_start + total_data_bytes - 1)
                self._file.write(b"\x00")
        except BaseException:
            self.abort()
            raise

        self._tensor_offsets = tensor_offsets
        self._tensor_specs = tensor_specs
        self._written = set()
        self._total_count = len(sorted_names)

    def write_tensor(self, name: str, tensor: torch.Tensor) -> None:
        """Write a single tensor's data to the file at its pre-computed offset.

        Can be called in any order — the header defines each tensor's byte
        position.  The tensor can be released after this call returns.

        Validates the tensor's dtype, shape, and byte length against the
        manifest before writing.  A mismatch is rejected immediately so
        that corrupt or incomplete artifacts cannot be finalized and later
        accepted as valid cache hits.

        Args:
            name: Tensor name (must be a key from the manifest).
            tensor: The tensor data to write.

        Raises:
            RuntimeError: If sink not open, already finalized/aborted, or
                name is not in the manifest / already written.
            ValueError: If the tensor's dtype, shape, or byte length does
                not match the manifest.
        """
        if self._file is None or self._tensor_offsets is None or self._written is None:
            raise RuntimeError("MaterializationSink not open")
        if self._finalized or self._aborted:
            raise RuntimeError("MaterializationSink already finalized or aborted")
        if name not in self._tensor_offsets:
            raise RuntimeError(f"write_tensor called with unknown tensor name: {name!r}")
        if name in self._written:
            raise RuntimeError(f"write_tensor called twice for tensor: {name!r}")

        # Validate tensor against manifest expectations.
        assert self._tensor_specs is not None
        expected_dtype, expected_shape, expected_nbytes = self._tensor_specs[name]
        if tensor.dtype != expected_dtype:
            raise ValueError(
                f"write_tensor {name!r}: dtype mismatch — "
                f"expected {expected_dtype}, got {tensor.dtype}"
            )
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"write_tensor {name!r}: shape mismatch — "
                f"expected {expected_shape}, got {tuple(tensor.shape)}"
            )
        actual_nbytes = tensor.nelement() * tensor.element_size()
        if actual_nbytes != expected_nbytes:
            raise ValueError(
                f"write_tensor {name!r}: byte length mismatch — "
                f"expected {expected_nbytes}, got {actual_nbytes}"
            )

        offset = self._data_start + self._tensor_offsets[name]
        self._file.seek(offset)
        self._file.write(_tensor_bytes(tensor))
        self._written.add(name)

    def finalize(
        self,
        save_path: str,
        *,
        pre_publish_check: Callable[[str], None] | None = None,
    ) -> None:
        """Flush, fsync, and atomically replace save_path with the completed file.

        AC: @streaming-full-model-materialization ac-incomplete-write-not-reused
        AC: @saved-model-artifact-safety ac-no-partial-publication

        Args:
            save_path: Target file path to atomically replace.
            pre_publish_check: Optional callable that runs against the temp
                file path between fsync and atomic replace.  Raising in the
                callback aborts the publication; the caller is responsible
                for cleaning up the temp file via abort() in that case.

        Raises:
            RuntimeError: If not all tensors were written, or already finalized.
        """
        if self._file is None or self._written is None:
            raise RuntimeError("MaterializationSink not open")
        if self._finalized or self._aborted:
            raise RuntimeError("MaterializationSink already finalized or aborted")
        if len(self._written) != self._total_count:
            raise RuntimeError(
                f"Cannot finalize: wrote {len(self._written)}/{self._total_count} tensors"
            )

        self._file.flush()
        os.fsync(self._file.fileno())
        self._file.close()
        self._file = None

        if pre_publish_check is not None:
            assert self._tmp_path is not None
            pre_publish_check(self._tmp_path)

        os.replace(self._tmp_path, save_path)
        self._finalized = True
        logger.info("MaterializationSink finalized: %s", save_path)

    def abort(self) -> None:
        """Clean up the temp file without replacing the target.

        AC: @streaming-full-model-materialization ac-incomplete-write-not-reused

        Safe to call multiple times. Safe to call after finalize (no-op).
        """
        if self._finalized:
            return
        self._aborted = True
        if self._file is not None:
            try:
                self._file.close()
            except OSError:
                pass
            self._file = None
        if self._tmp_path is not None:
            try:
                os.unlink(self._tmp_path)
            except OSError:
                pass
