"""Incremental safetensors artifact writer.

Writes a valid safetensors file one tensor at a time from a predetermined
manifest.  Tensors may arrive in any order; the writer places each at its
precomputed file offset.  The final artifact is published atomically only
after every expected tensor has been written and validated.

Implements MergeResultSink so it can be used as a drop-in consumer for
merge evaluation.

AC: @streaming-full-model-materialization ac-incomplete-write-not-reused
"""

from __future__ import annotations

import json
import os
import secrets
import struct
from dataclasses import dataclass
from types import TracebackType

import torch

from .streaming_save import _DTYPE_MAP, _tensor_bytes

__all__ = ["IncrementalSafetensorsWriter", "TensorSpec"]


@dataclass(frozen=True)
class TensorSpec:
    """Expected shape and dtype for a single tensor in the manifest."""

    shape: tuple[int, ...]
    dtype: torch.dtype


class IncrementalSafetensorsWriter:
    """Writes a safetensors file incrementally from a known manifest.

    The manifest declares every tensor's key, shape, and dtype before any
    data is written.  The writer precomputes the safetensors header and
    byte offsets, writes the header to a temporary file, then accepts
    tensors one at a time via write_tensor().

    finalize() verifies completeness, fsyncs, and atomically replaces
    the destination.  abort() removes the temporary file.

    Lifecycle::

        writer = IncrementalSafetensorsWriter(manifest, dest_path)
        try:
            for key, tensor in source:
                writer.write_tensor(key, tensor)
            writer.finalize()
        except:
            writer.abort()
            raise
    """

    def __init__(
        self,
        manifest: dict[str, TensorSpec],
        dest_path: str,
        metadata: dict[str, str] | None = None,
    ) -> None:
        self._dest_path = dest_path
        self._finalized = False
        self._aborted = False
        self._manifest = dict(manifest)
        self._written: set[str] = set()

        # Sort keys for deterministic layout (safetensors convention).
        sorted_keys = sorted(manifest.keys())

        # Precompute header info with data offsets.
        header_info: dict[str, dict] = {}
        data_offset = 0
        # Map key -> (file_offset_from_data_start, nbytes) for seek writes.
        self._key_offsets: dict[str, tuple[int, int]] = {}

        for key in sorted_keys:
            spec = manifest[key]
            dtype_str = _DTYPE_MAP.get(spec.dtype)
            if dtype_str is None:
                raise ValueError(
                    f"Unsupported dtype {spec.dtype} for tensor '{key}'"
                )
            numel = 1
            for dim in spec.shape:
                numel *= dim
            element_size = torch.tensor([], dtype=spec.dtype).element_size()
            nbytes = numel * element_size

            header_info[key] = {
                "dtype": dtype_str,
                "shape": list(spec.shape),
                "data_offsets": [data_offset, data_offset + nbytes],
            }
            self._key_offsets[key] = (data_offset, nbytes)
            data_offset += nbytes

        # Build padded JSON header (8-byte aligned).
        combined: dict = {}
        if metadata:
            combined["__metadata__"] = metadata
        combined.update(header_info)
        header_json = json.dumps(combined, separators=(",", ":")).encode()
        pad = (8 - ((8 + len(header_json)) % 8)) % 8
        padded_header = header_json + b" " * pad
        self._header_size = 8 + len(padded_header)  # 8-byte length prefix + header

        # Write header to temp file.
        directory = os.path.dirname(dest_path) or "."
        suffix = secrets.token_hex(4)
        self._tmp_path = os.path.join(
            directory,
            f".ecaj_tmp_{suffix}_{os.path.basename(dest_path)}",
        )

        try:
            self._file = open(self._tmp_path, "wb")
            self._file.write(struct.pack("<Q", len(padded_header)))
            self._file.write(padded_header)
            # Pre-allocate space for all tensor data so seeks are valid.
            if data_offset > 0:
                self._file.seek(self._header_size + data_offset - 1)
                self._file.write(b"\x00")
        except BaseException:
            # Clean up if header write fails.
            try:
                os.unlink(self._tmp_path)
            except OSError:
                pass
            raise

    def write_tensor(self, key: str, tensor: torch.Tensor) -> None:
        """Write a single tensor to the artifact.

        Validates the tensor against the manifest and writes its bytes
        at the precomputed file offset.

        Args:
            key: Tensor name matching a manifest entry.
            tensor: CPU tensor with the expected shape and dtype.

        Raises:
            RuntimeError: On validation failure, duplicate key, or
                write after finalize/abort.
        """
        if self._finalized:
            raise RuntimeError(
                "Cannot write to a writer that has been finalized"
            )
        if self._aborted:
            raise RuntimeError(
                "Cannot write to a writer that has been aborted"
            )

        if key not in self._manifest:
            raise RuntimeError(
                f"Unexpected key '{key}': not in manifest"
            )

        if key in self._written:
            raise RuntimeError(
                f"Duplicate key '{key}': already written"
            )

        spec = self._manifest[key]

        if tuple(tensor.shape) != tuple(spec.shape):
            raise RuntimeError(
                f"Shape mismatch for '{key}': "
                f"expected {spec.shape}, got {tuple(tensor.shape)}"
            )

        if tensor.dtype != spec.dtype:
            raise RuntimeError(
                f"Dtype mismatch for '{key}': "
                f"expected {spec.dtype}, got {tensor.dtype}"
            )

        data_bytes = _tensor_bytes(tensor)
        offset, expected_nbytes = self._key_offsets[key]

        if len(data_bytes) != expected_nbytes:
            raise RuntimeError(
                f"Byte size mismatch for '{key}': "
                f"expected {expected_nbytes}, got {len(data_bytes)}"
            )

        self._file.seek(self._header_size + offset)
        self._file.write(data_bytes)
        self._written.add(key)

    def finalize(self) -> dict[str, torch.Tensor]:
        """Verify completeness and atomically publish the artifact.

        Returns:
            Empty dict (tensors are on disk, not in memory).

        Raises:
            RuntimeError: If any expected keys are missing, or if already
                finalized/aborted.
        """
        if self._finalized:
            raise RuntimeError("Writer has already been finalized")
        if self._aborted:
            raise RuntimeError("Cannot finalize an aborted writer")

        missing = set(self._manifest.keys()) - self._written
        if missing:
            # Incomplete — abort and raise.
            self.abort()
            sorted_missing = sorted(missing)
            raise RuntimeError(
                f"Cannot finalize: {len(sorted_missing)} missing key(s): "
                f"{sorted_missing}"
            )

        try:
            self._file.flush()
            os.fsync(self._file.fileno())
            self._file.close()
            os.replace(self._tmp_path, self._dest_path)
        except BaseException:
            self._cleanup_tmp()
            raise
        finally:
            self._finalized = True

        return {}

    def abort(self) -> None:
        """Remove the temporary file without publishing.

        Safe to call multiple times and after finalize.
        """
        self._aborted = True
        self._close_file()
        self._cleanup_tmp()

    # -- Context manager support --

    def __enter__(self) -> IncrementalSafetensorsWriter:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if exc_type is not None and not self._finalized:
            self.abort()

    # -- Internal helpers --

    def _close_file(self) -> None:
        try:
            if not self._file.closed:
                self._file.close()
        except Exception:
            pass

    def _cleanup_tmp(self) -> None:
        try:
            os.unlink(self._tmp_path)
        except OSError:
            pass
