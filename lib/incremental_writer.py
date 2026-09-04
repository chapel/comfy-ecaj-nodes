"""Atomic incremental safetensors writer for safe artifact publication.

A manifest-driven writer that accepts tensor writes in any order, validates
every submission against the declared manifest, and publishes the final
artifact atomically.  Invalid submissions (unknown keys, duplicate keys,
wrong shapes, wrong dtypes) poison the writer so finalize() cannot publish
a corrupt or incomplete artifact.

AC: @saved-model-artifact-safety ac-no-partial-publication
AC: @saved-model-artifact-safety ac-existing-valid-artifact-preserved
AC: @streaming-full-model-materialization ac-incomplete-write-not-reused
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import struct

import torch

from .streaming_save import _DTYPE_MAP, _tensor_bytes

logger = logging.getLogger("ecaj.incremental_writer")

__all__ = ["IncrementalWriter"]


class IncrementalWriter:
    """Manifest-driven incremental safetensors writer with poison semantics.

    The writer is created from a manifest of expected tensor keys, shapes,
    and dtypes.  Tensors may be written in any order.  On finalize(), the
    completed temp file atomically replaces the target path.

    Validation errors poison the writer — a poisoned writer removes its temp
    file immediately and refuses to finalize, ensuring no partial or corrupt
    artifact is ever published.

    Usage::

        manifest = {"weight": (torch.float32, (768, 768)), ...}
        w = IncrementalWriter(manifest, "/path/to/model.safetensors")
        for name, tensor in tensors.items():
            w.write_tensor(name, tensor)
        w.finalize()

    Or as a context manager for automatic cleanup on failure::

        with IncrementalWriter(manifest, save_path) as w:
            for name, tensor in tensors.items():
                w.write_tensor(name, tensor)
            w.finalize()
    """

    def __init__(
        self,
        manifest: dict[str, tuple[torch.dtype, tuple[int, ...]]],
        save_path: str,
        metadata: dict[str, str] | None = None,
    ) -> None:
        """Create a writer and write the safetensors header.

        Args:
            manifest: Dict of tensor_name -> (dtype, shape) for every tensor
                that will be written.
            save_path: Target file path.  The temp file is created in the same
                directory.  On finalize(), the temp file atomically replaces
                this path.
            metadata: Optional string->string metadata dict to embed in the
                safetensors header.
        """
        if metadata is not None:
            for k, v in metadata.items():
                if not isinstance(k, str):
                    raise TypeError(f"metadata key must be str, got {type(k).__name__}: {k!r}")
                if not isinstance(v, str):
                    raise TypeError(
                        f"metadata value must be str, got {type(v).__name__} for key {k!r}"
                    )

        self._save_path = save_path
        self._poisoned = False
        self._finalized = False
        self._aborted = False

        directory = os.path.dirname(save_path) or "."
        suffix = secrets.token_hex(4)
        self._tmp_path = os.path.join(
            directory, f".ecaj_tmp_{suffix}_{os.path.basename(save_path)}"
        )

        sorted_names = sorted(manifest.keys())

        # Compute header and record per-tensor byte offsets and expected specs.
        header_info: dict[str, dict] = {}
        self._tensor_offsets: dict[str, int] = {}
        self._tensor_specs: dict[str, tuple[torch.dtype, tuple[int, ...], int]] = {}
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
            self._tensor_offsets[name] = current_offset
            self._tensor_specs[name] = (dt, shape, nbytes)
            current_offset += nbytes

        total_data_bytes = current_offset

        # Build padded header (8-byte aligned).
        combined: dict = {}
        if metadata:
            combined["__metadata__"] = metadata
        combined.update(header_info)
        header_json = json.dumps(combined, separators=(",", ":")).encode()
        pad = (8 - ((8 + len(header_json)) % 8)) % 8
        padded = header_json + b" " * pad

        self._file = open(self._tmp_path, "wb")
        self._file.write(struct.pack("<Q", len(padded)))
        self._file.write(padded)

        # Record where the data region starts (after header).
        self._data_start = 8 + len(padded)

        # Pre-allocate file to full size so seek-based writes land correctly.
        if total_data_bytes > 0:
            self._file.seek(self._data_start + total_data_bytes - 1)
            self._file.write(b"\x00")

        self._written: set[str] = set()
        self._total_count = len(sorted_names)

    # -- Context manager --

    def __enter__(self) -> IncrementalWriter:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if not self._finalized:
            self.abort()

    # -- Public API --

    def write_tensor(self, name: str, tensor: torch.Tensor) -> None:
        """Write a single tensor's data at its pre-computed offset.

        Validates the tensor's name, dtype, shape, and byte length against the
        manifest.  A validation failure poisons the writer so finalize() cannot
        publish a corrupt artifact.

        Args:
            name: Tensor name (must be a key from the manifest).
            tensor: The tensor data to write.

        Raises:
            RuntimeError: If writer is finalized, aborted, poisoned, or the
                name is unknown / already written.
            ValueError: If the tensor's dtype or shape does not match the
                manifest.
        """
        self._check_writable()

        if name not in self._tensor_offsets:
            self._poison(f"write_tensor called with unknown tensor name not in manifest: {name!r}")
            raise RuntimeError(
                f"write_tensor called with unknown tensor name not in manifest: {name!r}"
            )

        if name in self._written:
            self._poison(f"write_tensor called twice for already written tensor: {name!r}")
            raise RuntimeError(f"write_tensor called twice for already written tensor: {name!r}")

        expected_dtype, expected_shape, expected_nbytes = self._tensor_specs[name]
        if tensor.dtype != expected_dtype:
            self._poison(
                f"write_tensor {name!r}: dtype mismatch — "
                f"expected {expected_dtype}, got {tensor.dtype}"
            )
            raise ValueError(
                f"write_tensor {name!r}: dtype mismatch — "
                f"expected {expected_dtype}, got {tensor.dtype}"
            )
        if tuple(tensor.shape) != expected_shape:
            self._poison(
                f"write_tensor {name!r}: shape mismatch — "
                f"expected {expected_shape}, got {tuple(tensor.shape)}"
            )
            raise ValueError(
                f"write_tensor {name!r}: shape mismatch — "
                f"expected {expected_shape}, got {tuple(tensor.shape)}"
            )

        actual_nbytes = tensor.nelement() * tensor.element_size()
        if actual_nbytes != expected_nbytes:
            self._poison(
                f"write_tensor {name!r}: byte length mismatch — "
                f"expected {expected_nbytes}, got {actual_nbytes}"
            )
            raise ValueError(
                f"write_tensor {name!r}: byte length mismatch — "
                f"expected {expected_nbytes}, got {actual_nbytes}"
            )

        offset = self._data_start + self._tensor_offsets[name]
        try:
            self._file.seek(offset)
            self._file.write(_tensor_bytes(tensor))
        except Exception as exc:
            self._poison(f"write_tensor {name!r}: I/O error — {exc}")
            raise
        self._written.add(name)

    def finalize(self) -> None:
        """Flush, fsync, and atomically replace save_path with the completed file.

        AC: @saved-model-artifact-safety ac-no-partial-publication
        AC: @streaming-full-model-materialization ac-incomplete-write-not-reused

        Raises:
            RuntimeError: If poisoned, aborted, already finalized, or not all
                manifest keys have been written.
        """
        if self._aborted:
            raise RuntimeError("IncrementalWriter already aborted")
        if self._finalized:
            raise RuntimeError("IncrementalWriter already finalized")
        if self._poisoned:
            raise RuntimeError("IncrementalWriter is poisoned — cannot finalize")
        if len(self._written) != self._total_count:
            self._aborted = True
            self._cleanup_temp()
            raise RuntimeError(
                f"Cannot finalize: wrote {len(self._written)}/{self._total_count} tensors"
            )

        self._file.flush()
        os.fsync(self._file.fileno())
        self._file.close()
        self._file = None

        os.replace(self._tmp_path, self._save_path)
        self._finalized = True
        logger.info("IncrementalWriter finalized: %s", self._save_path)

    def abort(self) -> None:
        """Clean up the temp file without replacing the target.

        AC: @saved-model-artifact-safety ac-existing-valid-artifact-preserved

        Safe to call multiple times.  Safe to call after finalize (no-op).
        """
        if self._finalized:
            return
        self._aborted = True
        self._cleanup_temp()

    # -- Internal --

    def _check_writable(self) -> None:
        """Raise if the writer cannot accept more writes."""
        if self._finalized:
            raise RuntimeError("IncrementalWriter already finalized")
        if self._aborted:
            raise RuntimeError("IncrementalWriter already aborted")
        if self._poisoned:
            raise RuntimeError("IncrementalWriter is poisoned — cannot write")

    def _poison(self, reason: str) -> None:
        """Mark the writer as poisoned and clean up the temp file.

        A poisoned writer cannot finalize or accept further writes.
        The temp file is removed immediately to avoid leaving partial artifacts.
        """
        logger.warning("IncrementalWriter poisoned: %s", reason)
        self._poisoned = True
        self._cleanup_temp()

    def _cleanup_temp(self) -> None:
        """Close the file handle and remove the temp file if it exists."""
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
