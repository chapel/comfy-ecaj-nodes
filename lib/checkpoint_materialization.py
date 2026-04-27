"""Full checkpoint materialization sink.

Creates a complete merged diffusion model artifact by writing unaffected base
weights and affected merged weights through an incremental safetensors writer.

The materialization never constructs a second full-model-sized dict in memory:
- Unaffected base weights are written one at a time via write_base_weights().
- Affected merged weights arrive individually via write_tensor() (sink protocol).
- The artifact is finalized atomically only when all keys have been recorded.

Implements MergeResultSink so it can be wired into the merge evaluation loop.

AC: @streaming-full-model-materialization ac-base-weight-bounded-copying
AC: @streaming-full-model-materialization ac-affected-results-released
AC: @streaming-full-model-materialization ac-incomplete-write-not-reused
"""

from __future__ import annotations

from types import TracebackType

import torch

from .incremental_writer import IncrementalSafetensorsWriter, TensorSpec
from .result_sink import MergeResultSink

__all__ = ["CheckpointMaterializationSink"]


class CheckpointMaterializationSink(MergeResultSink):
    """Sink that materializes a complete checkpoint artifact incrementally.

    Combines unaffected base weights with affected merged weights into a single
    safetensors file.  The writer receives tensors one at a time so neither the
    base pass nor the merge pass needs to hold a full model copy in memory.

    Lifecycle::

        sink = CheckpointMaterializationSink(
            base_state=base_state,
            affected_keys=affected_keys,
            dest_path=save_path,
            storage_dtype=storage_dtype,
            metadata=metadata,
        )
        try:
            sink.write_base_weights()          # bounded pass over unaffected keys
            for key, tensor in evaluation:     # merge eval hands off affected keys
                sink.write_tensor(key, tensor)
            sink.finalize()
        except:
            sink.abort()
            raise
    """

    def __init__(
        self,
        base_state: dict[str, torch.Tensor],
        affected_keys: set[str],
        dest_path: str,
        storage_dtype: torch.dtype,
        metadata: dict[str, str] | None = None,
    ) -> None:
        """Initialize the materialization sink.

        Args:
            base_state: Full base model state dict (all keys).  Only read
                during write_base_weights(); the caller retains ownership.
            affected_keys: Keys that will be replaced by merge evaluation.
                These keys are skipped during write_base_weights() and must
                be provided later via write_tensor().
            dest_path: Target safetensors file path.
            storage_dtype: Dtype for all tensors in the artifact.  Base
                weights are converted to this dtype during the bounded pass.
            metadata: Optional safetensors metadata dict.
        """
        self._base_state = base_state
        self._affected_keys = set(affected_keys)
        self._storage_dtype = storage_dtype

        # Build manifest from base state — every key at storage_dtype.
        manifest: dict[str, TensorSpec] = {}
        for key, tensor in base_state.items():
            if storage_dtype != tensor.dtype:
                shape = tuple(tensor.shape)
            else:
                shape = tuple(tensor.shape)
            manifest[key] = TensorSpec(shape=shape, dtype=storage_dtype)

        self._writer = IncrementalSafetensorsWriter(
            manifest, dest_path, metadata=metadata
        )

    def write_base_weights(self) -> None:
        """Write unaffected base weights to the artifact one at a time.

        Only writes keys that are NOT in the affected set.  Each tensor is
        converted to storage_dtype, written, and then the local reference is
        released — no duplicate full-model dict is constructed.

        Raises:
            RuntimeError: If the writer has been finalized, aborted, or
                poisoned by a previous validation error.
        """
        for key, tensor in self._base_state.items():
            if key in self._affected_keys:
                continue
            converted = tensor.to(self._storage_dtype)
            self._writer.write_tensor(key, converted)

    def write_tensor(self, key: str, tensor: torch.Tensor) -> None:
        """Record a completed affected tensor.

        Called by merge evaluation for each affected key as it is produced.
        The tensor is written directly to the artifact file.

        Args:
            key: Parameter name matching a manifest entry.
            tensor: CPU tensor in storage dtype.

        Raises:
            RuntimeError: If the writer has been finalized, aborted, or
                poisoned by a previous validation error.
        """
        self._writer.write_tensor(key, tensor)

    def finalize(self) -> dict[str, torch.Tensor]:
        """Verify completeness and atomically publish the artifact.

        Returns:
            Empty dict — tensors are on disk, not in memory.

        Raises:
            RuntimeError: If any keys are missing or the writer is in an
                error state.
        """
        return self._writer.finalize()

    def abort(self) -> None:
        """Remove temporary file without publishing.

        Safe to call multiple times and after finalize.
        """
        self._writer.abort()

    # -- Context manager support --

    def __enter__(self) -> CheckpointMaterializationSink:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if exc_type is not None and not self._writer._finalized:
            self.abort()
