"""Merge result sink — bounded handoff of completed tensors from evaluation.

Provides:
- MergeResultSink: ABC defining write_tensor / finalize / abort contract
- InMemorySink: collects tensors into a dict (preserves current behavior)

This module is pure Python/PyTorch — no ComfyUI imports.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from types import TracebackType

import torch


class MergeResultSink(ABC):
    """Abstract sink for merge evaluation results.

    Merge evaluation hands off each completed affected tensor via
    write_tensor(). The consumer finalizes when all tensors are written,
    or aborts on error to release temporary resources.

    Lifecycle:
        sink = SomeSink()
        try:
            for key, tensor in evaluation:
                sink.write_tensor(key, tensor)
            result = sink.finalize()
        except:
            sink.abort()
            raise
    """

    @abstractmethod
    def write_tensor(self, key: str, tensor: torch.Tensor) -> None:
        """Record a completed affected tensor.

        Args:
            key: Parameter name (e.g. "diffusion_model.input_blocks.0.0.weight")
            tensor: CPU tensor in storage dtype

        Raises:
            RuntimeError: If the sink has been finalized or aborted.
        """

    @abstractmethod
    def finalize(self) -> dict[str, torch.Tensor]:
        """Seal the sink and return collected results.

        Returns:
            Dict of key -> tensor for all written tensors.

        Raises:
            RuntimeError: If the sink has already been finalized or aborted.
        """

    @abstractmethod
    def abort(self) -> None:
        """Release temporary resources without producing a result.

        Safe to call multiple times and after finalize (for finally blocks).
        """


class InMemorySink(MergeResultSink):
    """Collects tensors into a dict, preserving current merged_state behavior.

    After finalize(), returns the accumulated dict. After abort(), clears
    internal state to release tensor references.
    """

    def __init__(self) -> None:
        self._tensors: dict[str, torch.Tensor] = {}
        self._finalized = False
        self._aborted = False

    def write_tensor(self, key: str, tensor: torch.Tensor) -> None:
        if self._finalized or self._aborted:
            raise RuntimeError(
                "Cannot write to a sink that has been "
                f"{'finalized' if self._finalized else 'aborted'}"
            )
        self._tensors[key] = tensor

    def finalize(self) -> dict[str, torch.Tensor]:
        if self._finalized:
            raise RuntimeError("Sink has already been finalized")
        if self._aborted:
            raise RuntimeError("Cannot finalize an aborted sink")
        self._finalized = True
        return dict(self._tensors)

    def abort(self) -> None:
        self._aborted = True
        self._tensors.clear()

    # Context manager support for safe cleanup in error paths.

    def __enter__(self) -> InMemorySink:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        if exc_type is not None and not self._finalized:
            self.abort()
