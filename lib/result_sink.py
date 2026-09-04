"""Sink protocol for bounded affected-result handoff.

Provides:
- ResultSink: Protocol that receives completed affected tensors by key.
- DictResultSink: Collects tensors into a dict for backward compatibility
  with the dict-returning evaluation path (save_model=False patch mode).

AC: @streaming-full-model-materialization ac-affected-results-released
AC: @streaming-full-model-materialization ac-direct-artifact-handoff
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

import torch

__all__ = ["ResultSink", "DictResultSink", "WriteFnSink"]


@runtime_checkable
class ResultSink(Protocol):
    """Protocol for receiving completed affected tensors by key.

    AC: @streaming-full-model-materialization ac-direct-artifact-handoff
    AC: @streaming-full-model-materialization ac-affected-results-released

    Implementations receive each completed tensor via ``receive`` as soon as
    it is evaluated and converted to the storage dtype.  The caller does not
    accumulate a full affected-result payload before handing tensors off.

    Known implementations:
    - ``DictResultSink``: collects tensors into a dict (patch mode compat).
    - ``MaterializationSink.write_tensor``: writes to the incremental
      safetensors artifact (full saved model mode).
    - ``IncrementalWriter.write_tensor``: writes to the atomic incremental
      writer (full saved model mode, newer writer).
    """

    def receive(self, key: str, tensor: torch.Tensor) -> None:
        """Accept a completed affected tensor.

        Args:
            key: The parameter key (e.g. "diffusion_model.input_blocks.0.0.weight").
            tensor: CPU tensor in the target storage dtype.  The caller may
                release its reference after this call returns.
        """
        ...


class DictResultSink:
    """Sink that collects tensors into a dict for backward compatibility.

    AC: @streaming-full-model-materialization ac-direct-artifact-handoff

    Used by patch mode (save_model=False) so the dict-returning evaluation
    path can be expressed as a thin wrapper around the sink-based path.

    The collected dict is accessible via ``.results`` after evaluation
    completes.
    """

    def __init__(self) -> None:
        self._results: dict[str, torch.Tensor] = {}

    def receive(self, key: str, tensor: torch.Tensor) -> None:
        """Store the tensor in the internal dict."""
        self._results[key] = tensor

    @property
    def results(self) -> dict[str, torch.Tensor]:
        """Return the collected tensors."""
        return self._results


class WriteFnSink:
    """Adapter that wraps a ``write_tensor``-style callback as a ResultSink.

    AC: @streaming-full-model-materialization ac-direct-artifact-handoff

    Allows objects with a ``write_tensor(name, tensor)`` interface (such as
    ``MaterializationSink`` and ``IncrementalWriter``) to be used with
    :func:`evaluate_to_sink` without modification.

    Usage::

        sink = MaterializationSink()
        sink.open(manifest, save_path, metadata)
        evaluate_to_sink(..., sink=WriteFnSink(sink.write_tensor))
    """

    def __init__(self, write_fn: Callable[[str, torch.Tensor], None]) -> None:
        self._write_fn = write_fn

    def receive(self, key: str, tensor: torch.Tensor) -> None:
        """Delegate to the wrapped write function."""
        self._write_fn(key, tensor)
