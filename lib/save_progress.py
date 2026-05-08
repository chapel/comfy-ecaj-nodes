"""Saved-model materialization progress reporter.

AC: @streaming-materialization-progress ac-progress-during-streaming-writes
AC: @streaming-materialization-progress ac-no-op-save-progress
AC: @streaming-materialization-progress ac-affected-write-progress
AC: @streaming-materialization-progress ac-finalization-status-visible
AC: @streaming-materialization-progress ac-cache-reuse-status-visible
AC: @streaming-materialization-progress ac-failure-status-not-success

Lifecycle helper for the WIDEN Exit node's saved-model paths so users can
distinguish active model saving from a stalled execution.  Wraps an optional
ComfyUI ``ProgressBar`` (no-op when ComfyUI is not present) and emits a small
status string per phase that can be inspected by tests and logs.

Phases reported:
- ``prepare``           — manifest open / artifact preparation (1 unit)
- ``write_tensor``      — each successful tensor write (1 unit per tensor)
- ``finalize``          — fsync + atomic publish (1 unit)
- ``reload``            — reloading the published artifact through Comfy (1 unit)
- ``cache_reuse``       — a saved artifact already satisfies the request
- ``failure``           — materialization failed before publication

The helper does not log absolute paths; the caller already validates and
formats user-facing names so messages stay free of sensitive filesystem
information.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Protocol

logger = logging.getLogger("ecaj.save_progress")

__all__ = ["SavedModelProgress", "ProgressBarLike"]


class ProgressBarLike(Protocol):
    """Structural type for anything that quacks like ComfyUI's ProgressBar."""

    def update(self, value: int) -> None:  # pragma: no cover - protocol
        ...


class SavedModelProgress:
    """Reports lifecycle progress for a single saved-model materialization.

    The helper is intentionally tiny: it advances an underlying progress bar
    by 1 unit per phase event and records the most recent phase name plus
    every status message it has emitted.  Tests can construct it with a
    ``progress_bar_factory`` that returns a recording fake, or supply
    ``factory=None`` to verify the no-ComfyUI path remains safe.

    Args:
        total_units: Total number of progress units this materialization will
            advance through (manifest size + non-tensor phase units).
        progress_bar_factory: Optional callable taking the integer total and
            returning a ProgressBar-like object.  Pass ``None`` to disable
            the bar entirely (still safe to call advance/status methods).
        artifact_name: Optional short artifact identifier used in status
            messages — must be a basename or relative path; callers must not
            pass absolute filesystem paths.
    """

    def __init__(
        self,
        total_units: int,
        progress_bar_factory: Callable[[int], ProgressBarLike] | None,
        *,
        artifact_name: str | None = None,
    ) -> None:
        if total_units < 1:
            total_units = 1
        self._total = total_units
        self._artifact_name = artifact_name
        self._pbar: ProgressBarLike | None = None
        if progress_bar_factory is not None:
            try:
                self._pbar = progress_bar_factory(total_units)
            except Exception:
                # ProgressBar construction must never break a save.
                logger.debug(
                    "ProgressBar factory raised; continuing without a bar",
                    exc_info=True,
                )
                self._pbar = None

        self._advanced: int = 0
        self._phase: str | None = None
        self._published: bool = False
        self._failed: bool = False
        self._messages: list[tuple[str, str]] = []

    # ------------------------------------------------------------------
    # Inspection — used by tests and logs.
    # ------------------------------------------------------------------

    @property
    def total_units(self) -> int:
        return self._total

    @property
    def advanced(self) -> int:
        return self._advanced

    @property
    def phase(self) -> str | None:
        return self._phase

    @property
    def messages(self) -> tuple[tuple[str, str], ...]:
        return tuple(self._messages)

    @property
    def published(self) -> bool:
        return self._published

    @property
    def failed(self) -> bool:
        return self._failed

    # ------------------------------------------------------------------
    # Phase transitions.
    # ------------------------------------------------------------------

    def prepare(self) -> None:
        self._enter("prepare", "preparing artifact")
        self._tick()

    def tensor_written(self, name: str) -> None:
        # We only advance the bar; the phase label stays at write_tensor.
        if self._phase != "write_tensor":
            self._enter("write_tensor", "writing tensors")
        self._tick()
        # Per-tensor logging is at debug to avoid log floods on big models.
        logger.debug("save progress: wrote tensor %s", name)

    def finalize(self) -> None:
        self._enter("finalize", "finalizing artifact")
        self._tick()
        self._published = True

    def reload(self) -> None:
        self._enter("reload", "reloading artifact")
        self._tick()

    def cache_reuse(self) -> None:
        self._enter("cache_reuse", "reusing cached artifact")
        self._tick()

    def failure(self, reason: str) -> None:
        # Failure must clear the published flag if it was somehow set, and
        # must never advance the bar past the success path.
        self._published = False
        self._failed = True
        # Use the artifact name in the message but never an absolute path —
        # callers are responsible for passing user-facing names only.
        msg = f"materialization failed: {reason}"
        self._enter("failure", msg)

    # ------------------------------------------------------------------
    # Internals.
    # ------------------------------------------------------------------

    def _enter(self, phase: str, message: str) -> None:
        self._phase = phase
        formatted = (
            f"[{self._artifact_name}] {message}"
            if self._artifact_name
            else message
        )
        self._messages.append((phase, formatted))
        logger.info("save progress: %s", formatted)

    def _tick(self) -> None:
        if self._advanced >= self._total:
            return
        self._advanced += 1
        if self._pbar is not None:
            try:
                self._pbar.update(1)
            except Exception:  # pragma: no cover - defensive
                logger.debug("ProgressBar.update raised", exc_info=True)
