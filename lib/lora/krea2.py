"""Krea 2 architecture LoRA loader entry point.

Krea 2 LoRA package mapping is intentionally implemented in the dependent
compatibility task. This loader exists so Krea 2 recipes route to an explicit
Krea-specific branch and fail clearly if a LoRA package is applied before that
mapping exists.
"""

from collections.abc import Sequence

from ..executor import DeltaSpec
from .base import LoRALoader

__all__ = ["Krea2Loader"]


class Krea2Loader(LoRALoader):
    """Krea 2 loader placeholder with explicit unsupported-package diagnostics."""

    _MESSAGE = (
        "Krea 2 LoRA package compatibility is not implemented yet. "
        "This architecture is recognized as 'krea2', but Krea 2 LoRA tensor "
        "normalization and package compatibility belong to "
        "@task-krea2-lora-package-compatibility."
    )

    def load(self, path: str, strength: float = 1.0, set_id: str | None = None) -> None:
        raise NotImplementedError(self._MESSAGE)

    @property
    def affected_keys(self) -> frozenset[str]:
        return frozenset()

    def affected_keys_for_set(self, set_id: str) -> set[str]:
        return set()

    def get_delta_specs(
        self,
        keys: Sequence[str],
        key_indices: dict[str, int],
        set_id: str | None = None,
    ) -> list[DeltaSpec]:
        return []

    @property
    def loaded_bytes(self) -> int:
        return 0

    def cleanup(self) -> None:
        return None
