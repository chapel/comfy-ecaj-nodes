"""LoRA Loader Interface -- base class for architecture-specific loaders.

Loaders are responsible for:
1. Loading LoRA safetensors files and parsing keys to architecture format
2. Tracking which base model keys are affected, scoped by set_id
3. Producing DeltaSpec objects for batched GPU evaluation, filtered by set_id
4. Releasing resources when done

# AC: @lora-loaders ac-4
Interface provides load(path, strength, set_id), affected_keys property,
affected_keys_for_set(set_id), get_delta_specs(keys, key_indices, set_id),
and cleanup().
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from collections.abc import Set as AbstractSet

from ..executor import DeltaSpec

__all__ = ["LoRALoader"]


class LoRALoader(ABC):
    """Abstract base class for architecture-specific LoRA loaders.

    Each architecture subclass handles key mapping from LoRA format
    to base model format. The loader accumulates LoRAs via load()
    and then produces DeltaSpec objects for batched GPU evaluation.

    LoRA data is segmented by set_id so that get_delta_specs() can
    return only deltas belonging to a specific LoRA set. This prevents
    compose branches with overlapping keys from leaking deltas across
    sets.

    # AC: @lora-loaders ac-4
    Interface contract:
    - load(path, strength, set_id): load a LoRA file at given strength into a set
    - affected_keys: set of base model keys modified by loaded LoRAs
    - affected_keys_for_set(set_id): keys modified by a specific set
    - get_delta_specs(keys, key_indices, set_id): produce DeltaSpec objects for given keys and set
    - cleanup(): release resources (tensors, file handles)
    """

    @abstractmethod
    def load(self, path: str, strength: float = 1.0, set_id: str | None = None) -> None:
        """Load a LoRA file and accumulate its deltas into the given set.

        Args:
            path: Path to the LoRA safetensors file
            strength: Global strength multiplier for this LoRA
            set_id: Identifier for the LoRA set this file belongs to.
                    Required for correct set scoping.

        # AC: @lora-loaders ac-1
        Subclasses implement architecture-specific key mapping.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def affected_keys(self) -> AbstractSet[str]:
        """Return set of base model keys that loaded LoRAs modify (all sets).

        Keys should be in base model format (e.g. 'diffusion_model.X').
        Used by the executor to determine which parameters need LoRA
        deltas applied during batched evaluation. Returns a frozen view
        to prevent external mutation.

        # AC: @lora-loaders ac-4
        """
        raise NotImplementedError

    @abstractmethod
    def affected_keys_for_set(self, set_id: str) -> set[str]:
        """Return set of base model keys modified by a specific LoRA set.

        Args:
            set_id: Identifier for the LoRA set

        Returns:
            Set of base model keys affected by that set

        # AC: @lora-loaders ac-4
        """
        raise NotImplementedError

    @abstractmethod
    def get_delta_specs(
        self,
        keys: Sequence[str],
        key_indices: dict[str, int],
        set_id: str | None = None,
    ) -> list[DeltaSpec]:
        """Produce DeltaSpec objects for the given parameter keys and set.

        Args:
            keys: List of base model parameter keys to get deltas for
            key_indices: Mapping from key -> batch index for DeltaSpec
            set_id: If provided, only return deltas from this LoRA set.
                    If None, returns deltas from all sets (legacy behavior).

        Returns:
            List of DeltaSpec objects for batched GPU evaluation

        # AC: @lora-loaders ac-2
        Produces DeltaSpec objects compatible with batched executor.
        """
        raise NotImplementedError

    @abstractmethod
    def cleanup(self) -> None:
        """Release resources held by the loader.

        Should clear loaded tensors and any cached state.
        Called after batch evaluation is complete.

        # AC: @lora-loaders ac-4
        """
        raise NotImplementedError

    def __enter__(self) -> "LoRALoader":
        """Support context manager usage for automatic cleanup."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Cleanup on context exit."""
        self.cleanup()
