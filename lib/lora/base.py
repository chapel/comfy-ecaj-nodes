"""LoRA Loader Interface — base class for architecture-specific loaders.

Loaders are responsible for:
1. Loading LoRA safetensors files and parsing keys to architecture format
2. Tracking which base model keys are affected
3. Producing DeltaSpec objects for batched GPU evaluation
4. Releasing resources when done

# AC: @lora-loaders ac-4
Interface provides load(path, strength), affected_keys property,
get_delta_specs(keys) returning DeltaSpecs, and cleanup().
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from lib.executor import DeltaSpec

__all__ = ["LoRALoader"]


class LoRALoader(ABC):
    """Abstract base class for architecture-specific LoRA loaders.

    Each architecture subclass handles key mapping from LoRA format
    to base model format. The loader accumulates LoRAs via load()
    and then produces DeltaSpec objects for batched GPU evaluation.

    # AC: @lora-loaders ac-4
    Interface contract:
    - load(path, strength): load a LoRA file at given strength
    - affected_keys: set of base model keys modified by loaded LoRAs
    - get_delta_specs(keys): produce DeltaSpec objects for given keys
    - cleanup(): release resources (tensors, file handles)
    """

    @abstractmethod
    def load(self, path: str, strength: float = 1.0) -> None:
        """Load a LoRA file and accumulate its deltas.

        Args:
            path: Path to the LoRA safetensors file
            strength: Global strength multiplier for this LoRA

        # AC: @lora-loaders ac-1
        Subclasses implement architecture-specific key mapping.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def affected_keys(self) -> set[str]:
        """Return set of base model keys that loaded LoRAs modify.

        Keys should be in base model format (e.g. 'diffusion_model.X').
        Used by the executor to determine which parameters need LoRA
        deltas applied during batched evaluation.

        # AC: @lora-loaders ac-4
        """
        raise NotImplementedError

    @abstractmethod
    def get_delta_specs(
        self,
        keys: Sequence[str],
        key_indices: dict[str, int],
    ) -> list[DeltaSpec]:
        """Produce DeltaSpec objects for the given parameter keys.

        Args:
            keys: List of base model parameter keys to get deltas for
            key_indices: Mapping from key -> batch index for DeltaSpec

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
