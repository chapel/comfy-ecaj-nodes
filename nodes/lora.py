"""WIDEN LoRA Node — Declares a LoRA spec in the recipe tree."""

try:
    from ..lib.recipe import BlockConfig, RecipeLoRA
except ImportError:
    from lib.recipe import BlockConfig, RecipeLoRA


class WIDENLoRANode:
    """Produces RecipeLoRA. Chains via optional prev input to form sets."""

    @classmethod
    def INPUT_TYPES(cls):
        # Deferred import: folder_paths only exists in ComfyUI runtime
        import folder_paths

        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "prev": ("WIDEN",),
                "block_config": ("BLOCK_CONFIG",),
            },
        }

    RETURN_TYPES = ("WIDEN",)
    RETURN_NAMES = ("widen",)
    FUNCTION = "add_lora"
    CATEGORY = "ecaj/merge"

    def add_lora(
        self,
        lora_name: str,
        strength: float,
        prev: RecipeLoRA | None = None,
        block_config: BlockConfig | None = None,
    ) -> tuple:
        """Build RecipeLoRA, chaining with prev if provided.

        Returns a single-element tuple as required by ComfyUI node protocol.

        AC: @per-block-control ac-1 — when block_config is None, behavior is identical
        AC: @per-block-control ac-3 — BLOCK_CONFIG fans out correctly to each consumer
        """
        if prev is not None and not isinstance(prev, RecipeLoRA):
            raise TypeError(f"Expected RecipeLoRA for prev input, got {type(prev).__name__}")

        new_lora = {"path": lora_name, "strength": strength}

        if prev is not None:
            # Chain: append new LoRA to existing set
            loras = prev.loras + (new_lora,)
            # Preserve block_config from prev if new one not provided
            if block_config is None:
                block_config = prev.block_config
            elif prev.block_config is not None:
                # Both prev and new block_config provided — architectures must match
                if block_config.arch != prev.block_config.arch:
                    raise ValueError(
                        f"Block config architecture mismatch: prev has '{prev.block_config.arch}' "
                        f"but new block_config has '{block_config.arch}'. "
                        f"All LoRAs in a chain must use the same architecture."
                    )
        else:
            # First in chain: single-element tuple
            loras = (new_lora,)

        return (RecipeLoRA(loras=loras, block_config=block_config),)
