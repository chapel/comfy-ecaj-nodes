"""WIDEN LoRA Node — Declares a LoRA spec in the recipe tree."""

from lib.recipe import RecipeLoRA


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
            },
        }

    RETURN_TYPES = ("WIDEN",)
    RETURN_NAMES = ("widen",)
    FUNCTION = "add_lora"
    CATEGORY = "ecaj/merge"

    def add_lora(self, lora_name: str, strength: float, prev: RecipeLoRA | None = None) -> tuple:
        """Build RecipeLoRA, chaining with prev if provided.

        Returns a single-element tuple as required by ComfyUI node protocol.
        """
        new_lora = {"path": lora_name, "strength": strength}

        if prev is not None and isinstance(prev, RecipeLoRA):
            # Chain: append new LoRA to existing set
            loras = prev.loras + (new_lora,)
        else:
            # First in chain: single-element tuple
            loras = (new_lora,)

        return (RecipeLoRA(loras=loras),)
