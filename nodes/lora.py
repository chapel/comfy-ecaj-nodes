"""WIDEN LoRA Node — Declares a LoRA spec in the recipe tree."""


class WIDENLoRANode:
    """Produces RecipeLoRA. Chains via optional prev input to form sets."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_name": ("STRING", {"default": ""}),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "prev": ("WIDEN",),
            },
        }

    RETURN_TYPES = ("WIDEN",)
    RETURN_NAMES = ("widen",)
    FUNCTION = "add_lora"
    CATEGORY = "ECAJ/merge"

    def add_lora(self, lora_name, strength, prev=None):
        raise NotImplementedError("LoRA node not yet implemented")
