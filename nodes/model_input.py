"""WIDEN Model Input Node — Declares a full model spec in the recipe tree."""

from ..lib.recipe import BlockConfig, RecipeModel


class WIDENModelInputNode:
    """Produces RecipeModel from checkpoint file picker.

    Pure recipe building — no GPU memory allocation, no file I/O.
    The checkpoint path is stored for deferred loading at Exit time.
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Deferred import: folder_paths only exists in ComfyUI runtime
        import folder_paths

        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("checkpoints"),),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01},
                ),
            },
            "optional": {
                "block_config": ("BLOCK_CONFIG",),
            },
        }

    RETURN_TYPES = ("WIDEN",)
    RETURN_NAMES = ("widen",)
    FUNCTION = "create_model"
    CATEGORY = "ecaj/merge"

    def create_model(
        self,
        model_name: str,
        strength: float,
        block_config: BlockConfig | None = None,
    ) -> tuple:
        """Build RecipeModel with checkpoint path and optional block config.

        Returns a single-element tuple as required by ComfyUI node protocol.
        """
        return (RecipeModel(path=model_name, strength=strength, block_config=block_config),)
