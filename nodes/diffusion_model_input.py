"""WIDEN Diffusion Model Input Node — Declares a diffusion model spec in the recipe tree."""

from ..lib.recipe import BlockConfig, RecipeModel


class WIDENDiffusionModelInputNode:
    """Produces RecipeModel from diffusion model file picker.

    Pure recipe building — no GPU memory allocation, no file I/O.
    The model path is stored for deferred loading at Exit time.
    Uses the diffusion_models directory (or unet for older ComfyUI).
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Deferred import: folder_paths only exists in ComfyUI runtime
        import folder_paths

        # Try diffusion_models first, fall back to unet for older ComfyUI
        try:
            model_list = folder_paths.get_filename_list("diffusion_models")
        except Exception:
            model_list = folder_paths.get_filename_list("unet")

        return {
            "required": {
                "model_name": (model_list,),
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
        """Build RecipeModel with diffusion model path and optional block config.

        Returns a single-element tuple as required by ComfyUI node protocol.
        """
        return (
            RecipeModel(
                path=model_name,
                strength=strength,
                block_config=block_config,
                source_dir="diffusion_models",
            ),
        )
