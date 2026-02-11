"""WIDEN Entry Node — Boundary from ComfyUI MODEL to WIDEN recipe world."""


class WIDENEntryNode:
    """Snapshots base model, auto-detects architecture, produces RecipeBase."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
            },
        }

    RETURN_TYPES = ("WIDEN",)
    RETURN_NAMES = ("widen",)
    FUNCTION = "entry"
    CATEGORY = "ECAJ/merge"

    def entry(self, model):
        raise NotImplementedError("Entry node not yet implemented")
