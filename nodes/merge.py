"""WIDEN Merge Node — Defines a merge step in the recipe tree."""


class WIDENMergeNode:
    """Builds RecipeMerge. Compose target -> merge_weights, single -> filter_delta."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base": ("WIDEN",),
                "target": ("WIDEN",),
                "t_factor": ("FLOAT", {"default": 1.0, "min": -1.0, "max": 5.0, "step": 0.05}),
            },
            "optional": {
                "backbone": ("WIDEN",),
            },
        }

    RETURN_TYPES = ("WIDEN",)
    RETURN_NAMES = ("widen",)
    FUNCTION = "merge"
    CATEGORY = "ECAJ/merge"

    def merge(self, base, target, t_factor, backbone=None):
        raise NotImplementedError("Merge node not yet implemented")
