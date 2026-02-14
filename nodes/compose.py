"""WIDEN Compose Node — Accumulates branches for simultaneous merge."""

from ..lib.recipe import RecipeBase, RecipeCompose, RecipeLoRA, RecipeMerge, RecipeModel


class WIDENComposeNode:
    """Appends branch to compose list. Pure data manipulation, no GPU work."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "branch": ("WIDEN",),
            },
            "optional": {
                "compose": ("WIDEN",),
            },
        }

    RETURN_TYPES = ("WIDEN",)
    RETURN_NAMES = ("widen",)
    FUNCTION = "compose"
    CATEGORY = "ecaj/merge"

    def compose(self, branch, compose=None):
        """Accumulate branches for WIDEN merge.

        Args:
            branch: A WIDEN input (RecipeLoRA, RecipeCompose, or RecipeMerge).
                    Cannot be a raw RecipeBase.
            compose: Optional previous RecipeCompose to chain with.

        Returns:
            Tuple containing RecipeCompose with accumulated branches.

        Raises:
            ValueError: If branch is RecipeBase (needs LoRAs applied first).
            TypeError: If compose is provided but not a RecipeCompose.
        """
        # AC-4: Reject raw RecipeBase — must have LoRAs applied
        if isinstance(branch, RecipeBase):
            raise ValueError(
                "Cannot compose a raw base model. "
                "Apply LoRAs first using a LoRA node, or use this as a Merge base input."
            )

        # Validate branch is a valid recipe type
        if not isinstance(branch, (RecipeLoRA, RecipeModel, RecipeCompose, RecipeMerge)):
            raise TypeError(
                f"branch must be RecipeLoRA, RecipeModel, RecipeCompose, or RecipeMerge, "
                f"got {type(branch).__name__}"
            )

        # Validate compose input if provided
        if compose is not None and not isinstance(compose, RecipeCompose):
            raise TypeError(
                f"compose must be RecipeCompose or None, got {type(compose).__name__}"
            )

        # AC-1: No compose → single-element branches
        # AC-2: With compose → append to existing branches
        if compose is None:
            result = RecipeCompose(branches=(branch,))
        else:
            result = compose.with_branch(branch)

        return (result,)
