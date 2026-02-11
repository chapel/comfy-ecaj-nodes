"""WIDEN Merge Node — Defines a merge step in the recipe tree."""

from lib.recipe import BlockConfig, RecipeBase, RecipeCompose, RecipeLoRA, RecipeMerge


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
                "block_config": ("BLOCK_CONFIG",),
            },
        }

    RETURN_TYPES = ("WIDEN",)
    RETURN_NAMES = ("widen",)
    FUNCTION = "merge"
    CATEGORY = "ecaj/merge"

    def merge(
        self,
        base,
        target,
        t_factor,
        backbone=None,
        block_config: BlockConfig | None = None,
    ):
        """Build a RecipeMerge from base and target recipes.

        Args:
            base: The base WIDEN recipe (must be RecipeBase or RecipeMerge).
            target: The target WIDEN recipe to merge (RecipeLoRA, RecipeCompose, or RecipeMerge).
            t_factor: Merge strength factor. -1.0 means passthrough (no WIDEN).
            backbone: Optional explicit backbone reference for WIDEN importance.
            block_config: Optional block configuration for per-block weight control.

        Returns:
            Tuple containing RecipeMerge with the merge configuration.

        Raises:
            ValueError: If base is RecipeLoRA or RecipeCompose (must be base model or merge chain).

        AC: @per-block-control ac-1 — when block_config is None, behavior is identical
        AC: @per-block-control ac-3 — BLOCK_CONFIG fans out correctly to each consumer
        """
        # AC-5: base must be RecipeBase or RecipeMerge, not RecipeLoRA/RecipeCompose
        if isinstance(base, (RecipeLoRA, RecipeCompose)):
            raise ValueError(
                f"base must be RecipeBase or RecipeMerge (a base model or merge chain), "
                f"got {type(base).__name__}. Use Entry node or Merge output as base."
            )

        # Validate base is a known recipe type
        if not isinstance(base, (RecipeBase, RecipeMerge)):
            raise TypeError(
                f"base must be RecipeBase or RecipeMerge, got {type(base).__name__}"
            )

        # Validate target is a valid merge target
        if not isinstance(target, (RecipeLoRA, RecipeCompose, RecipeMerge)):
            raise TypeError(
                f"target must be RecipeLoRA, RecipeCompose, or RecipeMerge, "
                f"got {type(target).__name__}"
            )

        # AC-1, AC-2, AC-3, AC-6: Build RecipeMerge with all fields
        # backbone is None when not connected (AC-2), stored when connected (AC-3)
        # t_factor of -1.0 is preserved as-is (AC-6) — Exit node interprets it
        return (
            RecipeMerge(
                base=base,
                target=target,
                backbone=backbone,
                t_factor=t_factor,
                block_config=block_config,
            ),
        )
