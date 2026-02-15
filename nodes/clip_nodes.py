"""WIDEN CLIP Graph Nodes — CLIP-typed variants for graph-level type safety.

These nodes use WIDEN_CLIP instead of WIDEN for ComfyUI type checking,
ensuring CLIP pipelines cannot be accidentally connected to diffusion
pipelines. The execute logic is identical to their WIDEN counterparts.
"""

from ..lib.recipe import BlockConfig, RecipeLoRA, RecipeModel

# Import originals for delegation
from .compose import WIDENComposeNode
from .lora import WIDENLoRANode
from .merge import WIDENMergeNode


class WIDENCLIPLoRANode:
    """CLIP variant of LoRA node. Uses WIDEN_CLIP type for CLIP pipelines."""

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths

        return {
            "required": {
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "prev": ("WIDEN_CLIP",),
                "block_config": ("BLOCK_CONFIG",),
            },
        }

    RETURN_TYPES = ("WIDEN_CLIP",)
    RETURN_NAMES = ("clip_recipe",)
    FUNCTION = "add_lora"
    CATEGORY = "ecaj/merge/clip"

    def add_lora(
        self,
        lora_name: str,
        strength: float,
        prev: RecipeLoRA | None = None,
        block_config: BlockConfig | None = None,
    ) -> tuple:
        """Build RecipeLoRA for CLIP pipeline.

        AC: @clip-graph-nodes ac-1 — accepts WIDEN_CLIP prev, returns WIDEN_CLIP
        AC: @clip-graph-nodes ac-7 — produces same RecipeLoRA as WIDEN variant
        """
        # Delegate to original implementation
        delegate = WIDENLoRANode()
        return delegate.add_lora(lora_name, strength, prev, block_config)


class WIDENCLIPComposeNode:
    """CLIP variant of Compose node. Uses WIDEN_CLIP type for CLIP pipelines."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "branch": ("WIDEN_CLIP",),
            },
            "optional": {
                "compose": ("WIDEN_CLIP",),
            },
        }

    RETURN_TYPES = ("WIDEN_CLIP",)
    RETURN_NAMES = ("clip_recipe",)
    FUNCTION = "compose"
    CATEGORY = "ecaj/merge/clip"

    def compose(self, branch, compose=None) -> tuple:
        """Accumulate CLIP branches for merge.

        AC: @clip-graph-nodes ac-2 — accepts WIDEN_CLIP branch/compose, returns WIDEN_CLIP
        AC: @clip-graph-nodes ac-7 — produces same RecipeCompose as WIDEN variant
        """
        # Delegate to original implementation
        delegate = WIDENComposeNode()
        return delegate.compose(branch, compose)


class WIDENCLIPMergeNode:
    """CLIP variant of Merge node. Uses WIDEN_CLIP type for CLIP pipelines."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base": ("WIDEN_CLIP",),
                "target": ("WIDEN_CLIP",),
                "t_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05}),
            },
            "optional": {
                "backbone": ("WIDEN_CLIP",),
                "block_config": ("BLOCK_CONFIG",),
            },
        }

    RETURN_TYPES = ("WIDEN_CLIP",)
    RETURN_NAMES = ("clip_recipe",)
    FUNCTION = "merge"
    CATEGORY = "ecaj/merge/clip"

    def merge(
        self,
        base,
        target,
        t_factor: float,
        backbone=None,
        block_config: BlockConfig | None = None,
    ) -> tuple:
        """Build RecipeMerge for CLIP pipeline.

        AC: @clip-graph-nodes ac-3 — accepts WIDEN_CLIP base/target, returns WIDEN_CLIP
        AC: @clip-graph-nodes ac-7 — produces same RecipeMerge as WIDEN variant
        """
        # Delegate to original implementation
        delegate = WIDENMergeNode()
        return delegate.merge(base, target, t_factor, backbone, block_config)


class WIDENCLIPModelInputNode:
    """CLIP variant of Model Input node.

    Reads from checkpoints folder since CLIP weights come from full checkpoints.
    Uses WIDEN_CLIP type for CLIP pipelines.
    """

    @classmethod
    def INPUT_TYPES(cls):
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

    RETURN_TYPES = ("WIDEN_CLIP",)
    RETURN_NAMES = ("clip_recipe",)
    FUNCTION = "create_model"
    CATEGORY = "ecaj/merge/clip"

    def create_model(
        self,
        model_name: str,
        strength: float,
        block_config: BlockConfig | None = None,
    ) -> tuple:
        """Build RecipeModel for CLIP pipeline from checkpoints folder.

        AC: @clip-graph-nodes ac-4 — returns WIDEN_CLIP type
        AC: @clip-graph-nodes ac-5 — reads from checkpoints folder
        AC: @clip-graph-nodes ac-7 — produces same RecipeModel as WIDEN variant
        """
        # Same as WIDENModelInputNode but with WIDEN_CLIP return type
        return (RecipeModel(path=model_name, strength=strength, block_config=block_config),)
