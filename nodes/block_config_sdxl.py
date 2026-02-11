"""WIDEN Block Config SDXL Node — Architecture-specific block weight configuration."""

try:
    from ..lib.recipe import BlockConfig
except ImportError:
    from lib.recipe import BlockConfig


class WIDENBlockConfigSDXLNode:
    """Produces BlockConfig for SDXL architecture with grouped block sliders.

    SDXL block structure:
    - input_blocks: IN00-02, IN03-05, IN06-08 (3 groups of 3 blocks each)
    - middle_block: MID (single block)
    - output_blocks: OUT00-02, OUT03-05, OUT06-08 (3 groups of 3 blocks each)

    Each slider is FLOAT range 0.0-2.0 with step 0.05.
    ComfyUI allows typing values outside slider range so -1.0 is accessible.
    """

    @classmethod
    def INPUT_TYPES(cls):
        slider_config = {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}
        return {
            "required": {
                "IN00_02": ("FLOAT", slider_config),
                "IN03_05": ("FLOAT", slider_config),
                "IN06_08": ("FLOAT", slider_config),
                "MID": ("FLOAT", slider_config),
                "OUT00_02": ("FLOAT", slider_config),
                "OUT03_05": ("FLOAT", slider_config),
                "OUT06_08": ("FLOAT", slider_config),
            },
        }

    RETURN_TYPES = ("BLOCK_CONFIG",)
    RETURN_NAMES = ("block_config",)
    FUNCTION = "create_config"
    CATEGORY = "ecaj/merge"

    def create_config(
        self,
        IN00_02: float,
        IN03_05: float,
        IN06_08: float,
        MID: float,
        OUT00_02: float,
        OUT03_05: float,
        OUT06_08: float,
    ) -> tuple[BlockConfig]:
        """Create BlockConfig with SDXL block overrides.

        AC: @per-block-control ac-2 — block group sliders with float range 0.0 to 2.0
        """
        block_overrides = (
            ("IN00-02", IN00_02),
            ("IN03-05", IN03_05),
            ("IN06-08", IN06_08),
            ("MID", MID),
            ("OUT00-02", OUT00_02),
            ("OUT03-05", OUT03_05),
            ("OUT06-08", OUT06_08),
        )

        return (BlockConfig(arch="sdxl", block_overrides=block_overrides),)
