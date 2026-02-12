"""WIDEN Block Config SDXL Node — Architecture-specific block weight configuration."""

from .block_config import make_block_config_node

_SDXL_BLOCKS = (
    ("IN00_02", "IN00-02"),
    ("IN03_05", "IN03-05"),
    ("IN06_08", "IN06-08"),
    ("MID", "MID"),
    ("OUT00_02", "OUT00-02"),
    ("OUT03_05", "OUT03-05"),
    ("OUT06_08", "OUT06-08"),
)

WIDENBlockConfigSDXLNode = make_block_config_node(
    arch="sdxl",
    block_groups=_SDXL_BLOCKS,
    docstring="""\
Produces BlockConfig for SDXL architecture with grouped block sliders.

SDXL block structure:
- input_blocks: IN00-02, IN03-05, IN06-08 (3 groups of 3 blocks each)
- middle_block: MID (single block)
- output_blocks: OUT00-02, OUT03-05, OUT06-08 (3 groups of 3 blocks each)

Each slider is FLOAT range 0.0-2.0 with step 0.05.
ComfyUI allows typing values outside slider range so -1.0 is accessible.
""",
)
