"""WIDEN Block Config SDXL Node — Architecture-specific block weight configuration."""

from .block_config import make_block_config_node

_SDXL_BLOCKS = (
    *((f"IN{i:02d}", f"IN{i:02d}") for i in range(9)),
    ("MID", "MID"),
    *((f"OUT{i:02d}", f"OUT{i:02d}") for i in range(9)),
)

WIDENBlockConfigSDXLNode = make_block_config_node(
    arch="sdxl",
    block_groups=_SDXL_BLOCKS,
    docstring="""\
Produces BlockConfig for SDXL architecture with individual block sliders.

SDXL block structure:
- input_blocks: IN00-IN08 (9 individual blocks)
- middle_block: MID (single block)
- output_blocks: OUT00-OUT08 (9 individual blocks)

Each slider is FLOAT range 0.0-2.0 with step 0.05.
ComfyUI allows typing values outside slider range so -1.0 is accessible.
""",
)
