"""WIDEN Block Config SDXL Node — Architecture-specific block weight configuration."""

from .block_config import make_block_config_node

_SDXL_BLOCKS = (
    *((f"IN{i:02d}", f"IN{i:02d}") for i in range(9)),
    ("MID", "MID"),
    *((f"OUT{i:02d}", f"OUT{i:02d}") for i in range(9)),
)

# Layer type overrides for cross-cutting control (ac-5)
_LAYER_TYPES = (
    ("attention", "attention"),
    ("feed_forward", "feed_forward"),
    ("norm", "norm"),
)

WIDENBlockConfigSDXLNode = make_block_config_node(
    arch="sdxl",
    block_groups=_SDXL_BLOCKS,
    layer_types=_LAYER_TYPES,
    docstring="""\
Produces BlockConfig for SDXL architecture with individual block sliders.

SDXL block structure:
- input_blocks: IN00-IN08 (9 individual blocks)
- middle_block: MID (single block)
- output_blocks: OUT00-OUT08 (9 individual blocks)

Layer type overrides:
- attention: Controls attention layers (attn1, attn2, to_q, to_k, to_v, proj_in, proj_out)
- feed_forward: Controls feed-forward layers (ff., ff.net)
- norm: Controls normalization layers (norm, ln_)

Each slider is FLOAT range 0.0-2.0 with step 0.05.
ComfyUI allows typing values outside slider range so -1.0 is accessible.
""",
)
