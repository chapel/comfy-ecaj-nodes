"""WIDEN Block Config Z-Image Node — Architecture-specific block weight configuration."""

from .block_config import make_block_config_node

_ZIMAGE_BLOCKS = (
    ("L00_04", "L00-04"),
    ("L05_09", "L05-09"),
    ("L10_14", "L10-14"),
    ("L15_19", "L15-19"),
    ("L20_24", "L20-24"),
    ("L25_29", "L25-29"),
    ("noise_refiner", "noise_refiner"),
    ("context_refiner", "context_refiner"),
)

WIDENBlockConfigZImageNode = make_block_config_node(
    arch="zimage",
    block_groups=_ZIMAGE_BLOCKS,
    docstring="""\
Produces BlockConfig for Z-Image/S3-DiT architecture with grouped block sliders.

Z-Image block structure:
- layers: L00-04, L05-09, L10-14, L15-19, L20-24, L25-29 (6 groups of 5 layers)
- noise_refiner: single refiner block
- context_refiner: single refiner block

Each slider is FLOAT range 0.0-2.0 with step 0.05.
ComfyUI allows typing values outside slider range so -1.0 is accessible.
""",
)
