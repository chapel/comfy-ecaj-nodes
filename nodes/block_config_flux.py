"""WIDEN Block Config Flux Node — Architecture-specific block weight configuration."""

from .block_config import make_block_config_node

# Flux Klein block structure:
# - Double blocks: DB00-DB07 (8 blocks for 9B, 5 for 4B - unused sliders default to 1.0)
# - Single blocks: SB00-SB23 (24 blocks for 9B, 20 for 4B - unused sliders default to 1.0)
# Total: 32 block sliders for Klein 9B max
_FLUX_DOUBLE_BLOCKS = tuple((f"DB{i:02d}", f"DB{i:02d}") for i in range(8))
_FLUX_SINGLE_BLOCKS = tuple((f"SB{i:02d}", f"SB{i:02d}") for i in range(24))
_FLUX_BLOCKS = _FLUX_DOUBLE_BLOCKS + _FLUX_SINGLE_BLOCKS

# Layer type overrides for cross-cutting control (ac-5)
_LAYER_TYPES = (
    ("attention", "attention"),
    ("feed_forward", "feed_forward"),
    ("norm", "norm"),
)

WIDENBlockConfigFluxNode = make_block_config_node(
    arch="flux",
    block_groups=_FLUX_BLOCKS,
    layer_types=_LAYER_TYPES,
    docstring="""\
Produces BlockConfig for Flux Klein architecture with individual block sliders.

Flux Klein block structure:
- double_blocks: DB00-DB07 (8 blocks for 9B, 5 for 4B)
- single_blocks: SB00-SB23 (24 blocks for 9B, 20 for 4B)

Klein 4B models have 5 double + 20 single blocks; unused sliders (DB05-DB07, SB20-SB23)
remain at default 1.0 and are effectively no-ops for that model variant.

Layer type overrides:
- attention: Controls attention layers (img_attn, txt_attn, qkv, proj, query_norm, key_norm)
- feed_forward: Controls feed-forward layers (img_mlp, txt_mlp, linear2)
- norm: Controls normalization layers (img_mod, txt_mod, modulation)

Each slider is FLOAT range 0.0-2.0 with step 0.05.
ComfyUI allows typing values outside slider range so -1.0 is accessible.
""",
)
