"""WIDEN Block Config SDXL CLIP Node — Per-block weight control for SDXL text encoders."""

from .block_config import make_block_config_node

# AC: @sdxl-clip-block-config ac-1
# CLIP-L: 12 transformer blocks (CL00-CL11)
# CLIP-G: 32 transformer blocks (CG00-CG31)
# Structural keys: CL_EMBED, CL_FINAL, CG_EMBED, CG_FINAL, CG_PROJ
_SDXL_CLIP_BLOCKS = (
    # CLIP-L blocks (12 transformer blocks)
    *((f"CL{i:02d}", f"CL{i:02d}") for i in range(12)),
    # CLIP-L structural keys
    ("CL_EMBED", "CL_EMBED"),
    ("CL_FINAL", "CL_FINAL"),
    # CLIP-G blocks (32 transformer blocks)
    *((f"CG{i:02d}", f"CG{i:02d}") for i in range(32)),
    # CLIP-G structural keys
    ("CG_EMBED", "CG_EMBED"),
    ("CG_FINAL", "CG_FINAL"),
    ("CG_PROJ", "CG_PROJ"),
)

# Layer type overrides for cross-cutting control (ac-10)
_LAYER_TYPES = (
    ("attention", "attention"),
    ("feed_forward", "feed_forward"),
    ("norm", "norm"),
)

WIDENBlockConfigSDXLCLIPNode = make_block_config_node(
    arch="sdxl",
    block_groups=_SDXL_CLIP_BLOCKS,
    layer_types=_LAYER_TYPES,
    docstring="""\
Produces BlockConfig for SDXL CLIP text encoders with individual block sliders.

SDXL CLIP block structure:
- CLIP-L: CL00-CL11 (12 transformer blocks)
- CLIP-L structural: CL_EMBED (embeddings), CL_FINAL (final layer norm)
- CLIP-G: CG00-CG31 (32 transformer blocks)
- CLIP-G structural: CG_EMBED (embeddings), CG_FINAL (final layer norm), CG_PROJ (text projection)

Layer type overrides:
- attention: Controls attention layers (self_attn.q_proj, k_proj, v_proj, out_proj)
- feed_forward: Controls MLP layers (mlp.fc1, mlp.fc2)
- norm: Controls normalization layers (layer_norm1, layer_norm2, final_layer_norm)

Each slider is FLOAT range 0.0-2.0 with step 0.05.
ComfyUI allows typing values outside slider range so -1.0 is accessible.

AC: @sdxl-clip-block-config ac-1, ac-4
""",
)
