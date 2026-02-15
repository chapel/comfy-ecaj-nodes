"""WIDEN Block Config Z-Image Node — Architecture-specific block weight configuration."""

from .block_config import make_block_config_node

_ZIMAGE_BLOCKS = (
    *((f"L{i:02d}", f"L{i:02d}") for i in range(30)),
    ("NOISE_REF0", "NOISE_REF0"),
    ("NOISE_REF1", "NOISE_REF1"),
    ("CTX_REF0", "CTX_REF0"),
    ("CTX_REF1", "CTX_REF1"),
    ("PATCH_EMBED", "PATCH_EMBED"),
    ("FINAL_NORM", "FINAL_NORM"),
)

# Layer type overrides for cross-cutting control (ac-5)
_LAYER_TYPES = (
    ("attention", "attention"),
    ("feed_forward", "feed_forward"),
    ("norm", "norm"),
)

WIDENBlockConfigZImageNode = make_block_config_node(
    arch="zimage",
    block_groups=_ZIMAGE_BLOCKS,
    layer_types=_LAYER_TYPES,
    docstring="""\
Produces BlockConfig for Z-Image/S3-DiT architecture with individual block sliders.

Z-Image block structure:
- layers: L00-L29 (30 individual blocks)
- noise_refiner: NOISE_REF0, NOISE_REF1 (2 blocks)
- context_refiner: CTX_REF0, CTX_REF1 (2 blocks)
- non-block: PATCH_EMBED, FINAL_NORM (structural keys)

Layer type overrides:
- attention: Controls attention layers (attn.qkv, attn.out, q_norm, k_norm)
- feed_forward: Controls feed-forward layers (feed_forward, mlp, w1, w2, w3, fc1, fc2)
- norm: Controls normalization layers (norm, ln, rms)

Each slider is FLOAT range 0.0-2.0 with step 0.05.
ComfyUI allows typing values outside slider range so -1.0 is accessible.
""",
)
