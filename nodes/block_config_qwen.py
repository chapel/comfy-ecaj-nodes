"""WIDEN Block Config Qwen Node — Architecture-specific block weight configuration."""

from .block_config import make_block_config_node

_QWEN_BLOCKS = (
    *((f"TB{i:02d}", f"TB{i:02d}") for i in range(60)),
    ("TIME_EMBED", "TIME_EMBED"),
    ("FINAL_NORM", "FINAL_NORM"),
)

# Layer type overrides for cross-cutting control (ac-5)
_LAYER_TYPES = (
    ("attention", "attention"),
    ("feed_forward", "feed_forward"),
    ("norm", "norm"),
)

WIDENBlockConfigQwenNode = make_block_config_node(
    arch="qwen",
    block_groups=_QWEN_BLOCKS,
    layer_types=_LAYER_TYPES,
    docstring="""\
Produces BlockConfig for Qwen architecture with individual block sliders.

Qwen block structure:
- transformer_blocks: TB00-TB59 (60 individual blocks)
- non-block: TIME_EMBED, FINAL_NORM (structural keys)

Layer type overrides:
- attention: Controls attention layers (to_q, to_k, to_v, to_out, attn)
- feed_forward: Controls feed-forward layers (ff., mlp)
- norm: Controls normalization layers (norm, ln, img_mod, txt_mod)

Each slider is FLOAT range 0.0-2.0 with step 0.05.
ComfyUI allows typing values outside slider range so -1.0 is accessible.
""",
)
