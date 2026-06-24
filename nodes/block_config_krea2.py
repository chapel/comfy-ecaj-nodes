"""WIDEN Block Config Krea 2 Node."""

from .block_config import make_block_config_node

# Krea 2 default ComfyUI/reference configuration uses 28 single-stream MMDiT
# blocks. Text fusion has two layerwise blocks, a projector, and two refiner
# blocks.
_KREA2_MAIN_BLOCKS = tuple((f"B{i:02d}", f"B{i:02d}") for i in range(28))
_KREA2_TEXT_FUSION = (
    ("TF_LW0", "TF_LW0"),
    ("TF_LW1", "TF_LW1"),
    ("TF_PROJECTOR", "TF_PROJECTOR"),
    ("TF_REF0", "TF_REF0"),
    ("TF_REF1", "TF_REF1"),
)
_KREA2_STRUCTURAL = (
    ("PE_EMBEDDER", "PE_EMBEDDER"),
    ("FIRST", "FIRST"),
    ("TMLP", "TMLP"),
    ("TXTMLP", "TXTMLP"),
    ("TPROJ", "TPROJ"),
    ("LAST", "LAST"),
)
_KREA2_BLOCKS = _KREA2_MAIN_BLOCKS + _KREA2_TEXT_FUSION + _KREA2_STRUCTURAL

_LAYER_TYPES = (
    ("attention", "attention"),
    ("feed_forward", "feed_forward"),
    ("norm", "norm"),
    ("embedding_projection", "embedding_projection"),
    ("structural", "structural"),
)

WIDENBlockConfigKrea2Node = make_block_config_node(
    arch="krea2",
    block_groups=_KREA2_BLOCKS,
    layer_types=_LAYER_TYPES,
    docstring="""\
Produces BlockConfig for Krea 2 architecture with main, text-fusion, and structural sliders.

Krea 2 block structure:
- main denoiser: B00-B27
- text fusion: TF_LW0, TF_LW1, TF_PROJECTOR, TF_REF0, TF_REF1
- structural/projection: PE_EMBEDDER, FIRST, TMLP, TXTMLP, TPROJ, LAST

Layer type overrides:
- attention: Controls Krea attention projections, QK norm, and attention gates
- feed_forward: Controls SwiGLU/MLP gate, up, and down projections
- norm: Controls pre/post/final normalization layers
- embedding_projection: Controls input/output embedding and timestep/text projections
- structural: Controls modulation and other structural adapter weights

Keys outside these documented groups remain deterministic and use the neutral default.
Each slider is FLOAT range 0.0-2.0 with step 0.05.
ComfyUI allows typing values outside slider range so -1.0 is accessible.
""",
)
