# Krea 2 Implementation Reference

This resource captures project-owned planning references for Krea 2 support. It avoids requiring agents to resolve local operator paths. Local model/checkpoint files may be used during optional validation when available, but tasks should accept those paths as operator-provided inputs and must not hard-code machine-specific locations.

## Public source references

### Krea technical report

- Krea 2 Technical Report: <https://www.krea.ai/blog/krea-2-technical-report>
- Imported companion resource for planning: `./resources/references/krea2-technical-report-deep-dive.md`

### ComfyUI Krea 2 implementation

Pinned to the local ComfyUI commit inspected during planning:

- Krea 2 diffusion model: <https://github.com/Comfy-Org/ComfyUI/blob/2a610155821d670a2d8047e654e5fce96b790eb5/comfy/ldm/krea2/model.py>
- Krea 2 text encoder: <https://github.com/Comfy-Org/ComfyUI/blob/2a610155821d670a2d8047e654e5fce96b790eb5/comfy/text_encoders/krea2.py>

### ai-toolkit Krea 2 implementation

Pinned to upstream `origin/main` commit inspected during planning: `724e67d63428a7daddc77355b88d90fe99ea9fd2`.

- Krea 2 extension entry: <https://github.com/ostris/ai-toolkit/blob/724e67d63428a7daddc77355b88d90fe99ea9fd2/extensions_built_in/diffusion_models/krea2/krea2.py>
- MMDiT source: <https://github.com/ostris/ai-toolkit/blob/724e67d63428a7daddc77355b88d90fe99ea9fd2/extensions_built_in/diffusion_models/krea2/src/mmdit.py>
- Pipeline source: <https://github.com/ostris/ai-toolkit/blob/724e67d63428a7daddc77355b88d90fe99ea9fd2/extensions_built_in/diffusion_models/krea2/src/pipeline.py>
- Text encoder source: <https://github.com/ostris/ai-toolkit/blob/724e67d63428a7daddc77355b88d90fe99ea9fd2/extensions_built_in/diffusion_models/krea2/src/text_encoder.py>
- LoKr helper model: <https://github.com/ostris/ai-toolkit/blob/724e67d63428a7daddc77355b88d90fe99ea9fd2/toolkit/models/lokr.py>

### Public LoRA samples

- Comfy-Org Krea 2 LoRAs: <https://huggingface.co/Comfy-Org/Krea-2/tree/9b05e613f06f5ee45d97b362ba3478fec5488b5a/loras>
- Krea RetroAnime LoRA: <https://huggingface.co/krea/Krea-2-LoRA-retroanime/tree/23336fcff3bac43028918a2df795fbb63fdc1ff3>

## Planning facts from header/source inspection

These are planning facts to verify in implementation tests, not a substitute for code-level detection rules:

- Krea 2 should be treated as its own architecture family for WIDEN recipes.
- Krea 2 diffusion checkpoints expose main denoiser blocks and text-fusion substructures. Text-fusion names such as `layerwise_blocks` and `refiner_blocks` are compound names and must not be split into unrelated dotted fragments during key normalization.
- The inspected BF16 base header had 430 tensors and approximately 12.82B elements.
- Main denoiser block keys were observed as `blocks.0` through `blocks.27`.
- Text-fusion keys included `txtfusion.layerwise_blocks.*`, `txtfusion.refiner_blocks.*`, and `txtfusion.projector.*`.
- Attention uses GQA-style dimensions in the inspected model family: model width 6144, 48 query heads, 12 key/value heads, and head dim 128.
- Public Krea 2 LoRA examples appear in at least two naming/export families: Comfy/native-style keys and ai-toolkit-style keys. Implementation should support the public families through explicit compatibility rules and fail before producing a partial merge when a package is outside the supported set.
- Precision/quantization handling should be compatibility-oriented: like-with-like inputs should not be rejected solely because they are quantized, but mixed precision/quantization classes should produce a warning or explicit compatibility decision rather than silently merging incompatible representations.

## Implementation guidance

- Keep public URL references in plan/docs. Do not require task agents to know Jacob's local model directory layout.
- Optional real-file validation should accept explicit operator-provided paths or environment/config inputs.
- The plan should describe durable behavior in specs. Detailed key spelling, alias tables, file names, and fixture paths belong in task descriptions, tests, or this supporting reference.
