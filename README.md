# comfy-ecaj-nodes

> **Work in progress.** Implemented capabilities are not a guarantee of compatibility with every checkpoint or LoRA variant.

A ComfyUI custom node pack for **WIDEN-based model and LoRA merging**. WIDEN uses parameter-level importance rather than simple linear interpolation. Recipe-building nodes defer merge computation until an Exit node executes the recipe.

## Install in ComfyUI

Place this repository checkout at `ComfyUI/custom_nodes/comfy-ecaj-nodes`, then restart ComfyUI. Nodes appear under `ecaj/merge` and `ecaj/merge/clip`.

The intended plugin distribution is a **repository/Comfy registry checkout**, not a standalone Python wheel. Python **3.10 or newer** is declared. ComfyUI supplies the runtime environment, including PyTorch, NumPy and safetensors; `requirements.txt` and runtime package dependencies intentionally do not install or replace the host's PyTorch build. Do not install the development extra into an existing ComfyUI environment.

## Build a recipe

1. Connect a ComfyUI loader's `MODEL` to **WIDEN Entry**.
2. Connect Entry to **WIDEN Merge**'s `base`. Use **WIDEN LoRA**, **WIDEN Checkpoint Input**, or **WIDEN Diffusion Model Input** as its `target`.
3. Use **WIDEN Compose** to accumulate simultaneous targets, or feed a Merge output into the next Merge's `base` for a chain. Architecture-specific **WIDEN Block Config** nodes provide per-block controls.
4. Connect the final `WIDEN` recipe to **WIDEN Exit**, then use its `MODEL` downstream.

Entry keeps a reference to the supplied model; it does not copy its weights. Recipe construction avoids merge/GPU computation but is not a guarantee that no model or tensor references are held.

SDXL text-encoder merging uses a separate **WIDEN CLIP Entry → CLIP Merge → CLIP Exit** graph with `WIDEN_CLIP` sockets. CLIP LoRA, Compose, Model Input and SDXL CLIP Block Config nodes supply its targets/controls. Do not mix diffusion and CLIP recipe sockets. CLIP Exit returns `CLIP` and does not offer saved-model output.

### Exit output modes

| Setting | Behavior |
|---|---|
| `save_model=false` (default) | Return a cloned model patcher with merged weights installed as set patches; no full saved-model artifact. |
| `save_model=true` | Materialize a complete saved artifact and load the returned `MODEL` through the corresponding ComfyUI loader. This requires disk space for the full artifact and a temporary file. |
| `model_name` | Saved filename, not a path; `.safetensors` is appended if absent. |
| `save_workflow=true` (default) | Embed available workflow metadata when saving. |
| `enable_cache=true` (default) | Allow cache reuse; saved-model reuse also checks artifact metadata/kind. This is distinct from ComfyUI's graph cache. |

For checkpoint-style output, connect the base checkpoint's **CLIP and VAE** to Entry's optional inputs. Checkpoint-intent recipes without required companion components fail validation. Standalone diffusion output uses the diffusion-model folder (`diffusion_models`, with legacy `unet` fallback); checkpoint output uses `checkpoints`. Full artifacts are published with an atomic replace only after writing and validation. Existing non-ECAJ files are not intended overwrite targets.

## Implemented architecture surfaces

| Domain | Implemented loaders/detection and controls |
|---|---|
| Diffusion | SDXL, Z-Image, Flux, Qwen, Krea 2 |
| Text encoders | SDXL CLIP (`clip_l` and `clip_g`) |

These entries describe code paths, **not equal levels of real-model validation**. The normal suite uses CPU tensors and bounded safetensors fixtures. Its registration smoke loads all 19 stable node IDs with a small host API stub; it does not prove end-to-end ComfyUI execution, GPU memory bounds, image quality, or support for arbitrary quantized/FP8 model layouts. Real-model/GPU validation is separate, optional evidence.

## Development (separate CPU environment)

From a checkout, with `uv` installed:

```bash
uv sync --extra dev --python 3.12
uv run --no-sync pytest
uv run --no-sync python -m compileall lib nodes tests
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
```

The development PyTorch source is explicitly the CPU index. CI runs Python 3.10 and 3.12, verifies a CPU-only PyTorch build, and runs pytest plus lint/format gates. No running ComfyUI or GPU is required by the normal suite. `uv.lock` is ignored and is not a committed reproducibility contract; dependency resolution remains within the declared ranges.
