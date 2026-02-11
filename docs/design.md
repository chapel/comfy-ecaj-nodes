# ComfyUI WIDEN Merge Node Pack — Design Document

## 1. Motivation

The merge-router project implements WIDEN-based model merging with a hierarchical
config system (LoRAs → Sets → Chain → Compose). The JSON config + CLI approach works
well for batch merging, but iteration is slow: change a parameter, re-run the whole
merge, load the result in ComfyUI, generate test images, evaluate, repeat.

A ComfyUI node pack would allow:
- **Live iteration**: change a t_factor slider, only the affected branch re-merges
- **Visual composition**: the node graph IS the merge config — no JSON indirection
- **Caching**: ComfyUI's DAG executor skips unchanged branches automatically
- **Immediate testing**: merged model flows directly into sampling nodes

Current merge performance (~28s for a full 7-LoRA Z-Image compose on RTX 4090)
is fast enough for interactive use in ComfyUI.

## 2. ComfyUI Internals Reference

### 2.1 The MODEL Type: ModelPatcher

ComfyUI's `MODEL` type is an instance of `ModelPatcher` (defined in
`comfy/model_patcher.py`, aliased as `CoreModelPatcher`).

**Construction** — `CheckpointLoaderSimple` calls `comfy.sd.load_checkpoint_guess_config()`:
1. Loads safetensors → raw state dict via `comfy.utils.load_torch_file()`
2. Auto-detects architecture via `model_detection.model_config_from_unet()`
3. Instantiates inner model (`BaseModel` subclass) with `model_config.get_model()`
4. Wraps in `ModelPatcher(model, load_device, offload_device)`

**Key attributes**:
```python
class ModelPatcher:
    self.model           # BaseModel (has self.model.diffusion_model = the UNet/DiT)
    self.patches         # Dict[str, List[patch_tuple]] — pending weight patches
    self.backup          # Dict[str, namedtuple] — original weights before patching
    self.object_patches  # Non-weight patches (module replacements)
    self.model_options   # {"transformer_options": {...}}
    self.load_device     # GPU
    self.offload_device  # CPU
    self.patches_uuid    # uuid4, regenerated when patches change
```

**Key methods**:
- `clone()` — shallow copy: shares underlying nn.Module, independent patch dicts.
  Sets `n.parent = self`. Cheap — no tensor copies.
- `model_state_dict(filter_prefix=None)` — returns raw state dict (unpatched weights,
  or backup weights if patching is active)
- `get_key_patches(filter_prefix=None)` — returns `Dict[str, List]` where each value
  is `[(original_weight, convert_func), patch_tuple_1, ...]`. This is what merge
  nodes use to extract model2's weights.
- `add_patches(patches, strength_patch=1.0, strength_model=1.0)` — registers patches.
  Only adds for keys that exist in model's state dict. Appends to existing patch
  lists. Updates `patches_uuid`.
- `patch_weight_to_device(key, device_to)` — materializes patches for one key.
  Backs up original weight, casts to compute dtype, calls `calculate_weight()`,
  stochastic rounds back to storage dtype, writes to model.
- `unpatch_model()` — restores all original weights from backup.

### 2.2 Patch System

Every patch is a 5-tuple: `(strength_patch, patch_data, strength_model, offset, function)`

- `strength_patch`: multiplier for the incoming patch
- `patch_data`: tensor, list (nested), or `WeightAdapterBase` (LoRA/LoHa/LoKr)
- `strength_model`: multiplier for the original model weight
- `offset`: optional `(dimension, start, length)` for partial application
- `function`: optional transform on patch before merging

**Patch types** (determined by `patch_data` format):
- `"diff"` — additive: `weight += strength * patch` (used by model merge)
- `"set"` — replacement: `weight.copy_(patch)`
- `"model_as_lora"` — computes diff from original at apply time
- `WeightAdapterBase` — delegates to adapter's `calculate_weight()` (LoRA etc.)

**Patches are lazy** — `add_patches()` only stores metadata. Actual computation
happens in `patch_weight_to_device()` during model load, or via `LowVramPatch`
hooks during forward passes in low-VRAM mode.

**`calculate_weight()`** (in `comfy/lora.py`) iterates all patches for a key:
```python
for p in patches:
    strength, v, strength_model, offset, function = p
    if strength_model != 1.0:
        weight *= strength_model  # scale original
    # then add patch contribution based on type
```

### 2.3 How Built-In Merge Nodes Work

**ModelMergeSimple**:
```python
def merge(self, model1, model2, ratio):
    m = model1.clone()
    kp = model2.get_key_patches("diffusion_model.")
    for k in kp:
        m.add_patches({k: kp[k]}, 1.0 - ratio, ratio)
    return (m,)
```
Flow: clone model1 → extract model2's weights+patches → register as patches
with blend ratio. No weight computation happens here — it's deferred.

**ModelMergeBlocks**: Same pattern but uses longest-prefix-match on key names
to assign per-block ratios from kwargs.

### 2.4 How LoRA Loading Works

LoRA uses the **same patch system**. `LoraLoader` calls
`comfy.sd.load_lora_for_models()` which:
1. Builds key map (LoRA keys → model keys, handles diffusers/A1111/kohya naming)
2. Converts LoRA format via `comfy.lora_convert.convert_lora()`
3. Parses into `WeightAdapterBase` instances via `comfy.lora.load_lora()`
4. `model.clone()` + `add_patches(loaded, strength)`

LoRA weights are never applied directly — they're stored as adapter patches.

### 2.5 Memory Management

`comfy/model_management.py` manages GPU/CPU placement:
- `load_models_gpu([model_patcher, ...])` — ensures models are on GPU,
  evicts others as needed
- VRAM states: DISABLED, NO_VRAM, LOW_VRAM, NORMAL_VRAM, HIGH_VRAM, SHARED
- Priority eviction: by offloaded amount, ref count, total size, age
- `maximum_vram_for_weights(device)` = 88% of total VRAM minus inference reserve
- `minimum_inference_memory()` ~800MB + reserved (600MB Win, 400MB Linux)

### 2.6 Caching

Defined in `comfy_execution/caching.py`. Cache keys are computed by traversing
the node graph:
- Widget inputs (INT, FLOAT, STRING) are hashed directly
- Link inputs (MODEL, LATENT, etc.) are resolved as `("ANCESTOR", index, socket)`
- The full ancestor chain is part of the cache signature

This means: if you change a t_factor slider, only that node and its downstream
nodes re-execute. Upstream branches that feed into an unchanged input are cached.

`IS_CHANGED` classmethod can force re-execution (return `float("NaN")`) or
conditional re-execution (return a hash).

### 2.7 Inner Model Hierarchy

```
ModelPatcher
  └── .model: BaseModel (e.g. SDXL, Flux, or custom)
        └── .diffusion_model: nn.Module (UNet, DiT, etc.)
```

State dict keys are prefixed `diffusion_model.` in the ModelPatcher namespace.

**SDXL UNet keys**: `diffusion_model.input_blocks.N.M.weight`, etc.
**Flux DiT keys**: `diffusion_model.double_blocks.N.*.weight`, etc.
**Z-Image**: Not natively supported by ComfyUI's auto-detection. Would need
custom model loading or wrapping.

### 2.8 Saving Models

`ModelPatcher.state_dict_for_saving()` builds an export state dict with all
patches baked in. Uses `LazyCastingParam` for on-demand patch materialization
during save.

## 3. Existing ComfyUI Merge Packs

### ComfyUI-DareMerge (DARE-TIES)
- Repo: github.com/54rt1n/ComfyUI-DareMerge
- Per-block merging, MBW support, attention-level merging, magnitude masking
- DARE-TIES algorithm: compute deltas → stochastic sparsify → resolve ties → rescale

### comfy-mecha (Recipe-Based)
- Repo: github.com/ljleb/comfy-mecha (nodes), github.com/ljleb/sd-mecha (engine)
- Recipes compose as a DAG, executed key-by-key at merge time
- Most memory-efficient: never holds more than a few tensors simultaneously

### LoRA-Merger-ComfyUI (Mergekit Integration)
- Repo: github.com/larsupb/LoRA-Merger-ComfyUI
- 10+ algorithms via mergekit: TIES, DARE, DELLA, SLERP, SCE, etc.
- Pluggable method node system

**No existing WIDEN-based merge nodes in any public pack.**

## 4. Architecture Constraints and Design Considerations

### 4.1 WIDEN Requires Eager Computation

WIDEN's importance ranking operates across all parameters simultaneously:
- `filter_delta`: ranks per-column importance across the full model, builds
  continuous mask, applies to delta
- `merge_weights`: ranks importance per model, uses calibrated softmax across
  models, routes parameters to most-relevant model

This cross-key analysis cannot work with ComfyUI's lazy per-key patch system.
We must compute the merge eagerly (when the node executes) and store results
as `"set"` patches (weight replacement).

This is acceptable because:
- Our batched GPU pipeline is fast (~28s full compose)
- ComfyUI's caching means the merge only re-runs when inputs change
- DareMerge and other advanced packs also do eager computation

### 4.2 Z-Image Is Not Native to ComfyUI

ComfyUI auto-detects SDXL, SD1.5, Flux, etc. Z-Image (S3-DiT, 6B params)
is not a standard architecture. Options:
- Custom loader node that wraps Z-Image in a ModelPatcher manually
- Or: operate on raw state dicts and bypass ModelPatcher for Z-Image
- ComfyUI forks exist that support Z-Image (e.g., via custom model configs)

SDXL and Flux would work natively with the standard CheckpointLoaderSimple.

### 4.3 QKV Fusing Is Z-Image Specific

Z-Image base model uses fused `attention.qkv.weight` (11520×3840 = 3×3840),
but LoRAs use separate `to_q`/`to_k`/`to_v` keys. Our QKV fusing logic in
`zimage_lora_merge.py::_parse_lora_key()` handles this mapping. This would
need to be ported but kept as Z-Image-specific code.

### 4.4 LoRA Loading: ComfyUI's vs Ours

ComfyUI already has LoRA loading with key mapping for standard architectures
(SDXL, Flux). We have our own loaders with:
- Z-Image specific key parsing and QKV fusing
- DeltaSpec-based batched GPU apply (for performance)
- LoKr support
- get_delta_specs() for raw factor extraction

For SDXL: we could potentially use ComfyUI's LoRA loading and extract the
applied deltas. For Z-Image: we need our own loader.

### 4.5 Memory Implications

WIDEN eager merge means we materialize the full merged state dict in memory
during node execution. For Z-Image (11.46GB bf16), this requires:
- Base model weights (already loaded by ComfyUI)
- LoRA deltas (small, <100MB typically)
- Working memory for WIDEN computation (~2-4GB for batched ops)
- Merged result stored as patches on the clone

The merged result as "set" patches would hold references to new tensors
(not sharing with the original model), so total memory ≈ 2× model size
while the merge node's output is alive. ComfyUI's memory manager should
handle eviction, but this is worth monitoring.

### 4.6 Batched Pipeline vs Per-Key

Our current batched pipeline groups same-shape parameters and processes them
with GPU bmm. This is what gives us the ~28s performance. In ComfyUI, we'd
want to preserve this — the node's `merge()` function would run the full
batched pipeline internally, not go key-by-key.

## 5. What to Port from merge-router

Source project: `~/Projects/merge-router/`

### Core Algorithm (required)
- `src/core/widen.py` — WIDEN class, WIDENConfig, filter_delta, merge_weights,
  batched variants, disentangle, ranking, calibration
- `src/core/divergence.py` — divergence metrics
- `src/core/ranking.py` — ranking mechanisms
- `src/core/sparsity.py` — sparsemax, entmax (if used)
- `src/core/numerical_config.py` — NumericalConfig

### LoRA Loading (architecture-dependent)
- Z-Image: `scripts/zimage_lora_merge.py` key parsing, `_parse_lora_key()`, QKV fusing
- SDXL: could reuse ComfyUI's native LoRA loading
- Qwen: `scripts/qwen_merge.py` loader (if supporting Qwen)

### Batched GPU Pipeline (performance)
- `scripts/lora_chain_merge.py`: `_apply_lora_set_batched_gpu()`, DeltaSpec,
  `OpSignature` grouping, `compute_batch_size()`, OOM backoff
- `scripts/lora_chain_merge.py`: `evaluate_node_batched()` tree walker
  (adapted for ComfyUI node graph instead of config tree)

### Not Porting
- `scripts/merge.py` — JSON config runner (replaced by ComfyUI node graph)
- `scripts/lora_chain_merge.py` — CLI, config parsing, legacy engine, resolve_branch
  (replaced by ComfyUI execution)
- Training infrastructure (stubs, trainer, checkpointing, etc.)
- Nu scripts

## 6. Node Design

### 6.1 Core Insight: Two Fundamental Operations

The config hierarchy (LoRAs → Sets → Chain → Compose) collapses in node-graph
land to just two operations:

- **Serial (chain)**: node output feeds into next node's input. ComfyUI already
  does this — just wire sequentially. Each step's result becomes the next base.
- **Simultaneous (compose)**: N inputs merged at once against a common backbone
  via WIDEN. This is the node we build.

Sets and compose unify: a "set" is "apply LoRA(s) to backbone, producing a
virtual model." A compose merges N virtual models. Whether inputs came from
LoRA application or upstream merges doesn't matter — the flowing type is
always "model state."

The `filter_delta` case (single-LoRA WIDEN filtering) is just compose with
N=1. No separate node needed — the Merge node calls `filter_delta` internally
when it receives a single branch instead of a compose group.

### 6.2 Deferred Execution Architecture

**The node graph computes the config. The Exit node runs it.**

Compose and Merge nodes do zero GPU work. They build a recipe — a tree
structure equivalent to the JSON config. The Exit node receives the complete
recipe and runs the full batched GPU pipeline in one shot.

This preserves the same execution strategy as the CLI tool:
1. Scan all base model keys
2. Group by `OpSignature` (same shape, same affecting sets)
3. Batched GPU LoRA apply via `torch.bmm`
4. Batched WIDEN operations across the batch dimension
5. Write results as `"set"` patches on a `ModelPatcher.clone()`

**Why deferred**: This is a performance optimization. WIDEN itself operates
per-parameter (not globally across all keys), but the batched pipeline
groups same-shape parameters and processes them via GPU bmm. Per-node eager
execution would lose this batching opportunity, forcing per-key CPU
round-trips. Deferring to the Exit node means we have the complete recipe
and can batch optimally — same performance as the CLI tool.

**Performance**: The merge computation is ~10s (LoRA apply ~3.8s + WIDEN
~6.6s). In ComfyUI we skip the ~21s safetensors save, so the user-facing
latency is ~10s per re-merge. ComfyUI caches the Exit node's output, so
unchanged workflows return instantly. Any slider change re-runs the full
pipeline — fast enough for iteration.

**Known limitation**: Because all compute is deferred to Exit, changing
*any* upstream parameter re-runs the *entire* pipeline. There is no
partial re-computation. For v2, a recipe-level sub-tree cache inside
Exit could hash each sub-tree and only recompute changed branches.

### 6.3 Unified Type System

One custom ComfyUI type flows through the entire WIDEN subgraph: **`WIDEN`**.

It represents a recipe node — a lightweight data structure describing what
to compute. The Python objects inside vary (base reference, LoRA spec,
compose branch list, merge step) but the ComfyUI wire type is uniform.
This gives one consistent wire color and avoids naming confusion (a LoRA
spec is not a "model", a compose list is not a "model").

`WIDEN` objects hold no GPU tensors. They are pure recipe descriptions.
All tensor work happens in the Exit node.

Entry and Exit are the boundaries between ComfyUI's `MODEL` type and
the `WIDEN` recipe world.

### 6.4 Node Types

Five nodes:

#### Entry Node
- **Purpose**: Boundary from ComfyUI world to WIDEN world. Snapshots the
  base model for use as backbone and chain starting point.
- **Inputs**: ComfyUI `MODEL`
- **Outputs**: `WIDEN`
- **Behavior**: Wraps the ModelPatcher reference in a `RecipeBase` node.
  Auto-detects architecture (SDXL, Z-Image, Flux, Qwen) by inspecting
  the model's state dict key patterns. Stores the detected arch so
  downstream LoRA nodes know which key mapping to use.
  No tensor work — just stores the reference and arch tag.

#### LoRA Node
- **Purpose**: Declare a LoRA to be applied as part of the recipe.
  Our own loader — does not take a MODEL input like ComfyUI's built-in
  LoRA node. Instead it produces a recipe spec that the Exit node
  evaluates later.
- **Inputs**:
  - `lora_name`: file selector via `folder_paths.get_filename_list("loras")`
    — standard ComfyUI dropdown showing all LoRAs in configured dirs.
    Works with ComfyUI-Manager's LoRA browser.
  - `strength`: FLOAT slider (global strength, applied uniformly)
  - `prev` (optional): `WIDEN` from a previous LoRA node
  - `block_config` (optional): `BLOCK_CONFIG` for per-block LoRA
    strength scaling. Overrides global `strength` per block group.
    See §6.10.
- **Outputs**: `WIDEN`
- **Behavior**: Creates a `RecipeLoRA` spec (file path + strength +
  optional block config). When `prev` is connected, chains with the
  previous LoRA(s) — this builds what the config system calls a "set"
  (multiple LoRAs applied together to the same base, forming one
  virtual model).
- **Why our own loader**: ComfyUI's LoRA nodes eagerly clone + patch a
  MODEL. We need the LoRA as a deferred spec so the Exit node can apply
  it via the batched GPU bmm pipeline. Using our own loader also lets
  us handle Z-Image's QKV fusing and other architecture-specific key
  mapping that ComfyUI's loader doesn't know about.
- **Architecture awareness**: The LoRA node doesn't need an arch selector.
  The architecture tag flows from the Entry node through the recipe tree.
  At Exit time, the executor reads the arch tag to select the correct
  LoRA key mapping and loader.

#### Compose Node
- **Purpose**: Accumulate branches for simultaneous WIDEN merging.
  Pure recipe building — no computation.
- **Inputs**:
  - `branch`: `WIDEN` — the thing being added (LoRA spec, merge result, etc.)
  - `compose` (optional): `WIDEN` — from a previous Compose node (accumulation chain)
- **Outputs**: `WIDEN`
- **Behavior**: Appends the branch to the compose list. Instant — just
  data structure manipulation. Chain multiple Compose nodes to accumulate
  any number of branches.
- **Two named inputs of the same type**: `branch` is what's being added,
  `compose` is the chain from previous Compose nodes. The input names
  (and UI labels) distinguish them. This avoids needing a separate
  `WIDEN_COMPOSE` type.

#### Merge Node
- **Purpose**: The workhorse. Defines a WIDEN merge step in the recipe.
  Still no computation — deferred to Exit.
- **Inputs**:
  - `base`: `WIDEN` (from Entry or a previous Merge)
  - `target`: `WIDEN` (what to merge — a compose group, a LoRA spec, or a merge result)
  - `backbone` (optional): `WIDEN` — explicit backbone override for WIDEN
    importance analysis. Defaults to `base` if not connected.
  - `t_factor`: FLOAT slider
- **Outputs**: `WIDEN`
- **Behavior based on what `target` contains internally**:
  - Compose group (from Compose node) → recipe for `merge_weights`
  - Single branch (LoRA spec or merge result) → recipe for `filter_delta`
- **The t_factor lives here** — it's a property of the merge operation,
  not of individual branches.
- **Backbone override**: When `backbone` is not connected, WIDEN uses the
  `base` input as the backbone (normal chain behavior — each step's
  importance is measured against the result of the previous step).
  When `backbone` IS connected, WIDEN uses that explicit model as the
  importance reference instead. This enables workflows with separate
  model+LoRA pathways composed against independent backbones:
  ```
  [Model A + LoRAs] ──→ target ──→ [Merge, backbone=Model B] → ...
  [Model B + LoRAs] ──→ base ──╯         ↑ backbone
  [Entry: Model B] ────────────────────────╯
  ```
- **Chain output**: the Merge output feeds into the next Merge's `base`
  input for sequential chaining, or into Exit to leave the WIDEN world.

#### Exit Node
- **Purpose**: Boundary back to ComfyUI world. **The only node that
  computes.** Receives the complete recipe tree, executes the full batched
  GPU pipeline, returns a ComfyUI MODEL.
- **Inputs**: `WIDEN`
- **Outputs**: ComfyUI `MODEL`
- **Behavior**:
  1. Validates the recipe tree structure (type-check each node)
  2. Walks the recipe tree to collect all `RecipeLoRA` groups, assigns
     synthetic set IDs, loads LoRA files, computes affected key sets
  3. Groups parameters by `OpSignature` for batching
  4. Evaluates the recipe tree per-batch (see §6.8 for algorithm)
  5. Clones the original ModelPatcher, adds merged weights as `"set"`
     patches with **tensors on CPU** (see §6.9 for memory strategy)
  6. Returns the patched MODEL for downstream sampling
- **Caching**: ComfyUI's DAG cache handles this automatically — if no
  upstream node changed, the Exit returns its cached MODEL instantly.
- **`IS_CHANGED`**: Returns a hash of upstream LoRA file mtimes and
  sizes, so that re-training a LoRA on disk triggers re-execution even
  if the node graph wiring hasn't changed.
- **Progress**: Reports progress via `comfy.utils.ProgressBar` during
  the batched pipeline (LoRA apply phase, WIDEN phase).
- **Downstream compatibility**: The `"set"` patches work correctly with
  downstream ComfyUI LoRA nodes. ComfyUI's `calculate_weight()` processes
  patches in list order — `"set"` replaces the weight first, then any
  subsequent LoRA patches apply additively on top. Verified against
  `comfy/lora.py`'s `calculate_weight()` implementation.

### 6.5 Example Workflows

**Simple single-LoRA filter** (equivalent to zimage_lora_merge with 1 LoRA):
```
[CheckpointLoader] → [Entry] ── base ──→ [Merge t=1.0] → [Exit] → [Sampler]
                                              ↑ target
                                         [LoRA: nicegirls @0.8]
```

**LoRA set** (multiple LoRAs forming one virtual model, then filtered):
```
[Entry] ── base ──→ [Merge t=1.0] → [Exit]
                         ↑ target
        [LoRA: nsfw_v1 @0.5] → [LoRA: nsfw_v2 @0.5]
                                  (chained = one "set")
```

**Multi-branch compose** (two branches merged simultaneously):
```
[Entry] ── base ──→ [Merge t=1.0] → [Exit]
                         ↑ target
                    [Compose] ←── compose ── [Compose]
                         ↑ branch                 ↑ branch
                    [LoRA: painting @1]      [LoRA: realism @0.8]
```

**The full hyphoria config** (compose + sequential chain):
```
[Entry] ── base ──→ [Merge t=1.0] ── base ──→ [Merge t=1.0] ── base ──→ [Merge t=0.5] → [Exit]
                         ↑ target                   ↑ target                  ↑ target
                    [Compose]                  [LoRA: nipples @1]        [LoRA: Mystic @1]
                    [+ Compose]
                         ↑ branches
    branch A: [LoRA: nicegirls @0.8] → [LoRA: nsfw1 @0.5] → [LoRA: nsfw2 @0.5]
    branch B: [LoRA: painting @1] → [LoRA: mecha @1]
```
Reading left to right:
1. Merge t=1.0: compose branch A (realism+nsfw LoRAs) with branch B (style LoRAs)
2. Merge t=1.0: filter_delta nipples LoRA on top of compose result
3. Merge t=0.5: filter_delta Mystic LoRA last with light filtering

**Sequential chain** (LoRA A then LoRA B, each WIDEN-filtered independently):
```
[Entry] → [Merge t=1.0] → [Merge t=0.5] → [Exit]
               ↑ target        ↑ target
          [LoRA: A @1]    [LoRA: B @1]
```

### 6.6 Recipe Data Structures

The `WIDEN` ComfyUI type wraps Python dataclasses that form a recipe tree.
These are defined in `lib/recipe.py`:

All recipe dataclasses are **frozen** (immutable) to prevent aliasing bugs
with ComfyUI's caching and graph fan-out. Fields use tuples, not lists.

```python
@dataclass(frozen=True)
class RecipeBase:
    """Entry node output — wraps the ModelPatcher reference."""
    model_patcher: object  # ComfyUI ModelPatcher (holds state dict ref)
    arch: str              # auto-detected: "sdxl", "zimage", "flux", "qwen"

@dataclass(frozen=True)
class RecipeLoRA:
    """LoRA node output — one or more LoRAs to apply as a group (a "set")."""
    loras: tuple  # ({"path": str, "strength": float}, ...)

@dataclass(frozen=True)
class RecipeCompose:
    """Compose node output — accumulated branch list."""
    branches: tuple  # (WIDEN, WIDEN, ...) — each is a recipe node

@dataclass(frozen=True)
class RecipeMerge:
    """Merge node output — a merge step in the recipe."""
    base: object      # WIDEN (RecipeBase or RecipeMerge)
    target: object    # WIDEN (RecipeLoRA, RecipeCompose, or RecipeMerge)
    backbone: object  # WIDEN or None — explicit backbone override for WIDEN
    t_factor: float
```

Compose node creates new tuples on each append (persistent tree semantics):
```python
# In Compose node:
new_branches = prev_compose.branches + (branch,) if prev_compose else (branch,)
return RecipeCompose(branches=new_branches)
```

The Exit node receives a `RecipeMerge` (or chain of them) and walks the
tree to build the equivalent of the JSON config, then runs the batched
pipeline.

### 6.7 Design Decisions (Resolved)

1. **Architecture handling**: Auto-detect at Entry time by inspecting the
   ModelPatcher's state dict key patterns. The arch tag is stored in the
   `RecipeBase` and flows through the recipe tree. The Exit node uses it
   to select the correct LoRA loader/key mapping. No manual dropdown needed.

2. **Backbone for compose**: Explicit optional `backbone` input on the Merge
   node. Defaults to the `base` input when not connected (normal chain
   behavior). When connected, overrides the WIDEN importance reference.
   This enables multi-model compose workflows where separate model+LoRA
   pathways are composed against independent backbones.

3. **LoRA file selection**: Uses `folder_paths.get_filename_list("loras")`
   for the standard ComfyUI dropdown. Works with ComfyUI-Manager's LoRA
   browser. No arbitrary path input needed initially.

4. **Save-to-disk**: The Exit node produces a ComfyUI `MODEL` that can
   be saved with ComfyUI's built-in save checkpoint nodes. No custom save
   node needed.

### 6.8 Exit Node Evaluation Algorithm

The Exit node is the only node that performs GPU computation. It receives
a recipe tree (chain of `RecipeMerge` nodes) and executes the full batched
pipeline. The algorithm mirrors the CLI tool's `evaluate_node_batched()`
but is driven by recipe dataclasses instead of JSON config dicts.

**Phase 1: Recipe Analysis**

Walk the recipe tree to extract metadata needed for batching:

1. **Collect RecipeBase** — follow `base` links to the root `RecipeBase`.
   Extract the `model_patcher` reference and `arch` tag.
2. **Assign synthetic set IDs** — find every unique `RecipeLoRA` node in
   the tree. Each gets a synthetic set ID (e.g. `"set_0"`, `"set_1"`).
   Two LoRA nodes chained via `prev` share the same set ID (they form
   a single "set" — multiple LoRAs applied to the same base).
3. **Load LoRA files** — for each set, instantiate the architecture-
   appropriate loader (selected by `arch` tag) and load all LoRA files
   in that set. This is the first I/O — LoRA safetensors read from disk.
4. **Build affected-key map** — `set_id → {param keys}` mapping. Each
   loader reports which base model keys its LoRAs modify. Keys not
   affected by any set are skipped entirely (no work needed).

**Phase 2: OpSignature Grouping**

Partition all affected keys into groups that can be batched together:

```python
@dataclass(frozen=True)
class OpSignature:
    affecting_sets: frozenset  # Which sets modify this param
    shape: tuple               # Parameter shape (e.g. (3840, 3840))
    ndim: int                  # 1D bias, 2D linear, 4D conv, etc.
```

Keys with identical `OpSignature` follow the same recipe tree path and
have the same tensor shape — they can be stacked along a batch dimension
and processed in a single vectorized pass. This is the key insight that
makes the pipeline fast.

**Phase 3: Batched Evaluation**

For each `OpSignature` group:

1. **Compute batch size** — query free VRAM, estimate peak memory as
   `B × numel(shape) × dtype_bytes × (3 + 3 × n_models)`, solve for
   max B. The `3 + 3n` multiplier accounts for: base batch, per-set
   LoRA-applied copies, and WIDEN intermediates (disentangle, rank,
   calibrate).

2. **Chunk the group** into batches of size B.

3. **For each chunk** of B keys:

   a. **Stack**: gather B tensors from the base model's state dict,
      stack into `base_batch: [B, *shape]`, transfer to GPU.

   b. **Evaluate**: walk the recipe tree recursively:

   - **RecipeMerge with RecipeCompose target** (multi-branch):
     - Recurse into each branch against `backbone` (the original base
       or explicit backbone override)
     - Each branch returns `[B, *shape]` on GPU
     - Single branch → call `widen.filter_delta_batched(branch, backbone)`
     - Multiple branches → call `widen.merge_weights_batched(branches, backbone)`

   - **RecipeMerge with RecipeLoRA target** (single set, filter_delta):
     - Apply LoRA set to base via batched GPU bmm
       (`_apply_lora_set_batched_gpu`)
     - Call `widen.filter_delta_batched(lora_applied, backbone)`

   - **Chain** (RecipeMerge whose base is another RecipeMerge):
     - Evaluate inner merge first → its result becomes the `base` for
       the outer merge. Pure recursion.

   c. **GPU LoRA apply** (`_apply_lora_set_batched_gpu`): for each set
      contributing to this batch:
      - Loader provides `DeltaSpec` objects (raw LoRA factors: up, down,
        scale, kind)
      - Partition specs by `(kind, rank)` for uniform stacking
      - Standard LoRA: stack up/down matrices → single `torch.bmm` call
        → `[B, out, in]` deltas
      - QKV: apply to correct third of fused weight via offset indexing
      - LoKr: per-key `torch.kron` on GPU (small params, not worth bmm)
      - Scatter deltas back by `key_index` → result `[B, *shape]`

   d. **Unstack**: transfer merged `[B, *shape]` to CPU, scatter back
      into individual tensors in the output dict.

   e. **Cleanup**: delete GPU tensors, allow GC.

4. **OOM backoff**: if `torch.cuda.OutOfMemoryError` during any chunk,
   catch it, clear CUDA cache, and fall back to per-key (B=1) evaluation
   for that specific chunk only. Other chunks continue at full batch size.

**Phase 4: Patch Installation**

Clone the original `ModelPatcher` and install merged weights:

```python
merged_model = model_patcher.clone()
patches = {}
for key, merged_tensor in merged_state.items():
    # "set" patch: (strength=1.0, tensor, strength_model=1.0, None, None)
    patches["diffusion_model." + key] = ("set", merged_tensor)
merged_model.add_patches(patches, strength_patch=1.0)
return (merged_model,)
```

The `"set"` patch type means "replace weight with this tensor" — when
ComfyUI later calls `patch_weight_to_device(key)`, it copies our merged
tensor directly into the model weight, discarding the original.

### 6.9 Memory Strategy

**During merge execution** (Exit node's `merge()` call):

The batched pipeline's GPU memory is transient — tensors are allocated
per-chunk and freed immediately after results transfer to CPU. Peak GPU
usage per chunk is approximately:

```
B × numel(shape) × dtype_bytes × (1 + n_sets + ~3 WIDEN intermediates)
```

For Z-Image (shape 3840×3840, bf16, 2 sets, B=50): ~5.3 GB peak.
The `compute_batch_size()` function targets 70% of free VRAM to leave
headroom for PyTorch allocator fragmentation.

Base model weights are read from the `ModelPatcher`'s state dict via
`model_state_dict()` — this returns the original (unpatched) weights.
They stay on CPU; only the current chunk's slice is transferred to GPU.

**After merge execution** (steady state):

The Exit node returns a `ModelPatcher.clone()` with `"set"` patches.
Memory implications:

- **Original model**: The inner `nn.Module` weights remain in the
  original `ModelPatcher`. ComfyUI's memory manager controls whether
  these are on GPU or CPU.
- **Merged patches**: Stored as CPU tensors in the clone's `patches`
  dict. These are the full merged weights — one tensor per modified
  parameter.
- **Total footprint**: ~2× model size while both the original and
  merged `ModelPatcher` are alive. For Z-Image (11.5 GB bf16), that's
  ~23 GB CPU RAM.

**Why CPU patches work**: ComfyUI's `patch_weight_to_device(key)`
handles the GPU transfer lazily at sampling time. When the model is
loaded for inference via `load_models_gpu([merged_model])`:

1. ComfyUI evicts other models from GPU if needed
2. For each key, `patch_weight_to_device(key, device_to=GPU)`:
   - Backs up the original weight
   - Sees our `"set"` patch → copies the merged tensor to GPU
   - Writes it into the model's parameter
3. After sampling, `unpatch_model()` restores originals from backup

This is the standard ComfyUI pattern — DareMerge and model merge nodes
also produce `"set"` patches on CPU.

**Reducing memory pressure**: If the original `ModelPatcher` (from the
Entry node) is not used downstream by any other node, ComfyUI's cache
can evict it, freeing ~1× model size of CPU RAM. In practice, the most
common workflow is `Entry → ... → Exit → Sampler`, so only the merged
clone needs to stay alive during sampling.

**Low-VRAM mode**: ComfyUI's `LowVramPatch` system applies patches
on-the-fly during forward passes instead of pre-patching the entire
model. Our `"set"` patches are compatible with this — they're just
tensor replacements. No special handling needed.

**Downstream LoRA stacking**: The `"set"` patches are applied first
(weight replacement), then any downstream ComfyUI LoRA patches apply
additively on top. This is correct — `calculate_weight()` processes
patches in list order, and `clone()` + `add_patches()` appends to the
list. A user can wire `Exit → LoraLoader → Sampler` and get
WIDEN-merged base + additional LoRA, with the LoRA riding on top of
the merged weights.

### 6.10 Per-Block Control (Open Design)

The current Merge node has a single `t_factor` that applies uniformly to
all parameters. Users need finer control: different t_factor per block
group, or disabling WIDEN entirely for certain blocks (passing the LoRA
delta through unfiltered).

#### Why Per-Block Matters

Different model regions serve different purposes:
- **SDXL**: Input blocks handle encoding, middle block is the bottleneck,
  output blocks handle decoding. Style LoRAs often concentrate in attention
  blocks 4-8, while structural changes affect input/output blocks.
- **Z-Image**: 30 transformer layers with different roles. Layers 0-5
  handle low-level features, 25-29 handle high-level composition. The
  noise_refiner and context_refiner blocks are distillation-critical.
- **Qwen**: 60 transformer blocks with gradual feature hierarchy.

A user might want t_factor=1.0 for attention-heavy middle blocks (strong
WIDEN filtering) but t_factor=-1 (passthrough) for early input blocks.

#### Architecture Block Structures

Each architecture has distinct block organization:

**SDXL UNet**:
```
input_blocks[0..8]    — 9 blocks (3 downsampling stages of 3)
middle_block[0..2]    — 3 blocks (ResNet + attention + ResNet)
output_blocks[0..8]   — 9 blocks (3 upsampling stages of 3)
+ time_embed, label_emb, out.* (embeddings/projections)
```

**Z-Image S3-DiT**:
```
layers[0..29]                — 30 transformer blocks
noise_refiner.blocks[0..1]   — 2 noise refiner blocks
context_refiner.blocks[0..1] — 2 context refiner blocks
+ embeddings, norms, projections
```

**Qwen Image Transformer**:
```
transformer_blocks[0..59]    — 60 transformer blocks
+ img_in, txt_in, pos_embed, time_text_embed, norm_out, proj_out
```

Within any block, parameters are classified by **layer type**:
`attention`, `feed_forward`, `norm`, `embedder`, `refiner`, `final`.
This cross-cuts the block structure — a user might want to filter
attention weights but leave FFN weights unfiltered within the same block.

#### Design Options

Two approaches, not yet decided:

**Option A: Inline Filter Node** (`WIDEN` in, `WIDEN` out)

A filter node sits inline in the WIDEN chain and annotates the recipe
with block-level overrides. Architecture-specific: one variant per
model type (SDXL Block Filter, Z-Image Block Filter, etc.).

```
[Entry] → [Z-Image Block Filter] → base → [Merge t=1.0] → [Exit]
              ↑ config                         ↑ target
   layers 0-14: t=0.5                    [LoRA @1.0]
   layers 15-29: t=1.2
   noise_refiner: disabled
```

The filter node wraps its input recipe in a `RecipeBlockFilter` that
the Exit node reads during evaluation. "Disabled" means that block's
parameters skip WIDEN and get raw LoRA delta (or passthrough base).

**Pros**:
- Simple wiring — just drop it in the chain anywhere
- Works as a general-purpose annotation on any recipe subtree
- Could apply to Merge target side too (filter how a LoRA set is
  applied before it reaches the merge)

**Cons**:
- Ambiguous scope — does it affect just the next downstream Merge,
  or all Merges? Need clear scoping rules.
- If placed on a branch input to Compose, does it affect only that
  branch's contribution or the whole compose operation?
- "Invisible" effect — the filter modifies behavior of a node it's
  not directly wired to (the Merge/Exit downstream).

**Option B: Explicit Config Type** (separate `BLOCK_CONFIG` type)

Architecture-specific config nodes produce a new `BLOCK_CONFIG` type
that feeds into Merge (and optionally LoRA/Compose) as an explicit
optional input.

```
[Entry] → base → [Merge t=1.0] → [Exit]
                      ↑ target
                 [LoRA @1.0]
                      ↑ block_config
              [Z-Image Block Config]
                 layers 0-14: t=0.5
                 layers 15-29: t=1.2
                 noise_refiner: disabled
```

The Merge node stores the `BLOCK_CONFIG` in its `RecipeMerge`. The
Exit node reads it per-merge-step and applies per-block t_factor
overrides during evaluation. Each architecture exposes its own node:

- **SDXL Block Config**: input blocks (0-8), middle, output blocks
  (0-8). Groups of 3 (matching MBW convention) or individual.
- **Z-Image Block Config**: layers (0-29), noise_refiner, context_refiner.
  Groups of 5 (matching existing blocks_per_stage).
- **Qwen Block Config**: transformer blocks (0-59). Groups of 10.

Each block group gets a FLOAT slider (t_factor override) plus an
enable/disable toggle. Unconnected = use the Merge node's global
t_factor for all blocks (current behavior, backwards compatible).

**Pros**:
- Explicit — you see exactly which Merge step the config applies to
- Clear scope — each Merge has its own optional block config
- No ambiguity about what affects what
- Could also wire into LoRA nodes for per-block LoRA strength
  (separate from WIDEN filtering)

**Cons**:
- Extra wire per Merge node when using block control
- Second custom type (`BLOCK_CONFIG`) adds complexity
- Architecture-specific nodes are a separate node per arch (3+ nodes)
  — but this is unavoidable either way since block structures differ

#### Layer-Type Filtering (Both Options)

Orthogonal to block-level control, users may want layer-type filtering:
"apply WIDEN to attention weights but pass through FFN weights." This
could be a toggle set on the block config (per-block × per-layer-type
matrix) or a simpler global toggle on the Merge node.

The existing `_get_layer_type()` classification in the CLI tool already
handles this — each parameter key is classified as `attention`,
`feed_forward`, `norm`, `embedder`, `refiner`, or `final`. This
classification could feed into OpSignature grouping so that attention
and FFN params within the same block get different t_factor values.

#### Per-Block LoRA Strength

Separate from WIDEN's t_factor, users need per-block **LoRA strength**
control — scaling how much of the LoRA delta is applied before WIDEN
even sees it. This is the ComfyUI equivalent of "LoRA Block Weight"
(MBW) from A1111/Forge.

Example: a style LoRA that's great for composition (output blocks) but
causes artifacts in fine details (input blocks). The user wants
strength=1.0 for output blocks but strength=0.3 for input blocks.

This is a different control from t_factor:
- **t_factor** (Merge node): Controls WIDEN's importance threshold —
  how aggressively WIDEN filters or routes parameters.
- **LoRA block strength** (LoRA node): Controls how much of the LoRA
  delta is applied per block — a simple scalar multiplier on the delta
  before it enters the WIDEN pipeline.

The same `BLOCK_CONFIG` type works for both. The LoRA node gains an
optional `block_config` input alongside the Merge node:

```
[LoRA: style @1.0] ──→ target ──→ [Merge t=1.0] → [Exit]
     ↑ block_config                    ↑ block_config
[SDXL Block Config]              [SDXL Block Config]
  IN00-02: 0.3                     IN00-02: t=-1 (off)
  IN03-05: 0.5                     IN03-05: t=0.5
  MID: 1.0                         MID: t=1.0
  OUT00-02: 1.0                    OUT00-02: t=1.0
  OUT03-05: 1.0                    OUT03-05: t=1.2
  OUT06-08: 1.0                    OUT06-08: t=1.2
```

Left side: LoRA applied at reduced strength for early blocks.
Right side: WIDEN disabled for early blocks, aggressive for late blocks.
Two independent `BLOCK_CONFIG` instances, each wired to the node it
controls.

At Exit time, the executor reads each `RecipeLoRA`'s block config to
scale LoRA deltas per-block during the batched apply phase, and each
`RecipeMerge`'s block config for per-block t_factor during WIDEN.

#### Recipe Dataclass Extension

```python
@dataclass(frozen=True)
class BlockConfig:
    """Per-block overrides. Used for both t_factor and LoRA strength."""
    arch: str                          # Must match RecipeBase.arch
    block_overrides: tuple             # ((block_pattern, value), ...)
    layer_type_overrides: tuple        # ((layer_type, value), ...)
    # Interpretation depends on context:
    #   On RecipeMerge: value is t_factor (-1.0 = disabled/passthrough)
    #   On RecipeLoRA:  value is strength multiplier (0.0 = skip block)
    # None means "use the node's global value"

@dataclass(frozen=True)
class RecipeLoRA:
    loras: tuple           # ({"path": str, "strength": float}, ...)
    block_config: object   # BlockConfig or None (new field)

@dataclass(frozen=True)
class RecipeMerge:
    base: object
    target: object
    backbone: object
    t_factor: float
    block_config: object   # BlockConfig or None (new field)
```

#### Decisions

**Wiring: Option B (Explicit Config Type).** The explicit `BLOCK_CONFIG`
wiring fits the "recipe building" philosophy — config is data attached
to a specific node, not an ambient side-effect. Option A's scope
ambiguity ("does this filter affect just the next Merge or everything
downstream?") would be confusing and hard to reason about in complex
graphs.

**Node count: One generic node per architecture.** The Block Config
node is purely structural — it describes the architecture's block
groups and provides a float per group. It doesn't know whether those
floats will be used as LoRA strength or t_factor. The consuming node's
input name carries the semantics:

- LoRA node input: `block_strength` — values scale LoRA delta per block
- Merge node input: `block_t_factor` — values override t_factor per block

Why one node, not two (Strength + T-Factor) per architecture:

- **Maintenance**: 3 node classes total (one per arch), not 6. Adding
  an architecture means one file, not two.
- **Code simplicity**: `BLOCK_CONFIG` is a clean structural type —
  "per-block float map" with no conditional logic or mode branching.
- **Reusability**: If we later add another per-block control (e.g.,
  per-block delta budget for distilled models), same node works.
- **Input names are the documentation**: `block_strength` and
  `block_t_factor` are unambiguous about what the values mean at
  each connection point.

**Slider range**: ComfyUI's `INPUT_TYPES` is static — slider min/max
can't adapt to where the output is wired. Compromise: sliders use a
0.0 to 2.0 range (covers both LoRA strength and normal t_factor use).
ComfyUI allows typing values outside the slider range, so -1.0 (WIDEN
disabled) and higher t_factor values are accessible. Per-block
enable/disable could also be a separate boolean column if the float
range proves too awkward.

Summary:
- One generic Block Config node per architecture → `BLOCK_CONFIG` type
- Merge node gains optional `block_t_factor` input
- LoRA node gains optional `block_strength` input
- Each node has its own independent block config instance
- Unconnected = global value for all blocks (backwards compatible)
- `RecipeMerge` and `RecipeLoRA` dataclasses gain `block_config` fields
- One `BLOCK_CONFIG` output can fan out to multiple consumers

## 7. Project Structure

```
comfy-ecaj-nodes/
├── __init__.py              # NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
├── pyproject.toml           # ComfyUI registry metadata
├── requirements.txt         # torch, safetensors (comfy provides these?)
├── nodes/
│   ├── entry.py             # Entry node (MODEL → WIDEN)
│   ├── lora.py              # LoRA node (file + strength → WIDEN)
│   ├── compose.py           # Compose node (branch accumulator)
│   ├── merge.py             # Merge node (recipe builder)
│   └── exit.py              # Exit node (recipe executor → MODEL)
├── lib/
│   ├── widen.py             # WIDEN algorithm (ported from src/core/widen.py)
│   ├── divergence.py        # (ported from src/core/divergence.py)
│   ├── ranking.py           # (ported from src/core/ranking.py)
│   ├── numerical_config.py  # (ported from src/core/numerical_config.py)
│   ├── recipe.py            # Recipe tree dataclasses (§6.6)
│   ├── executor.py          # Batched pipeline executor (ported from lora_chain_merge.py)
│   └── lora/                # Architecture-specific LoRA handling
│       ├── base.py          # Loader interface
│       ├── zimage.py        # Z-Image QKV fusing, key mapping
│       └── sdxl.py          # SDXL key mapping (or delegate to ComfyUI's)
├── examples/                # Example ComfyUI workflow JSONs
└── tests/                   # Test suite (ported + new)
```

## 8. References

### ComfyUI Source (key files)
- `comfy/model_patcher.py` — ModelPatcher class
- `comfy/model_base.py` — BaseModel, SDXL, Flux classes
- `comfy/sd.py` — checkpoint loading, LoRA application
- `comfy/lora.py` — calculate_weight, load_lora, key mappings
- `comfy_extras/nodes_model_merging.py` — ModelMergeSimple, ModelMergeBlocks
- `comfy_extras/nodes_model_merging_model_specific.py` — per-block definitions
- `comfy/model_management.py` — GPU/CPU memory management
- `comfy_execution/caching.py` — execution cache system

### Existing Merge Packs
- github.com/54rt1n/ComfyUI-DareMerge (DARE-TIES)
- github.com/ljleb/comfy-mecha (recipe-based)
- github.com/larsupb/LoRA-Merger-ComfyUI (mergekit integration)
- github.com/ntc-ai/ComfyUI-DARE-LoRA-Merge (DARE for LoRA stacks)

### merge-router Source (to port from ~/Projects/merge-router/)
- `src/core/widen.py` — WIDEN algorithm
- `src/core/divergence.py`, `ranking.py`, `sparsity.py`, `numerical_config.py`
- `scripts/lora_chain_merge.py` — batched pipeline, DeltaSpec, evaluate_node
- `scripts/zimage_lora_merge.py` — Z-Image LoRA loading, QKV fusing
