# Model Loader Split & CLIP Merging

Two enhancements: (1) split model input node by source directory, (2) add CLIP text encoder merging with full WIDEN pipeline.

## Specs

```yaml
# ── Enhancement 1: Diffusion Model Input ──────────────────────────

- title: Diffusion Model Input Node
  slug: diffusion-model-input-node
  type: feature
  parent: "@widen"
  description: |
    New ComfyUI node that produces a RecipeModel from the diffusion_models/
    directory instead of checkpoints/. Mirrors the existing Model Input Node
    but targets standalone diffusion model files (Flux, Qwen, Z-Image, etc.).
    Both nodes are directory-based only with no architecture restriction --
    arch detection happens at Exit time. The existing Model Input Node keeps
    reading from checkpoints/ and gets a display name clarification to
    "Checkpoint Input" in DISPLAY_NAME.
  acceptance_criteria:
    - id: ac-1
      given: the node's INPUT_TYPES
      when: inspected
      then: |
        it has model_name as a combo populated from
        folder_paths.get_filename_list for the diffusion_models
        (or unet) folder
    - id: ac-2
      given: the node's INPUT_TYPES
      when: inspected
      then: |
        it has strength as FLOAT with default 1.0 and range 0.0-2.0
    - id: ac-3
      given: the node executes with a valid diffusion model filename
      when: output is inspected
      then: |
        it returns a RecipeModel with the filename stored in path,
        the strength stored, and source_dir set to diffusion_models
    - id: ac-4
      given: the node executes
      when: checking GPU memory and disk I/O
      then: no GPU memory is allocated and no file is opened (deferred to Exit)
    - id: ac-5
      given: the node class
      when: inspecting CATEGORY
      then: it is ecaj/merge
    - id: ac-6
      given: the node's RETURN_TYPES
      when: inspected
      then: it returns WIDEN type (compatible with Compose and Merge inputs)
    - id: ac-7
      given: an optional BLOCK_CONFIG input
      when: connected
      then: the BlockConfig is stored in RecipeModel.block_config
    - id: ac-8
      given: the node is registered in __init__.py
      when: NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS are inspected
      then: |
        the node appears with a class key and a display name containing
        Diffusion Model
  implementation_notes: |
    Create nodes/diffusion_model_input.py mirroring model_input.py.
    Use folder_paths.get_filename_list("diffusion_models") with fallback
    to "unet" for older ComfyUI versions (try/except at runtime).
    Set source_dir="diffusion_models" on the RecipeModel.
    Register in __init__.py with display name "WIDEN Diffusion Model Input".

- title: Diffusion Model Path Resolution
  slug: diffusion-model-path-resolution
  type: requirement
  parent: "@diffusion-model-input-node"
  description: |
    The Exit node and supporting infrastructure must resolve RecipeModel
    paths from either checkpoints/ or diffusion_models/ directories.
    The source_dir field on RecipeModel distinguishes which source a
    RecipeModel came from. All call sites that resolve or hash model
    paths must be source_dir-aware.
  acceptance_criteria:
    - id: ac-1
      given: a RecipeModel with source_dir="checkpoints"
      when: the Exit node resolves its path
      then: the path is resolved against the checkpoints folder
    - id: ac-2
      given: a RecipeModel with source_dir="diffusion_models"
      when: the Exit node resolves its path
      then: the path is resolved against the diffusion_models folder
    - id: ac-3
      given: both checkpoint and diffusion_models RecipeModels in the same tree
      when: the Exit node processes them
      then: each resolves against its correct source directory independently
    - id: ac-4
      given: the RecipeModel frozen dataclass
      when: inspecting its fields
      then: |
        it has a source_dir string field with default "checkpoints"
    - id: ac-5
      given: _build_model_resolver in exit.py
      when: called with a RecipeModel
      then: |
        it reads source_dir from the RecipeModel and resolves the
        path against the corresponding ComfyUI folder
    - id: ac-6
      given: serialize_recipe in persistence.py
      when: serializing a RecipeModel
      then: source_dir is included in the JSON output
    - id: ac-7
      given: _compute_recipe_hash in exit.py
      when: hashing a recipe tree containing RecipeModels
      then: |
        source_dir is included in the hash so that identical filenames
        in different folders produce different hashes
    - id: ac-8
      given: compute_lora_stats in persistence.py
      when: resolving a RecipeModel path for stats
      then: it uses the correct folder based on source_dir
    - id: ac-9
      given: a previously cached merged model (pre-source_dir)
      when: the updated code loads the cache
      then: |
        the cache miss is silent (stale hash, re-evaluated cleanly)
        and no error is raised
  implementation_notes: |
    Add source_dir field to RecipeModel dataclass (default "checkpoints"
    for backward compat). Update _build_model_resolver to accept source_dir.
    Update serialize_recipe to include source_dir. Update _compute_recipe_hash
    to include source_dir. Update compute_lora_stats model path resolution.
    Cache invalidation from the new field is expected and acceptable.

# ── Enhancement 2: CLIP Merging ───────────────────────────────────

- title: Recipe Domain Field
  slug: recipe-domain-field
  type: requirement
  parent: "@widen"
  description: |
    Add a domain field to RecipeBase to distinguish diffusion model
    pipelines from CLIP pipelines. The domain flows from RecipeBase
    at the root of the recipe tree and is used by loader registries,
    block classifiers, and analysis functions to select the correct
    architecture-specific implementations. Defaults to "diffusion"
    for backward compatibility.
  acceptance_criteria:
    - id: ac-1
      given: the RecipeBase frozen dataclass
      when: inspecting its fields
      then: |
        it has a domain string field with default "diffusion"
    - id: ac-2
      given: an existing recipe tree (pre-domain field)
      when: domain is not explicitly set
      then: it defaults to "diffusion" and all existing behavior is unchanged
    - id: ac-3
      given: analyze_recipe in lib/analysis.py
      when: selecting a LoRA loader
      then: it dispatches on (arch, domain) not just arch
    - id: ac-4
      given: analyze_recipe_models in lib/analysis.py
      when: selecting a model loader
      then: it dispatches on (arch, domain) not just arch
    - id: ac-5
      given: get_loader in lib/lora/__init__.py
      when: called with arch="sdxl" and domain="clip"
      then: it returns the SDXL CLIP LoRA loader (not the UNet loader)
    - id: ac-6
      given: get_loader in lib/lora/__init__.py
      when: called with arch="sdxl" and domain="diffusion" (or no domain)
      then: it returns the existing SDXL UNet LoRA loader (backward compat)
    - id: ac-7
      given: serialize_recipe in persistence.py
      when: serializing a RecipeBase
      then: domain is included in the JSON output
    - id: ac-8
      given: classify_key in lib/block_classify.py
      when: called with arch="sdxl" and domain="clip"
      then: |
        it dispatches to CLIP-specific key classification
        (CL00-CL11, CG00-CG31, structural keys)
    - id: ac-9
      given: classify_key in lib/block_classify.py
      when: called with arch="sdxl" and domain="diffusion"
      then: it dispatches to existing UNet key classification (unchanged)
  implementation_notes: |
    Add domain: str = "diffusion" to RecipeBase. Update get_loader()
    signature to accept domain param. Update analyze_recipe() and
    analyze_recipe_models() to pass domain through. Update
    classify_key() to accept and dispatch on domain. Update
    persistence serialization. This is a cross-cutting change that
    enables the entire CLIP pipeline.

- title: CLIP Merging Pipeline
  slug: clip-merge-pipeline
  type: feature
  parent: "@widen"
  description: |
    Full WIDEN-based text encoder merging pipeline for SDXL CLIP models.
    Uses a separate WIDEN_CLIP ComfyUI type for graph-level type safety,
    preventing accidental cross-connection with the diffusion model
    pipeline. Reuses the existing recipe dataclasses (RecipeBase,
    RecipeLoRA, RecipeModel, RecipeMerge, RecipeCompose) with the
    domain field set to "clip". LoRAs are declared separately in each
    pipeline chain with independent strength control. SDXL only for v1.
  acceptance_criteria:
    - id: ac-1
      given: the CLIP pipeline nodes
      when: their RETURN_TYPES and input types are inspected
      then: they use WIDEN_CLIP type (not WIDEN) for recipe connections
    - id: ac-2
      given: a complete CLIP merge workflow (entry -> lora -> merge -> exit)
      when: executed end-to-end
      then: |
        the output CLIP object produces different text embeddings
        from the input CLIP when encoding the same text prompt
    - id: ac-3
      given: a WIDEN_CLIP output
      when: a user attempts to connect it to a model Exit node input
      then: ComfyUI prevents the connection at the graph level (type mismatch)
    - id: ac-4
      given: the CLIP Entry node
      when: it creates a RecipeBase
      then: the domain field is set to "clip"

- title: CLIP Entry Node
  slug: clip-entry-node
  type: requirement
  parent: "@clip-merge-pipeline"
  description: |
    Boundary from ComfyUI CLIP world to WIDEN_CLIP recipe world. Wraps
    a CLIP object reference in a RecipeBase with domain="clip". Detects
    architecture from text encoder state dict key patterns. Zero GPU work.
  acceptance_criteria:
    - id: ac-1
      given: a ComfyUI CLIP input
      when: CLIP Entry node executes
      then: |
        it returns a RecipeBase wrapping the CLIP reference with
        arch set and domain="clip"
    - id: ac-2
      given: an SDXL CLIP with clip_l and clip_g encoder keys
      when: architecture detection runs
      then: arch field is set to sdxl
    - id: ac-3
      given: CLIP Entry node executes
      when: output is inspected
      then: no GPU memory is allocated and no tensors are copied
    - id: ac-4
      given: the node's RETURN_TYPES
      when: inspected
      then: it returns WIDEN_CLIP type
    - id: ac-5
      given: a non-SDXL CLIP
      when: architecture detection runs
      then: |
        it raises a clear error stating only SDXL CLIP merging
        is supported in v1
    - id: ac-6
      given: the node's INPUT_TYPES
      when: inspected
      then: it accepts a CLIP input type (ComfyUI standard CLIP type)
    - id: ac-7
      given: the CLIP object
      when: its state dict keys are accessed for architecture detection
      then: |
        keys are accessed via the CLIP object's patcher or load_model
        API without loading weights to GPU
  implementation_notes: |
    Create nodes/clip_entry.py. ComfyUI CLIP objects have a
    .cond_stage_model or .patcher attribute. Investigate the exact API
    to access state dict keys. Detect SDXL by presence of clip_l and
    clip_g key prefixes. Set domain="clip" on RecipeBase.

- title: CLIP Exit Node
  slug: clip-exit-node
  type: requirement
  parent: "@clip-merge-pipeline"
  description: |
    GPU execution node for CLIP merging. Evaluates the CLIP recipe tree
    using the same batched WIDEN pipeline but targeting text encoder
    weights. Returns a ComfyUI CLIP with merged weights. Selects
    CLIP-specific loaders and classifiers via the domain="clip" field
    on RecipeBase.
  acceptance_criteria:
    - id: ac-1
      given: a valid CLIP recipe tree ending in RecipeMerge
      when: CLIP Exit node executes
      then: |
        it returns a ComfyUI CLIP object that is usable by all
        downstream nodes that accept CLIP input
    - id: ac-2
      given: a CLIP recipe tree with LoRA target
      when: CLIP Exit evaluates
      then: |
        it selects the CLIP LoRA loader (not UNet loader) based
        on domain="clip" from RecipeBase
    - id: ac-3
      given: a CLIP recipe tree with RecipeModel target
      when: CLIP Exit evaluates
      then: |
        it selects the CLIP model loader (not diffusion model
        loader) based on domain="clip"
    - id: ac-4
      given: the CLIP Exit node
      when: installing merged weights on the CLIP object
      then: |
        it clones the base CLIP object and applies merged text
        encoder weights via ComfyUI's supported patch mechanism
    - id: ac-5
      given: the output CLIP from CLIP Exit
      when: used in a standard ComfyUI text encoding workflow
      then: |
        it functions as a valid CLIP object (encode text, return
        conditioning) without errors
    - id: ac-6
      given: an invalid CLIP recipe tree
      when: CLIP Exit validates
      then: it raises ValueError naming the invalid type and position
    - id: ac-7
      given: the CLIP Exit node's input type
      when: inspected
      then: it accepts WIDEN_CLIP type (not WIDEN)
    - id: ac-8
      given: the CLIP Exit node
      when: LoRA or model file mtimes change between executions
      then: IS_CHANGED returns a different value triggering re-execution
    - id: ac-9
      given: the CLIP Exit node processes multiple batch groups
      when: each batch group completes
      then: |
        progress is reported via ComfyUI ProgressBar
  implementation_notes: |
    Create nodes/clip_exit.py. Mirrors exit.py structure but operates on
    CLIP state dict. Reuses recipe_eval, gpu_ops, per_block, batch_groups
    from lib/executor. The domain="clip" on RecipeBase drives loader and
    classifier selection via analyze_recipe(). Investigate ComfyUI CLIP
    clone/patch API early -- this is the highest risk area. If CLIP uses
    ModelPatcher internally, the same add_patches("set", ...) mechanism
    works. If not, may need load_sd() or direct state_dict manipulation.
    v1 does NOT include incremental block recompute for CLIP (add later).

- title: CLIP Graph Nodes
  slug: clip-graph-nodes
  type: requirement
  parent: "@clip-merge-pipeline"
  description: |
    WIDEN_CLIP-typed versions of LoRA, Compose, Merge, and Model Input
    nodes. Identical logic to their WIDEN counterparts but use WIDEN_CLIP
    ComfyUI type for graph-level type safety. Can be generated via factory
    or thin wrappers.
  acceptance_criteria:
    - id: ac-1
      given: CLIP LoRA node
      when: its types are inspected
      then: it accepts optional WIDEN_CLIP prev input and returns WIDEN_CLIP
    - id: ac-2
      given: CLIP Compose node
      when: its types are inspected
      then: |
        it accepts WIDEN_CLIP branch and optional WIDEN_CLIP compose,
        returns WIDEN_CLIP
    - id: ac-3
      given: CLIP Merge node
      when: its types are inspected
      then: |
        it accepts WIDEN_CLIP base and WIDEN_CLIP target,
        returns WIDEN_CLIP
    - id: ac-4
      given: CLIP Model Input node
      when: its types are inspected
      then: it returns WIDEN_CLIP type
    - id: ac-5
      given: CLIP Model Input node
      when: it creates a RecipeModel
      then: it reads from the checkpoints folder (CLIP weights come from full checkpoints)
    - id: ac-6
      given: any CLIP graph node
      when: inspecting CATEGORY
      then: it is ecaj/merge/clip
    - id: ac-7
      given: the CLIP graph nodes
      when: their execute methods run
      then: they produce the same recipe dataclass types as their WIDEN counterparts
    - id: ac-8
      given: all CLIP graph nodes
      when: registered in __init__.py
      then: |
        they appear in NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS
        with CLIP-prefixed display names
  implementation_notes: |
    Create nodes/clip_nodes.py with a make_clip_variant() factory that
    wraps existing node classes, swapping WIDEN for WIDEN_CLIP in
    INPUT_TYPES and RETURN_TYPES. Each CLIP variant gets CATEGORY
    "ecaj/merge/clip". Register all variants in __init__.py.

- title: SDXL CLIP Block Config
  slug: sdxl-clip-block-config
  type: requirement
  parent: "@clip-merge-pipeline"
  description: |
    Per-block strength control for SDXL text encoders. Covers both
    CLIP-L (12 transformer blocks) and CLIP-G (32 transformer blocks)
    with per-block sliders. Uses the existing block config factory.
    Block config uses arch="sdxl" with domain="clip" to prevent
    cross-connection with UNet block configs.
  acceptance_criteria:
    - id: ac-1
      given: the SDXL CLIP block config node
      when: its INPUT_TYPES are inspected
      then: |
        it has sliders for CLIP-L blocks (CL00-CL11),
        CLIP-G blocks (CG00-CG31), and structural keys
        (CL_EMBED, CL_FINAL, CG_EMBED, CG_FINAL, CG_PROJ)
    - id: ac-2
      given: the block config node
      when: a slider is set to 0.0
      then: that block's text encoder weights are not merged (base preserved)
    - id: ac-3
      given: the block config node
      when: a slider is set to 2.0
      then: that block's merge strength is doubled
    - id: ac-4
      given: the block config node
      when: it produces a BlockConfig
      then: |
        BlockConfig.arch is sdxl and block_overrides contains
        the per-block values
    - id: ac-5
      given: the SDXL CLIP block config
      when: connected to a CLIP Merge node
      then: the merge applies per-block text encoder strength control
    - id: ac-6
      given: no block config connected to CLIP merge nodes
      when: the workflow executes
      then: all text encoder blocks merge at uniform strength
    - id: ac-7
      given: a CLIP key like clip_l.transformer.text_model.encoder.layers.5.X
      when: classify_key is called with arch="sdxl" and domain="clip"
      then: it returns "CL05"
    - id: ac-8
      given: a CLIP key like clip_g.transformer.text_model.encoder.layers.20.X
      when: classify_key is called with arch="sdxl" and domain="clip"
      then: it returns "CG20"
    - id: ac-9
      given: a CLIP embedding key like clip_l.transformer.text_model.embeddings.X
      when: classify_key is called with arch="sdxl" and domain="clip"
      then: it returns "CL_EMBED"
    - id: ac-10
      given: a CLIP key
      when: classify_layer_type is called with arch="sdxl" and domain="clip"
      then: |
        it returns the correct layer type (attention, feed_forward,
        or norm) based on the key's component suffix
  implementation_notes: |
    Create nodes/block_config_sdxl_clip.py using make_block_config_node().
    Add classify_key_sdxl_clip() and classify_layer_type_sdxl_clip() to
    lib/block_classify.py. Update classify_key() and classify_layer_type()
    dispatch to check domain parameter. CLIP-L blocks map to
    clip_l.transformer.text_model.encoder.layers.{N}. CLIP-G blocks map
    to clip_g.transformer.text_model.encoder.layers.{N}. Verify exact
    key patterns against real SDXL CLIP state dicts.

- title: SDXL CLIP LoRA Loader
  slug: sdxl-clip-lora-loader
  type: requirement
  parent: "@clip-merge-pipeline"
  description: |
    Architecture-specific LoRA loader for SDXL text encoder weights.
    Handles kohya/A1111 LoRA key mapping for text encoder components
    (lora_te1_* for CLIP-L, lora_te2_* for CLIP-G). Produces DeltaSpec
    objects for the batched executor pipeline. Registered in the loader
    registry under (arch="sdxl", domain="clip").
  acceptance_criteria:
    - id: ac-1
      given: an SDXL LoRA file with lora_te1_* keys
      when: the CLIP loader processes it
      then: |
        it maps them to CLIP-L base model keys
        (clip_l.transformer.text_model.encoder.layers.N.*)
    - id: ac-2
      given: an SDXL LoRA file with lora_te2_* keys
      when: the CLIP loader processes it
      then: |
        it maps them to CLIP-G base model keys
        (clip_g.transformer.text_model.encoder.layers.N.*)
    - id: ac-3
      given: an SDXL LoRA file with only UNet keys (no te1/te2)
      when: the CLIP loader processes it
      then: affected_keys returns an empty frozenset
    - id: ac-4
      given: the CLIP loader after processing a LoRA with te1/te2 keys
      when: get_delta_specs is called with matching keys
      then: |
        it returns DeltaSpec objects with correct up/down weight
        tensors and dimensions matching the base model parameters
    - id: ac-5
      given: the CLIP loader
      when: it implements the loader interface
      then: |
        it provides load(path, strength, set_id), affected_keys,
        affected_keys_for_set(set_id),
        get_delta_specs(keys, key_indices, set_id), and cleanup()
    - id: ac-6
      given: a LoRA with both UNet and CLIP keys
      when: loaded by the CLIP loader
      then: only text encoder keys are extracted (UNet keys ignored)
    - id: ac-7
      given: the CLIP loader class
      when: registered in lib/lora/__init__.py
      then: |
        it is accessible via get_loader(arch="sdxl", domain="clip")
  implementation_notes: |
    Create lib/lora/sdxl_clip.py following the SDXLLoader pattern.
    Key mapping for kohya format:
    lora_te1_text_model_encoder_layers_{N}_{component} maps to
    clip_l.transformer.text_model.encoder.layers.{N}.{component}.
    Similar for lora_te2_ -> clip_g. Handle compound tokens the same
    way as sdxl.py handles UNet compound tokens. Register in
    lib/lora/__init__.py LOADER_REGISTRY with key ("sdxl", "clip").

- title: CLIP Model Loader
  slug: clip-model-loader
  type: requirement
  parent: "@clip-merge-pipeline"
  description: |
    Streaming loader for text encoder weights from checkpoint files.
    Inverse of the existing ModelLoader -- includes only text encoder
    keys and excludes diffusion model and VAE keys. Uses safetensors
    safe_open() for memory-efficient access. Has its own key
    normalization function (NOT shared with ModelLoader).
  acceptance_criteria:
    - id: ac-1
      given: an SDXL checkpoint file
      when: the CLIP model loader opens it
      then: |
        it exposes text encoder keys (conditioner.embedders.*)
        and excludes diffusion model and VAE keys
    - id: ac-2
      given: CLIP model loader
      when: get_weights(keys) is called
      then: |
        it returns text encoder weight tensors mapped to CLIP base
        model key format (clip_l.*, clip_g.*)
    - id: ac-3
      given: the CLIP model loader
      when: affected_keys is accessed
      then: |
        it returns the frozenset of CLIP base model keys present
        in the checkpoint
    - id: ac-4
      given: the CLIP model loader
      when: cleanup() is called
      then: the safe_open file handle is closed
    - id: ac-5
      given: a non-safetensors file
      when: the loader attempts to open it
      then: UnsupportedFormatError is raised
    - id: ac-6
      given: an SDXL checkpoint with conditioner.embedders.0.* keys
      when: key normalization runs
      then: |
        embedders.0 keys map to clip_l.* and embedders.1 keys map
        to clip_g.* (with validation that embedder ordering is correct)
    - id: ac-7
      given: a checkpoint with unexpected embedder structure
      when: key normalization cannot determine CLIP-L vs CLIP-G mapping
      then: a clear error is raised describing the unexpected key structure
  implementation_notes: |
    Create lib/clip_model_loader.py with its own _normalize_key and
    _INCLUDED_PREFIXES (inverse of ModelLoader's _EXCLUDED_PREFIXES).
    Do NOT reuse ModelLoader._normalize_key -- they filter opposite key
    sets. Key mapping: conditioner.embedders.0.transformer.* -> clip_l.*,
    conditioner.embedders.1.transformer.* -> clip_g.*. Validate embedder
    count and ordering during init. Verify against real SDXL checkpoints.
```

## Tasks

derive_from_specs: true

```yaml
- title: Rename existing Model Input node display name
  slug: rename-model-input-display
  priority: 1
  tags:
    - cleanup
    - enhancement-1

- title: Investigate ComfyUI CLIP clone and patch API
  slug: investigate-clip-api
  priority: 1
  description: |
    Spike task: investigate how ComfyUI CLIP objects support cloning
    and weight patching. Determine if CLIP uses ModelPatcher internally
    (in which case add_patches with set works) or needs a different
    mechanism (load_sd, direct state_dict manipulation). Document
    findings as task notes. This blocks CLIP Exit Node implementation.
  tags:
    - spike
    - enhancement-2
```

## Implementation Notes

### Enhancement 1: Diffusion Model Input
Small scope. Add source_dir field to RecipeModel, create new node file, update exit path resolver and all serialization/hashing call sites. Rename existing node display name. The ComfyUI folder name may be "diffusion_models" or "unet" depending on version -- use whichever folder_paths supports (check at runtime with try/except fallback). Cache invalidation from the new source_dir field is expected and acceptable (silent cache miss, clean re-evaluation).

### Enhancement 2: CLIP Merging
Larger scope but leverages existing infrastructure heavily. Key architectural addition: the `domain` field on RecipeBase enables (arch, domain) dispatch throughout the pipeline without changing arch strings.

**Domain-based routing:**
- RecipeBase gets `domain: str = "diffusion"` (backward compat default)
- CLIP Entry sets `domain="clip"`
- `get_loader(arch, domain)` selects correct LoRA loader
- `classify_key(key, arch, domain)` selects correct block classifier
- `analyze_recipe()` / `analyze_recipe_models()` pass domain through
- Persistence serialization includes domain in hashes

**Reusable infrastructure (no changes needed):**
- recipe_eval.py: operates on arbitrary key sets, domain-agnostic
- gpu_ops.py: DeltaSpec, apply_lora_batch_gpu, chunked_evaluation
- per_block.py: uses classify_key() which handles domain dispatch
- batch_groups.py: uses classify_key() which handles domain dispatch

**Architecture-specific new code:**
1. Entry/Exit nodes (CLIP object wrapping and patching)
2. LoRA key mapping (te1/te2 prefix handling)
3. Model loader key filtering (include CLIP, exclude UNet)
4. Block classification (CLIP-L and CLIP-G block patterns)

The CLIP graph nodes (LoRA, Compose, Merge, ModelInput) are thin wrappers via a make_clip_variant() factory that swaps WIDEN for WIDEN_CLIP.

**Risk areas:**
- CLIP patching: highest risk. Spike task investigates API before implementation.
- Key mapping verification: CLIP-L/CLIP-G embedder ordering in checkpoints and LoRA te1/te2 mapping need verification against real files.
- v1 scope: no incremental block recompute for CLIP Exit (add later).
- v1 scope: no persistence cache for CLIP Exit (add later).
