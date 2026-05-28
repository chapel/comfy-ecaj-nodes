# Full Model Merging — WIDEN on Full Checkpoints

Extends the WIDEN merge pipeline to support full model checkpoint merging
alongside existing LoRA merging. Models are loaded from disk out-of-band
from ComfyUI using safetensors streaming (safe_open), matching the deferred
loading pattern used for LoRAs. This avoids requiring ComfyUI to load
additional models into memory and enables per-batch streaming for large
checkpoints.

merge-router (sibling project at ../merge-router, used for offline merge
experiments) already proves WIDEN works on full model state dicts. The gap
is plumbing full model weights through the ComfyUI recipe tree and exit
node execution pipeline.

Architecture scope: SDXL and Z-Image (current supported set). Flux/Qwen
architecture support is tracked separately.

Key terminology: "WIDEN" is the custom ComfyUI type string used for all
recipe nodes (RecipeBase, RecipeLoRA, RecipeCompose, RecipeMerge). Nodes
with RETURN_TYPES = ("WIDEN",) can connect to any input accepting WIDEN.

## Specs

```yaml
- title: Full Model Recipe Type
  slug: full-model-recipe
  type: feature
  description: |
    Frozen recipe dataclass representing a full model checkpoint to merge.
    Stores a file path and strength (like RecipeLoRA stores LoRA paths),
    not a ComfyUI MODEL reference. This enables deferred disk-based loading
    at Exit time via safetensors streaming. Follows all recipe conventions:
    frozen, tuples not lists, no GPU tensors.
  acceptance_criteria:
    - id: ac-1
      given: a RecipeModel instance
      when: a field is assigned after construction
      then: a FrozenInstanceError is raised
    - id: ac-2
      given: a RecipeModel
      when: inspecting its fields
      then: |
        it has path (str), strength (float, default 1.0), and
        block_config (BlockConfig or None, default None)
    - id: ac-3
      given: a RecipeModel instance
      when: passed to RecipeCompose.with_branch()
      then: a new RecipeCompose is returned containing it as a branch
    - id: ac-4
      given: a RecipeMerge constructed with target=RecipeModel
      when: the tree is inspected
      then: construction succeeds and target is the RecipeModel
    - id: ac-5
      given: the RecipeNode type alias
      when: inspected
      then: RecipeModel is included in the union
    - id: ac-6
      given: a RecipeModel
      when: inspected for GPU tensors
      then: no torch.Tensor objects are found (path and strength only)
  implementation_notes: |
    Add to lib/recipe.py alongside existing dataclasses. Pattern follows
    RecipeLoRA but simpler -- single path+strength instead of tuple of
    LoRA dicts. No MappingProxyType needed (only scalar fields). Update
    __all__ and RecipeNode type alias to include RecipeModel.

    The path field stores the checkpoint filename (resolved to full path
    at Exit time via folder_paths, same as LoRA path resolution).
    block_config enables per-block strength control, reusing existing
    BlockConfig type.

    Files: lib/recipe.py.

- title: Model Input Node
  slug: model-input-node
  type: feature
  description: |
    ComfyUI node that produces a RecipeModel from a checkpoint file picker.
    Loads from disk out-of-band from ComfyUI -- the model is NOT loaded at
    node execution time. Like the LoRA node, this is pure recipe building
    with zero GPU work and zero file I/O. The file path is stored in the
    recipe and resolved at Exit time for deferred streaming access.
  acceptance_criteria:
    - id: ac-1
      given: the node's INPUT_TYPES
      when: inspected
      then: |
        it has model_name (checkpoint file combo via folder_paths) and
        strength (FLOAT, default 1.0, range 0.0-2.0)
    - id: ac-2
      given: the node executes with a valid checkpoint name
      when: output is inspected
      then: it returns a RecipeModel with the filename and strength stored
    - id: ac-3
      given: the node executes
      when: checking GPU memory and disk I/O
      then: no GPU memory is allocated and no file is opened (deferred to Exit)
    - id: ac-4
      given: the node class
      when: inspecting CATEGORY
      then: it is ecaj/merge
    - id: ac-5
      given: the node's RETURN_TYPES
      when: inspected
      then: it returns WIDEN type (compatible with Compose and Merge inputs)
    - id: ac-6
      given: an optional BLOCK_CONFIG input
      when: connected
      then: the BlockConfig is stored in RecipeModel.block_config
  implementation_notes: |
    New file nodes/model_input.py. Pattern follows nodes/lora.py closely:
    checkpoint file combo via folder_paths.get_filename_list("checkpoints"),
    strength slider, optional BLOCK_CONFIG input. No chaining (unlike LoRA
    node) -- each RecipeModel represents one model. To merge multiple models,
    use Compose node.

    Register in __init__.py NODE_CLASS_MAPPINGS as
    "WIDENModelInput": WIDENModelInputNode with display name
    "WIDEN Model Input". The file path is stored as-is (the filename,
    not full path) -- Exit resolves via
    folder_paths.get_full_path("checkpoints", name) at execution time.

    Files: nodes/model_input.py, __init__.py.

- title: Full Model Loader
  slug: full-model-loader
  type: feature
  description: |
    Streaming model loader using safetensors.safe_open() for memory-efficient
    per-batch access to full checkpoint weights. Matches the LoRALoader
    interface pattern but provides full weight tensors instead of low-rank
    factors. Handles key normalization between checkpoint file format and
    base model state dict format. Architecture-specific key mapping for
    SDXL and Z-Image. Only supports safetensors format (non-safetensors
    checkpoints raise a clear error).
  acceptance_criteria:
    - id: ac-1
      given: a safetensors checkpoint path
      when: the loader opens it
      then: |
        it uses safe_open() for memory-mapped access without loading the
        full file into memory
    - id: ac-2
      given: a list of base model parameter keys
      when: get_weights(keys) is called
      then: |
        it returns the corresponding weight tensors from the checkpoint
        file, correctly mapped from file key format to base model key format
    - id: ac-3
      given: an SDXL checkpoint file with model.diffusion_model prefix
      when: key mapping runs
      then: |
        file keys are normalized to match base model state dict keys
        (e.g., model.diffusion_model.input_blocks.0 maps to input_blocks.0)
    - id: ac-4
      given: a Z-Image checkpoint file
      when: key mapping runs
      then: |
        file keys are normalized to match base model state dict keys,
        handling the diffusion_model or transformer prefix variants
    - id: ac-5
      given: the loader
      when: affected_keys is accessed
      then: |
        it returns the set of base model keys that have corresponding
        diffusion model weights in the checkpoint file, excluding
        VAE and text encoder keys
    - id: ac-6
      given: the loader is no longer needed
      when: cleanup() is called
      then: the safe_open file handle is closed and resources freed
    - id: ac-7
      given: a checkpoint file with keys that don't match the base model
      when: the mismatch is detected
      then: a clear error is raised listing unmatched keys
    - id: ac-8
      given: the loader
      when: detecting architecture from file keys
      then: |
        it can determine architecture without loading any tensor data
        by inspecting normalized keys against architecture patterns
    - id: ac-9
      given: a non-safetensors checkpoint file (e.g., .ckpt, .pt)
      when: the loader attempts to open it
      then: |
        a clear error is raised explaining that only safetensors
        format is supported for model merging
  implementation_notes: |
    New file lib/model_loader.py. Uses safetensors.safe_open(path,
    framework="pt", device="cpu") context manager.

    Key normalization pipeline (runs once at open time):
    1. Read all keys from safe_open metadata (no tensor loading).
    2. Normalize: strip architecture-specific prefixes to canonical form.
       - SDXL files: strip "model.diffusion_model." prefix.
       - Z-Image files: strip "model.diffusion_model." or similar prefix.
       - Reference merge-router src/models/ for format-specific patterns.
    3. Filter: keep only diffusion model keys (drop VAE "first_stage_model."
       and text encoder "conditioner." / "cond_stage_model." keys).
    4. Build forward map: file_key -> base_model_key (normalized).
    5. Build reverse map: base_model_key -> file_key (for lookups).

    Architecture detection: run architecture patterns on NORMALIZED keys
    (post prefix-stripping), so the same _ARCH_PATTERNS from nodes/entry.py
    work. Detection must happen AFTER normalization, not before, since
    patterns expect state_dict-format keys (e.g., "diffusion_model.input_blocks"
    not "model.diffusion_model.input_blocks").

    get_weights(keys) calls f.get_tensor(reverse_map[key]) per key, returns
    list of tensors. Streaming means per-batch disk I/O but avoids full model
    in memory. The safe_open handle is kept open for the duration of execution
    (closed in cleanup()).

    Note: safetensors safe_open uses memory-mapping on supported platforms,
    but actual behavior may vary. The key guarantee is that tensors are not
    allocated in Python memory until get_tensor() is called.

    Files: lib/model_loader.py, reference lib/lora/base.py for interface
    pattern, reference merge-router src/models/ for key normalization.

- title: Full Model Execution
  slug: full-model-execution
  type: feature
  description: |
    Exit node extension for executing WIDEN merge on full model checkpoints.
    Adds OpApplyModel operation to the recipe evaluation engine. During
    recipe analysis, detects RecipeModel nodes and opens streaming loaders.
    During batched evaluation, OpApplyModel loads model weights per-batch
    into registers, then existing OpFilterDelta/OpMergeWeights apply WIDEN
    importance routing unchanged. Validates architecture consistency between
    base model and merge models. Adding RecipeModel as a 5th recipe type
    requires updating all isinstance dispatch points across the codebase.
  acceptance_criteria:
    - id: ac-1
      given: a recipe tree containing RecipeModel nodes
      when: recipe analysis runs (lib/analysis.py)
      then: |
        it detects RecipeModel nodes, opens FullModelLoader instances for
        each unique path, and builds affected-key maps per model
    - id: ac-2
      given: a RecipeModel in a recipe tree
      when: compile_plan processes it
      then: an OpApplyModel op is emitted referencing the model's loader ID
    - id: ac-3
      given: execute_plan encounters OpApplyModel
      when: executing a batch of keys
      then: |
        it loads the model weight tensors for those keys from the streaming
        loader into a register (the raw weights, no arithmetic)
    - id: ac-4
      given: OpApplyModel result is in a register
      when: OpFilterDelta or OpMergeWeights uses it with a backbone register
      then: |
        WIDEN computes delta (model_weights - backbone) internally and
        routes by importance -- existing algorithm, unchanged
    - id: ac-5
      given: a recipe that mixes RecipeModel and RecipeLoRA nodes
      when: the exit node processes it
      then: |
        both paths execute correctly -- LoRA via existing DeltaSpec path,
        models via OpApplyModel streaming path
    - id: ac-6
      given: a checkpoint file whose detected architecture
      when: it differs from the base model architecture
      then: a clear error is raised naming both architectures and both file paths
    - id: ac-7
      given: full model weights loaded per-batch via streaming
      when: GPU evaluation completes for a batch
      then: |
        loaded weights are freed after use, not held resident
        (streaming loader re-reads from disk as needed)
    - id: ac-8
      given: GPU runs out of memory during full model evaluation
      when: OOM is caught
      then: |
        existing chunked_evaluation backoff retries at batch_size=1
        (compatible with streaming loader -- just re-reads fewer keys)
    - id: ac-9
      given: a RecipeModel with block_config
      when: per-block control is applied during execution
      then: |
        block-level strength scaling is applied to full model deltas
        the same way it applies to LoRA deltas
    - id: ac-10
      given: the checkpoint file does not exist or is not a valid safetensors file
      when: the exit node attempts to open it
      then: |
        a clear error is raised naming the missing file and which
        Model Input node referenced it
    - id: ac-11
      given: IS_CHANGED is called
      when: the recipe tree contains RecipeModel nodes
      then: |
        checkpoint file (mtime, size) is included in the hash alongside
        any LoRA file hashes
    - id: ac-12
      given: a recipe with only RecipeModel targets (no LoRAs)
      when: affected keys are computed
      then: |
        all diffusion model keys present in both base and merge model
        are processed (not just LoRA-affected subset)
    - id: ac-13
      given: a recipe composing 3 full models for merge
      when: execution runs
      then: |
        only one batch of model weights per loader is on GPU at a time
        (streaming loaders are read sequentially, not all at once)
  implementation_notes: |
    This is the largest spec -- it touches every isinstance dispatch
    point that currently hardcodes the 4 recipe types. All changes needed:

    RECIPE TYPE SYSTEM (lib/recipe.py):
    - Add RecipeModel to RecipeNode type alias (line 86).
    - Add RecipeModel to __all__ exports.

    VALIDATION (nodes/exit.py _validate_recipe_tree):
    - Add RecipeModel as valid Compose branch type (line 62).
    - Add RecipeModel as valid Merge target type (line 81).
    - Add RecipeModel leaf case (no children to validate).

    RECIPE ANALYSIS (lib/analysis.py):
    - _walk_to_base(): RecipeModel cannot be tree root -- add case
      that raises ValueError like RecipeLoRA (line 71-75).
    - _collect_lora_sets(): Add RecipeModel case that skips (no LoRAs
      to collect) instead of raising ValueError (line 124).
    - New function _collect_model_refs(): Walk tree, collect unique
      RecipeModel nodes with synthetic model IDs (parallel to
      _collect_lora_sets pattern).
    - Extend analyze_recipe() or create analyze_recipe_models() to
      open FullModelLoader per unique path, validate arch match,
      build model affected-key maps. Return extended AnalysisResult
      with model_loaders and model_affected fields.
    - Extend get_keys_to_process(): Union of LoRA affected keys AND
      model affected keys determines which keys need processing.

    EXIT NODE (nodes/exit.py):
    - _collect_lora_paths(): Add RecipeModel skip case so tree walk
      doesn't raise on model nodes.
    - New _collect_model_paths(): Parallel function that collects
      checkpoint file paths from RecipeModel nodes.
    - _compute_recipe_hash(): Include model file (mtime, size) in hash.
    - execute(): Call model analysis, pass model_loaders to executor.
    - Cleanup: Close model loaders after execution completes.

    PLAN COMPILER (lib/recipe_eval.py):
    - New OpApplyModel frozen dataclass: model_id (str), block_config,
      input_reg (int), out_reg (int).
    - Add OpApplyModel to _Op type alias (line 78).
    - _input_regs(): Add OpApplyModel case returning (input_reg,).
    - _PlanCompiler.compile_node(): Add RecipeModel dispatch that
      emits OpApplyModel (parallel to _compile_lora pattern).
    - _PlanCompiler._compile_merge(): Add RecipeModel to valid
      target isinstance check (line 283).

    PLAN EXECUTOR (lib/recipe_eval.py):
    - execute_plan() gains new parameter:
      model_loaders: dict[str, FullModelLoader] | None = None
      This keeps backward compatibility (None = no models).
    - OpApplyModel handler: look up loader by model_id, call
      loader.get_weights(keys), stack into [B, *shape] tensor,
      move to device/dtype, store in register.
    - evaluate_recipe() wrapper passes model_loaders through.

    COMPOSE NODE (nodes/compose.py):
    - Add RecipeModel to valid branch types (line 48).

    MERGE NODE (nodes/merge.py):
    - Add RecipeModel to valid target types (line 77-81).

    KEY INSIGHT: OpApplyModel just loads weights into a register.
    It does NOT compute deltas. OpFilterDelta internally computes
    delta = input_reg - backbone_reg via WIDEN's filter_delta_batched.
    So full model merge reuses 100% of the WIDEN algorithm unchanged.

    MIXED RECIPES: compile_plan handles both node types independently.
    A chained merge like Merge(base=Merge(base=Entry, target=LoRA),
    target=Model) evaluates inner merge (LoRA path) first, then outer
    merge (model path) with inner result as base. The only naming
    confusion: execute_plan's variable "lora_applied" (line 397) is
    actually "the weights in input_reg" which may be raw model weights
    when from OpApplyModel. Consider renaming to "applied" for clarity.

    PERFORMANCE NOTE: For recipes with only RecipeModel targets (no
    LoRAs), affected_keys is the full diffusion model key set. This
    means ALL keys are processed, not a small LoRA subset. Batch sizing
    via compute_batch_size() handles this, but total processing time is
    proportionally higher. For SDXL: ~1500 keys vs ~200-400 for typical
    LoRA merges.

    MEMORY: Streaming loaders mean only 1 batch of model weights on GPU
    at a time. For SDXL with batch_size=64, this is ~100-500MB GPU per
    batch vs 4.2GB for full model. With 3 models in a Compose, each
    batch loads 3 × batch_size weights sequentially.

    Files: lib/recipe.py, lib/recipe_eval.py, lib/analysis.py,
    nodes/exit.py, nodes/merge.py, nodes/compose.py.
```

## Tasks

derive_from_specs: true

## Implementation Notes

The dependency chain for implementation:
1. full-model-recipe (standalone, extends lib/recipe.py)
2. model-input-node (depends on full-model-recipe)
3. full-model-loader (standalone, new lib/model_loader.py)
4. full-model-execution (depends on all above + existing exit-node + batched-executor)

Key architecture decisions:
- Disk-based loading, not ComfyUI MODEL: checkpoint files are opened
  directly via safetensors.safe_open() at Exit time, bypassing ComfyUI's
  model loading. This enables streaming per-batch access without full
  model in memory. Same pattern as LoRA deferred loading.
- OpApplyModel is a "load" not "compute": it loads model weights into a
  register. WIDEN's filter_delta/merge_weights handles the delta internally.
- Mixed recipes work naturally: LoRA and model nodes coexist in the same
  recipe tree via chained merges.
- Per-block control reuses existing BlockConfig and block_classify
  infrastructure. No new classifiers needed for SDXL/Z-Image.
- Only safetensors format supported. Non-safetensors checkpoints get a
  clear error message.
- VAE/CLIP/text encoder keys in checkpoint files are filtered out --
  only diffusion model weights are merged.
- Architecture detection runs on normalized keys (post prefix-stripping)
  so existing _ARCH_PATTERNS work unchanged.

Reference implementations:
- merge-router/scripts/sdxl_merge.py: SDXL full model merge pattern
- merge-router/scripts/qwen_merge.py: Streaming safe_open() pattern
- merge-router/src/models/qwen_key_mapper.py: Key normalization pattern
