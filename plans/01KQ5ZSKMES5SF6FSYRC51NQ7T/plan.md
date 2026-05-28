# Full Saved Model Output Plan

## Specs

```yaml
- title: Full Saved Model Output
  slug: full-saved-model-output
  type: feature
  parent: "@exit-node"
  tags: [persistence, memory, comfy]
  description: |
    The Exit node can produce a complete saved diffusion model artifact for a
    merge recipe and return a MODEL that behaves like a normally loaded ComfyUI
    diffusion model for downstream execution.
  acceptance_criteria:
    - id: ac-complete-artifact
      given: |
        A valid merge recipe affects some or all diffusion model weights.
      when: |
        The Exit node produces a full saved model output for that recipe.
      then: |
        The saved artifact contains every diffusion model weight required for the
        merged model to be loaded without relying on the original in-memory patch
        payload.
    - id: ac-return-loaded-model
      given: |
        A full saved model artifact is produced or reused for the current recipe.
      when: |
        The Exit node returns its MODEL output.
      then: |
        Downstream nodes receive a MODEL representing the saved merged artifact,
        and the result remains usable after temporary merge outputs have been
        released by the Exit node.
    - id: ac-cache-reuses-artifact
      given: |
        A saved full model artifact matches the current recipe identity.
      when: |
        The Exit node executes in full saved model mode.
      then: |
        The existing artifact is reused without recomputing the merged weights.

- title: Streaming Full Model Materialization
  slug: streaming-full-model-materialization
  type: requirement
  parent: "@full-saved-model-output"
  tags: [persistence, memory]
  description: |
    Full model materialization keeps merge output memory bounded while creating
    the complete saved artifact.
  acceptance_criteria:
    - id: ac-affected-results-released
      given: |
        A merge produces affected weights across multiple groups of model keys.
      when: |
        Full saved model mode records affected weights for the artifact.
      then: |
        The operation can complete without requiring all affected weights to
        remain resident until every later group has been evaluated.
    - id: ac-base-weight-bounded-copying
      given: |
        The saved artifact must include base weights that are not affected by the
        merge recipe.
      when: |
        Full saved model mode creates the artifact.
      then: |
        The operation does not require a second full model-sized in-memory copy
        of all base weights at once.
    - id: ac-incomplete-write-not-reused
      given: |
        Artifact creation fails before the full saved model is complete.
      when: |
        The Exit node handles the failure.
      then: |
        No incomplete artifact is accepted as a successful cache hit on a later
        execution.

- title: Comfy Memory Manager Compatibility
  slug: comfy-memory-manager-compatibility
  type: requirement
  parent: "@full-saved-model-output"
  tags: [comfy, memory, compatibility]
  description: |
    Full saved model output remains compatible with ComfyUI's current model
    memory lifecycle, including dynamic memory management when ComfyUI enables
    it.
  acceptance_criteria:
    - id: ac-no-dynamic-vram-opt-out
      given: |
        ComfyUI is running with its dynamic model memory management enabled.
      when: |
        A workflow uses full saved model output.
      then: |
        The workflow can run without requiring users to disable ComfyUI's dynamic
        memory management.
    - id: ac-comfy-owns-returned-model-memory
      given: |
        The Exit node has returned a MODEL from a full saved model artifact.
      when: |
        ComfyUI loads, unloads, partially loads, or prioritizes models for a
        workflow.
      then: |
        The returned MODEL remains compatible with ComfyUI's model memory
        lifecycle and does not require all merged affected weights to remain
        resident solely because the Exit node returned.
    - id: ac-measured-memory-behavior
      given: |
        The same representative merge workflow is run with the current patch
        output and with full saved model output.
      when: |
        memory behavior is measured under ComfyUI with dynamic memory management
        enabled.
      then: |
        The validation output reports the active ComfyUI memory mode, peak RAM,
        peak GPU memory, returned-model behavior, and whether merged affected
        weights remain persistently resident after the Exit node returns; the
        validation fails if full saved model output requires disabling ComfyUI's
        dynamic memory management or requires all merged affected weights to
        remain resident solely because the Exit node returned.
```

## Tasks

derive_from_specs: false

```yaml
- title: Update Exit output-mode specs for full saved model behavior
  slug: task-full-model-update-existing-specs
  priority: 1
  tags: [specs, exit-node, persistence]
  spec_ref: "@full-saved-model-output"
  depends_on: []
  description: |
    What: Update existing project specs so the current set-patch Exit behavior is
    scoped as the default in-memory output mode and full saved model output is a
    distinct supported output behavior.

    Why: Existing specs such as @exit-node, @exit-patch-install,
    @exit-model-persistence, and @memory-management predate multiple output
    modes. They currently describe the Exit result as set patches and saved
    artifacts as persistence/cache support, which can conflict with returning a
    Comfy-loadable saved MODEL.

    How:
    - Review existing specs for Exit return behavior, patch installation,
      persistence, cache hits, and memory management.
    - Scope existing set-patch acceptance criteria to the in-memory patch output
      mode instead of making them universal Exit behavior.
    - Extend or refine saved-model persistence specs so a full saved model output
      can return a MODEL loaded from the saved artifact.
    - Add relationships from new full saved model requirements to existing
      persistence and memory-management specs where supported by kspec.
    - Keep spec wording behavioral: describe user-visible output mode behavior,
      reloadability, cache reuse, and memory observations rather than internal
      writer or patcher classes.

    Testing:
    - Run kspec validation or plan import checks after the spec edits.
    - Confirm the resulting specs do not contain contradictory requirements for
      the same Exit output mode.

    Reconciliation evidence, not final product coverage: confirms existing Exit,
      patch-installation, persistence, and memory-management specs no longer
      contradict the full saved model output mode before implementation begins.
- title: Validate saved-artifact reload before full materialization work
  slug: task-full-model-saved-artifact-reload-spike
  priority: 1
  tags: [research, comfy, memory, persistence]
  spec_ref: "@comfy-memory-manager-compatibility"
  depends_on:
    - "@task-full-model-update-existing-specs"
  description: |
    What: Run a deliberately bounded technical spike that proves whether a WIDEN
    merged diffusion artifact can be loaded through ComfyUI's normal model
    loading path and returned as the Exit node's MODEL output with equivalent
    downstream behavior.

    Why: The full saved model plan depends on ComfyUI accepting the saved merged
    artifact as a normal loaded model. That compatibility should be proven before
    investing in bounded merge-result sinks, incremental artifact writing, and
    full materialization refactors. The spike is intentionally narrow so it can
    stop the plan early if saved-artifact reload is not viable.

    How:
    - Use the existing save_model/atomic_save path or a temporary test-only path
      to create a representative merged safetensors artifact. Do not optimize
      peak RAM in this task.
    - Load the saved artifact with the same public ComfyUI diffusion-model loading
      path that regular model nodes use.
    - Compare the loaded-artifact MODEL against the current in-memory patch
      output for the same recipe and assert affected weights are equivalent.
    - Exercise the existing cache-hit shape: when a saved artifact matches the
      current recipe identity, prove the GPU merge can be skipped and the MODEL
      can be loaded from that artifact.
    - In a ComfyUI environment with Dynamic VRAM enabled, capture whether the
      returned loaded model participates in ComfyUI's normal memory lifecycle.
      When Dynamic VRAM is unavailable, record the fallback mode explicitly
      rather than treating that as a failure.
    - Produce a go/no-go note in the task evidence with the exact loader API or
      path used, returned MODEL contract, cache-hit behavior, parity result,
      memory-mode observation, and decision outcome.
    - If the saved artifact cannot be loaded as a usable Comfy MODEL, stop the
      full saved model plan and revise the design before starting the
      materialization tasks.
    - Remove or isolate any temporary proof-of-concept output mode before the
      task is complete unless a later approved task explicitly keeps it.

    Testing:
    - Add unit tests that monkeypatch ComfyUI's diffusion-model loader and assert
      the saved path is loaded for the spike path and compatible cache-hit path.
    - Add parity coverage comparing affected weights from current in-memory patch
      output and saved-artifact reload output.
    - Add an optional integration script or documented command for a local
      ComfyUI checkout that prints the active patcher/memory mode and whether the
      loaded artifact can feed downstream Comfy nodes.
    - Run the focused persistence/Exit pytest slice.

    Gate evidence, not final product coverage: records whether
      @full-saved-model-output ac-return-loaded-model and ac-cache-reuses-artifact
      are viable before the implementation tasks proceed; final product coverage
      remains with Exit integration and memory validation tasks.
- title: Introduce a bounded merge result sink
  slug: task-full-model-result-sink-interface
  priority: 1
  tags: [persistence, memory]
  spec_ref: "@streaming-full-model-materialization"
  depends_on:
    - "@task-full-model-saved-artifact-reload-spike"
  description: |
    What: Introduce an internal sink interface so merge evaluation can hand off
    each completed affected tensor without requiring the whole affected result
    set to stay resident in one dictionary.

    Why: The current Exit implementation builds merged_state and then installs or
    saves it. Full saved model output cannot reduce memory until merge
    evaluation can emit each completed tensor to a consumer that may immediately
    persist it and release references.

    How:
    - Inspect the current merge execution path in nodes/exit.py and
      lib/gpu_ops.py.
    - Add a pure Python/PyTorch result sink abstraction with write_tensor(key,
      tensor), finalize(), and abort() behavior.
    - Provide an in-memory sink that preserves current in_memory_patches behavior
      and lets existing callers continue receiving a dict of affected tensors.
    - Keep the sink layer free of ComfyUI imports so it can be unit-tested
      without a running ComfyUI instance.
    - Ensure abort paths release temporary resources and preserve existing error
      propagation.

    Testing:
    - Add focused unit tests for the in-memory sink.
    - Assert the sink records the exact affected keys and tensors passed to it.
    - Assert abort/finalize behavior is deterministic and safe to call from
      error paths.

    Covers: @streaming-full-model-materialization ac-affected-results-released.
- title: Add sink-writing chunk evaluation
  slug: task-full-model-chunked-evaluation-to-sink
  priority: 1
  tags: [executor, memory]
  spec_ref: "@streaming-full-model-materialization"
  depends_on:
    - "@task-full-model-result-sink-interface"
  description: |
    What: Add a sink-writing variant of the chunked evaluation loop used by the
    Exit node.

    Why: The existing chunked evaluation returns a dict, which keeps all affected
    CPU tensors alive until later install or save work completes. A sink-writing
    path allows full saved model output to bound affected-result memory while
    preserving the current computation semantics.

    How:
    - Modify lib/gpu_ops.py or add a focused streaming evaluation module.
    - Implement a function with the same merge evaluation inputs as the current
      chunked evaluation path plus a result sink.
    - For every successfully evaluated key, convert the result to the base-model
      storage dtype on CPU, pass it to the sink, and then release local chunk
      references before later chunks are evaluated.
    - Preserve current GPU out-of-memory retry behavior, system memory error
      handling, dtype conversion, CPU output guarantees, and cleanup behavior.
    - Keep the existing dict-returning function as a compatibility wrapper over
      the in-memory sink or otherwise prove existing behavior remains unchanged.

    Testing:
    - Compare dict-returning evaluation and sink-writing evaluation on identical
      small fake tensors.
    - Add a retry-path test showing the sink receives exactly the completed keys
      after an out-of-memory retry succeeds.
    - Run the existing executor and Exit tests to prove current patch output is
      unchanged.

    Covers: @streaming-full-model-materialization ac-affected-results-released.
- title: Implement an incremental safetensors artifact writer
  slug: task-full-model-incremental-writer
  priority: 1
  tags: [persistence, storage]
  spec_ref: "@streaming-full-model-materialization"
  depends_on:
    - "@task-full-model-saved-artifact-reload-spike"
  description: |
    What: Implement a safetensors artifact writer that can publish a complete
    file only after every required tensor has been written successfully.

    Why: Full saved model output needs a complete model artifact, but a simple
    all-at-once save can create a second model-sized memory spike. The writer
    needs to accept tensors incrementally while preserving safetensors validity
    and cache safety.

    How:
    - Extend the existing streaming save support or add a new persistence helper
      for predetermined tensor metadata and incremental writes.
    - Precompute each required tensor's name, shape, dtype, and byte length from
      base model metadata and the selected storage dtype.
    - Write to a temporary file in the destination directory and publish the
      final file atomically only after all required tensors have been written.
    - Validate every write against the expected key, shape, dtype, and byte size.
    - Track missing or duplicate keys and fail before publishing if the artifact
      would be incomplete or inconsistent.
    - Remove temporary files on abort or failure.

    Testing:
    - Round-trip artifacts with multiple keys and dtypes through safetensors.
    - Assert missing-key, duplicate-key, wrong-shape, and wrong-dtype cases fail
      with clear errors and do not publish a cacheable artifact.
    - Assert a failure leaves no completed artifact at the destination path.

    Covers: @streaming-full-model-materialization ac-incomplete-write-not-reused.
- title: Implement full checkpoint materialization
  slug: task-full-model-checkpoint-materialization
  priority: 2
  tags: [persistence, memory]
  spec_ref: "@streaming-full-model-materialization"
  depends_on:
    - "@task-full-model-result-sink-interface"
    - "@task-full-model-incremental-writer"
  description: |
    What: Add the materialization path that creates a complete merged diffusion
    artifact while receiving affected tensors from merge evaluation.

    Why: The full saved model option must include both affected merged weights
    and unaffected base weights without constructing a second complete state dict
    in memory.

    How:
    - Build the artifact's required key list from the base model state dict.
    - Record unaffected base weights directly to the artifact writer in a bounded
      pass that does not also retain a duplicate full-model dictionary.
    - For affected keys, write merged tensors through the result sink as merge
      evaluation produces them.
    - Ensure affected keys replace only their corresponding base entries in the
      final artifact.
    - Finalize the artifact only after all affected and unaffected keys have been
      recorded.
    - On errors, abort the writer and leave any previous valid cache artifact
      untouched.

    Testing:
    - Use a fake base model state with affected and unaffected keys.
    - Load the resulting safetensors artifact and assert unaffected keys equal
      the base tensors while affected keys equal the merged tensors.
    - Assert materialization does not expose or require a full merged_state dict.
    - Assert a write failure leaves no partially published artifact.

    Covers: @streaming-full-model-materialization ac-base-weight-bounded-copying,
      ac-affected-results-released, ac-incomplete-write-not-reused.
- title: Add mode-aware persistence metadata and cache checks
  slug: task-full-model-cache-metadata
  priority: 2
  tags: [persistence, cache]
  spec_ref: "@full-saved-model-output"
  depends_on:
    - "@task-full-model-checkpoint-materialization"
  description: |
    What: Extend persistence metadata so full saved model artifacts are safely
    distinguished from other ecaj artifacts and from older saved files.

    Why: A matching recipe hash is not enough once multiple output modes can
    write artifacts. Cache hits must only reuse artifacts whose kind and required
    metadata match the active output behavior.

    How:
    - Update metadata construction and cache checks in lib/persistence.py to
      include and verify an artifact kind for full saved model outputs.
    - Reject or recompute when an artifact has the right recipe hash but the
      wrong artifact kind, missing kind, or incompatible metadata schema.
    - Preserve protection against overwriting non-ecaj safetensors files.
    - Include enough metadata for later diagnostics, such as affected keys and
      the saved recipe identity.
    - Ensure older ecaj files without an artifact kind do not become silent cache
      hits for full saved model output.

    Testing:
    - Add persistence tests for matching kind, mismatched kind, missing kind, and
      non-ecaj files.
    - Assert cache hits skip merge work only for compatible full saved model
      artifacts.
    - Run the persistence test slice and existing save/cache tests.

    Covers: @full-saved-model-output ac-cache-reuses-artifact.
- title: Integrate full saved model output into Exit
  slug: task-full-model-exit-integration
  priority: 2
  tags: [exit-node, persistence, comfy]
  spec_ref: "@full-saved-model-output"
  depends_on:
    - "@task-full-model-saved-artifact-reload-spike"
    - "@task-full-model-chunked-evaluation-to-sink"
    - "@task-full-model-checkpoint-materialization"
    - "@task-full-model-cache-metadata"
  description: |
    What: Add a user-selectable full saved model output mode to the Exit node.

    Why: Users need an option that returns a reusable Comfy-loadable merged model
    and does not keep the merged result as the original in-memory set patch
    payload.

    How:
    - Modify nodes/exit.py to expose an output mode that includes full saved
      model output while preserving the current in-memory patch output as the
      default behavior.
    - Validate that the mode has a usable output artifact name or path before
      any expensive merge work begins.
    - On a cache hit for the current recipe and full-model artifact kind, load
      the saved artifact as a Comfy MODEL and skip GPU merge work.
    - On a cache miss, run merge evaluation through the sink-writing path,
      materialize the complete artifact, then load the artifact as a Comfy MODEL
      through the supported loading path identified by the research task.
    - Do not call the current in-memory patch installation path when this output
      mode is active.
    - Preserve existing save/cache flags or migrate them in a backwards-compatible
      way so existing workflows continue to use their prior output mode.

    Testing:
    - Unit-test that full saved model mode uses the full checkpoint
      materialization path and does not call the current in-memory patch
      installer.
    - Monkeypatch the Comfy loading path and assert it receives the completed
      artifact path after successful materialization and on cache hit.
    - Test validation errors for missing or invalid artifact names.
    - Run existing in-memory patch output tests unchanged.

    Covers: @full-saved-model-output ac-complete-artifact,
      ac-return-loaded-model, ac-cache-reuses-artifact;
      @comfy-memory-manager-compatibility ac-comfy-owns-returned-model-memory.
- title: Validate full saved model memory behavior end to end
  slug: task-full-model-memory-validation
  priority: 3
  tags: [testing, validation, memory]
  spec_ref: "@comfy-memory-manager-compatibility"
  depends_on:
    - "@task-full-model-exit-integration"
  description: |
    What: Add end-to-end-style validation showing full saved model output's
    memory and behavioral impact compared with current in-memory patch output.

    Why: The main reason to implement this option is to reduce persistent merge
    payload memory and let Comfy own the returned model's weight lifecycle. The
    plan should not be considered complete until that claim is measured.

    How:
    - Build a representative fake or local workflow that exercises multiple
      affected and unaffected diffusion model keys.
    - Run the workflow with current in-memory patch output and with full saved
      model output.
    - Assert downstream patch/application results are equivalent for affected
      keys.
    - Capture process RAM, GPU memory, and whether the returned model is managed
      by Comfy's normal model lifecycle in both Dynamic VRAM and fallback modes
      when the local environment supports them.
    - Document expected tradeoffs: full saved model output can use more disk and
      may add an artifact load step, but should remove the persistent in-memory
      patch payload for the returned MODEL.

    Testing:
    - Add automated assertions for output equivalence and cache reuse.
    - Add a manual or optional integration script for Dynamic VRAM memory
      measurement, skipped with a clear reason when ComfyUI or compatible
      hardware is not available.
    - Run pytest for persistence, executor, and Exit tests.

    Covers: @comfy-memory-manager-compatibility ac-no-dynamic-vram-opt-out,
      ac-comfy-owns-returned-model-memory, ac-measured-memory-behavior.
```

## Implementation Notes

This option is the strongest candidate for actually using ComfyUI's Dynamic
VRAM system because the merged result becomes a normal saved diffusion artifact
loaded through Comfy's own model loading path. Dynamic VRAM is implemented by
Comfy/aimdo around Comfy-managed model weights and ModelPatcher lifecycle; it is
not a general-purpose allocator for arbitrary tensors held by custom nodes.

The tradeoff is disk usage and an extra artifact load path. The plan therefore
keeps the full saved model path explicit instead of replacing the current patch
output by default. The first implementation gate is the saved-artifact reload spike: before any
bounded materialization refactor begins, the project must prove that a WIDEN
saved artifact can be loaded as a usable Comfy MODEL with parity against the
current in-memory patch output. After that gate passes, the proof point for
promotion is measured behavior under a current ComfyUI build with Dynamic VRAM
enabled: the returned model should no longer depend on the original in-memory
set patch payload, and any new load-time or write-time peak should be visible in
the validation output.
