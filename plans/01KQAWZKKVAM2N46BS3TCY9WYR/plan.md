# Checkpoint Save Reconciliation Plan

## Specs

```yaml
- title: Checkpoint-Loadable Saved Model Output
  slug: checkpoint-loadable-saved-model-output
  type: requirement
  parent: "@full-saved-model-output"
  tags: [persistence, comfy, checkpoint]
  description: |
    Saved model output produces artifacts that can be loaded as the same ComfyUI
    model kind users expect from the source workflow, including checkpoint-style
    models whose usable workflow requires more than diffusion weights alone.
  acceptance_criteria:
    - id: ac-terminal-save-executes
      given: |
        A ComfyUI workflow ends at the WIDEN Exit node and the Exit node is
        configured to save the merged model.
      when: |
        the workflow is queued without any downstream consumer of the Exit
        node's MODEL output.
      then: |
        ComfyUI treats the Exit node as an executable side-effecting output and
        attempts the save instead of rejecting the workflow because it has no
        outputs.
    - id: ac-artifact-matches-source-model-kind
      given: |
        A merge recipe starts from a checkpoint-style source model whose normal
        ComfyUI use depends on model, conditioning, and decode components.
      when: |
        the Exit node saves the merged result for that recipe.
      then: |
        the saved artifact is loadable through the appropriate checkpoint-style
        ComfyUI loader with the components required for that model kind, rather
        than being only a diffusion-weight artifact.
    - id: ac-generated-workflow-round-trip
      given: |
        A saved artifact was produced for a checkpoint-style merge recipe.
      when: |
        the artifact is loaded through the appropriate ComfyUI loader and used
        in a bounded KSampler, VAEDecode, and SaveImage-style workflow for that
        model kind.
      then: |
        the workflow reaches the image-save output without relying on the
        original in-memory WIDEN merge payload.
    - id: ac-downstream-return-remains-usable
      given: |
        A workflow both saves through the WIDEN Exit node and connects the Exit
        node's MODEL output to downstream nodes in the same queued execution.
      when: |
        the save completes successfully.
      then: |
        downstream nodes receive a usable MODEL for the saved merge result.

- title: Saved Model Artifact Safety
  slug: saved-model-artifact-safety
  type: requirement
  parent: "@full-saved-model-output"
  tags: [persistence, cache, safety]
  description: |
    Saved model output publishes and reuses only complete artifacts whose kind,
    contents, and metadata match the requested saved-model behavior.
  acceptance_criteria:
    - id: ac-missing-metadata-not-reused
      given: |
        A saved file has the same recipe identity as the current request but is
        missing required saved-model metadata.
      when: |
        saved model output checks whether it can reuse that file.
      then: |
        the file is not accepted as a cache hit for the current saved-model
        behavior.
    - id: ac-wrong-artifact-kind-not-reused
      given: |
        A saved file has the same recipe identity as the current request but was
        written for a different artifact kind.
      when: |
        saved model output checks whether it can reuse that file.
      then: |
        the file is not accepted as a cache hit for the current saved-model
        behavior.
    - id: ac-internal-format-not-checkpoint-cache
      given: |
        A saved file contains only internal diffusion-weight keys for a
        checkpoint-style source model.
      when: |
        checkpoint-style saved model output checks whether it can reuse that
        file.
      then: |
        the file is rejected or recomputed with a clear diagnostic rather than
        being treated as a usable checkpoint-style artifact.
    - id: ac-no-partial-publication
      given: |
        saved model artifact creation fails before every required component has
        been written and validated.
      when: |
        the failed save returns control to ComfyUI.
      then: |
        no newly incomplete artifact is published as a reusable saved model.
    - id: ac-existing-valid-artifact-preserved
      given: |
        a previously valid saved model artifact exists at the requested logical
        target.
      when: |
        a later save to the same logical target fails before publication.
      then: |
        the previously valid saved model artifact remains safe for later reuse.
    - id: ac-missing-components-fail-before-work
      given: |
        a checkpoint-style save is requested without the components required to
        create a loadable checkpoint-style artifact.
      when: |
        the Exit node validates the save request.
      then: |
        the workflow fails before expensive merge work or artifact publication
        begins.
    - id: ac-missing-component-diagnostic-names-requirement
      given: |
        a checkpoint-style save is requested without the components required to
        create a loadable checkpoint-style artifact.
      when: |
        the Exit node reports the validation failure.
      then: |
        the error names the missing requirement.
    - id: ac-missing-component-diagnostic-gives-guidance
      given: |
        a checkpoint-style save is requested without the components required to
        create a loadable checkpoint-style artifact.
      when: |
        the Exit node reports the validation failure.
      then: |
        the error explains how the workflow can provide the missing requirement.

- title: Live Comfy Saved Output Validation
  slug: live-comfy-saved-output-validation
  type: requirement
  parent: "@comfy-memory-manager-compatibility"
  tags: [comfy, validation, memory]
  description: |
    Real ComfyUI validation proves saved model output through the same scheduler,
    loader, memory-management mode, and downstream workflow shapes that users run.
  acceptance_criteria:
    - id: ac-explicit-real-comfy-opt-in
      given: |
        a validation harness or command can interact with a real ComfyUI
        installation, service, model directory, or API.
      when: |
        normal automated tests, CI, or automated task dispatch run.
      then: |
        the real ComfyUI validation does not run unless an operator supplies
        explicit opt-in inputs for the target environment.
    - id: ac-report-identifies-environment
      given: |
        an operator intentionally runs saved-model validation against a real
        ComfyUI environment.
      when: |
        the validation report is produced.
      then: |
        it records the ComfyUI version and active memory-management mode.
    - id: ac-report-records-save-outcome
      given: |
        an operator intentionally runs saved-model validation against a real
        ComfyUI environment.
      when: |
        the validation report is produced.
      then: |
        it records the queued save workflow shape and saved artifact
        classification.
    - id: ac-report-records-loader-downstream-outcome
      given: |
        an operator intentionally runs saved-model validation against a real
        ComfyUI environment.
      when: |
        the validation report is produced.
      then: |
        it records the loader and downstream workflow results for the saved
        artifact.
    - id: ac-report-records-cache-reuse-outcome
      given: |
        an operator intentionally runs saved-model validation against a real
        ComfyUI environment.
      when: |
        the validation report is produced.
      then: |
        it records the cache-reuse result for the saved artifact.
    - id: ac-memory-mode-not-changed-for-success
      given: |
        ComfyUI is running in a supported memory-management mode.
      when: |
        saved model output is validated in that environment.
      then: |
        success does not require disabling or changing the active memory mode;
        if the save can only succeed after changing the mode, the report records
        that as a validation failure for the original mode.
    - id: ac-primitive-tests-do-not-substitute-for-live-validation
      given: |
        lower-level tests prove artifact helpers, metadata checks, or mock model
        behavior.
      when: |
        saved model output compatibility is assessed for release.
      then: |
        those primitive tests are treated as supporting evidence only; release
        compatibility requires a live workflow validation result or an explicit
        documented deferral.
```


## Tasks

derive_from_specs: false

```yaml
- title: Rewrite full saved model output spec for requested model kind
  slug: task-checkpoint-save-rewrite-full-saved-model-output-spec
  priority: 1
  tags: [specs, persistence, checkpoint]
  spec_ref: "@full-saved-model-output"
  depends_on: []
  description: |
    What: Update @full-saved-model-output so its existing full-saved-output
    contract is model-kind aware and no longer treats diffusion weights alone as
    sufficient for checkpoint-style source workflows.

    Why: Live SDXL testing proved that a diffusion-only artifact is not a
    checkpoint-style saved model. The existing spec currently says the saved
    artifact contains every diffusion model weight and can be returned as a
    normally loaded diffusion model, which leaves room for the old internal file
    format to be treated as done.

    Exact spec changes:
    - Target spec: @full-saved-model-output.
    - Set implementation status to in_progress.
    - Replace the description with exactly:
      The Exit node can produce a complete saved model artifact for a merge
      recipe in the model kind requested by the source workflow and return a
      MODEL that behaves like a normally loaded ComfyUI model for downstream
      execution.

      For checkpoint-style source workflows, complete means the saved artifact
      remains loadable through the appropriate checkpoint-style ComfyUI loader
      with the non-diffusion components needed for normal generation and decode
      use. A diffusion-weight-only artifact is not sufficient for
      checkpoint-style saved model output.
    - Replace ac-complete-artifact with:
      given: A valid merge recipe starts from a source workflow whose normal
        ComfyUI use has a defined saved model kind.
      when: The Exit node produces full saved model output for that recipe.
      then: The saved artifact is complete for that same model kind and can be
        loaded without relying on the original in-memory WIDEN merge payload.
        For checkpoint-style source workflows, the artifact includes the
        conditioning and decode components required by the appropriate
        checkpoint-style ComfyUI loader, not only the merged diffusion weights.
    - Replace ac-return-loaded-model with:
      given: A full saved model artifact is produced or reused for the current
        recipe and requested saved model kind.
      when: The Exit node returns its MODEL output.
      then: Downstream nodes receive a MODEL representing the saved merged
        artifact through a supported return or load path for that artifact kind,
        and the result remains usable after temporary merge outputs have been
        released by the Exit node.
    - Replace ac-cache-reuses-artifact with:
      given: A saved full model artifact matches the current recipe identity,
        requested artifact kind, and required saved-model metadata.
      when: The Exit node executes with save_model enabled for that same saved
        model kind.
      then: The existing artifact is reused without recomputing the merged
        weights.
    - Replace ac-no-op-produces-full-artifact with:
      given: A valid full saved model workflow selects a recipe that does not
        change any diffusion model weights.
      when: The Exit node produces full saved model output for that recipe.
      then: Downstream nodes receive a MODEL backed by a complete saved model
        artifact for the selected recipe and requested saved model kind.
    - Replace ac-cache-reuse-is-artifact-backed with:
      given: A saved full model artifact matches the current recipe identity,
        requested artifact kind, and required saved-model metadata.
      when: The Exit node reuses cached full saved model output.
      then: The reused result comes from the saved artifact rather than from a
        process-resident affected-weight payload.
    - Keep the existing slug, title, type, tags, and relationships to
      @exit-model-persistence, @memory-management, and @exit-patch-install.
    - Do not add a new output mode or output-mode dropdown to this spec.

    Testing:
    - Run kspec validate.
    - Run kspec item get @full-saved-model-output and verify the description and
      five listed ACs exactly match the requested model-kind-aware wording.
    - Verify no remaining @full-saved-model-output AC says or implies that a
      diffusion-weight-only artifact satisfies checkpoint-style saved output.

    Covers: @full-saved-model-output ac-complete-artifact,
      ac-return-loaded-model, ac-cache-reuses-artifact,
      ac-no-op-produces-full-artifact, ac-cache-reuse-is-artifact-backed;
      @checkpoint-loadable-saved-model-output ac-artifact-matches-source-model-kind,
      ac-downstream-return-remains-usable.

- title: Rewrite Exit model persistence spec for artifact-kind-safe save_model
  slug: task-checkpoint-save-rewrite-exit-model-persistence-spec
  priority: 1
  tags: [specs, persistence, cache, checkpoint]
  spec_ref: "@exit-model-persistence"
  depends_on:
    - "@task-checkpoint-save-rewrite-full-saved-model-output-spec"
  description: |
    What: Update @exit-model-persistence so save_model remains the boolean save
    control while cache safety and checkpoint-style loadability are stated
    explicitly.

    Why: The existing spec still describes saving a fully merged state dict with
    all base model keys as a complete standalone model loadable by a standard
    loader. For SDXL/checkpoint-style workflows that is too broad and lets an
    internal diffusion-only safetensors file masquerade as a complete checkpoint.

    Exact spec changes:
    - Target spec: @exit-model-persistence.
    - Set implementation status to in_progress.
    - Replace the description with exactly:
      Opt-in save/cache for merged model output through the existing save_model
      boolean. When save_model is enabled, the Exit node writes or reuses an
      ecaj-owned saved model artifact for the current recipe and requested
      artifact kind using a safe model_name.

      For checkpoint-style source workflows, the saved artifact must be loadable
      through the appropriate ComfyUI checkpoint-style loader with the components
      required for normal generation and decode use. A diffusion-weight-only
      internal state file is not a complete checkpoint-style saved model
      artifact.

      For model kinds that are truly diffusion-only, a diffusion-weight artifact
      may be valid only when it is explicitly classified and validated as that
      artifact kind. Cache validation uses ecaj metadata including recipe
      identity, artifact kind, base model identity, and dependency fingerprints.
      The default save_model off path performs no file I/O.
    - Replace ac-2 with:
      given: save_model is enabled, model_name is provided, and the Exit node
        has the components required for the requested saved model kind.
      when: The Exit node completes the merge for the current recipe.
      then: The node publishes an ecaj-owned saved model artifact for the
        requested artifact kind using model_name. For checkpoint-style source
        workflows, the artifact is loadable through the appropriate
        checkpoint-style ComfyUI loader with the required conditioning and decode
        components, not merely as a diffusion-weight-only file.
    - Replace ac-3 with:
      given: A saved artifact exists at the expected path and save_model is
        enabled.
      when: The Exit node executes and the artifact metadata matches the current
        recipe, requested artifact kind, base model identity, and dependency
        fingerprints.
      then: The saved artifact is accepted as a cache hit, the GPU merge pipeline
        is skipped, and subsequent output behavior follows the saved-output
        contract for that artifact kind.
    - Replace ac-4 with:
      given: A saved artifact exists at the expected path but its ecaj metadata
        does not match the current recipe, requested artifact kind, base model
        identity, or dependency fingerprints.
      when: The Exit node executes with save_model enabled.
      then: The existing artifact is not accepted as a cache hit. The merge
        recomputes and publication proceeds only if the target is safe for
        ecaj-owned replacement.
    - Replace ac-6 with:
      given: A saved artifact written by the Exit node.
      when: The artifact is examined for metadata.
      then: The metadata identifies the serialized recipe tree with model objects
        replaced by stable source identities, the recipe identity hash, the
        requested artifact kind, the base model identity, and dependency
        fingerprints used for cache validation.
    - Replace ac-8 with:
      given: A saved artifact written by the Exit node for a checkpoint-style
        source workflow.
      when: The artifact is loaded through the appropriate ComfyUI
        checkpoint-style loader.
      then: The loaded result functions as a complete checkpoint-style model for
        normal downstream generation and decode use, including the merged
        diffusion weights and required conditioning and decode components,
        without relying on the original in-memory WIDEN merge payload.
    - Replace ac-10 with:
      given: save_model is enabled and artifact creation reaches publication.
      when: The Exit node writes the saved artifact to disk.
      then: Publication is atomic for the logical artifact target, so partial
        writes are not accepted as successful saved-model cache entries and do
        not replace a previously valid ecaj-owned artifact.
    - Leave ac-1, ac-5, ac-7, ac-9, ac-11, ac-12, ac-13, and ac-14 unchanged.
    - Do not add output_mode or any output-mode selector to the spec.

    Testing:
    - Run kspec validate.
    - Run kspec item get @exit-model-persistence and verify the description and
      ACs 2, 3, 4, 6, 8, and 10 match the exact wording above.
    - Verify ac-9 still protects non-ecaj files.
    - Verify no @exit-model-persistence AC treats diffusion-only loading as
      checkpoint-style completion.

    Covers: @exit-model-persistence ac-2, ac-3, ac-4, ac-6, ac-8, ac-10;
      @saved-model-artifact-safety ac-missing-metadata-not-reused,
      ac-wrong-artifact-kind-not-reused, ac-no-partial-publication,
      ac-existing-valid-artifact-preserved;
      @checkpoint-loadable-saved-model-output ac-artifact-matches-source-model-kind.

- title: Rewrite Comfy memory compatibility spec for checkpoint-loaded artifacts
  slug: task-checkpoint-save-rewrite-comfy-memory-compatibility-spec
  priority: 1
  tags: [specs, comfy, memory, validation]
  spec_ref: "@comfy-memory-manager-compatibility"
  depends_on:
    - "@task-checkpoint-save-rewrite-full-saved-model-output-spec"
  description: |
    What: Update @comfy-memory-manager-compatibility so Dynamic VRAM compatibility
    is assessed through real saved artifacts and Comfy loader/downstream paths,
    not primitive helper success.

    Why: Dynamic VRAM manages Comfy-owned model loading and inference residency.
    The compatibility claim must be about saved-output behavior running without
    changing the active Comfy memory mode and without retaining transient merge
    payloads or deep-copied model internals.

    Exact spec changes:
    - Target spec: @comfy-memory-manager-compatibility.
    - Set implementation status to in_progress.
    - Replace the description with exactly:
      Saved model output remains compatible with ComfyUI's model memory lifecycle
      across supported memory-management modes for the requested saved model
      artifact kind. For checkpoint-style saved output, compatibility requires
      that ComfyUI can load and use the saved checkpoint-style artifact through
      normal loader and downstream workflow paths without changing the active
      memory mode.
    - Replace ac-no-dynamic-vram-opt-out with:
      given: ComfyUI is running with its dynamic model memory management enabled.
      when: A workflow uses saved model output for a supported artifact kind.
      then: The workflow can run without requiring users to disable ComfyUI's
        dynamic memory management. If checkpoint-style output is requested, the
        saved artifact remains loadable through the appropriate checkpoint-style
        ComfyUI loader in that memory mode.
    - Replace ac-comfy-owns-returned-model-memory with:
      given: The Exit node has returned a MODEL after producing or reusing saved
        model output.
      when: ComfyUI loads, unloads, partially loads, or prioritizes models for a
        workflow.
      then: The returned MODEL remains compatible with ComfyUI's model memory
        lifecycle and does not require the transient WIDEN merge payload or a
        deep-copied ComfyUI model object to remain resident solely because the
        Exit node returned.
    - Replace ac-measured-memory-report with:
      given: A representative merge workflow is run in an explicitly selected
        ComfyUI memory-management mode.
      when: memory behavior is measured for patch output, first saved model
        output, and saved artifact reuse.
      then: The validation output reports the active memory mode, saved artifact
        classification, loader and downstream workflow result when applicable,
        returned-model behavior, artifact reuse behavior, and whether transient
        merge outputs remain persistently resident after the Exit node returns.
    - Replace ac-non-dynamic-memory-mode-supported with:
      given: ComfyUI is running in a supported memory-management mode where
        Dynamic VRAM is not active.
      when: A workflow uses saved model output for a supported artifact kind.
      then: The workflow can produce and return the saved model output without
        requiring Dynamic VRAM to be enabled.
    - Replace ac-memory-mode-preserved with:
      given: ComfyUI has an active memory-management mode for the workflow.
      when: A workflow uses saved model output.
      then: The workflow does not require users or the node to change that memory
        mode in order to produce, load, reuse, or return the saved model output.
    - Replace ac-memory-validation-fails-on-mode-opt-out with:
      given: A validation run is measuring saved model output under an explicitly
        selected ComfyUI memory-management mode.
      when: saved model output can only complete by changing or disabling that
        selected memory mode.
      then: The validation reports the run as failed rather than treating the
        mode change as a successful compatibility result.

    Testing:
    - Run kspec validate.
    - Run kspec item get @comfy-memory-manager-compatibility and verify the
      description and six ACs above match the required wording.
    - Verify the spec still supports Dynamic VRAM and non-Dynamic-VRAM modes.
    - Verify primitive helper tests are not described as sufficient evidence for
      release compatibility.

    Covers: @comfy-memory-manager-compatibility ac-no-dynamic-vram-opt-out,
      ac-comfy-owns-returned-model-memory, ac-measured-memory-report,
      ac-non-dynamic-memory-mode-supported, ac-memory-mode-preserved,
      ac-memory-validation-fails-on-mode-opt-out;
      @live-comfy-saved-output-validation ac-memory-mode-not-changed-for-success.

- title: Preserve scoped streaming and manual-validation specs unchanged
  slug: task-checkpoint-save-preserve-scoped-existing-specs
  priority: 1
  tags: [specs, persistence, validation]
  spec_ref: "@streaming-full-model-materialization"
  depends_on:
    - "@task-checkpoint-save-rewrite-full-saved-model-output-spec"
    - "@task-checkpoint-save-rewrite-comfy-memory-compatibility-spec"
  description: |
    What: Explicitly preserve @streaming-full-model-materialization and
    @manual-comfy-memory-validation as scoped existing specs instead of letting
    an implementer decide whether to rewrite them.

    Why: @streaming-full-model-materialization is about bounded affected-weight
    handoff and avoiding resident affected-weight caches. @manual-comfy-memory-validation
    is about opt-in/refusal/reporting for real Comfy memory validation. Neither
    spec should be expanded to carry checkpoint component sourcing or artifact
    loadability; this plan adds those requirements in the new checkpoint and live
    validation specs.

    Exact spec changes:
    - Target spec: @streaming-full-model-materialization.
      - Make no description or acceptance-criteria text changes.
      - Keep implementation status implemented.
    - Target spec: @manual-comfy-memory-validation.
      - Make no description or acceptance-criteria text changes.
      - Keep implementation status implemented.
    - Add a task note, if task notes are used, stating that these specs are
      intentionally preserved unchanged because checkpoint artifact loadability
      is covered by @checkpoint-loadable-saved-model-output,
      @saved-model-artifact-safety, rewritten @full-saved-model-output,
      rewritten @exit-model-persistence, and @live-comfy-saved-output-validation.

    Testing:
    - Run kspec validate.
    - Run kspec item get @streaming-full-model-materialization and verify all ACs
      are unchanged.
    - Run kspec item get @manual-comfy-memory-validation and verify all ACs are
      unchanged.
    - Verify no task claims these two specs alone prove checkpoint-style saved
      output readiness.

    Covers: @streaming-full-model-materialization ac-affected-results-released,
      ac-base-weight-bounded-copying, ac-incomplete-write-not-reused,
      ac-direct-artifact-handoff, ac-full-cache-avoids-resident-payload,
      ac-failed-materialization-releases-resident-payload;
      @manual-comfy-memory-validation ac-explicit-opt-in-required,
      ac-refuses-before-comfy-work-without-opt-in,
      ac-report-identifies-memory-mode;
      @live-comfy-saved-output-validation ac-primitive-tests-do-not-substitute-for-live-validation.

- title: Preserve save_model-only Exit UX and terminal scheduling
  slug: task-checkpoint-save-exit-ui-and-terminal-scheduling
  priority: 1
  tags: [exit-node, comfy, ux, persistence]
  spec_ref: "@checkpoint-loadable-saved-model-output"
  depends_on:
    - "@task-checkpoint-save-rewrite-full-saved-model-output-spec"
    - "@task-checkpoint-save-rewrite-exit-model-persistence-spec"
  description: |
    What: Make terminal WIDEN Exit saves executable and lock the public save UX to
    the existing save_model boolean rather than adopting the prior branch's
    output-mode dropdown.

    Why: Live Comfy API testing showed terminal save workflows are rejected when
    WIDENExitNode is not an output node. The prior branch also introduced
    output_mode UI concepts that are outside approved scope.

    Required implementation:
    - In nodes/exit.py, set WIDENExitNode.OUTPUT_NODE = True.
    - Keep WIDENExitNode.RETURN_TYPES exactly ("MODEL",).
    - Keep save_model as the existing BOOLEAN optional input.
    - Keep model_name, save_workflow, and enable_cache behavior as existing save
      controls unless later tasks change their cache/artifact semantics.
    - Do not add output_mode, OUTPUT_MODE_PATCHES, OUTPUT_MODE_FULL_MODEL, or any
      output-mode dropdown to WIDENExitNode.INPUT_TYPES, IS_CHANGED, or execute.
    - If any prior branch code is consulted, discard output-mode-specific code and
      tests instead of porting them.

    Testing:
    - Add or update tests/test_exit_node.py to assert WIDENExitNode.OUTPUT_NODE is
      True.
    - Add or update tests/test_exit_node.py to assert RETURN_TYPES remains
      ("MODEL",).
    - Add or update tests/test_exit_node.py to assert optional inputs include
      save_model and do not include output_mode.
    - Run the existing Exit-node tests that cover save_model=False patch output.

    Covers: @checkpoint-loadable-saved-model-output ac-terminal-save-executes;
      @exit-model-persistence ac-1, ac-5.

- title: Carry checkpoint CLIP and VAE through WIDEN Entry payload
  slug: task-checkpoint-save-component-sourcing
  priority: 1
  tags: [entry-node, exit-node, checkpoint, comfy]
  spec_ref: "@saved-model-artifact-safety"
  depends_on:
    - "@task-checkpoint-save-rewrite-exit-model-persistence-spec"
  description: |
    What: Add the concrete graph-visible component contract for checkpoint-style
    save_model: WIDEN Entry accepts optional CLIP and VAE inputs and stores them
    in the WIDEN recipe payload for WIDEN Exit to validate and use.

    Why: A WIDEN MODEL alone is not enough to write an SDXL/checkpoint-style
    artifact. The task agent must not decide how users provide companion
    components. The approved contract is explicit: users connect the MODEL, CLIP,
    and VAE outputs from CheckpointLoaderSimple-style workflows to WIDEN Entry.

    Required implementation:
    - In lib/recipe.py, add an immutable payload type for checkpoint companion
      components named CheckpointComponents or a similarly clear dataclass with:
        - clip: object
        - vae: object
    - Extend RecipeBase with an optional checkpoint_components field defaulting
      to None. Existing WIDEN recipes without checkpoint components must remain
      valid.
    - In nodes/entry.py, extend WIDENEntryNode.INPUT_TYPES so WIDEN Entry accepts:
        required:
          model: MODEL
        optional:
          clip: CLIP
          vae: VAE
    - In WIDENEntryNode.entry(), store checkpoint_components on the RecipeBase
      only from those visible optional clip and vae inputs.
    - Do not source CLIP or VAE from folder_paths, prompt JSON, current Comfy
      globals, object_info, model filename guessing, or hidden module state.
    - In nodes/exit.py, add an early validation helper for save_model=True
      checkpoint-style saves that walks to the RecipeBase and requires both
      checkpoint_components.clip and checkpoint_components.vae for SDXL
      checkpoint-style saves.
    - If CLIP, VAE, or both are missing for a checkpoint-style save, raise a
      Comfy-visible ValueError before recipe analysis, model_state_dict reads,
      GPU merge work, cache publication, temp file creation, or artifact writing.
    - The missing-component error must name the missing requirement with wording
      equivalent to: checkpoint save_model requires CLIP and VAE components.
    - The same error must give connection guidance with wording equivalent to:
      connect CheckpointLoaderSimple MODEL, CLIP, and VAE outputs to WIDEN Entry.
    - save_model=False must not require CLIP or VAE and must preserve current
      in-memory patch output behavior.
    - Do not add a mode dropdown or let the task agent choose any alternate UX.

    Testing:
    - Unit-test WIDEN Entry accepts optional CLIP and VAE and stores them on
      RecipeBase.checkpoint_components.
    - Unit-test WIDEN Entry with no CLIP/VAE still returns a valid WIDEN recipe.
    - Unit-test save_model=True checkpoint save fails before analyze_recipe,
      model_state_dict, MaterializationSink, Comfy save calls, or temp file
      creation when CLIP/VAE are missing.
    - Unit-test partial component cases: CLIP without VAE and VAE without CLIP.
    - Unit-test missing-component diagnostics name the missing requirement and
      include connection guidance.
    - Unit-test valid MODEL+CLIP+VAE proceeds to the checkpoint artifact writer
      seam without performing real Comfy checkpoint I/O.

    Covers: @saved-model-artifact-safety ac-missing-components-fail-before-work,
      ac-missing-component-diagnostic-names-requirement,
      ac-missing-component-diagnostic-gives-guidance.

- title: Implement checkpoint-aware artifact metadata and cache classification
  slug: task-checkpoint-save-artifact-cache-classification
  priority: 1
  tags: [persistence, cache, checkpoint]
  spec_ref: "@saved-model-artifact-safety"
  depends_on:
    - "@task-checkpoint-save-rewrite-exit-model-persistence-spec"
  description: |
    What: Rebuild the useful metadata and cache-classification hardening from the
    prior branch as checkpoint-aware persistence behavior.

    Why: The prior branch contains useful artifact_kind and metadata schema work,
    but checkpoint-style save_model must reject old internal/diffusion-only
    artifacts even when recipe hashes match.

    Required implementation:
    - Extend ecaj safetensors metadata so saved artifacts include:
        - __ecaj_version__
        - __ecaj_recipe__
        - __ecaj_recipe_hash__
        - __ecaj_affected_keys__ when affected-key manifests are relevant
        - __ecaj_artifact_kind__
        - base model identity metadata used for cache validation
        - dependency fingerprints used for cache validation
    - For checkpoint-style artifacts, include checkpoint-specific classification
      metadata that records the artifact kind as checkpoint and records that the
      artifact was written with checkpoint companion components.
    - In lib/persistence.py, update cache validation so a checkpoint-style
      save_model request is a cache miss when:
        - ecaj metadata is missing,
        - ecaj metadata version is unsupported,
        - artifact kind metadata is missing,
        - artifact kind does not equal checkpoint,
        - checkpoint component classification metadata is missing,
        - recipe identity does not match,
        - base model identity does not match,
        - dependency fingerprints do not match,
        - the file is an older ecaj internal-format or diffusion-only artifact.
    - Keep current manifest/header validation for any artifact kind that depends
      on key, shape, or dtype manifests; do not replace it with artifact_kind
      checks alone.
    - Preserve non-ecaj overwrite protection: an existing file at model_name that
      lacks ecaj ownership metadata must raise/refuse instead of being overwritten
      or accepted as a cache hit.
    - Consult the prior branch's lib/persistence.py only for concrete metadata
      hardening patterns; do not port output_mode behavior.

    Testing:
    - Keep or rewrite prior branch artifact-kind metadata tests in
      tests/test_persistence.py.
    - Test matching checkpoint artifact kind and matching metadata are accepted
      as cache hit.
    - Test missing artifact kind is cache miss.
    - Test wrong artifact kind is cache miss.
    - Test missing checkpoint component classification is cache miss.
    - Test unsupported ecaj version is cache miss or a clear unsupported-version
      result according to existing persistence conventions.
    - Test older internal-format artifacts with diffusion_model/noise_augmentor/
      model_sampling-only keys are rejected for checkpoint-style cache reuse.
    - Test existing non-ecaj file raises/refuses overwrite.

    Covers: @saved-model-artifact-safety ac-missing-metadata-not-reused,
      ac-wrong-artifact-kind-not-reused, ac-internal-format-not-checkpoint-cache;
      @exit-model-persistence ac-3, ac-4, ac-6, ac-9.

- title: Adopt atomic incremental writer for safe artifact publication
  slug: task-checkpoint-save-atomic-artifact-writer
  priority: 1
  tags: [persistence, checkpoint, safety]
  spec_ref: "@saved-model-artifact-safety"
  depends_on:
    - "@task-checkpoint-save-artifact-cache-classification"
  description: |
    What: Adopt or rebuild the prior branch's incremental safetensors writer as
    a low-level artifact lifecycle primitive for safe temporary writes, aborts,
    and atomic publication.

    Why: The prior branch has useful file-publication safety work. That safety
    should be retained as a concrete primitive, but it must not be mistaken for
    proving checkpoint-style output by itself.

    Required implementation:
    - Add or rebuild lib/incremental_writer.py from the prior branch as a writer
      created from a manifest of expected tensor keys, shapes, and dtypes.
    - The writer may accept tensor writes in any order.
    - Unknown keys, duplicate keys, wrong shapes, and wrong dtypes must poison or
      abort the writer so finalize cannot publish a public artifact.
    - finalize() must succeed only after every manifest key has been written.
    - Publication must use a temp file in the same directory as the target and
      atomically replace the final path only after the temp file is complete.
    - Explicit abort must remove the temp file.
    - Failed finalize, exceptions during write, and abort must leave no partial
      public artifact.
    - If a valid artifact already exists at the final target, a failed later
      write must leave that previous artifact untouched.
    - The writer is a lifecycle primitive only. The checkpoint artifact task must
      still use Comfy-compatible checkpoint save semantics for checkpoint files.

    Testing:
    - Add or adopt tests/test_incremental_writer.py.
    - Test random write order round-trip.
    - Test metadata is preserved.
    - Test missing key prevents finalize.
    - Test duplicate key aborts and prevents finalize.
    - Test unknown key aborts and prevents finalize.
    - Test wrong shape and wrong dtype abort and remove the temp file.
    - Test explicit abort removes the temp file.
    - Test failed write/finalize preserves a pre-existing valid artifact.

    Covers: @saved-model-artifact-safety ac-no-partial-publication,
      ac-existing-valid-artifact-preserved; @streaming-full-model-materialization
      ac-incomplete-write-not-reused.

- title: Adopt sink-based affected-result handoff for bounded merge memory
  slug: task-checkpoint-save-result-sink-handoff
  priority: 1
  tags: [memory, persistence, streaming]
  spec_ref: "@streaming-full-model-materialization"
  depends_on:
    - "@task-checkpoint-save-atomic-artifact-writer"
  description: |
    What: Adopt or rebuild the prior branch's result-sink evaluation handoff so
    completed affected tensors can be handed to a sink incrementally without
    accumulating a full affected-result payload.

    Why: Dynamic VRAM does not manage arbitrary WIDEN merge intermediates. The
    project still needs bounded merge-memory behavior for affected results, but
    that behavior is separate from checkpoint loadability.

    Required implementation:
    - Add or rebuild lib/result_sink.py from the prior branch with a sink protocol
      or class that receives each completed affected tensor by key.
    - Add or rebuild the chunked evaluation-to-sink path in lib/gpu_ops.py or the
      current equivalent execution layer.
    - Keep the current dict-returning patch path as a wrapper around the sink path
      or as an equivalent compatibility path so save_model=False behavior remains
      unchanged.
    - Preserve existing OOM retry behavior.
    - Preserve storage dtype conversion semantics.
    - Ensure sink outputs are CPU tensors when the existing patch path produces
      CPU tensors.
    - Ensure errors propagate and prevent sink finalization/publication.
    - Do not describe this task as proving checkpoint-loadable saved output; it
      only provides bounded affected-result handoff.

    Testing:
    - Add or adopt tests/test_result_sink.py.
    - Add or adopt tests/test_chunked_evaluation_to_sink.py against the current
      gpu_ops/executor API.
    - Test sink output equals dict-returning output for representative batches.
    - Test dtype conversion matches the existing patch path.
    - Test OOM retry still writes all expected keys.
    - Test the sink receives keys incrementally before later groups complete.
    - Test errors propagate and do not finalize the sink.
    - Test the dict-returning wrapper remains backward-compatible.

    Covers: @streaming-full-model-materialization ac-affected-results-released,
      ac-direct-artifact-handoff, ac-full-cache-avoids-resident-payload,
      ac-failed-materialization-releases-resident-payload.

- title: Save SDXL checkpoint artifacts through Comfy checkpoint semantics
  slug: task-checkpoint-save-artifact-format
  priority: 1
  tags: [persistence, checkpoint, comfy]
  spec_ref: "@checkpoint-loadable-saved-model-output"
  depends_on:
    - "@task-checkpoint-save-component-sourcing"
    - "@task-checkpoint-save-artifact-cache-classification"
    - "@task-checkpoint-save-atomic-artifact-writer"
    - "@task-checkpoint-save-result-sink-handoff"
  description: |
    What: Replace the SDXL/checkpoint-style save_model artifact writer with a
    Comfy-compatible checkpoint writer that produces artifacts loadable by
    CheckpointLoaderSimple and usable by KSampler/VAEDecode/SaveImage workflows.

    Why: The current save path writes an internal artifact with keys such as
    diffusion_model.*, noise_augmentor.*, and model_sampling.*. Live testing
    showed that file is not a valid SDXL checkpoint. Comfy-native CheckpointSave
    produced a valid checkpoint with diffusion, conditioning, and VAE/decode
    components.

    Required implementation:
    - For SDXL/checkpoint-style save_model, do not publish the current
      MaterializationSink/internal-format artifact as the checkpoint output.
    - Compute the merged diffusion MODEL using the existing WIDEN merge path and
      install the merged weights as a Comfy-compatible ModelPatcher result.
    - Save the checkpoint artifact using the same Comfy checkpoint save semantics
      used by Comfy's CheckpointSave path. The implementation must call
      comfy.sd.save_checkpoint or a project wrapper around that function, passing
      the merged MODEL plus checkpoint_components.clip and checkpoint_components.vae.
    - Write to a same-directory temp target and publish the final target only
      after the artifact is completely written, classified, and safe to reuse.
    - Include ecaj metadata from task-checkpoint-save-artifact-cache-classification,
      including artifact kind checkpoint.
    - The checkpoint artifact must contain Comfy checkpoint-style component
      content for diffusion model, conditioning/text encoder, and VAE/decode.
      Tests may assert representative prefixes produced by Comfy such as model.*,
      conditioner.*, and first_stage_model.*.
    - Internal WIDEN artifacts containing only diffusion_model.*, noise_augmentor.*,
      or model_sampling.* must be rejected as checkpoint cache hits.
    - Preserve refusal to overwrite non-ecaj files.
    - Preserve failed-publication safety: failures before final replacement must
      not clobber an existing valid target and must not leave a reusable partial
      artifact.
    - Preserve save_model=False in-memory patch output behavior.
    - Do not invent a parallel checkpoint key-mapping implementation from raw
      ModelPatcher.model_state_dict().
    - Do not add output_mode or ask the task agent to choose checkpoint vs
      diffusion output UX.

    Testing:
    - Unit-test that SDXL checkpoint save calls the Comfy checkpoint save helper
      with merged MODEL, CLIP, VAE, and ecaj metadata.
    - Unit-test artifact-kind metadata is written and required for checkpoint
      cache reuse.
    - Unit-test internal-format artifacts with diffusion_model/noise_augmentor/
      model_sampling-only keys are rejected as checkpoint cache hits.
    - Unit-test missing metadata and wrong artifact kind are rejected.
    - Unit-test failed temp save does not publish a new artifact and does not
      clobber a previous valid artifact.
    - Add a monkeypatched Comfy save/load smoke test proving the checkpoint path
      is selected without requiring real model weights.

    Covers: @checkpoint-loadable-saved-model-output ac-artifact-matches-source-model-kind;
      @saved-model-artifact-safety ac-missing-metadata-not-reused,
      ac-wrong-artifact-kind-not-reused, ac-internal-format-not-checkpoint-cache,
      ac-no-partial-publication, ac-existing-valid-artifact-preserved;
      @exit-model-persistence ac-2, ac-8, ac-10.

- title: Return save_model results without deep-copying Comfy model internals
  slug: task-checkpoint-save-returned-model
  priority: 2
  tags: [exit-node, comfy, memory]
  spec_ref: "@checkpoint-loadable-saved-model-output"
  depends_on:
    - "@task-checkpoint-save-artifact-format"
  description: |
    What: Replace the checkpoint save_model return path that deep-copies Comfy
    model internals with an explicit cache-miss/cache-hit return contract.

    Why: Live testing on current Comfy/Dynamic VRAM failed in
    _load_model_from_artifact -> deepcopy(cloned.model). Deep-copying Comfy model
    internals is not a supported ownership or memory-management boundary.

    Required return contract:
    - On SDXL/checkpoint-style save_model cache miss:
        1. Compute the merged MODEL.
        2. Save and publish a valid checkpoint artifact.
        3. Return the merged Comfy MODEL from the in-memory WIDEN merge result.
        4. Do not reload the just-saved artifact solely to produce the return
           value.
    - On SDXL/checkpoint-style save_model cache hit:
        1. Validate checkpoint metadata and artifact kind.
        2. Skip recipe analysis and WIDEN GPU merge computation.
        3. Load the checkpoint artifact through Comfy's supported checkpoint load
           path, using comfy.sd.load_checkpoint_guess_config or a project wrapper
           around that function with output_vae=True and output_clip=True.
        4. Return only the loaded MODEL as WIDEN Exit's MODEL output.
    - Remove or bypass _load_model_from_artifact for checkpoint-style save_model.
    - No checkpoint-style save_model path may call copy.deepcopy on cloned.model,
      ModelPatcher.model, diffusion_model, CLIP internals, or VAE internals.
    - If the cache-hit Comfy load path fails, raise a clear error naming the
      saved checkpoint path and do not treat the artifact as successful reuse.
    - save_model=False patch-mode return behavior remains unchanged.

    Testing:
    - Unit-test copy.deepcopy is not called anywhere in checkpoint-style
      save_model return paths; monkeypatch copy.deepcopy to raise if necessary.
    - Unit-test cache miss returns a usable merged MODEL and does not invoke the
      checkpoint loader solely for the return value.
    - Unit-test cache hit invokes Comfy's checkpoint load path and skips
      analyze_recipe/GPU merge work.
    - Unit-test loader failure on cache hit surfaces a clear diagnostic naming
      the checkpoint path.
    - Unit-test connected downstream MODEL consumers receive a non-None usable
      MODEL from both cache-miss and cache-hit paths.
    - Remove or rewrite diffusion-loader-only spike tests so they no longer prove
      checkpoint-style saved output.

    Covers: @checkpoint-loadable-saved-model-output ac-downstream-return-remains-usable,
      ac-generated-workflow-round-trip; @saved-model-artifact-safety
      ac-missing-metadata-not-reused, ac-wrong-artifact-kind-not-reused;
      @comfy-memory-manager-compatibility ac-comfy-owns-returned-model-memory.

- title: Add guarded live Comfy checkpoint-save validation harness
  slug: task-checkpoint-save-live-validation-harness
  priority: 2
  tags: [validation, comfy, memory]
  spec_ref: "@live-comfy-saved-output-validation"
  depends_on:
    - "@task-checkpoint-save-exit-ui-and-terminal-scheduling"
    - "@task-checkpoint-save-artifact-format"
    - "@task-checkpoint-save-returned-model"
  description: |
    What: Add a guarded validation harness that drives real Comfy API workflows
    for checkpoint-style save_model only when a maintainer explicitly opts in.

    Why: Existing primitive memory harnesses can pass while real Comfy scheduler,
    checkpoint loader, Dynamic VRAM mode, or KSampler downstream behavior fails.
    Release evidence must come from real workflow behavior or an explicit
    documented deferral.

    Required implementation:
    - The harness must refuse to run unless the operator provides all required
      opt-in inputs:
        - explicit opt-in environment variable or flag,
        - ComfyUI API URL,
        - source checkpoint/model name or path,
        - output/report path,
        - safe resource settings for the run.
    - Refusal must happen before importing ComfyUI modules, mutating a ComfyUI
      installation, submitting API prompts, starting processes, or writing large
      artifacts.
    - When opted in, query ComfyUI system stats and object_info to record ComfyUI
      version, active memory-management mode, and actual node class names.
    - Submit a terminal WIDEN Exit save workflow and classify whether Comfy
      accepted or rejected it for scheduler/no-output reasons.
    - Submit a checkpoint-style WIDEN save_model workflow using MODEL, CLIP, and
      VAE from the source checkpoint path.
    - Load the saved artifact through the appropriate Comfy checkpoint loader.
    - Run a bounded downstream workflow equivalent to CheckpointLoaderSimple ->
      CLIPTextEncode -> EmptyLatentImage -> KSampler -> VAEDecode -> SaveImage.
      Safe defaults are 256x256, batch_size 1, 1 step, cfg 1.0, euler/normal,
      and deterministic seed unless the operator overrides them.
    - Run or simulate a cache-reuse workflow that proves a compatible saved
      checkpoint artifact is reused without recomputing the WIDEN merge.
    - Produce report JSON recording: ComfyUI version, memory mode, terminal save
      scheduling result, saved artifact path, saved artifact classification,
      checkpoint loader result, downstream image-save result, cache-reuse result,
      and distinct failure categories for scheduling, save, loader, downstream,
      cache, and memory-mode failures.
    - Keep the real Comfy run out of normal automated test execution.

    Testing:
    - Unit-test harness refusal with missing opt-in flag.
    - Unit-test refusal with missing API URL, model name/path, or report path.
    - Unit-test refusal happens before Comfy import or prompt submission.
    - Unit-test report schema includes all required fields.
    - Unit-test mocked API classifications for scheduler/no-output rejection,
      save failure, loader failure, downstream failure, cache-reuse failure, and
      memory-mode opt-out failure.

    Covers: @live-comfy-saved-output-validation ac-explicit-real-comfy-opt-in,
      ac-report-identifies-environment, ac-report-records-save-outcome,
      ac-report-records-loader-downstream-outcome,
      ac-report-records-cache-reuse-outcome,
      ac-memory-mode-not-changed-for-success,
      ac-primitive-tests-do-not-substitute-for-live-validation.

- title: Manually validate checkpoint-style save_model in live ComfyUI
  slug: task-checkpoint-save-live-validation-manual-run
  priority: 3
  tags: [manual, validation, comfy, memory]
  spec_ref: "@live-comfy-saved-output-validation"
  depends_on:
    - "@task-checkpoint-save-live-validation-harness"
  description: |
    What: Run the guarded live validation harness against an explicitly approved
    real ComfyUI environment and record the resulting evidence.

    Why: The final release-relevant behavior depends on Comfy's live scheduler,
    checkpoint loaders, active memory mode, artifact directories, and bounded
    downstream KSampler/VAEDecode/SaveImage execution.

    Required manual run contract:
    - Do not run this task automatically. A human must explicitly authorize the
      ComfyUI API/service target, model file, output directory, report path, and
      resource limits.
    - Run the harness in the selected Comfy memory-management mode without
      changing or disabling that mode to make the test pass.
    - Validate terminal WIDEN Exit save workflow acceptance and execution.
    - Validate checkpoint-style save_model writes an artifact classified as a
      checkpoint artifact.
    - Validate the artifact loads through the appropriate Comfy checkpoint loader.
    - Validate the loaded checkpoint completes the bounded KSampler, VAEDecode,
      and SaveImage-style workflow.
    - Validate a compatible cache reuse run avoids recomputing the merge and
      returns/loads from the saved checkpoint artifact.
    - Record in task notes: report path, ComfyUI version, memory mode, model name,
      artifact path, output image path or downstream success evidence, cache
      reuse evidence, and any failures or explicit deferrals.
    - If the live run creates large artifacts, list them in task evidence and ask
      a maintainer before deleting them.

    Testing:
    - Attach or reference the live validation report produced by the harness.
    - Run focused unit and integration slices after the live run to confirm no
      temporary probes or debug instrumentation are required for normal behavior.

    Covers: @checkpoint-loadable-saved-model-output ac-terminal-save-executes,
      ac-artifact-matches-source-model-kind, ac-generated-workflow-round-trip;
      @live-comfy-saved-output-validation ac-report-identifies-environment,
      ac-report-records-save-outcome, ac-report-records-loader-downstream-outcome,
      ac-report-records-cache-reuse-outcome, ac-memory-mode-not-changed-for-success,
      ac-primitive-tests-do-not-substitute-for-live-validation.

- title: Set checkpoint-save spec statuses after live validation
  slug: task-checkpoint-save-final-spec-status-reconciliation
  priority: 3
  tags: [kspec, specs, reconciliation]
  spec_ref: "@live-comfy-saved-output-validation"
  depends_on:
    - "@task-checkpoint-save-live-validation-manual-run"
  description: |
    What: Apply an exact post-validation status map to the affected kspec specs
    and add notes superseding the earlier primitive validation baseline.

    Why: The older full-saved-model work left some specs marked implemented based
    on primitive/helper validation. Checkpoint-style saved output must not be
    marked implemented unless live checkpoint workflow evidence passes or the
    deferral is explicit.

    Exact status changes:
    - If live validation passes all checkpoint-style saved-output requirements,
      set these specs to implementation status implemented:
        - @full-saved-model-output
        - @checkpoint-loadable-saved-model-output
        - @saved-model-artifact-safety
        - @comfy-memory-manager-compatibility
        - @live-comfy-saved-output-validation
        - @exit-model-persistence
        - @streaming-full-model-materialization
        - @manual-comfy-memory-validation
    - If live validation is deferred or fails, set these specs to implementation
      status in_progress:
        - @full-saved-model-output
        - @checkpoint-loadable-saved-model-output
        - @saved-model-artifact-safety
        - @comfy-memory-manager-compatibility
        - @live-comfy-saved-output-validation
        - @exit-model-persistence
      and keep these specs implementation status implemented:
        - @streaming-full-model-materialization
        - @manual-comfy-memory-validation
    - Add notes to the prior full-saved-model plan or related tasks stating
      exactly:
        - Primitive helper validation was superseded by checkpoint-style live
          validation.
        - Diffusion-only loadability is not sufficient for checkpoint-style
          models.
        - This checkpoint-save reconciliation plan is now the authoritative
          validation baseline for checkpoint-style saved output.
    - Do not mark checkpoint-style saved output implemented if the only passing
      evidence is primitive tests, fake ModelPatcher tests, or diffusion-only
      loader success.

    Testing:
    - Run kspec validate.
    - Verify the implementation status for all eight listed specs.
    - Verify notes exist on the prior plan/tasks when the CLI supports notes for
      those records.
    - Verify no active task or completed prior-plan note claims checkpoint-style
      saved output is done solely from primitive or diffusion-only validation.

    Covers: @live-comfy-saved-output-validation ac-primitive-tests-do-not-substitute-for-live-validation.
```

## Implementation Notes

This plan supersedes the previous assumption that a saved WIDEN artifact is
acceptable for checkpoint-style workflows merely because some loader can consume
its diffusion weights. For SDXL-style checkpoints, the validated artifact must be
loadable as a checkpoint-style model and usable in a normal generation-and-decode
workflow. Diffusion-only artifacts may still be useful for model kinds that are
actually diffusion-only, but they must not masquerade as checkpoint-style saved
model output.

The user-facing control remains the existing save_model boolean. Do not add an
output-mode dropdown. The concrete graph contract for checkpoint-style saves is
that WIDEN Entry can accept visible optional CLIP and VAE inputs alongside MODEL
and carries those components through the WIDEN payload to Exit. Checkpoint-style
save_model fails early if the required components are missing.

ComfyUI Dynamic VRAM should be treated as Comfy-owned model weight lifecycle
management, not as a general allocator for arbitrary WIDEN merge tensors. The
saved-output path can benefit from Dynamic VRAM only after the artifact is
loadable through ComfyUI's normal model loading paths. Merge-time intermediates
still need bounded execution, cleanup, and artifact safety behavior.

The previous branch `plan/plan-full-saved-model-output-plan/01kq5zsk` is not a
merge target and no task asks an agent to classify or decide what to salvage.
Concrete prior-branch-derived deliverables in this plan are: checkpoint-aware
artifact metadata/cache classification, atomic incremental writer safety,
sink-based affected-result handoff, explicit rejection of the output_mode UI, and
replacement of diffusion-loader/deepcopy return behavior with checkpoint-compatible
save and return semantics.
