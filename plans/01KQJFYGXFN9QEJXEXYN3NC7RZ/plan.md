# Saved Model Materialization Progress

## Specs

```yaml
- title: Streaming Materialization Progress
  slug: streaming-materialization-progress
  type: requirement
  parent: "@streaming-full-model-materialization"
  tags: [persistence, progress, comfy]
  description: |
    Streaming full-model materialization reports progress for long-running saved
    artifact work so ComfyUI users can distinguish active model saving from a
    stalled Exit node execution.
  acceptance_criteria:
    - id: ac-progress-during-streaming-writes
      given: |
        A valid WIDEN saved-model workflow is materializing tensor data through
        the streaming full-model materialization path.
      when: |
        tensor data is written into the saved artifact.
      then: |
        ComfyUI-visible progress advances during the write phase.
    - id: ac-no-op-save-progress
      given: |
        A valid WIDEN saved-model workflow selects a recipe that does not change
        any source model weights.
      when: |
        the source model weights are copied into the saved artifact.
      then: |
        ComfyUI-visible progress advances during the base-weight copy phase.
    - id: ac-affected-write-progress
      given: |
        A valid WIDEN saved-model workflow produces affected tensors from one or
        more merge evaluation groups.
      when: |
        an affected tensor is handed to artifact materialization.
      then: |
        ComfyUI-visible progress advances for that tensor handoff.
    - id: ac-finalization-status-visible
      given: |
        all required tensor data has been written for a saved artifact.
      when: |
        the Exit node validates, publishes, or reloads the saved artifact.
      then: |
        the reported save status identifies the active post-write phase.
    - id: ac-cache-reuse-status-visible
      given: |
        a valid saved artifact already satisfies the current saved-model request.
      when: |
        the Exit node reuses that artifact instead of writing a new one.
      then: |
        the reported save status identifies artifact reuse as the active phase.
    - id: ac-failure-status-not-success
      given: |
        streaming full-model materialization fails before artifact publication.
      when: |
        the failure is reported to the workflow.
      then: |
        the reported save status does not present the artifact as published.
```

## Tasks

derive_from_specs: false

```yaml
- title: Report streaming saved-model materialization progress
  slug: task-streaming-materialization-progress
  priority: 1
  tags: [exit-node, persistence, progress]
  spec_ref: "@streaming-materialization-progress"
  depends_on: []
  description: |
    What:
    - Add progress reporting for WIDEN full saved model streaming
      materialization so large diffusion-model saves show visible activity while
      tensors are being written to the saved artifact.
    - Cover both diffusion-only save paths that use MaterializationSink:
      the normal merge path in WIDENExitNode._execute_diffusion_save and the
      no-op diffusion-model save path that writes the source model weights into
      a complete standalone diffusion artifact.
    - Report status for cache reuse, artifact finalization/publication, and
      artifact reload phases.
    - Preserve the existing public node interface, including the boolean
      save_model input. Do not add output-mode controls or change saved artifact
      semantics.

    Why:
    - Large standalone diffusion-model artifacts can take long enough to write
      that users cannot tell whether WIDEN is actively saving or stalled.
    - The existing merge progress covers evaluation groups, but a no-op or
      mostly-base save can spend most of its time writing source tensors with no
      visible progress.
    - Saved artifact reload behavior is already validated separately; this task
      makes the long-running save lifecycle observable without changing the
      model kind, cache, or loader contracts.

    How:
    - Inspect nodes/exit.py and lib/streaming_save.py before editing. In
      particular, review WIDENExitNode._execute_diffusion_save, the no-op
      diffusion-model save branch, and MaterializationSink.write_tensor/finalize.
    - Add an internal saved-model progress helper that can wrap ComfyUI's
      ProgressBar when available and can be exercised by unit tests without a
      running ComfyUI instance.
    - Size progress from the known materialization work for the current save:
      manifest tensor writes plus explicit phase units for open/preparation,
      finalization/publication, and returned-model reload. Cache-hit reuse should
      use a separate small progress/status path rather than pretending to write
      tensors.
    - Advance progress after each successful MaterializationSink.write_tensor
      call in the no-op diffusion save branch.
    - Advance progress after each successful base/unaffected tensor write in
      WIDENExitNode._execute_diffusion_save.
    - Advance progress when affected tensors are written through the
      streaming_evaluation_to_sink write callback in WIDENExitNode._execute_diffusion_save.
    - Replace or fold the existing evaluation-group-only ProgressBar updates in
      the diffusion saved-model path so progress is not double-counted and the
      visible total reflects the whole materialization lifecycle.
    - Emit status/log messages when entering cache reuse, artifact preparation,
      tensor writing, finalization/publication, artifact reload, and failure
      cleanup phases. The messages must not include secrets or absolute paths
      outside normal local model/output paths.
    - Keep MaterializationSink's artifact format, atomic publication behavior,
      abort behavior, metadata, and safetensors byte layout unchanged.
    - Do not run real ComfyUI validation from normal tests or automated dispatch;
      any real Comfy check must remain an explicit maintainer-invoked command.

    Testing:
    - Add focused unit tests with a fake progress object or fake ProgressBar that
      assert no-op diffusion saves advance progress while base tensors are
      written.
    - Add focused unit tests that assert normal diffusion saves advance progress
      for unaffected/base tensor writes and affected tensor handoffs.
    - Add focused unit tests that assert cache-hit reuse reports a reuse phase
      without reporting tensor-write progress.
    - Add focused unit tests that assert a materialization failure does not report
      a published/successful phase and still aborts the temporary artifact.
    - Add or update tests so ProgressBar absence outside ComfyUI remains safe.
    - Run the focused saved-model/materialization tests and the project's normal
      Python test gate before submitting.

    Covers: @streaming-materialization-progress ac-progress-during-streaming-writes,
      ac-no-op-save-progress, ac-affected-write-progress,
      ac-finalization-status-visible, ac-cache-reuse-status-visible,
      ac-failure-status-not-success.
```

## Implementation Notes

This plan intentionally adds observability to the streaming materialization
lifecycle only. It does not reopen the saved artifact kind decision, does not add
new public node controls, and does not change whether standalone diffusion
artifacts use external text encoder/CLIP/VAE components from the surrounding
workflow.

The task should remain automation-eligible after approval because it can be
validated with mocked/fake progress objects and normal unit tests. Real ComfyUI
validation is optional maintainer evidence and must remain explicitly invoked.
