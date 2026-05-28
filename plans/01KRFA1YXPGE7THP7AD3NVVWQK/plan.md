# Checkpoint Save Memory Contract Hardening

## Specs

```yaml
[]
```

This plan updates existing WIDEN saved-model specs through explicit spec-update
tasks. It intentionally does not define new spec items in this section: the
missing contracts belong to the existing memory compatibility, live validation,
and artifact safety requirements rather than to duplicate sibling specs.

The spec-update task below is the source of truth for the exact AC additions.

## Tasks

derive_from_specs: false

```yaml
- title: Add checkpoint save memory hardening ACs
  slug: task-checkpoint-save-memory-contract-ac-updates
  priority: 1
  tags: [specs, memory, checkpoint, validation]
  spec_ref: "@comfy-memory-manager-compatibility"
  depends_on: []
  description: |
    What:
    - Add explicit acceptance criteria for the checkpoint save_model=true memory
      contracts that were previously only implied by broad memory-lifecycle
      wording.
    - Update implementation status on any tightened existing spec to in_progress
      until the follow-up tasks and manual validation evidence are complete.
    - Keep the existing public WIDEN node interface unchanged; this is a contract
      hardening pass, not a new output mode or product-control change.

    Why:
    - Live SDXL checkpoint save_model=true runs exposed a failure mode where a
      checkpoint cache-miss save could release Python references yet still retain
      native anonymous CPU arenas or swap inside the ComfyUI process.
    - The current @comfy-memory-manager-compatibility ACs say the returned MODEL
      should be Comfy-owned, but they do not explicitly name the temporary
      checkpoint serialization payload as save-time state that must be released.
    - The current live validation ACs record model/memory-mode outcomes, but do
      not require repeated-run process-memory observations that distinguish RSS
      from anonymous swap where the platform exposes those fields.

    Exact spec changes:
    - Target spec: @comfy-memory-manager-compatibility.
    - Ensure implementation status is in_progress.
    - Add AC ac-checkpoint-cache-miss-releases-save-payload:
      given: |
        Checkpoint-style saved model output creates a new saved artifact for a
        cache miss.
      when: |
        the Exit node returns its MODEL output after the save.
      then: |
        the returned MODEL can be used without keeping the temporary save-time
        merge payload resident solely for serialization or return.
    - Add AC ac-checkpoint-save-failure-releases-temp-payload:
      given: |
        checkpoint-style saved model output has created temporary save-time
        merge payloads.
      when: |
        save, publication, or artifact reload fails before a usable MODEL is
        returned.
      then: |
        temporary save-time merge payloads are released before control returns
        to ComfyUI.
    - Add AC ac-report-os-memory-observations:
      given: |
        checkpoint-style saved model output creates a new saved artifact in a
        real ComfyUI process.
      when: |
        memory behavior is validated with operating-system process memory
        observation available.
      then: |
        the validation output records process resident memory and swapped
        anonymous memory at the observed checkpoint-save lifecycle points.
    - Add AC ac-repeated-cache-miss-memory-bounded:
      given: |
        repeated checkpoint-style cache-miss saves are run in the same ComfyUI
        process.
      when: |
        each run completes and the Exit node returns.
      then: |
        post-release memory observations do not show accumulating temporary
        save-time payloads across runs.
    - Target spec: @live-comfy-saved-output-validation.
    - Set implementation status to in_progress because the live report contract
      is being tightened.
    - Add AC ac-report-records-process-memory-points:
      given: |
        an operator intentionally runs checkpoint-style saved-output validation
        against a real ComfyUI environment and supplies explicit process-memory
        observation inputs.
      when: |
        the validation report is produced.
      then: |
        it records per-run process-memory observations for before execution,
        after save-time checkpoint writing when available, and after the Exit
        node returns.
    - Add AC ac-report-records-repeated-cache-miss-memory:
      given: |
        an operator intentionally runs repeated checkpoint-style cache-miss
        saved-output validation in one ComfyUI process.
      when: |
        the validation report is produced.
      then: |
        it records each run's cache-miss identity and post-return memory
        observation so accumulation across runs can be assessed.
    - Target spec: @saved-model-artifact-safety.
    - Set implementation status to in_progress because the successful-return
      contract is being tightened.
    - Add AC ac-failed-return-not-successful:
      given: |
        saved model output writes an artifact but fails before returning a usable
        MODEL for that execution.
      when: |
        the failed execution returns control to ComfyUI.
      then: |
        that execution is not reported as a successful returned saved model.

    How:
    - Use `kspec item ac add` for new ACs and `kspec item set` for status
      changes; do not create duplicate spec items.
    - After applying spec updates, run `kspec item get` for each target spec and
      verify the new AC IDs appear exactly once.
    - Run `kspec validate --refs` and `kspec validate --alignment --warnings-ok`.

    Testing:
    - This task is a metadata/spec task. Validation is kspec metadata validation,
      not Python runtime testing.
    - Do not mark implementation complete in this task; the implementation and
      validation tasks below provide the behavior/evidence for the new ACs.

    Covers: @comfy-memory-manager-compatibility
      ac-checkpoint-cache-miss-releases-save-payload,
      ac-checkpoint-save-failure-releases-temp-payload,
      ac-report-os-memory-observations,
      ac-repeated-cache-miss-memory-bounded;
      @live-comfy-saved-output-validation
      ac-report-records-process-memory-points,
      ac-report-records-repeated-cache-miss-memory;
      @saved-model-artifact-safety ac-failed-return-not-successful.

- title: Harden checkpoint save temporary payload release paths
  slug: task-checkpoint-save-temp-payload-release-hardening
  priority: 1
  tags: [exit-node, memory, checkpoint]
  spec_ref: "@comfy-memory-manager-compatibility"
  depends_on:
    - "@task-checkpoint-save-memory-contract-ac-updates"
  description: |
    What:
    - Ensure checkpoint-style save_model=true cache-miss execution releases the
      temporary save-time merge payload in both success and failure paths.
    - Ensure the returned MODEL after a successful checkpoint save is loaded from
      the saved artifact and is not the temporary checkpoint serialization clone.
    - Ensure temporary ModelPatcher-style payload containers are explicitly
      cleared before garbage collection and best-effort native heap trimming.

    Why:
    - The live failure was not only a returned-model ownership issue. A temporary
      checkpoint serialization model could keep dense set-patch tensors alive
      while cleanup ran, and freed native CPU arenas could remain committed or
      swapped by the process allocator.
    - The successful path needs to release the serialization payload before
      loading/returning the artifact-backed MODEL. Failure paths need the same
      release guarantee so a failed save, publication, or artifact reload does
      not leave a dense temporary payload resident.

    How:
    - Inspect `nodes/exit.py`, especially `WIDENExitNode._execute_checkpoint_save`,
      `_release_temporary_checkpoint_model`, `_clear_temporary_model_patch_payloads`,
      `_trim_native_heap`, `_load_checkpoint_artifact`, and the loader cleanup
      paths.
    - Keep cache-hit behavior unchanged: checkpoint cache hits should load via
      `_load_checkpoint_artifact` and must not run merge/materialization work.
    - In the checkpoint cache-miss path, track the temporary merged checkpoint
      serialization model separately from the artifact-loaded return model.
    - Release the temporary model exactly once after it has been created and is
      no longer needed, including when `save_comfy_checkpoint`, artifact
      publication, or `_load_checkpoint_artifact` raises.
    - Do not attempt to release a temporary model before it exists, and do not
      release the artifact-loaded returned model.
    - Keep `_release_temporary_checkpoint_model` best-effort for platform-specific
      native heap behavior: unpatch loaded clones, clear patch-payload
      containers, run Python GC, empty CUDA cache when available, and call native
      heap trim only where supported.
    - Clear all known temporary patch-payload containers on the ModelPatcher clone:
      `patches`, `object_patches`, and `weight_wrapper_patches` when those
      attributes exist and support `clear()`.
    - Do not change saved artifact format, metadata, cache identity, atomic
      publication semantics, or the boolean `save_model` node input.

    Testing:
    - Extend `tests/test_checkpoint_returned_model.py` with focused tests for:
      - cache-miss success order: checkpoint save, temporary-model release, then
        artifact load for the returned MODEL;
      - failure during checkpoint save releases the temporary model and propagates
        the original failure;
      - failure during artifact publication or finalization after the temporary
        model exists releases the temporary model and propagates the original
        failure;
      - failure during artifact reload releases the temporary model and propagates
        the original failure;
      - `_clear_temporary_model_patch_payloads` clears `patches`,
        `object_patches`, and `weight_wrapper_patches` without requiring all
        attributes to exist;
      - `_release_temporary_checkpoint_model` still calls GC and native heap trim
        after severing temporary patch payloads.
    - Update AC annotations in the affected tests/source comments to reference
      the new AC IDs from the spec-update task.
    - Run `pytest tests/test_checkpoint_returned_model.py tests/test_checkpoint_save_format.py`.
    - Run `pytest tests/test_real_comfy_memory_harness.py tests/test_checkpoint_validation_harness.py` if validation helper changes are also present in the same branch.
    - Run `kspec validate --refs` before submitting.

    Covers: @comfy-memory-manager-compatibility
      ac-checkpoint-cache-miss-releases-save-payload,
      ac-checkpoint-save-failure-releases-temp-payload,
      ac-comfy-owns-returned-model-memory;
      @saved-model-artifact-safety ac-failed-return-not-successful.

- title: Extend live checkpoint memory validation reporting
  slug: task-checkpoint-save-memory-validation-reporting
  priority: 1
  tags: [validation, memory, checkpoint, manual-harness]
  spec_ref: "@live-comfy-saved-output-validation"
  depends_on:
    - "@task-checkpoint-save-memory-contract-ac-updates"
    - "@task-checkpoint-save-temp-payload-release-hardening"
  description: |
    What:
    - Extend the guarded live checkpoint validation harness so an operator can
      collect process-memory evidence for repeated checkpoint cache-miss saves
      in one ComfyUI process.
    - Keep real ComfyUI validation manual/opt-in only. Normal tests, CI, and
      automated dispatch must not contact a live ComfyUI server, inspect an
      ambient process, mutate a ComfyUI install, or write large model artifacts.

    Why:
    - The observed bug required repeated runs and OS-visible memory inspection to
      diagnose. A single functional checkpoint round-trip report can pass while
      native anonymous memory or swap accumulates across cache-miss saves.
    - The validation report needs enough evidence to distinguish supported
      memory behavior from allocator retention that remains visible in the
      ComfyUI process after WIDENExit returns.

    How:
    - Inspect `scripts/manual/real_comfy_checkpoint_validation.py`,
      `scripts/manual/real_comfy_memory_validation.py`,
      `tests/test_checkpoint_validation_harness.py`, and
      `tests/test_real_comfy_memory_harness.py` before editing.
    - Add explicit optional inputs to `real_comfy_checkpoint_validation.py`:
      `--comfy-pid <pid>` for reading `/proc/<pid>/status` and
      `/proc/<pid>/smaps_rollup` on Linux, `--comfy-log-path <path>` for parsing
      WIDEN memory log labels when the operator chooses to provide a log, and
      `--repeat-cache-miss-runs <n>` for repeated cache-miss validation.
    - Do not auto-discover a ComfyUI pid, systemd unit, log file, model folder, or
      process by default. The operator must supply every live-process/log input.
    - Preserve the existing required guards:
      `COMFY_ECAJ_CHECKPOINT_VALIDATION=1`, `--run-checkpoint-validation`,
      `--comfy-api-url`, `--source-model`, and `--report-output`.
    - Extend the JSON report schema with a per-run memory section that can record
      unavailable observations explicitly. When `--comfy-pid` is supplied and
      Linux procfs fields are readable, record at least RSS and Swap from
      `/proc/<pid>/smaps_rollup`; also include Anonymous when available.
    - When `--comfy-log-path` is supplied, parse the existing WIDEN memory labels
      `after-checkpoint-save` and `after-checkpoint-temp-model-release` for the
      runs covered by the validation report. If labels are missing, record that
      the internal log observation was unavailable rather than treating missing
      logs as a harness crash.
    - Ensure repeated cache-miss runs use distinct saved artifact identities so
      the requested repetitions do not collapse into artifact cache hits. Keep
      the normal cache-reuse check as a separate reported outcome.
    - Add a validation result field that marks repeated cache-miss memory behavior
      as failed when provided post-return observations show accumulating
      temporary save-time payloads across runs. The comparison threshold may be a
      harness parameter or documented constant, but it must be recorded in the
      report.
    - Keep the harness pure-Python and safe to run without ComfyUI when guards are
      missing; refusal must happen before API prompts or live-process reads.

    Testing:
    - Extend `tests/test_checkpoint_validation_harness.py` with pure unit tests
      that do not contact ComfyUI:
      - guard refusal still happens before API calls and before process/log reads;
      - report formatting includes process-memory fields when supplied;
      - unreadable procfs/log observations are recorded as unavailable rather than
        crashing the harness;
      - repeated cache-miss report entries preserve per-run identities;
      - accumulation failure is reported when fake post-return observations exceed
        the configured threshold.
    - Add or update tests in `tests/test_real_comfy_memory_harness.py` only for
      shared pure parsing/report helpers; do not run a real ComfyUI process from
      automated tests.
    - Run `pytest tests/test_checkpoint_validation_harness.py tests/test_real_comfy_memory_harness.py`.
    - Run `pytest tests/test_checkpoint_returned_model.py` if helper annotations
      or shared memory-report structures are touched.
    - Run `kspec validate --refs` before submitting.

    Covers: @live-comfy-saved-output-validation
      ac-report-records-process-memory-points,
      ac-report-records-repeated-cache-miss-memory,
      ac-explicit-real-comfy-opt-in;
      @comfy-memory-manager-compatibility ac-report-os-memory-observations,
      ac-repeated-cache-miss-memory-bounded;
      @manual-comfy-memory-validation ac-explicit-opt-in-required,
      ac-refuses-before-comfy-work-without-opt-in.

- title: Run manual repeated checkpoint memory validation
  slug: task-checkpoint-save-memory-manual-validation-run
  priority: 2
  tags: [manual, validation, memory, checkpoint]
  spec_ref: "@live-comfy-saved-output-validation"
  depends_on:
    - "@task-checkpoint-save-memory-validation-reporting"
  description: |
    What:
    - Run the guarded live checkpoint validation harness against an explicitly
      selected local ComfyUI environment and record evidence for repeated
      checkpoint cache-miss save_model=true runs.
    - This task is manual-only in practice. It requires a maintainer-selected
      ComfyUI server, source checkpoint, process id, and optional log path.

    Why:
    - Pure tests can prove cleanup ordering and harness/report behavior, but they
      do not prove that a real ComfyUI process running a representative SDXL
      checkpoint workflow releases OS-visible native memory and avoids repeated
      cache-miss accumulation.
    - The manual evidence is the release-confidence check for the memory
      compatibility ACs tightened by this plan.

    How:
    - Do not auto-discover or start ComfyUI. Use only explicit operator inputs.
    - Ensure the current branch is installed or symlinked into the target ComfyUI
      custom_nodes environment before the run.
    - Identify the target ComfyUI API URL, source checkpoint name, process id,
      and log path if log parsing is desired.
    - Run a command in this shape, substituting explicit maintainer-selected
      values:
      COMFY_ECAJ_CHECKPOINT_VALIDATION=1 \
        python scripts/manual/real_comfy_checkpoint_validation.py \
        --run-checkpoint-validation \
        --comfy-api-url http://127.0.0.1:8188 \
        --source-model <checkpoint-name> \
        --report-output reports/checkpoint-memory-validation.json \
        --comfy-pid <pid> \
        --comfy-log-path <path-to-comfy-log> \
        --repeat-cache-miss-runs 3
    - If a log path is not available, omit `--comfy-log-path` and document that
      internal after-save/after-release log observations were unavailable.
    - If `/proc/<pid>/smaps_rollup` is not available on the platform, document the
      unavailable OS memory fields in the report rather than inventing values.
    - Add a kspec note to @comfy-memory-manager-compatibility and to this task
      summarizing the validation result, the ComfyUI version, active memory mode,
      number of cache-miss repetitions, post-return RSS/swap observations, and
      whether the run passed or was deferred.

    Testing:
    - Verify the generated report exists and includes the ComfyUI version, active
      memory mode, checkpoint save outcome, loader/downstream outcome, cache-reuse
      outcome, per-run cache-miss identities, and process-memory observations or
      explicit unavailable markers.
    - Verify the report marks the repeated cache-miss memory behavior as passed,
      failed, or explicitly deferred; do not treat missing required live evidence
      as a successful compatibility result.
    - Run `kspec validate --refs` after recording notes.

    Covers: @live-comfy-saved-output-validation
      ac-report-identifies-environment,
      ac-report-records-save-outcome,
      ac-report-records-loader-downstream-outcome,
      ac-report-records-cache-reuse-outcome,
      ac-memory-mode-not-changed-for-success,
      ac-primitive-tests-do-not-substitute-for-live-validation,
      ac-report-records-process-memory-points,
      ac-report-records-repeated-cache-miss-memory;
      @comfy-memory-manager-compatibility ac-report-os-memory-observations,
      ac-repeated-cache-miss-memory-bounded.
```

## Implementation Notes

This plan hardens the contracts around a discovered checkpoint-style
save_model=true memory failure. It does not add output-mode controls, does not
change the boolean `save_model` interface, and does not change saved artifact
semantics.

The cleanup progress/status idea from the retrospective is intentionally not in
scope. Existing progress specs already cover write, finalization, cache reuse,
and failure status. Add a separate UX plan only if post-save cleanup becomes
long-running enough that users need a visible cleanup/release phase.

The implementation tasks should remain small and ordered:

1. update the specs first so new code/tests can truthfully annotate the new ACs;
2. harden success and failure cleanup in `nodes/exit.py`;
3. extend the guarded live validation harness/report shape without running real
   ComfyUI from automated tests;
4. run the real repeated-cache-miss validation manually with explicit inputs.

Real ComfyUI validation remains opt-in. Automated workers can implement and test
the harness, but the final live evidence task requires maintainer-provided local
environment details.
