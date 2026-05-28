# Disk-Backed Patch Output Plan

## Specs

```yaml
- title: Disk-Backed Patch Output
  slug: disk-backed-patch-output
  type: feature
  parent: "@exit-node"
  tags: [persistence, memory, comfy]
  description: |
    The Exit node can return a patched MODEL that can supply merged affected
    weights without retaining all merged affected weights in process memory after
    the Exit node returns.
  acceptance_criteria:
    - id: ac-return-patched-model
      given: |
        A merge recipe produces affected diffusion model weights.
      when: |
        The Exit node runs in disk-backed patch output mode.
      then: |
        Downstream nodes receive a patched MODEL that can provide the merged
        weight for each affected key when that weight is requested.
    - id: ac-no-resident-patch-payload
      given: |
        The disk-backed patch MODEL has been returned.
      when: |
        memory behavior is observed after the Exit node returns and before
        downstream execution requests affected weights.
      then: |
        The returned MODEL does not require all merged affected weights to remain
        resident in process memory.
    - id: ac-downstream-merge-equivalence
      given: |
        The same recipe is executed with the current in-memory patch output and
        with disk-backed patch output.
      when: |
        Downstream execution asks for the affected weights.
      then: |
        The resulting affected weights are equivalent for every affected key.

- title: Disk Patch Artifact Lifecycle
  slug: disk-patch-artifact-lifecycle
  type: requirement
  parent: "@disk-backed-patch-output"
  tags: [persistence, cache]
  description: |
    Disk-backed patch artifacts remain valid for as long as returned patch
    objects may need them and are reused only when they match the current recipe
    and output behavior.
  acceptance_criteria:
    - id: ac-artifact-reuse
      given: |
        A disk-backed patch output artifact matches the current recipe identity
        and requested output behavior.
      when: |
        The Exit node runs in disk-backed patch output mode.
      then: |
        The existing patch artifact is reused without recomputing affected
        tensors.
    - id: ac-artifact-missing-error
      given: |
        A returned disk-backed patch MODEL needs an artifact that is no longer
        available.
      when: |
        Downstream execution asks for an affected weight from that artifact.
      then: |
        Patch application fails with clear guidance identifying the missing
        artifact and affected key.
    - id: ac-artifact-mode-is-distinct
      given: |
        A saved artifact exists for a different output behavior.
      when: |
        Disk-backed patch mode checks for a reusable cache artifact.
      then: |
        The artifact is not accepted as a disk-backed patch cache hit.

- title: Streaming Affected Patch Materialization
  slug: streaming-affected-patch-materialization
  type: requirement
  parent: "@disk-backed-patch-output"
  tags: [persistence, memory]
  description: |
    Disk-backed patch artifact creation keeps affected merge output memory
    bounded while writing only the affected weights required by the patch output.
  acceptance_criteria:
    - id: ac-affected-results-released
      given: |
        A merge produces affected weights across multiple groups of model keys.
      when: |
        Disk-backed patch mode records the affected patch artifact.
      then: |
        The operation can complete without requiring all affected weights to
        remain resident until every later group has been evaluated.
    - id: ac-only-affected-keys-stored
      given: |
        A merge recipe affects a subset of the base model keys.
      when: |
        Disk-backed patch mode finalizes the patch artifact.
      then: |
        The patch artifact contains the affected keys needed for patch output and
        does not require unaffected base keys to be stored in that artifact.

- title: Dynamic Memory Compatibility for Disk Patches
  slug: disk-patch-dynamic-memory-compatibility
  type: requirement
  parent: "@disk-backed-patch-output"
  tags: [comfy, memory, compatibility]
  description: |
    Disk-backed patch output remains compatible with ComfyUI's model memory
    lifecycle and clearly reports unsupported ComfyUI patch paths when
    compatibility is not available.
  acceptance_criteria:
    - id: ac-no-dynamic-vram-opt-out
      given: |
        ComfyUI is running with its dynamic model memory management enabled.
      when: |
        A workflow uses disk-backed patch output.
      then: |
        The workflow can run without requiring users to disable ComfyUI's dynamic
        memory management.
    - id: ac-patch-materializes-on-demand
      given: |
        A disk-backed patch MODEL has been returned.
      when: |
        Downstream execution requests one affected weight.
      then: |
        The returned MODEL provides only the affected weight needed for that
        request rather than preloading every affected weight into process memory.
    - id: ac-memory-behavior-is-measured
      given: |
        The same representative merge workflow is run with current in-memory
        patch output and disk-backed patch output.
      when: |
        memory behavior is measured under ComfyUI with dynamic memory management
        enabled.
      then: |
        The validation output reports the active ComfyUI memory mode, peak RAM,
        peak GPU memory, whether affected weights remain persistently resident
        after the Exit node returns, and the temporary memory behavior observed
        while downstream execution requests affected weights; the validation
        fails if disk-backed patch output requires disabling ComfyUI's dynamic
        memory management or preloads every affected weight into process memory
        before downstream execution requests them.
```

## Tasks

derive_from_specs: false

```yaml
- title: Update Exit output-mode specs for disk-backed patch behavior
  slug: task-disk-patch-update-existing-specs
  priority: 1
  tags: [specs, exit-node, patching]
  spec_ref: "@disk-backed-patch-output"
  depends_on: []
  description: |
    What: Update existing project specs so the current set-patch Exit behavior is
    scoped as the default in-memory output mode and disk-backed patch output is a
    distinct supported output behavior.

    Why: Existing specs such as @exit-node, @exit-patch-install,
    @exit-model-persistence, and @memory-management predate multiple output
    modes. They currently make set patches look universal for Exit, which can
    conflict with a mode that supplies affected weights without retaining all
    affected merged tensors in memory after the Exit node returns.

    How:
    - Review existing specs for Exit return behavior, patch installation,
      persistence/cache behavior, and memory management.
    - Scope set-patch requirements to the in-memory patch output mode.
    - Add or refine specs for disk-backed patch output, artifact lifecycle,
      bounded affected-result materialization, and Comfy memory compatibility.
    - Add relationships from disk-backed patch requirements to existing
      persistence and memory-management specs where supported by kspec.
    - Keep spec wording behavioral: describe downstream MODEL behavior, cache
      reuse, clear errors, and memory observations rather than internal adapter,
      writer, or artifact-kind implementation details.

    Testing:
    - Run kspec validation or plan import checks after the spec edits.
    - Confirm the resulting specs do not contain contradictory requirements for
      the same Exit output mode.

    Covers: @disk-backed-patch-output ac-return-patched-model,
      ac-no-resident-patch-payload; @disk-patch-dynamic-memory-compatibility
      ac-no-dynamic-vram-opt-out.
- title: Validate Comfy patching and Dynamic VRAM behavior for disk-backed patches
  slug: task-disk-patch-dynamic-vram-spike
  priority: 1
  tags: [research, comfy, memory]
  spec_ref: "@disk-patch-dynamic-memory-compatibility"
  depends_on: []
  description: |
    What: Prove how a custom disk-backed patch object behaves when ComfyUI
    applies patches with Dynamic VRAM enabled and document any constraints before
    building the output mode.

    Why: Dynamic VRAM manages Comfy-owned model weights. Disk-backed patches keep
    merged tensors outside the normal model weight artifact until a patch is
    applied, so they may be compatible with Dynamic VRAM without directly using
    it for the patch artifact payload. The implementation must know whether
    custom patch objects are supported in the current Comfy patch path and what
    temporary memory peak they introduce.

    How:
    - Inspect the current ComfyUI patch API used by ModelPatcher for patch data
      and weight adapters.
    - Build a minimal local patch object that returns a replacement tensor for a
      fake or small real model weight only when that key is applied.
    - Run the patch through ComfyUI with Dynamic VRAM enabled when the local
      environment supports it, and through the fallback patcher when it does not.
    - Confirm whether the patch can participate in normal downstream execution
      without requiring Dynamic VRAM to be disabled.
    - Measure whether patch application reads one affected tensor at a time and
      whether temporary CPU/GPU memory is released after each requested key.
    - Record any Comfy version or patch API assumptions that implementation tasks
      must enforce with compatibility checks or clear errors.

    Testing:
    - Add a small optional integration script or test that prints the active
      Comfy patcher class, applies the minimal patch, and reports per-key memory
      observations.
    - The script should skip with a clear reason when ComfyUI or compatible
      Dynamic VRAM hardware is unavailable.

    Covers: @disk-patch-dynamic-memory-compatibility ac-no-dynamic-vram-opt-out,
      ac-patch-materializes-on-demand, ac-memory-behavior-is-measured.
- title: Introduce or reuse a bounded merge result sink for affected patch output
  slug: task-disk-patch-result-sink-foundation
  priority: 1
  tags: [persistence, memory]
  spec_ref: "@streaming-affected-patch-materialization"
  depends_on: []
  description: |
    What: Ensure the merge evaluation path can hand off each completed affected
    tensor to a consumer without retaining all affected tensors in one resident
    dictionary.

    Why: Disk-backed patch output only reduces memory if affected weights are
    recorded to the patch artifact as they are produced and local references can
    be released before later groups are evaluated.

    How:
    - Inspect the repository for an existing result sink or sink-writing merge
      evaluation helper.
    - If a compatible bounded sink already exists, extend or reuse it for
      affected patch artifacts.
    - If none exists, add a pure Python/PyTorch sink abstraction with
      write_tensor(key, tensor), finalize(), and abort() behavior plus an
      in-memory sink preserving the current patch output behavior.
    - Add or reuse sink-writing chunk evaluation that preserves current dtype,
      out-of-memory retry, system memory error, and cleanup behavior.
    - Keep the sink layer free of ComfyUI imports so it can be tested without a
      running ComfyUI instance.

    Testing:
    - Compare sink-written outputs against the current dict-returning evaluation
      path for small fake tensors.
    - Add a retry-path test showing every affected key is recorded exactly once
      after a recoverable out-of-memory path.
    - Run existing executor and Exit tests to confirm current output behavior is
      unchanged.

    Covers: @streaming-affected-patch-materialization ac-affected-results-released.
- title: Implement affected-key patch artifact materialization
  slug: task-disk-patch-affected-artifact
  priority: 1
  tags: [persistence, storage]
  spec_ref: "@streaming-affected-patch-materialization"
  depends_on:
    - "@task-disk-patch-result-sink-foundation"
  description: |
    What: Implement the persistence path that writes only affected merged weights
    to a disk-backed patch artifact.

    Why: Disk-backed patch output should avoid both full checkpoint disk usage
    and persistent in-memory storage of the merged patch tensors.

    How:
    - Build expected key, shape, dtype, and byte-length metadata for the affected
      keys before merge evaluation begins.
    - Use an existing incremental safetensors writer if one exists; otherwise add
      a focused writer that publishes only after every affected key is written
      successfully.
    - During sink-writing evaluation, write each affected merged tensor to the
      patch artifact and release local references after the write succeeds.
    - Store metadata that identifies the recipe, artifact kind, affected key
      list, and the shape/dtype information needed to recreate patch objects on
      a cache hit.
    - Fail clearly when a tensor has an unexpected key, shape, dtype, or byte
      length.
    - Abort and remove temporary files on failure.

    Testing:
    - Write a patch artifact with a subset of fake base keys and assert loading
      the artifact exposes only that subset.
    - Assert finalization fails if an affected key was never written.
    - Assert wrong shape/dtype writes fail with clear errors.
    - Assert aborted or failed writes do not publish a cacheable artifact.

    Covers: @streaming-affected-patch-materialization ac-only-affected-keys-stored,
      ac-affected-results-released.
- title: Implement disk patch cache semantics
  slug: task-disk-patch-cache-semantics
  priority: 2
  tags: [persistence, cache]
  spec_ref: "@disk-patch-artifact-lifecycle"
  depends_on:
    - "@task-disk-patch-affected-artifact"
  description: |
    What: Extend persistence metadata and cache checks for disk-backed patch
    artifacts.

    Why: Disk-backed patch mode must reuse matching artifacts without accepting
    full saved models, older affected-key files, or incompatible metadata as
    equivalent cache hits.

    How:
    - Update lib/persistence.py metadata helpers to include an artifact kind for
      disk-backed patch outputs.
    - Add or update cache-checking behavior so callers can require a specific
      artifact kind and metadata schema.
    - Store the affected key list plus each key's expected shape and dtype so
      patch objects can be recreated after a cache hit.
    - Reject or recompute when recipe hash matches but artifact kind, key list,
      shape metadata, dtype metadata, or schema version is incompatible.
    - Preserve existing refusal to overwrite non-ecaj files.
    - Emit clear diagnostics that distinguish cache miss, cache incompatibility,
      and unsafe overwrite cases.

    Testing:
    - Cache hit with matching disk patch metadata installs patch objects and
      skips merge work.
    - Matching recipe hash with full-model artifact kind is rejected for
      disk-backed patch mode.
    - Missing shape/dtype metadata produces a clear incompatibility error or
      forced recompute, according to the behavior chosen in code.
    - Non-ecaj files are still protected from overwrite.

    Covers: @disk-patch-artifact-lifecycle ac-artifact-reuse,
      ac-artifact-mode-is-distinct.
- title: Add disk-backed patch object
  slug: task-disk-patch-object
  priority: 1
  tags: [patching, persistence, comfy]
  spec_ref: "@disk-backed-patch-output"
  depends_on:
    - "@task-disk-patch-dynamic-vram-spike"
    - "@task-disk-patch-affected-artifact"
  description: |
    What: Implement the patch object that supplies a merged replacement tensor
    for one affected key by reading it from the patch artifact at patch
    application time.

    Why: The returned MODEL must behave like the current patched clone while the
    patch list itself no longer holds every merged tensor in memory.

    How:
    - Add a focused module for disk-backed patch objects.
    - Use the Comfy patch/weight-adapter API proven by the research task, with a
      small fallback base for unit tests that run without ComfyUI installed.
    - Store only durable identifiers and expected metadata on the patch object:
      artifact path, key, expected shape, and expected dtype.
    - When Comfy asks for the patched weight, open or access the artifact, load
      that key, validate shape and dtype, move it to the requested device/dtype,
      and return the replacement weight according to the patch API contract.
    - If the artifact path is missing, the key is absent, or metadata is
      incompatible, raise a RuntimeError naming the artifact and key and giving
      recovery guidance.
    - Start with the simplest correct file access strategy. Add handle caching
      only after profiling proves repeated open cost matters and tests cover
      cleanup.

    Testing:
    - Unit-test shape reporting and replacement-weight behavior against a
      temporary safetensors artifact.
    - Unit-test missing artifact, missing key, and incompatible metadata errors.
    - Unit-test that the patch object does not keep the full tensor payload in an
      instance field after construction.

    Covers: @disk-backed-patch-output ac-return-patched-model,
      ac-no-resident-patch-payload, ac-downstream-merge-equivalence;
      @disk-patch-artifact-lifecycle ac-artifact-missing-error;
      @disk-patch-dynamic-memory-compatibility ac-patch-materializes-on-demand.
- title: Install disk-backed patches from Exit
  slug: task-disk-patch-exit-installation
  priority: 2
  tags: [exit-node, patching]
  spec_ref: "@disk-backed-patch-output"
  depends_on:
    - "@task-disk-patch-object"
    - "@task-disk-patch-cache-semantics"
  description: |
    What: Add an Exit path that returns a cloned patched MODEL whose affected
    keys use disk-backed patch objects instead of in-memory set patch tensors.

    Why: This preserves the current Exit contract of returning a MODEL while
    moving the merged payload out of the persistent in-memory patch list.

    How:
    - Modify nodes/exit.py to expose a disk-backed patch output mode while
      preserving the current in-memory patch output as the default.
    - On a cache miss, run affected-key merge evaluation through the bounded
      result sink, finalize the patch artifact, create one disk-backed patch
      object per affected key, and install those patch objects on a cloned model
      patcher.
    - On a cache hit for a compatible disk-patch artifact, recreate patch objects
      from artifact metadata and skip GPU merge work.
    - Do not call the current in-memory patch installation helper when
      disk-backed patch output is active.
    - Ensure the patch payload on the clone contains patch objects or equivalent
      lightweight descriptors rather than replacement tensors.
    - Keep existing save/cache controls backwards-compatible for workflows that
      do not select disk-backed patch output.

    Testing:
    - Unit-test that disk-backed patch mode does not call the current
      in-memory patch installer.
    - Unit-test that the clone receives one lightweight patch object for each
      affected key.
    - Unit-test that patch payload inspection shows no full merged tensor tuple
      retained in memory.
    - Run existing in-memory patch output tests unchanged.

    Covers: @disk-backed-patch-output ac-return-patched-model,
      ac-no-resident-patch-payload; @disk-patch-artifact-lifecycle
      ac-artifact-reuse.
- title: Validate disk-backed patch equivalence and lifecycle
  slug: task-disk-patch-equivalence-validation
  priority: 3
  tags: [testing, validation]
  spec_ref: "@disk-backed-patch-output"
  depends_on:
    - "@task-disk-patch-exit-installation"
  description: |
    What: Add end-to-end-style tests for disk-backed patch output against the
    existing in-memory patch output.

    Why: Disk-backed patches are only acceptable if downstream execution receives
    equivalent weights and lifecycle failures are understandable to users.

    How:
    - Build a small fake base model state with multiple affected and unaffected
      keys.
    - Run the current in-memory patch output and record expected affected weights
      after patch application.
    - Run disk-backed patch output using a temporary artifact and apply each
      disk-backed patch object to the corresponding fake weight.
    - Compare affected results for equality and confirm unaffected keys are not
      present in the patch artifact.
    - Delete the patch artifact after the MODEL is returned and confirm applying
      a patch raises a RuntimeError naming the missing artifact and key.
    - Confirm that a compatible cache hit recreates lightweight patch objects
      without recomputing merged tensors.

    Testing:
    - Run pytest for disk-backed patch, persistence, executor, and Exit tests.
    - Existing in-memory patch output tests should continue to pass unchanged.

    Covers: @disk-backed-patch-output ac-downstream-merge-equivalence;
      @disk-patch-artifact-lifecycle ac-artifact-missing-error,
      ac-artifact-reuse.
- title: Measure disk-backed patch Dynamic VRAM impact
  slug: task-disk-patch-memory-validation
  priority: 3
  tags: [testing, validation, memory, comfy]
  spec_ref: "@disk-patch-dynamic-memory-compatibility"
  depends_on:
    - "@task-disk-patch-equivalence-validation"
  description: |
    What: Measure disk-backed patch output under a current ComfyUI build with
    Dynamic VRAM enabled and compare it with current in-memory patch output.

    Why: Disk-backed patches may be compatible with Dynamic VRAM while not being
    directly managed by it. The implementation should ship with evidence of the
    real memory tradeoff: less persistent patch payload, plus any temporary cost
    while each affected weight is read and applied.

    How:
    - Run a representative merge workflow with current in-memory patch output and
      disk-backed patch output.
    - Capture process RAM, GPU memory, and whether Dynamic VRAM is active for the
      base model in the local ComfyUI environment.
    - Confirm disk-backed patch output does not require the
      --disable-dynamic-vram workaround.
    - Confirm affected tensors are read on demand rather than loaded as a full
      affected-key payload at MODEL return time.
    - Document expected limitations: base model weights may be Comfy-managed by
      Dynamic VRAM, while the disk patch artifact payload is managed by this
      project's patch object and may create temporary tensors during application.

    Testing:
    - Add an optional integration script or documented test command that prints
      the active Comfy memory mode, memory measurements, and pass/fail summary.
    - Skip with a clear reason when ComfyUI, compatible NVIDIA/PyTorch support,
      or test models are unavailable.
    - Run the normal unit test suite regardless of local Dynamic VRAM support.

    Covers: @disk-patch-dynamic-memory-compatibility ac-no-dynamic-vram-opt-out,
      ac-patch-materializes-on-demand, ac-memory-behavior-is-measured.
```

## Implementation Notes

This option is the best fit when preserving the current WIDEN Exit contract is
more important than producing a standalone full model file. It should use less
disk than a full saved model because only affected weights are stored. It should
also remove the current persistent in-memory set patch payload if patch objects
hold only artifact references and per-key metadata.

The Dynamic VRAM fit is weaker than the full saved model option. ComfyUI's
Dynamic VRAM/aimdo machinery primarily manages Comfy-owned model weights and the
ModelPatcher lifecycle. A disk-backed patch artifact is external to that system;
its benefit comes from lazy per-key materialization, not from aimdo managing the
patch artifact itself. Treat this path as experimental until the validation task
shows that Comfy's current patch API and Dynamic VRAM mode apply the custom patch
without requiring users to disable Dynamic VRAM and without introducing an
unacceptable per-weight memory spike.
