# Saved Model Reload Proof of Concept Plan

## Specs

```yaml
- title: Saved Merge Reload Parity
  slug: saved-merge-reload-parity
  type: feature
  tags: [persistence, memory]
  description: |
    A saved merge artifact can be used as the MODEL returned by the Exit node
    without changing the merge result observed by downstream ComfyUI nodes.
  acceptance_criteria:
    - id: ac-returned-model-origin
      given: |
        An Exit execution is configured to save a merged diffusion artifact and
        return the saved artifact as the MODEL output.
      when: |
        The merge finishes successfully.
      then: |
        The returned MODEL is constructed from the saved artifact rather than
        from in-memory set patches.
    - id: ac-merge-result-parity
      given: |
        The same recipe is executed once using in-memory patch output and once
        using saved-artifact reload output.
      when: |
        The affected model weights are compared after ComfyUI applies patches or
        loads the artifact.
      then: |
        The resulting weights are equivalent for every affected key.
    - id: ac-cache-hit-reload
      given: |
        A saved artifact already matches the current recipe identity.
      when: |
        The Exit node is executed in saved-artifact reload mode.
      then: |
        The GPU merge is skipped and the returned MODEL is loaded from the
        matching saved artifact.
```

## Tasks

derive_from_specs: false

```yaml
- title: Add saved-artifact reload proof-of-concept mode
  slug: task-poc-saved-model-reload-mode
  priority: 1
  tags: [persistence, poc]
  spec_ref: "@saved-merge-reload-parity"
  depends_on: []
  description: |
    What: Add a proof-of-concept output mode to WIDENExitNode that saves the
    merged diffusion model using the existing save_model pipeline and returns a
    MODEL loaded from the saved artifact instead of returning a clone with
    in-memory set patches.

    Why: This validates whether ComfyUI can consume WIDEN's saved output through
    its normal efficient loading path before investing in larger streaming or
    disk-backed patch refactors.

    How:
    - Modify nodes/exit.py.
    - Add an optional enum input, for example output_mode with values
      in_memory_patches and saved_model_reload. Preserve the existing default
      behavior as in_memory_patches.
    - Require save_model=True and a valid model_name when output_mode is
      saved_model_reload. Raise ValueError with a clear message if the user asks
      for saved_model_reload without a save target.
    - After the existing atomic_save(save_state, save_path, metadata) call,
      load the saved artifact as a diffusion MODEL using Comfy's loader:
        import comfy.sd
        return (comfy.sd.load_diffusion_model(save_path),)
    - In the pre-GPU cache-hit branch, if output_mode is saved_model_reload,
      skip load_affected_keys()/install_merged_patches() and return
      comfy.sd.load_diffusion_model(save_path) directly.
    - Do not attempt to optimize peak RAM in this task. This task intentionally
      proves reload compatibility while leaving the existing merged_state and
      save_state flow intact.

    Testing:
    - Add unit tests in tests/test_persistence.py or tests/test_exit_patch_install.py
      that monkeypatch comfy.sd.load_diffusion_model and assert it is called with
      the saved path when output_mode is saved_model_reload.
    - Add a cache-hit test where check_cache returns matching metadata and assert
      load_affected_keys is not called in saved_model_reload mode.
    - Add a validation test asserting saved_model_reload without save_model=True
      raises a clear ValueError.

    Covers: @saved-merge-reload-parity ac-returned-model-origin,
      ac-cache-hit-reload.

- title: Verify saved-artifact reload parity on representative tensors
  slug: task-poc-saved-model-reload-parity-tests
  priority: 2
  tags: [persistence, testing]
  spec_ref: "@saved-merge-reload-parity"
  depends_on:
    - "@task-poc-saved-model-reload-mode"
  description: |
    What: Add parity coverage proving the saved-artifact reload mode returns the
    same affected weights as the existing in-memory set-patch output path.

    Why: The proof of concept only matters if saved reload preserves merge
    semantics for downstream ComfyUI consumers.

    How:
    - Add a focused test using the existing MockModelPatcher helpers and a small
      fake merged_state.
    - Exercise install_merged_patches for the in-memory path to obtain expected
      per-key tensors.
    - Exercise the saved-model reload branch with a monkeypatched loader that
      returns a mock ModelPatcher whose state dict contains the saved tensors.
    - Compare each affected key from __ecaj_affected_keys__ for equality and
      dtype preservation.
    - If a full ComfyUI integration test harness already exists locally, add an
      optional manual verification note that runs a tiny workflow and confirms
      the returned MODEL can be fed to a sampler. Do not make GPU integration a
      required unit test.

    Testing:
    - Run the focused pytest slice for persistence/exit behavior.
    - Expected result: tests pass without requiring CUDA or ComfyUI runtime
      imports beyond monkeypatched modules.

    Covers: @saved-merge-reload-parity ac-merge-result-parity.
```

## Implementation Notes

This is intentionally a low-risk proof of concept. It does not solve current
peak RAM behavior because nodes/exit.py still accumulates merged_state and then
builds save_state before atomic_save. The only question this plan answers is:
"If WIDEN saves a merged artifact, can Exit return a Comfy-loaded MODEL from
that artifact and preserve behavior?" If this fails, the larger saved-model path
should not proceed until the loader incompatibility is understood.
