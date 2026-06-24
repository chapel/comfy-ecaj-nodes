# Krea 2 Architecture Support Plan

## Context

This draft adds first-class Krea 2 support to WIDEN / comfy-ecaj-nodes without relying on non-project local file paths in the plan. Local checkpoints and LoRA files may be used for optional validation, but task agents should receive those as explicit operator-provided inputs rather than hard-coded assumptions.

## Resources

- [Krea 2 technical report deep dive](./resources/references/krea2-technical-report-deep-dive.md) — imported project-owned copy of the Obsidian research note based on the public Krea report.
- [Krea 2 implementation reference](./resources/references/krea2-implementation-reference.md) — project-owned reference with public GitHub/Hugging Face links, planning facts from header/source inspection, and implementation guardrails.

Public sources referenced by the imported implementation reference include:

- Krea report: <https://www.krea.ai/blog/krea-2-technical-report>
- ComfyUI Krea 2 diffusion model: <https://github.com/Comfy-Org/ComfyUI/blob/2a610155821d670a2d8047e654e5fce96b790eb5/comfy/ldm/krea2/model.py>
- ComfyUI Krea 2 text encoder: <https://github.com/Comfy-Org/ComfyUI/blob/2a610155821d670a2d8047e654e5fce96b790eb5/comfy/text_encoders/krea2.py>
- ai-toolkit Krea 2 implementation: <https://github.com/ostris/ai-toolkit/blob/724e67d63428a7daddc77355b88d90fe99ea9fd2/extensions_built_in/diffusion_models/krea2/krea2.py>
- ai-toolkit Krea 2 MMDiT source: <https://github.com/ostris/ai-toolkit/blob/724e67d63428a7daddc77355b88d90fe99ea9fd2/extensions_built_in/diffusion_models/krea2/src/mmdit.py>
- ai-toolkit LoKr helper: <https://github.com/ostris/ai-toolkit/blob/724e67d63428a7daddc77355b88d90fe99ea9fd2/toolkit/models/lokr.py>
- Comfy-Org Krea 2 LoRAs: <https://huggingface.co/Comfy-Org/Krea-2/tree/main/loras>
- Krea RetroAnime LoRA: <https://huggingface.co/krea/Krea-2-LoRA-retroanime/tree/main>

## Scope

In scope:

- Detect Krea 2 diffusion models as `krea2` from Krea-specific structural evidence.
- Route Krea 2 recipes through Krea-compatible loader, LoRA, block/layer, and validation paths.
- Support public Krea 2 LoRA families without requiring users to manually rename LoRA tensors.
- Provide Krea 2 block/layer controls that expose meaningful user-facing control groups.
- Add Krea-specific tests and optional smoke evidence as task validation, not as a separate Krea-only validation spec.

Out of scope for this plan:

- A general project-wide validation-gate taxonomy; that belongs to the separate trait/taxonomy planning work.
- A LoRA conversion/save utility for rewriting public LoRAs into another key format.
- Hard-coded references to local checkpoint paths.
- A blanket ban on FP8 or quantized inputs. Precision/quantization behavior should be compatibility-oriented: like-with-like should be allowed when the merge path supports it, and mixed or unsupported precision combinations should warn or fail explicitly.

## Specs

```yaml
- title: Krea 2 Architecture Support
  slug: krea2-architecture-support
  type: feature
  parent: "@widen"
  tags: [krea2, architecture]
  description: |
    WIDEN treats Krea 2 as a supported architecture family with its own observable
    model signature and recipe behavior. Krea 2 support is based on Krea-specific
    model features and compatibility rules rather than reusing another
    architecture's assumptions.
  acceptance_criteria:
    - id: ac-krea2-detected-from-krea-signature
      given: |
        A ComfyUI model exposes Krea 2 structural evidence such as Krea denoiser
        blocks, Krea text-fusion structures, and Krea-specific structural
        projections.
      when: |
        the WIDEN Entry node snapshots the model for a recipe.
      then: |
        the recipe records the base architecture as `krea2` only when the observed
        signature satisfies the documented Krea 2 recognition rules.
    - id: ac-ambiguous-or-unknown-architecture-rejected
      given: |
        A model lacks a complete supported architecture signature or presents
        conflicting evidence for multiple supported architectures.
      when: |
        the WIDEN Entry node attempts to create a recipe.
      then: |
        recipe creation fails before merge execution and reports the architecture
        evidence that prevented a confident supported-architecture decision.
    - id: ac-krea2-recipe-uses-krea-compatible-paths
      given: |
        A recipe records its base architecture as `krea2`.
      when: |
        the recipe applies architecture-sensitive behavior such as LoRA loading,
        block/layer weighting, full-model access, or saved-output compatibility.
      then: |
        the behavior is selected from Krea 2-compatible rules whose supported
        feature set is documented and testable for Krea 2 inputs.

- title: Krea 2 LoRA Package Compatibility
  slug: krea2-lora-package-compatibility
  type: feature
  parent: "@lora-loaders"
  tags: [krea2, lora]
  description: |
    Users can apply supported public Krea 2 LoRA packages to Krea 2 WIDEN recipes
    without hand-renaming tensor keys, and incompatible packages fail before a
    partial or misleading merge is produced.
  acceptance_criteria:
    - id: ac-supported-krea2-lora-packages-load
      given: |
        A user selects a Krea 2 LoRA package from a supported public Krea 2 LoRA
        family for a Krea 2 recipe.
      when: |
        the LoRA node and Exit executor apply the recipe.
      then: |
        the LoRA contribution is applied to the intended Krea 2 weight groups
        without requiring the user to manually edit package tensor names.
    - id: ac-lora-compatibility-is-complete-or-rejected
      given: |
        A selected Krea 2 LoRA package contains tensor groups outside the supported
        compatibility rules for the current Krea 2 recipe.
      when: |
        the package is checked before merge execution publishes a result.
      then: |
        WIDEN either applies every required supported group or fails with a report
        of the unsupported or incompatible groups; it does not silently publish a
        partial merge as successful.
    - id: ac-krea2-lora-strength-controls-are-stable
      given: |
        A Krea 2 LoRA is supported for the current recipe.
      when: |
        the user changes LoRA strength or combines the LoRA with block/layer
        controls.
      then: |
        the resulting recipe changes only through the documented strength and
        block/layer semantics, with deterministic behavior across repeated runs.

- title: Krea 2 Block and Layer Controls
  slug: krea2-block-and-layer-controls
  type: feature
  parent: "@per-block-control"
  tags: [krea2, block-config, layer-controls]
  description: |
    Krea 2 recipes expose meaningful user controls for Krea 2 model regions and
    layer categories, while preserving deterministic behavior for keys that do
    not belong to a user-adjustable group.
  acceptance_criteria:
    - id: ac-main-model-regions-are-controllable
      given: |
        A Krea 2 recipe includes main denoiser model regions that are eligible for
        per-region merge or LoRA weighting.
      when: |
        the user configures Krea 2 block controls.
      then: |
        changes to a main-region control affect only the corresponding Krea 2
        main model region and leave unrelated regions at their configured values.
    - id: ac-text-fusion-regions-are-controllable
      given: |
        A Krea 2 recipe includes Krea text-fusion regions that are eligible for
        per-region merge or LoRA weighting.
      when: |
        the user configures Krea 2 text-fusion controls.
      then: |
        changes to a text-fusion control affect the corresponding Krea text-fusion
        region rather than being collapsed into an unrelated main-model or
        generic catch-all control.
    - id: ac-layer-category-controls-are-controllable
      given: |
        A Krea 2 recipe contains weights belonging to supported layer categories
        such as attention, feed-forward, normalization, embedding/projection, or
        structural/fallback groups.
      when: |
        the user configures Krea 2 layer-category controls.
      then: |
        each supported category modifies only its documented category of Krea 2
        weights, and unsupported categories remain deterministic and documented.
```

## Tasks

derive_from_specs: false

```yaml
- title: Add Krea 2 architecture recognition and recipe routing
  slug: task-krea2-architecture-recognition-routing
  priority: 1
  tags: [krea2, architecture, entry-node, routing]
  spec_ref: "@krea2-architecture-support"
  resource_refs:
    - krea2-technical-report-deep-dive
    - krea2-implementation-reference
  depends_on: []
  description: |
    What: Add first-class `krea2` architecture recognition and route Krea 2 recipes
    through Krea-compatible architecture paths.

    Resources:
    - [Krea 2 technical report deep dive](./resources/references/krea2-technical-report-deep-dive.md)
      for architecture context and non-claims.
    - [Krea 2 implementation reference](./resources/references/krea2-implementation-reference.md)
      for pinned public source links and planning facts.

    Why: Krea 2 has its own denoiser/text-fusion structure and should not be
    accepted by accidental overlap with another architecture family.

    How:
    - Inspect current architecture recognition in `nodes/entry.py` and any shared
      architecture registry/helpers used by loaders, classifiers, model loaders,
      and saved-output code.
    - Define Krea 2 recognition rules using Krea-specific positive evidence. The
      rule should require enough Krea evidence to distinguish Krea 2 from existing
      supported families and should not rely on a single generic transformer key.
    - Keep the unsupported/ambiguous architecture path separate from successful
      Krea detection so tests can assert the two behaviors independently.
    - Register `krea2` in the architecture-sensitive places that need to select
      Krea-compatible behavior.
    - Add compatibility handling for precision/quantization classes as a warning
      or explicit compatibility check. Do not add a blanket FP8 ban; prefer
      like-with-like compatibility and explicit diagnostics for mixed or
      unsupported combinations.

    Testing:
    - Add synthetic state-dict tests for positive Krea 2 recognition.
    - Add negative tests for incomplete, unknown, and ambiguous architecture
      signatures.
    - Add routing tests proving a `krea2` recipe reaches Krea-compatible loader,
      classifier, model-loader, and validation branches where those branches exist.
    - Run `uv run python -m compileall lib nodes tests`, focused pytest for the
      touched architecture tests, and the project review gates from the
      `comfy-widen-gates` skill.

    Covers: @krea2-architecture-support ac-krea2-detected-from-krea-signature,
      ac-ambiguous-or-unknown-architecture-rejected,
      ac-krea2-recipe-uses-krea-compatible-paths.

- title: Implement Krea 2 LoRA package compatibility
  slug: task-krea2-lora-package-compatibility
  priority: 1
  tags: [krea2, lora, compatibility]
  spec_ref: "@krea2-lora-package-compatibility"
  resource_refs:
    - krea2-implementation-reference
  depends_on:
    - "@task-krea2-architecture-recognition-routing"
  description: |
    What: Implement Krea 2 LoRA compatibility for the public Krea 2 LoRA package
    families identified in the imported implementation reference.

    Resources:
    - [Krea 2 implementation reference](./resources/references/krea2-implementation-reference.md)
      for pinned ComfyUI, ai-toolkit, LoKr, and Hugging Face references.

    Why: Users should be able to use supported Krea 2 LoRAs without manually
    rewriting tensor names, and unsupported packages must not produce silent
    partial merges.

    How:
    - Inspect `lib/lora/` loader interfaces and existing architecture loaders.
    - Add or update a Krea 2 loader module with an explicit compatibility table
      for supported public package families.
    - Put detailed tensor-name normalization rules and shape expectations in code
      comments/tests, not in timeless spec ACs.
    - Preserve compound Krea names such as text-fusion substructure names during
      normalization.
    - Validate package groups before publishing merge output. Unsupported or
      shape-incompatible groups must produce an actionable report instead of
      being silently skipped.
    - Ensure cleanup/resource-release behavior matches the existing loader
      contract.

    Testing:
    - Add CPU-safe synthetic fixtures for each supported package family.
    - Add tests for compound-name preservation and shape compatibility.
    - Add unsupported-package tests that fail before producing a merged result and
      report the incompatible groups.
    - Add optional header-only tests that accept operator-provided local sample
      paths or public downloaded headers without hard-coding machine-local paths.
    - Run focused loader tests and the project review gates.

    Covers: @krea2-lora-package-compatibility ac-supported-krea2-lora-packages-load,
      ac-lora-compatibility-is-complete-or-rejected,
      ac-krea2-lora-strength-controls-are-stable;
      @krea2-architecture-support ac-krea2-recipe-uses-krea-compatible-paths.

- title: Add Krea 2 block and layer controls
  slug: task-krea2-block-layer-controls
  priority: 2
  tags: [krea2, block-config, layer-controls]
  spec_ref: "@krea2-block-and-layer-controls"
  resource_refs:
    - krea2-implementation-reference
  depends_on:
    - "@task-krea2-architecture-recognition-routing"
  description: |
    What: Add Krea 2 key classification and user-facing block/layer controls that
    map to Krea model regions and layer categories.

    Resources:
    - [Krea 2 implementation reference](./resources/references/krea2-implementation-reference.md)
      for architecture structure facts and source links.

    Why: Per-block and per-layer controls should remain meaningful for Krea 2
    users. Krea text-fusion regions should not disappear into an unrelated or
    generic catch-all group.

    How:
    - Inspect `lib/block_classify.py`, existing block config node factories, and
      existing architecture-specific block config nodes.
    - Add Krea 2 model-region classification for main denoiser regions and
      text-fusion regions.
    - Add Krea 2 layer-category classification for categories the existing UI and
      executor can support.
    - Define deterministic fallback behavior for structural keys and keys outside
      user-adjustable groups.
    - Add or generate a Krea 2 block-config node and register it in ComfyUI node
      mappings.
    - Keep exact internal labels and regexes in implementation/tests. Specs should
      assert user-observable control behavior, not the regex mechanics.

    Testing:
    - Add unit tests proving main-region controls affect only main-region keys.
    - Add unit tests proving text-fusion controls affect only text-fusion keys.
    - Add layer-category tests for supported categories and fallback behavior.
    - Add node tests proving the Krea 2 block-config output is accepted by LoRA
      and merge consumers.
    - Run focused classifier/node tests and the project review gates.

    Covers: @krea2-block-and-layer-controls ac-main-model-regions-are-controllable,
      ac-text-fusion-regions-are-controllable,
      ac-layer-category-controls-are-controllable;
      @krea2-architecture-support ac-krea2-recipe-uses-krea-compatible-paths.

- title: Add Krea 2 validation fixtures and optional smoke evidence
  slug: task-krea2-validation-and-smoke-evidence
  priority: 3
  tags: [krea2, tests, validation, smoke]
  spec_ref: "@krea2-architecture-support"
  resource_refs:
    - krea2-implementation-reference
  depends_on:
    - "@task-krea2-lora-package-compatibility"
    - "@task-krea2-block-layer-controls"
  description: |
    What: Add Krea-specific test fixtures and optional validation evidence for the
    implemented architecture, LoRA, and block/layer behavior.

    Why: Krea 2 assets are large, but support still needs repeatable CPU-safe
    tests plus a clear path for optional real-file or Comfy smoke evidence.

    How:
    - Keep required tests CPU-safe and synthetic.
    - Add fixture data that exercises Krea architecture recognition, supported
      LoRA package families, compound-name preservation, block/layer controls,
      unsupported-package diagnostics, and precision/quantization compatibility
      warnings or failures.
    - Add an optional header/probe helper that accepts explicit operator-provided
      paths or downloaded public sample locations. Do not hard-code local model
      directories in tests, docs, or plan-derived task text.
    - Document a Comfy smoke workflow using public source references and
      operator-provided model assets. The smoke may be skipped when GPU/assets are
      unavailable, but the skip reason and replacement evidence must be recorded.
    - If the project-wide trait/validation plan creates a reusable validation
      trait before this task starts, consider applying it in a separate reviewed
      metadata update rather than expanding this task's product scope.

    Testing:
    - Run the required CPU-safe Krea tests.
    - Run optional header/probe validation when assets are available.
    - Run or document the Comfy smoke path with exact evidence or a concrete skip
      reason.
    - Run `kspec validate --warnings-ok` and the `comfy-widen-gates` review gates.

    Covers: @krea2-architecture-support ac-krea2-detected-from-krea-signature,
      ac-ambiguous-or-unknown-architecture-rejected,
      ac-krea2-recipe-uses-krea-compatible-paths;
      @krea2-lora-package-compatibility ac-supported-krea2-lora-packages-load,
      ac-lora-compatibility-is-complete-or-rejected,
      ac-krea2-lora-strength-controls-are-stable;
      @krea2-block-and-layer-controls ac-main-model-regions-are-controllable,
      ac-text-fusion-regions-are-controllable,
      ac-layer-category-controls-are-controllable.
```

## Notes

- Krea-specific implementation details are intentionally concentrated in tasks and
  imported resources. The specs describe durable user/system behavior.
- The separate project-wide trait taxonomy draft should decide whether validation,
  architecture support, LoRA loader safety, quantized-input safety, or block/layer
  control contracts become reusable traits across the whole project.
