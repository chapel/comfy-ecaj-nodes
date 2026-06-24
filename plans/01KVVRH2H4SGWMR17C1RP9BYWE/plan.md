# Krea 2 Architecture Support Plan

## Context

This is a quick draft for first-class `krea2` support in WIDEN / comfy-ecaj-nodes. It is based on:

- Local Krea 2 technical note: `/home/chapel/Documents/Obsidian Vault/Krea 2 Technical Report Deep Dive.md`
- Local Krea model assets:
  - `/mnt/big-data/AI/models/diffusion_models/krea/krea2_raw_bf16.safetensors`
  - `/mnt/big-data/AI/models/diffusion_models/krea/krea2_raw_fp8_scaled.safetensors`
  - `/mnt/big-data/AI/models/loras/krea/krea2_turbo_lora_rank_64_bf16.safetensors`
- ComfyUI implementation references:
  - `/home/chapel/Projects/ComfyUI/comfy/ldm/krea2/model.py`
  - `/home/chapel/Projects/ComfyUI/comfy/text_encoders/krea2.py`
- ai-toolkit current upstream reference: `~/Projects/ai-toolkit` fetched to `origin/main` at `724e67d Set krea 2 to use new lokr format` without rebasing the local dirty checkout.
- Hugging Face LoRAs:
  - `Comfy-Org/Krea-2/loras/krea2_turbo_lora_rank_64_bf16.safetensors` — Comfy/native-style keys, rank 64, `diffusion_model.blocks.*`, `lora_down/lora_up`.
  - `Comfy-Org/Krea-2/loras/krea2_coolblue.safetensors`, `krea2_darkbrush.safetensors`, `krea2_plasmoid.safetensors`, `krea2_warmpastel.safetensors` — ai-toolkit-style keys, rank 32, `transformer.transformer_blocks.*`, `lora_A/lora_B`.
  - `krea/Krea-2-LoRA-retroanime/retroanime.safetensors` — same ai-toolkit-style header shape as the style LoRAs above.

Important architecture facts from the checkpoint/header probes:

- Krea 2 should be a first-class architecture id: `krea2`.
- The BF16 base has 430 tensors and about 12.82B elements.
- Main denoiser blocks are local checkpoint keys `blocks.0` … `blocks.27`.
- The local checkpoint also has text-fusion keys: `txtfusion.layerwise_blocks.*`, `txtfusion.refiner_blocks.*`, `txtfusion.projector.*`.
- The model uses GQA attention: width 6144, 48 query heads, 12 key/value heads, head dim 128.
- The native turbo LoRA maps cleanly to 264 base parameters after preserving compound names like `layerwise_blocks` and `refiner_blocks`.
- Current Qwen-ish parsing is dangerous because it can almost work while corrupting text-fusion names, e.g. `layerwise_blocks` → `layerwise.blocks`.
- New ai-toolkit-style LoRAs use a different dialect: `transformer.transformer_blocks.N.attn.to_q.lora_A/B.weight`, `transformer.text_fusion.*`, `ff.*`, etc. Supporting those requires explicit alias normalization to the Comfy/Krea base keyspace.

## Specs

```yaml
- title: Krea 2 Diffusion Architecture Support
  slug: krea2-diffusion-architecture-support
  type: feature
  parent: "@widen"
  tags: [krea2, diffusion, architecture]
  description: |
    WIDEN can detect and execute merge recipes against Krea 2 diffusion models as
    a first-class architecture rather than treating them as Qwen, Flux, or a
    generic transformer. Initial support is LoRA-first against BF16 Krea 2
    diffusion checkpoints; direct FP8 math and independent full-model Krea2↔Krea2
    WIDEN are explicitly out of scope unless a later plan adds dequantization and
    full-model validation.
  acceptance_criteria:
    - id: ac-detect-krea2
      given: |
        A ComfyUI model patcher exposes a Krea 2 diffusion state dict with
        `blocks.N.*`, Krea text-fusion keys, and Krea structural keys.
      when: |
        the WIDEN Entry node snapshots the model.
      then: |
        the recipe base architecture is detected as `krea2` and unsupported
        architectures remain rejected with a useful key-prefix diagnostic.
    - id: ac-krea2-executor-routing
      given: |
        A recipe base has architecture `krea2`.
      when: |
        the Exit executor applies LoRA deltas or block/layer controls.
      then: |
        Krea 2-specific loader selection, key normalization, block classification,
        and validation paths are selected without routing through Qwen-specific
        assumptions.
    - id: ac-fp8-not-merged-implicitly
      given: |
        A Krea 2 checkpoint includes FP8-scaled weights or `weight_scale` sidecar
        tensors.
      when: |
        the checkpoint is used as a WIDEN base, LoRA merge target, or validation
        fixture in this initial Krea 2 support path.
      then: |
        the operation rejects direct FP8/scale-sidecar merge math with an explicit
        unsupported-input diagnostic; it does not silently merge FP8 sidecars as
        ordinary model weights.

- title: Krea 2 LoRA Keyspace Support
  slug: krea2-lora-keyspace-support
  type: feature
  parent: "@lora-loaders"
  tags: [krea2, lora, keyspace]
  description: |
    WIDEN can load Krea 2 LoRAs from both the native Comfy/Krea key dialect and
    the ai-toolkit training/export dialect, mapping every applicable LoRA group
    to an exact Krea 2 base tensor or failing explicitly.
  acceptance_criteria:
    - id: ac-native-turbo-lora-matches-base
      given: |
        The native Krea Turbo LoRA uses keys such as
        `diffusion_model.blocks.0.attn.wq.lora_down.weight` and
        `diffusion_model.txtfusion.layerwise_blocks.0.attn.wq.lora_up.weight`.
      when: |
        the Krea 2 LoRA loader parses the LoRA against the BF16 Krea 2 base.
      then: |
        every parsed LoRA group maps to an existing base key with a compatible
        matrix product shape; no text-fusion group is skipped or name-mangled.
    - id: ac-aitoolkit-style-lora-matches-base
      given: |
        An ai-toolkit-style Krea 2 LoRA uses keys such as
        `transformer.transformer_blocks.0.attn.to_q.lora_A.weight`,
        `transformer.text_fusion.layerwise_blocks.0.ff.up.lora_B.weight`, and
        `transformer.final_layer.linear.lora_A.weight`.
      when: |
        the Krea 2 LoRA loader parses the LoRA against the BF16 Krea 2 base.
      then: |
        aliases are normalized into the Comfy/Krea base keyspace, including
        `transformer_blocks`→`blocks`, `text_fusion`→`txtfusion`,
        `to_q/to_k/to_v/to_out.0/to_gate`→`wq/wk/wv/wo/gate`, and
        `ff.*`→`mlp.*`, with all supported groups either matched or explicitly
        reported as unsupported.
    - id: ac-lora-dialects-are-detected
      given: |
        A Krea 2 LoRA is loaded without an explicit dialect override.
      when: |
        the loader inspects its tensor names.
      then: |
        it distinguishes native `lora_down/lora_up` keys from ai-toolkit
        `lora_A/lora_B` keys and applies the correct rank, alpha, and matrix
        multiplication conventions for that dialect.

- title: Krea 2 Block and Layer Control
  slug: krea2-block-and-layer-control
  type: feature
  parent: "@per-block-control"
  tags: [krea2, block-config, per-block-control]
  description: |
    Krea 2 recipes can use per-block and per-layer controls that reflect the
    actual Krea 2 model layout, including main denoiser blocks and text-fusion
    blocks.
  acceptance_criteria:
    - id: ac-main-block-classification
      given: |
        Krea 2 diffusion keys under `blocks.0` through `blocks.27` are evaluated
        for block-specific merge weights.
      when: |
        the block classifier processes those keys.
      then: |
        each key is classified into stable block labels such as `B00` through
        `B27`.
    - id: ac-text-fusion-classification
      given: |
        Krea 2 keys under `txtfusion.layerwise_blocks.*` and
        `txtfusion.refiner_blocks.*` are evaluated for block-specific merge
        weights.
      when: |
        the block classifier processes those keys.
      then: |
        they are classified into distinct text-fusion block labels rather than
        collapsed into `other` or into main denoiser blocks.
    - id: ac-structural-keys-classified
      given: |
        Krea 2 structural keys such as `img_in`, `txt_in`, `time_embed`,
        `time_mod_proj`, `final_layer`, and `txtfusion.projector` are present.
      when: |
        layer/block controls are applied.
      then: |
        structural keys receive stable classifier labels or documented fallback
        behavior so merge recipes remain deterministic.
    - id: ac-layer-type-classification
      given: |
        Krea 2 attention, MLP, embedding/projection, text-fusion, and final-layer
        keys are evaluated for layer-type filtering.
      when: |
        the layer-type classifier or block-config node maps those keys.
      then: |
        keys receive stable layer-type categories such as attention, mlp,
        embedding/projection, text-fusion, final, or structural/fallback so
        layer-type sliders do not collapse all Krea 2 weights into `other`.
    - id: ac-krea2-block-config-sliders
      given: |
        A user opens the Krea 2 block-config node.
      when: |
        they configure global, main-block, text-fusion, structural, and
        layer-type merge weights.
      then: |
        the node emits a `BlockConfig(arch="krea2")` whose slider keys are accepted
        by the executor and applied to the matching Krea 2 key groups.

- title: Krea 2 Validation Fixtures and Smoke Gates
  slug: krea2-validation-fixtures-and-smoke-gates
  type: requirement
  parent: "@krea2-diffusion-architecture-support"
  tags: [krea2, tests, validation]
  description: |
    Krea 2 support has CPU-safe parser/classifier coverage and optional real-file
    smoke gates before being considered complete.
  acceptance_criteria:
    - id: ac-cpu-fixtures-cover-dialects
      given: |
        The test suite runs without GPU or full model downloads.
      when: |
        Krea 2 loader and classifier tests run.
      then: |
        synthetic fixtures cover native Comfy/Krea LoRA keys, ai-toolkit-style
        LoRA keys, text-fusion compound names, GQA projection shapes, and
        unsupported/missing-key diagnostics.
    - id: ac-real-header-probe
      given: |
        The local Krea BF16 base, local/native Turbo LoRA, and public HF style
        LoRA headers are accessible.
      when: |
        the optional validation harness runs.
      then: |
        it reports key counts, dialect, matched group count, missing group count,
        and incompatible shape count without loading all model tensors into GPU
        memory.
    - id: ac-comfy-smoke-documentation
      given: |
        Krea 2 support passes CPU-safe tests and header validation.
      when: |
        the implementation task is submitted for review.
      then: |
        the task evidence documents a ComfyUI smoke path or explains why the
        smoke was skipped, including required base model, LoRA, CLIP/text encoder,
        VAE, and output artifact expectations.
```

## Tasks

derive_from_specs: false

```yaml
- title: Add Krea 2 architecture detection and registry plumbing
  slug: task-krea2-architecture-detection
  priority: 1
  tags: [krea2, architecture, entry-node]
  spec_ref: "@krea2-diffusion-architecture-support"
  depends_on: []
  description: |
    What: Add a first-class `krea2` architecture id throughout the recipe and
    execution plumbing.

    Why: Krea 2 shares some transformer concepts with Qwen/Flux, but its base
    keyspace, text-fusion modules, and LoRA dialects are different enough that
    aliasing it to another architecture risks partial merges that look
    successful.

    How:
    - Add Krea 2 detection in `nodes/entry.py` using Krea-specific base keys,
      e.g. `diffusion_model.blocks.*` plus `txtfusion`/`text_fusion`/structural
      signatures that avoid false-positive Qwen detection.
    - Register `krea2` anywhere architecture-specific loaders, classifiers,
      block-config nodes, or executor branches are selected.
    - Keep FP8-scaled inputs out of the default path unless an explicit
      dequantization design is added.
    - Add focused tests in the existing entry/model-loader test style.

    Testing:
    - Run focused architecture-detection and model-loader tests.
    - Add negative coverage proving Qwen and Flux fixtures still detect as their
      own architectures.
    - Add a focused rejection test for FP8-scaled Krea checkpoints or
      `weight_scale` sidecars so unsupported inputs fail explicitly before merge
      math.

    Covers: @krea2-diffusion-architecture-support ac-detect-krea2,
      ac-krea2-executor-routing, ac-fp8-not-merged-implicitly.

- title: Implement Krea 2 native LoRA loader
  slug: task-krea2-native-lora-loader
  priority: 1
  tags: [krea2, lora, native-dialect]
  spec_ref: "@krea2-lora-keyspace-support"
  depends_on:
    - "@task-krea2-architecture-detection"
  description: |
    What: Implement a Krea 2 LoRA loader for native Comfy/Krea-style LoRAs using
    `diffusion_model.*` keys and `lora_down/lora_up` tensors.

    Why: The local Turbo LoRA is already in this dialect and should be the first
    real merge target. Existing Qwen parsing currently almost works but can
    corrupt compound text-fusion names.

    How:
    - Add `lib/lora/krea2.py` or equivalent and expose it from `lib/lora/__init__.py`.
    - Preserve compound names exactly: `layerwise_blocks`, `refiner_blocks`,
      `txtfusion`, `tmlp`, `txtmlp`, `tproj`, `wq`, `wk`, `wv`, `wo`, `gate`.
    - Normalize only safe prefixes such as `diffusion_model.`.
    - Validate every LoRA group against the base state dict and report missing or
      incompatible groups explicitly.
    - Add CPU-safe synthetic tests plus an optional header/key-only validation
      helper for `/mnt/big-data/AI/models/loras/krea/krea2_turbo_lora_rank_64_bf16.safetensors`.

    Testing:
    - New loader tests assert zero unmatched groups for the known native Turbo
      LoRA key set when the BF16 base key set is available.
    - Synthetic tests cover GQA shapes: `wq/wo/gate` 6144x6144 and `wk/wv`
      1536x6144.

    Covers: @krea2-lora-keyspace-support ac-native-turbo-lora-matches-base;
      @krea2-diffusion-architecture-support ac-krea2-executor-routing.

- title: Implement Krea 2 ai-toolkit LoRA dialect support
  slug: task-krea2-aitoolkit-lora-dialect
  priority: 2
  tags: [krea2, lora, ai-toolkit, huggingface]
  spec_ref: "@krea2-lora-keyspace-support"
  depends_on:
    - "@task-krea2-native-lora-loader"
  description: |
    What: Extend the Krea 2 loader to parse ai-toolkit-exported LoRAs from the
    public Krea 2 style LoRA repos.

    Why: The current public style LoRAs from Comfy-Org and krea use
    `transformer.*` prefixes, `transformer_blocks`, `text_fusion`, and
    `lora_A/lora_B` tensors rather than the native Comfy/Krea names. A Krea 2
    implementation that only supports the Turbo LoRA would miss these real-world
    LoRAs.

    How:
    - Add dialect detection for `lora_A/lora_B` pairs.
    - Normalize ai-toolkit module aliases into the Comfy/Krea base keyspace:
      `transformer.transformer_blocks.N` → `blocks.N`,
      `transformer.text_fusion` → `txtfusion`,
      `attn.to_q/to_k/to_v/to_out.0/to_gate` → `attn.wq/wk/wv/wo/gate`,
      `ff.up/ff.gate/ff.down` → `mlp.up/mlp.gate/mlp.down` where shapes match.
    - Audit and map structural modules seen in headers: `img_in`, `txt_in`,
      `time_embed`, `time_mod_proj`, `final_layer.linear`, and
      `text_fusion.projector`.
    - Decide whether unmapped structural modules are supported, skipped by
      explicit user option, or rejected.
    - Reference ai-toolkit upstream `origin/main` Krea2 code and keep the mapping
      table in tests/docs so future export-format changes are visible.

    Testing:
    - Header-only tests or fixtures derived from `krea2_coolblue.safetensors` and
      `retroanime.safetensors` assert dialect detection and alias mapping.
    - Shape tests prove `lora_A/lora_B` multiplication yields the target base
      tensor shape for attention, MLP, text-fusion, and structural modules.

    Covers: @krea2-lora-keyspace-support ac-aitoolkit-style-lora-matches-base,
      ac-lora-dialects-are-detected.

- title: Add Krea 2 block classifier and block-config node
  slug: task-krea2-block-config
  priority: 2
  tags: [krea2, block-config, ui]
  spec_ref: "@krea2-block-and-layer-control"
  depends_on:
    - "@task-krea2-architecture-detection"
  description: |
    What: Add Krea 2 key classification and a matching ComfyUI block-config node.

    Why: Without Krea-specific classification, per-block controls collapse into
    generic/other buckets and do not reflect the 28 denoiser blocks or the
    text-fusion stack.

    How:
    - Add `classify_key_krea2` to `lib/block_classify.py` and register it.
    - Label main blocks `B00`–`B27`.
    - Label text-fusion layerwise/refiner blocks distinctly, e.g. `TF_L00`– and
      `TF_R00`– style labels.
    - Add structural labels for `IMG_IN`, `TXT_IN`, `TIME_EMBED`,
      `TIME_MOD_PROJ`, `FINAL_LAYER`, and `TXT_FUSION_PROJECTOR` or document
      fallback handling.
    - Add Krea 2 layer-type classification for attention, MLP, embedding,
      projection/final, text-fusion, and structural/fallback keys so existing
      layer-type filters and sliders work with `arch="krea2"`.
    - Add `nodes/block_config_krea2.py` and expose it in node registration using
      the same generator pattern as existing Qwen/Flux/Z-Image block config nodes.
    - Include user-facing sliders for main denoiser blocks, text-fusion groups,
      structural groups, and layer-type weights; text-fusion sliders are included
      in the initial node under an advanced/clearly labeled section rather than
      left as an open product decision.

    Testing:
    - Add classifier tests for native and ai-toolkit-normalized key forms.
    - Add node tests proving default sliders produce a `BlockConfig(arch="krea2")`
      and per-block/per-layer overrides affect expected key groups.

    Covers: @krea2-block-and-layer-control ac-main-block-classification,
      ac-text-fusion-classification, ac-structural-keys-classified,
      ac-layer-type-classification, ac-krea2-block-config-sliders.

- title: Add Krea 2 real-file validation harness
  slug: task-krea2-real-file-validation
  priority: 3
  tags: [krea2, validation, real-files]
  spec_ref: "@krea2-validation-fixtures-and-smoke-gates"
  depends_on:
    - "@task-krea2-aitoolkit-lora-dialect"
    - "@task-krea2-block-config"
  description: |
    What: Add an optional validation command/test helper that can inspect the
    local BF16 Krea base, local Turbo LoRA, and remote/HF style LoRA headers.

    Why: Krea 2 checkpoints are large and should not be required for unit tests,
    but support should be checked against real keyspaces before it is considered
    ready for Comfy use.

    How:
    - Keep normal pytest CPU-safe and synthetic.
    - Add an opt-in script or pytest marker that reads safetensors headers/key
      metadata without moving tensors to GPU.
    - Report base key count, LoRA group count, dialect, matched count, missing
      count, incompatible shape count, and unsupported-group details.
    - Include explicit skip messages when `/mnt/big-data` assets are unavailable.

    Testing:
    - Run normal pytest without the real Krea assets.
    - Run the opt-in harness locally against `/mnt/big-data/AI/models/diffusion_models/krea/krea2_raw_bf16.safetensors`
      and the known native/ai-toolkit LoRA samples.

    Covers: @krea2-validation-fixtures-and-smoke-gates ac-cpu-fixtures-cover-dialects,
      ac-real-header-probe.

- title: Document and execute a ComfyUI Krea 2 smoke path
  slug: task-krea2-comfy-smoke
  priority: 4
  tags: [krea2, comfy, smoke]
  spec_ref: "@krea2-validation-fixtures-and-smoke-gates"
  depends_on:
    - "@task-krea2-real-file-validation"
  description: |
    What: Document and, when local model assets are present, execute the smallest
    ComfyUI workflow that proves a Krea 2 merged artifact can be loaded and used.

    Why: Parser and tensor-shape correctness do not prove the saved output is
    accepted by Comfy's Krea2 loader or that required text encoder/VAE components
    are wired correctly.

    How:
    - Use the BF16 raw Krea 2 base first; do not use FP8-scaled base as a merge
      input in the initial smoke.
    - Apply a known LoRA at a small strength and save a merged diffusion artifact.
    - Load the merged artifact through the local ComfyUI Krea2 path with the
      expected Qwen3-VL text encoder and Qwen-Image VAE components.
    - Record exact paths, loader names, output format, any Comfy log warnings,
      and whether image generation completes.

    Testing:
    - Attach smoke evidence to task notes.
    - If a GPU/Comfy session is not available, include the skipped reason and the
      validated replacement evidence from the real-file harness.

    Covers: @krea2-validation-fixtures-and-smoke-gates ac-comfy-smoke-documentation.
```

## Initial Scope Decisions

- ai-toolkit-style LoRAs are normalized on load only in this plan; a separate
  conversion/save utility is deferred until there is a demonstrated workflow need.
- Text-fusion block controls are exposed in the Krea 2 block-config node under an
  advanced or clearly labeled section so they are available without making them
  mandatory for simple workflows.
- Unsupported structural LoRA groups fail by default with explicit diagnostics.
  A future permissive "apply matched groups only" option may be planned later,
  but it is not part of the initial safe merge path.
- Initial implementation is LoRA-first against BF16 Krea 2 diffusion bases. Full
  Krea2 model-model WIDEN is deferred until at least two independent compatible
  BF16 Krea 2 checkpoints exist and can be validated.
- FP8 support is deferred. If required later, it should be designed as an
  explicit dequantize/merge/requantize path around BF16/FP32 merge math rather
  than direct FP8 sidecar merging.
