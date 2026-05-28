# Qwen and Flux 2 Klein Architecture Support

Add WIDEN merge support for Qwen-Image (20B MMDiT) and Flux 2 Klein (4B/9B) architectures.

## Specs

```yaml
- title: Qwen Architecture Support
  slug: qwen-support
  type: feature
  description: |
    Full WIDEN merge support for Qwen-Image (20B MMDiT, 60 dual-stream
    transformer blocks). Dual-stream (img/txt) architecture with separate
    Q/K/V projections -- no QKV fusion needed. Standard diffusers LoRA
    format plus A1111/kohya and LyCORIS variants.
    Reference implementation in ../merge-router (qwen_key_mapper.py,
    qwen_splitter.py, qwen_merge.py).
    Key files to modify: nodes/entry.py, lib/block_classify.py,
    lib/lora/ (new qwen.py), lib/model_loader.py,
    nodes/block_config_qwen.py (new), __init__.py.
  acceptance_criteria:
    - id: ac-1
      given: a ComfyUI model whose state dict contains transformer_blocks keys
      when: detect_architecture() runs
      then: |
        it returns "qwen" and "qwen" is in _SUPPORTED_ARCHITECTURES.
        Must not false-match Flux which uses double_blocks not transformer_blocks.
    - id: ac-2
      given: a parameter key from a Qwen model
      when: classify_key_qwen() is called
      then: |
        transformer_blocks.N maps to TBNN (zero-padded two digits, e.g.
        TB00-TB59). Block indices discovered dynamically via regex (not
        hardcoded to 60). Non-block keys (proj_out, norm_out, pos_embed,
        x_embedder, context_embedder, t_embedder, time_text_embed)
        return None.
    - id: ac-3
      given: a parameter key from a Qwen model
      when: classify_layer_type(key, "qwen") is called
      then: |
        returns "attention" for attn.to_q, attn.to_k, attn.to_v,
        attn.add_q_proj, attn.add_k_proj, attn.add_v_proj,
        attn.to_out, attn.to_add_out, attn.norm_q, attn.norm_k,
        attn.norm_added_q, attn.norm_added_k.
        Returns "feed_forward" for img_mlp, txt_mlp.
        Returns "norm" for img_norm1, img_norm2, txt_norm1, txt_norm2,
        img_mod, txt_mod.
    - id: ac-4
      given: a Qwen LoRA file in diffusers format
      when: QwenLoader parses it
      then: |
        keys like transformer.transformer_blocks.N.attn.to_q.lora_A.weight
        are mapped to base model key transformer_blocks.N.attn.to_q.weight
        and produce DeltaSpec with kind="standard", offset=None, and
        correct up/down tensor pairs.
    - id: ac-5
      given: a Qwen LoRA file in A1111/kohya format
      when: QwenLoader parses it
      then: |
        keys like lora_unet_transformer_blocks_N_attn_to_q.lora_down.weight
        are normalized with compound name preservation (transformer_blocks,
        add_k_proj, add_q_proj, add_v_proj, to_add_out, img_mod, txt_mod,
        img_mlp, txt_mlp, img_norm, txt_norm, time_text_embed) and
        produce DeltaSpec with kind="standard" and correct base model key.
    - id: ac-6
      given: a Qwen LoRA file in LyCORIS format
      when: QwenLoader parses it
      then: |
        keys like lycoris_transformer_blocks_N_attn_to_q.lokr_w1 are
        normalized using underscore-to-dot conversion with compound name
        handling and produce DeltaSpec with kind="lokr" and correct
        w1/w2 tensor pairs.
    - id: ac-7
      given: a safetensors checkpoint with Qwen architecture keys
      when: model_loader.py opens it
      then: |
        architecture is detected as "qwen" from transformer_blocks pattern.
        Key normalization strips diffusion_model. and transformer. prefixes.
        VAE and text encoder keys are excluded.
    - id: ac-8
      given: a WIDENBlockConfigQwen node in ComfyUI
      when: rendered
      then: |
        shows 60 individual block sliders (TB00-TB59, FLOAT 0.0-2.0,
        default 1.0) plus 3 layer-type sliders (attention, feed_forward,
        norm). Generated via make_block_config_node() factory.
    - id: ac-9
      given: QwenLoader, classify_key_qwen, classify_layer_type for qwen,
        model_loader qwen detection, and block config node all implemented
      when: all registries are checked
      then: |
        _SUPPORTED_ARCHITECTURES includes "qwen", _CLASSIFIERS has "qwen"
        entry, _LAYER_TYPE_PATTERNS has "qwen" patterns, LOADER_REGISTRY
        has "qwen" entry, _ARCH_PATTERNS has qwen detection,
        NODE_CLASS_MAPPINGS has WIDENBlockConfigQwen.

- title: Flux 2 Klein Architecture Support
  slug: flux-klein-support
  type: feature
  description: |
    Full WIDEN merge support for Flux 2 Klein (4B and 9B variants).
    Dual block types: double_blocks (joint img/txt attention with fused
    QKV per stream) and single_blocks (fused linear1 = QKV + MLP
    projection). Requires offset-based DeltaSpec for QKV fusing,
    following Z-Image's pattern in lib/lora/zimage.py.
    Klein 9B has 8 double + 24 single = 32 blocks.
    Klein 4B has 5 double + 20 single = 25 blocks.
    Block config sized for Klein 9B max. Flux.1 out of scope for now.
    Key files to modify: nodes/entry.py, lib/block_classify.py,
    lib/lora/ (new flux.py), lib/model_loader.py,
    nodes/block_config_flux.py (new), __init__.py.
  acceptance_criteria:
    - id: ac-1
      given: a ComfyUI model whose state dict contains double_blocks keys
      when: detect_architecture() runs
      then: it returns "flux" and "flux" is in _SUPPORTED_ARCHITECTURES
    - id: ac-2
      given: a parameter key from a Flux Klein model
      when: classify_key_flux() is called
      then: |
        double_blocks.N maps to DB00-DB07 (9B) or DB00-DB04 (4B).
        single_blocks.N maps to SB00-SB23 (9B) or SB00-SB19 (4B).
        Block indices are discovered dynamically from keys (not hardcoded).
        Non-block keys (guidance_in, time_in, vector_in, img_in, txt_in,
        final_layer) return None.
    - id: ac-3
      given: a parameter key from a Flux Klein model
      when: classify_layer_type(key, "flux") is called
      then: |
        returns "attention" for img_attn, txt_attn, qkv, proj,
        norm.query_norm, norm.key_norm.
        Returns "feed_forward" for img_mlp, txt_mlp, linear2.
        Returns "norm" for img_mod, txt_mod, modulation
        (excluding attention-specific norms already captured above).
    - id: ac-4
      given: a Flux LoRA targeting double_block attention (to_q, to_k, to_v)
      when: FluxLoader parses it
      then: |
        separate LoRA to_q/to_k/to_v components map to fused
        img_attn.qkv and txt_attn.qkv base model keys with
        DeltaSpec kind=qkv_q/qkv_k/qkv_v and offset=(start, length)
        tuples. Two QKV fusions per double_block (one per stream).
    - id: ac-5
      given: a Flux LoRA targeting single_block components
      when: FluxLoader parses it
      then: |
        LoRA to_q, to_k, to_v map to slices of fused linear1 weight
        with DeltaSpec kind=qkv_q/qkv_k/qkv_v and offset tuples.
        proj_mlp maps with a new offset-aware kind (e.g. offset_mlp)
        and offset tuple. The gpu_ops.py offset-slice branch (line 200)
        is extended to include the new kind.
    - id: ac-6
      given: a Flux LoRA in BFL/kohya format
      when: FluxLoader parses it
      then: |
        keys like lora_unet_double_blocks_N_img_attn_qkv.lora_down.weight
        are correctly normalized and mapped to base model keys.
    - id: ac-7
      given: a Flux LoRA in diffusers format
      when: FluxLoader parses it
      then: |
        keys like transformer.double_blocks.N.img_attn.to_q.lora_A.weight
        are correctly normalized and mapped to base model keys.
    - id: ac-8
      given: a safetensors checkpoint with Flux architecture keys
      when: model_loader.py opens it
      then: |
        architecture is detected as "flux" from double_blocks pattern.
        Key normalization strips diffusion_model. and transformer. prefixes.
    - id: ac-9
      given: a WIDENBlockConfigFlux node in ComfyUI
      when: rendered
      then: |
        shows 32 block sliders for Klein 9B max (DB00-DB07 + SB00-SB23,
        FLOAT 0.0-2.0, default 1.0) plus 3 layer-type sliders.
        Klein 4B models have unused block sliders default to 1.0.
        Generated via make_block_config_node() factory.
    - id: ac-10
      given: a Klein 4B model (5 double + 20 single blocks) and a Klein 9B
        model (8 double + 24 single blocks)
      when: both are processed with the same "flux" arch tag
      then: |
        the classifier returns block names matching the actual blocks
        present in each model (e.g. DB00-DB04 for 4B, DB00-DB07 for 9B).
        No error is raised for either variant.
    - id: ac-11
      given: all Flux Klein components implemented
      when: all registries are checked
      then: |
        _SUPPORTED_ARCHITECTURES includes "flux", _CLASSIFIERS has "flux"
        entry, _LAYER_TYPE_PATTERNS has "flux" patterns, LOADER_REGISTRY
        has "flux" entry, _ARCH_PATTERNS has flux detection,
        NODE_CLASS_MAPPINGS has WIDENBlockConfigFlux.
```

## Tasks

```yaml
- title: Implement Qwen detection and block classification
  slug: qwen-detect-classify
  spec_ref: "@qwen-support"
  priority: 2
  description: |
    Enable Qwen detection in nodes/entry.py (add "qwen" to
    _SUPPORTED_ARCHITECTURES). Add classify_key_qwen() in
    lib/block_classify.py mapping transformer_blocks.N to TB00-TB59
    with dynamic index discovery (regex, not hardcoded 60).
    Add Qwen layer type patterns (attention/feed_forward/norm).
    Register in _CLASSIFIERS and _LAYER_TYPE_PATTERNS.
    Update __all__ in block_classify.py to export new functions.

    BREAKING TESTS to update:
    - tests/test_entry.py: test_qwen_detected_but_unsupported (line 189)
      must become test_qwen_detected_and_supported.
    - tests/test_layer_type_classify.py: assertions that qwen returns
      None must become positive classification tests.
    - tests/test_merge_block_config.py: assertion that
      get_block_classifier("qwen") is None must test real classifier.
    - tests/test_lora_loaders.py: assertion that get_loader("qwen")
      raises ValueError must test real loader (covered by lora task).

    New tests in: tests/test_entry.py, tests/test_layer_type_classify.py,
    tests/test_merge_block_config.py.
    Covers ac-1, ac-2, ac-3.
  tags:
    - qwen
    - classification

- title: Implement Qwen LoRA loader
  slug: qwen-lora-loader
  spec_ref: "@qwen-support"
  priority: 2
  depends_on:
    - "@qwen-detect-classify"
  description: |
    Create lib/lora/qwen.py implementing QwenLoader (subclass LoRALoader).
    Handle 3 LoRA formats -- diffusers, A1111/kohya, LyCORIS -- with
    compound name preservation from merge-router reference
    (qwen_merge.py lines 254-283). No QKV fusion needed (separate
    to_q/to_k/to_v). Standard up/down DeltaSpec production.
    Register in LOADER_REGISTRY in lib/lora/__init__.py.

    Update tests/test_lora_loaders.py: replace ValueError assertion for
    get_loader("qwen") with real loader tests. Add new tests with
    synthetic safetensors LoRA files for each format.
    Covers ac-4, ac-5, ac-6.
  tags:
    - qwen
    - lora

- title: Implement Qwen model loader support
  slug: qwen-model-loader
  spec_ref: "@qwen-support"
  priority: 2
  description: |
    Add Qwen architecture detection pattern to _ARCH_PATTERNS in
    lib/model_loader.py (match transformer_blocks, distinguish from
    Flux double_blocks). Add Qwen-specific entries to _FILE_KEY_PREFIXES
    (e.g. model.transformer.) and _EXCLUDED_PREFIXES if needed.
    Update _normalize_key() for any Qwen-specific prefix stripping.

    Tests in tests/test_model_loader.py: add Qwen detection and
    key normalization tests.
    Covers ac-7.
  tags:
    - qwen
    - model-loader

- title: Implement Qwen block config node and registration
  slug: qwen-block-config
  spec_ref: "@qwen-support"
  priority: 2
  depends_on:
    - "@qwen-detect-classify"
    - "@qwen-lora-loader"
    - "@qwen-model-loader"
  description: |
    Create nodes/block_config_qwen.py using make_block_config_node()
    factory with 60 block definitions (TB00-TB59) and 3 layer-type
    sliders. Register WIDENBlockConfigQwen in NODE_CLASS_MAPPINGS and
    NODE_DISPLAY_NAME_MAPPINGS in __init__.py. Verify all registries
    are wired (ac-9). Add integration test exercising mock Qwen recipe
    through the full pipeline.

    Tests in: tests/test_per_block_control.py (new Qwen node tests).
    Covers ac-8, ac-9.
  tags:
    - qwen
    - node

- title: Implement Flux Klein detection and block classification
  slug: flux-detect-classify
  spec_ref: "@flux-klein-support"
  priority: 2
  description: |
    Enable Flux detection in nodes/entry.py (add "flux" to
    _SUPPORTED_ARCHITECTURES). Add classify_key_flux() in
    lib/block_classify.py mapping double_blocks.N to DB0N and
    single_blocks.N to SB0N with dynamic index discovery.
    Add Flux layer type patterns. Register in _CLASSIFIERS and
    _LAYER_TYPE_PATTERNS. Update __all__ exports.

    BREAKING TESTS to update:
    - tests/test_entry.py: test_flux_detected_but_unsupported (line 174)
      must become test_flux_detected_and_supported.
    - tests/test_layer_type_classify.py: assertions that flux returns
      None must become positive classification tests.
    - tests/test_merge_block_config.py: assertion that
      get_block_classifier("flux") is None must test real classifier.

    New tests in: tests/test_entry.py, tests/test_layer_type_classify.py,
    tests/test_merge_block_config.py.
    Covers ac-1, ac-2, ac-3.
  tags:
    - flux
    - classification

- title: Implement Flux Klein LoRA loader
  slug: flux-lora-loader
  spec_ref: "@flux-klein-support"
  priority: 2
  depends_on:
    - "@flux-detect-classify"
  description: |
    Create lib/lora/flux.py implementing FluxLoader (subclass LoRALoader).
    Handle double_block QKV fusing (img_attn.qkv and txt_attn.qkv per
    block, two streams) using existing qkv_q/qkv_k/qkv_v DeltaSpec kinds
    with offset=(start, length).

    Handle single_block linear1 fusing (4-way offset split for
    to_q/to_k/to_v/proj_mlp). Q/K/V components reuse qkv_q/qkv_k/qkv_v
    kinds. The MLP projection slice needs a new DeltaSpec kind (e.g.
    "offset_mlp") and a minor executor update in lib/gpu_ops.py to add
    it to the offset-aware kind set at line 200. This is a one-line
    change to extend the tuple.

    Follow Z-Image offset-based DeltaSpec pattern from lib/lora/zimage.py.
    Support both BFL/kohya and diffusers LoRA formats.
    Register in LOADER_REGISTRY.

    Update tests/test_lora_loaders.py: replace ValueError assertion for
    get_loader("flux") with real loader tests. Add synthetic safetensors
    LoRA files testing both double_block QKV and single_block linear1.
    Covers ac-4, ac-5, ac-6, ac-7.
  tags:
    - flux
    - lora

- title: Implement Flux Klein model loader support
  slug: flux-model-loader
  spec_ref: "@flux-klein-support"
  priority: 2
  description: |
    Add Flux architecture detection pattern to _ARCH_PATTERNS in
    lib/model_loader.py (match double_blocks). Add Flux-specific entries
    to _FILE_KEY_PREFIXES and _EXCLUDED_PREFIXES. Update _normalize_key()
    for Flux-specific prefix stripping.

    Tests in tests/test_model_loader.py: add Flux detection and
    key normalization tests.
    Covers ac-8.
  tags:
    - flux
    - model-loader

- title: Implement Flux Klein block config node and registration
  slug: flux-block-config
  spec_ref: "@flux-klein-support"
  priority: 2
  depends_on:
    - "@flux-detect-classify"
    - "@flux-lora-loader"
    - "@flux-model-loader"
  description: |
    Create nodes/block_config_flux.py using make_block_config_node()
    factory with 32 block definitions for Klein 9B max (DB00-DB07 +
    SB00-SB23) and 3 layer-type sliders. Register WIDENBlockConfigFlux
    in NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS in __init__.py.
    Verify both 4B and 9B variants work with same "flux" arch tag (ac-10).
    Verify all registries wired (ac-11). Add integration test.

    Tests in: tests/test_per_block_control.py (new Flux node tests).
    Covers ac-9, ac-10, ac-11.
  tags:
    - flux
    - node
```

## Implementation Notes

Implementation order: Qwen first (simpler, no QKV fusion, has reference impl in
../merge-router), then Flux Klein (complex QKV fusing in both block types).

Within each architecture, 3 tasks can run in parallel (detect-classify, model-loader,
and then lora-loader + block-config after classify completes).

Key reference files:
- Qwen reference: ../merge-router/src/models/qwen_key_mapper.py (compound names)
- Qwen reference: ../merge-router/scripts/qwen_merge.py (LoRA formats, lines 90-308)
- QKV fusion pattern: lib/lora/zimage.py (lines 34-44, _QKV_OFFSETS)
- Block config factory: nodes/block_config.py (make_block_config_node)
- LoRA base class: lib/lora/base.py (LoRALoader abstract)
- Current detection: nodes/entry.py (detect_architecture, _SUPPORTED_ARCHITECTURES)

Existing architecture patterns (SDXL + Z-Image) serve as templates. Each new
architecture adds: one classifier function, one set of layer type patterns, one
LoRA loader module, one block config node module, and registry entries.

Executor change for Flux linear1: lib/gpu_ops.py line 200 currently handles
qkv_q/qkv_k/qkv_v kinds with offset-based slicing. Flux single_block linear1
4-way split reuses qkv_* for Q/K/V and adds one new kind (e.g. "offset_mlp")
to the same elif branch. This is a one-line tuple extension, not a new code
path. The offset=(start, length) mechanism is already general-purpose.

No changes needed to: lib/analysis.py (dispatches via registry), lib/per_block.py
(dispatches via classify_key/classify_layer_type registries), nodes/exit.py
(architecture-agnostic), lib/persistence.py (recipe serialization is
architecture-agnostic).
