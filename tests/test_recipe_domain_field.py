"""Tests for Recipe Domain Field feature.

Covers all 9 acceptance criteria:
- AC-1: RecipeBase has domain field with default "diffusion"
- AC-2: Pre-domain recipe trees default to "diffusion"
- AC-3: analyze_recipe dispatches on (arch, domain) for LoRA loader
- AC-4: analyze_recipe_models dispatches on (arch, domain) for model loader
- AC-5: get_loader returns CLIP LoRA loader for domain="clip"
- AC-6: get_loader returns UNet LoRA loader for domain="diffusion"
- AC-7: serialize_recipe includes domain in JSON
- AC-8: classify_key dispatches to CLIP classifier for domain="clip"
- AC-9: classify_key dispatches to UNet classifier for domain="diffusion"
"""

import dataclasses
import json
import tempfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from lib.analysis import AnalysisResult, analyze_recipe, analyze_recipe_models
from lib.block_classify import classify_key, filter_changed_keys
from lib.lora import LOADER_REGISTRY, LoRALoader, SDXLLoader, get_loader
from lib.persistence import serialize_recipe
from lib.recipe import RecipeBase, RecipeLoRA, RecipeMerge

# ---------------------------------------------------------------------------
# AC-1: RecipeBase has domain field with default "diffusion"
# ---------------------------------------------------------------------------


class TestAC1DomainFieldDefault:
    """AC-1: RecipeBase has a domain string field with default 'diffusion'."""

    def test_recipe_base_has_domain_field(self):
        """RecipeBase dataclass has a domain field."""
        # AC: @recipe-domain-field ac-1
        fields = {f.name for f in dataclasses.fields(RecipeBase)}
        assert "domain" in fields

    def test_recipe_base_domain_defaults_to_diffusion(self):
        """RecipeBase.domain defaults to 'diffusion'."""
        # AC: @recipe-domain-field ac-1
        base = RecipeBase(model_patcher=object(), arch="sdxl")
        assert base.domain == "diffusion"

    def test_recipe_base_domain_can_be_clip(self):
        """RecipeBase.domain can be set to 'clip'."""
        # AC: @recipe-domain-field ac-1
        base = RecipeBase(model_patcher=object(), arch="sdxl", domain="clip")
        assert base.domain == "clip"

    def test_recipe_base_domain_is_string(self):
        """RecipeBase.domain is a string type."""
        # AC: @recipe-domain-field ac-1
        base = RecipeBase(model_patcher=object(), arch="sdxl")
        assert isinstance(base.domain, str)


# ---------------------------------------------------------------------------
# AC-2: Pre-domain recipe trees default to "diffusion"
# ---------------------------------------------------------------------------


class TestAC2BackwardCompatibility:
    """AC-2: Existing recipe trees default to 'diffusion'."""

    def test_existing_recipe_base_gets_default_domain(self):
        """RecipeBase created without domain field gets 'diffusion' default."""
        # AC: @recipe-domain-field ac-2
        # Simulate pre-domain code creating RecipeBase without domain kwarg
        base = RecipeBase(model_patcher=object(), arch="sdxl")
        assert base.domain == "diffusion"

    def test_recipe_merge_with_old_style_base(self):
        """RecipeMerge with base lacking explicit domain uses 'diffusion'."""
        # AC: @recipe-domain-field ac-2
        base = RecipeBase(model_patcher=object(), arch="sdxl")
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)
        assert merge.base.domain == "diffusion"


# ---------------------------------------------------------------------------
# AC-3: analyze_recipe dispatches on (arch, domain) for LoRA loader
# ---------------------------------------------------------------------------


class TestAC3AnalyzeRecipeLoRADispatch:
    """AC-3: analyze_recipe dispatches LoRA loader selection on (arch, domain)."""

    @pytest.fixture
    def sdxl_lora_file(self):
        """Create a temporary SDXL LoRA file."""
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            tensors = {
                "lora_unet_input_blocks_0_0_proj_in.lora_up.weight": torch.randn(64, 8),
                "lora_unet_input_blocks_0_0_proj_in.lora_down.weight": torch.randn(8, 32),
            }
            save_file(tensors, f.name)
            yield f.name
        Path(f.name).unlink(missing_ok=True)

    def test_analyze_recipe_extracts_domain(self, sdxl_lora_file: str):
        """analyze_recipe extracts domain from RecipeBase."""
        # AC: @recipe-domain-field ac-3
        base = RecipeBase(model_patcher=object(), arch="sdxl", domain="diffusion")
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        result = analyze_recipe(merge, lora_path_resolver=lambda _: sdxl_lora_file)

        assert result.domain == "diffusion"

    def test_analyze_recipe_result_includes_domain(self, sdxl_lora_file: str):
        """AnalysisResult includes domain field."""
        # AC: @recipe-domain-field ac-3
        assert hasattr(AnalysisResult, "__dataclass_fields__")
        fields = {f.name for f in dataclasses.fields(AnalysisResult)}
        assert "domain" in fields

    def test_analyze_recipe_uses_sdxl_loader_for_diffusion(self, sdxl_lora_file: str):
        """analyze_recipe uses SDXL loader for arch=sdxl, domain=diffusion."""
        # AC: @recipe-domain-field ac-3
        base = RecipeBase(model_patcher=object(), arch="sdxl", domain="diffusion")
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        result = analyze_recipe(merge, lora_path_resolver=lambda _: sdxl_lora_file)

        # Should use SDXLLoader (diffusion domain)
        assert isinstance(result.loader, SDXLLoader)
        result.loader.cleanup()


# ---------------------------------------------------------------------------
# AC-4: analyze_recipe_models dispatches on (arch, domain) for model loader
# ---------------------------------------------------------------------------


class TestAC4AnalyzeRecipeModelsDispatch:
    """AC-4: analyze_recipe_models dispatches model loader on (arch, domain)."""

    def test_analyze_recipe_models_accepts_domain_param(self):
        """analyze_recipe_models accepts domain parameter."""
        # AC: @recipe-domain-field ac-4
        import inspect

        sig = inspect.signature(analyze_recipe_models)
        params = list(sig.parameters.keys())
        assert "domain" in params

    def test_analyze_recipe_models_domain_defaults_to_diffusion(self):
        """analyze_recipe_models domain parameter defaults to 'diffusion'."""
        # AC: @recipe-domain-field ac-4
        import inspect

        sig = inspect.signature(analyze_recipe_models)
        domain_param = sig.parameters["domain"]
        assert domain_param.default == "diffusion"


# ---------------------------------------------------------------------------
# AC-5: get_loader returns CLIP LoRA loader for domain="clip"
# ---------------------------------------------------------------------------


class TestAC5GetLoaderCLIPDispatch:
    """AC-5: get_loader returns CLIP LoRA loader for domain='clip'."""

    def test_get_loader_raises_for_missing_clip_loader(self):
        """get_loader raises ValueError when CLIP loader not registered for arch."""
        # AC: @recipe-domain-field ac-5
        # zimage does not have a CLIP loader
        with pytest.raises(ValueError, match="No CLIP LoRA loader"):
            get_loader("zimage", domain="clip")

    def test_get_loader_clip_looks_for_arch_clip_key(self):
        """get_loader looks for '{arch}_clip' key in registry for CLIP domain."""
        # AC: @recipe-domain-field ac-5
        # Add a mock CLIP loader to verify dispatch
        class MockCLIPLoader(LoRALoader):
            def load(self, path, strength=1.0, set_id=None):
                pass

            @property
            def affected_keys(self):
                return set()

            def affected_keys_for_set(self, set_id):
                return set()

            def get_delta_specs(self, keys, key_indices, set_id=None):
                return []

            @property
            def loaded_bytes(self):
                return 0

            def cleanup(self):
                pass

        original = LOADER_REGISTRY.copy()
        LOADER_REGISTRY["sdxl_clip"] = MockCLIPLoader

        try:
            loader = get_loader("sdxl", domain="clip")
            assert isinstance(loader, MockCLIPLoader)
        finally:
            LOADER_REGISTRY.clear()
            LOADER_REGISTRY.update(original)


# ---------------------------------------------------------------------------
# AC-6: get_loader returns UNet LoRA loader for domain="diffusion"
# ---------------------------------------------------------------------------


class TestAC6GetLoaderDiffusionBackwardCompat:
    """AC-6: get_loader returns existing UNet loader for domain='diffusion'."""

    def test_get_loader_returns_sdxl_for_diffusion_domain(self):
        """get_loader returns SDXLLoader for arch=sdxl, domain=diffusion."""
        # AC: @recipe-domain-field ac-6
        loader = get_loader("sdxl", domain="diffusion")
        assert isinstance(loader, SDXLLoader)

    def test_get_loader_diffusion_is_default(self):
        """get_loader defaults to domain='diffusion' for backward compat."""
        # AC: @recipe-domain-field ac-6
        import inspect

        sig = inspect.signature(get_loader)
        domain_param = sig.parameters["domain"]
        assert domain_param.default == "diffusion"

    def test_get_loader_no_domain_uses_unet_loader(self):
        """get_loader without domain param uses UNet loader (backward compat)."""
        # AC: @recipe-domain-field ac-6
        loader = get_loader("sdxl")  # No domain specified
        assert isinstance(loader, SDXLLoader)


# ---------------------------------------------------------------------------
# AC-7: serialize_recipe includes domain in JSON
# ---------------------------------------------------------------------------


class TestAC7SerializeRecipeDomain:
    """AC-7: serialize_recipe includes domain in JSON output."""

    def test_serialize_recipe_includes_domain_in_base(self):
        """serialize_recipe includes domain field in RecipeBase JSON."""
        # AC: @recipe-domain-field ac-7
        base = RecipeBase(model_patcher=object(), arch="sdxl", domain="diffusion")
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        serialized = serialize_recipe(merge, "test_identity", {})
        data = json.loads(serialized)

        # Walk to find RecipeBase
        base_data = data["base"]
        assert "domain" in base_data
        assert base_data["domain"] == "diffusion"

    def test_serialize_recipe_domain_clip(self):
        """serialize_recipe preserves domain='clip' in JSON."""
        # AC: @recipe-domain-field ac-7
        base = RecipeBase(model_patcher=object(), arch="sdxl", domain="clip")
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        serialized = serialize_recipe(merge, "test_identity", {})
        data = json.loads(serialized)

        base_data = data["base"]
        assert base_data["domain"] == "clip"

    def test_serialize_recipe_domain_affects_hash(self):
        """Different domain values produce different recipe hashes."""
        # AC: @recipe-domain-field ac-7
        from lib.persistence import compute_recipe_hash

        base_diff = RecipeBase(model_patcher=object(), arch="sdxl", domain="diffusion")
        base_clip = RecipeBase(model_patcher=object(), arch="sdxl", domain="clip")
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))

        merge_diff = RecipeMerge(base=base_diff, target=lora, backbone=None, t_factor=1.0)
        merge_clip = RecipeMerge(base=base_clip, target=lora, backbone=None, t_factor=1.0)

        hash_diff = compute_recipe_hash(serialize_recipe(merge_diff, "id", {}))
        hash_clip = compute_recipe_hash(serialize_recipe(merge_clip, "id", {}))

        assert hash_diff != hash_clip


# ---------------------------------------------------------------------------
# AC-8: classify_key dispatches to CLIP classifier for domain="clip"
# ---------------------------------------------------------------------------


class TestAC8ClassifyKeyCLIPDispatch:
    """AC-8: classify_key dispatches to CLIP classifier for domain='clip'."""

    def test_classify_key_accepts_domain_param(self):
        """classify_key accepts domain parameter."""
        # AC: @recipe-domain-field ac-8
        import inspect

        sig = inspect.signature(classify_key)
        params = list(sig.parameters.keys())
        assert "domain" in params

    def test_classify_key_clip_domain_classifies_sdxl_clip_keys(self):
        """classify_key returns block group for CLIP domain with registered classifier."""
        # AC: @recipe-domain-field ac-8
        # SDXL CLIP classifier is registered as "sdxl_clip"
        clip_key = "clip_l.transformer.text_model.encoder.layers.0.weight"
        result = classify_key(clip_key, "sdxl", "clip")
        assert result == "CL00"

    def test_classify_key_clip_returns_none_for_unknown_arch(self):
        """classify_key returns None for CLIP domain with unknown architecture."""
        # AC: @recipe-domain-field ac-8
        # No "unknown_clip" classifier exists
        clip_key = "clip_l.transformer.text_model.encoder.layers.0.weight"
        result = classify_key(clip_key, "unknown", "clip")
        assert result is None

    def test_classify_key_clip_uses_arch_clip_key(self):
        """classify_key looks for '{arch}_clip' classifier for CLIP domain."""
        # AC: @recipe-domain-field ac-8
        from lib.block_classify import _CLASSIFIERS

        # Add a mock CLIP classifier
        def mock_clip_classifier(key: str) -> str | None:
            if "clip_l" in key:
                return "CL00"
            return None

        original = _CLASSIFIERS.copy()
        _CLASSIFIERS["sdxl_clip"] = mock_clip_classifier

        # Clear the LRU cache to pick up the new classifier
        classify_key.cache_clear()

        try:
            clip_key = "clip_l.transformer.text_model.encoder.layers.0.weight"
            result = classify_key(clip_key, "sdxl", "clip")
            assert result == "CL00"
        finally:
            _CLASSIFIERS.clear()
            _CLASSIFIERS.update(original)
            classify_key.cache_clear()  # Clear again after restoring


# ---------------------------------------------------------------------------
# AC-9: classify_key dispatches to UNet classifier for domain="diffusion"
# ---------------------------------------------------------------------------


class TestAC9ClassifyKeyDiffusionBackwardCompat:
    """AC-9: classify_key dispatches to existing UNet classifier for domain='diffusion'."""

    def test_classify_key_diffusion_uses_existing_classifier(self):
        """classify_key uses existing SDXL classifier for domain='diffusion'."""
        # AC: @recipe-domain-field ac-9
        result = classify_key("diffusion_model.input_blocks.0.0.weight", "sdxl", "diffusion")
        assert result == "IN00"

    def test_classify_key_diffusion_is_default(self):
        """classify_key defaults to domain='diffusion' for backward compat."""
        # AC: @recipe-domain-field ac-9
        import inspect

        sig = inspect.signature(classify_key)
        domain_param = sig.parameters["domain"]
        assert domain_param.default == "diffusion"

    def test_classify_key_no_domain_uses_unet_classifier(self):
        """classify_key without domain param uses UNet classifier."""
        # AC: @recipe-domain-field ac-9
        # This is the existing behavior that must be preserved
        result = classify_key("diffusion_model.input_blocks.0.0.weight", "sdxl")
        assert result == "IN00"

    def test_filter_changed_keys_accepts_domain(self):
        """filter_changed_keys accepts domain parameter."""
        # Related to AC-8, AC-9 - filter_changed_keys should pass domain through
        keys = {"diffusion_model.input_blocks.0.0.weight"}
        changed_blocks = {"IN00"}
        changed_layer_types: set[str] = set()

        result = filter_changed_keys(
            keys, changed_blocks, changed_layer_types, "sdxl", "diffusion"
        )
        assert result == keys


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestDomainFieldIntegration:
    """Integration tests for domain field across the system."""

    @pytest.fixture
    def sdxl_lora_file(self):
        """Create a temporary SDXL LoRA file."""
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            tensors = {
                "lora_unet_input_blocks_0_0_proj_in.lora_up.weight": torch.randn(64, 8),
                "lora_unet_input_blocks_0_0_proj_in.lora_down.weight": torch.randn(8, 32),
            }
            save_file(tensors, f.name)
            yield f.name
        Path(f.name).unlink(missing_ok=True)

    def test_full_diffusion_workflow(self, sdxl_lora_file: str):
        """Full workflow with domain='diffusion' works end-to-end."""
        # Create recipe with explicit diffusion domain
        base = RecipeBase(model_patcher=object(), arch="sdxl", domain="diffusion")
        lora = RecipeLoRA(loras=({"path": "test.safetensors", "strength": 1.0},))
        merge = RecipeMerge(base=base, target=lora, backbone=None, t_factor=1.0)

        # Analyze recipe
        result = analyze_recipe(merge, lora_path_resolver=lambda _: sdxl_lora_file)

        # Verify domain is preserved
        assert result.domain == "diffusion"

        # Serialize and verify domain in output
        serialized = serialize_recipe(merge, "test_identity", {})
        data = json.loads(serialized)
        assert data["base"]["domain"] == "diffusion"

        # Classify key with domain
        key_result = classify_key("diffusion_model.input_blocks.0.0.weight", "sdxl", "diffusion")
        assert key_result == "IN00"

        result.loader.cleanup()

    def test_domain_propagates_through_nested_merge(self, sdxl_lora_file: str):
        """Domain from RecipeBase propagates through nested RecipeMerge."""
        base = RecipeBase(model_patcher=object(), arch="sdxl", domain="clip")
        lora1 = RecipeLoRA(loras=({"path": "lora1.safetensors", "strength": 1.0},))
        lora2 = RecipeLoRA(loras=({"path": "lora2.safetensors", "strength": 0.5},))

        merge1 = RecipeMerge(base=base, target=lora1, backbone=None, t_factor=1.0)
        merge2 = RecipeMerge(base=merge1, target=lora2, backbone=None, t_factor=0.7)

        # Walk to base should find the original with domain="clip"
        from lib.analysis import walk_to_base

        found_base = walk_to_base(merge2)
        assert found_base.domain == "clip"
